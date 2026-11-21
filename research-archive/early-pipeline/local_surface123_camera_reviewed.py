"""Fixed-K candidate geometry consuming frozen reviewed native observations.

The legacy geometry body is preserved below. No detector or matcher exists in
this entry. A candidate is not physical calibration or permission to fit SMPL-X.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from local_surface123 import PROJECT,status as state,now
from local_surface123_frozen_observations import solver_tracks,ObservationContractError,sha
import sys
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt

def camera(root,bundle):
    train,held=solver_tracks(bundle)
    # The native input was validated before preparing a new run. Bind it again
    # to the copied decoded roster and image bytes before solver imports/calls.
    root=Path(root)
    bindings={k:h for k,p,h in bundle.bound_files}
    if sha(root/'decoded.json')!=bindings['decoded']:
        raise ObservationContractError('solver_decoded_roster_changed')
    decoded=rt.read(root/'decoded.json')['records'];n=len(decoded)
    if root.name!=bundle.subject_id:
        raise ObservationContractError('solver_subject_changed')
    for row,frame in zip(decoded,bundle.frames):
        if sha(root/'raw'/row['name'])!=frame['image_sha256']:
            raise ObservationContractError('solver_native_image_changed')
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.sparse import lil_matrix
    start=time.monotonic();out=root/'camera';subject=root.name
    K=np.asarray(rt.read(root/'intrinsic-contract.json')['raw_K_integer'])
    xy=bundle.xy
    tracks=[dict(r['nodes']) for r in train]
    def status(stage,**kw):
        if time.monotonic()-start>1800:raise RuntimeError('1800-second camera budget exceeded.')
        state(subject,state=stage,run_root=str(root),training_running=False,cameras_total=n,seconds=time.monotonic()-start,**kw)
    def project(E,X):
        q=X@E[:3,:3].T+E[:3,3];uv=q@K.T
        return uv[:,:2]/uv[:,2,None],q[:,2]
    def summary(e):
        a=np.asarray(e);return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(count=0)
    rt.save(out/'frozen-observation-audit.json',bundle.audit())
    rt.save(out/'plan.json',dict(created=now(),source_sha256=sha(Path(__file__)),pixels='integer full native decoded pixels',
        frame_count=n,observation_manifest_sha256=bundle.manifest_sha256,source_video_sha256=bundle.source_video_sha256,
        detector_called=False,matcher_called=False,original_partitions_preserved=True,
        source_segment_ids=[r['id'] for r in train],source_fit_families=sorted({r['family_id'] for r in train}),
        minimum_seed_direct_matches=160,seed_cheirality=80,min_pnp_inliers=50,min_BA_obs_per_camera=50,
        ransac_px=1.5,pnp_px=3,triangulation_px=3,min_parallax_degrees=1,max_BA_points=5000,max_BA_nfev=60,
        time_budget_seconds=1800,K_fixed=True,old_poses_loaded=False,training_started=False))
    cv2.setNumThreads(6);cv2.setRNGSeed(118)
    if len(held)<30:raise RuntimeError('Fewer than30 original whole-track holdouts; no geometry solve.')
    seed_pairs=[]
    for j in [1,2,4,8,16]:
        count=sum(0 in t and j in t for t in tracks)
        if j<n and count>=160:seed_pairs.append((j,count))
    if not seed_pairs:raise RuntimeError('No sufficiently supported fresh seed pair.')
    seed_j,seed_matches=max(seed_pairs)
    seed=[k for k,t in enumerate(tracks) if 0 in t and seed_j in t]
    a=np.array([xy[0][tracks[k][0]] for k in seed]);b=np.array([xy[seed_j][tracks[k][seed_j]] for k in seed])
    norm=lambda v:cv2.undistortPoints(v.reshape(-1,1,2),K,None).reshape(-1,2)
    essential,mask=cv2.findEssentialMat(norm(a),norm(b),np.eye(3),method=cv2.RANSAC,prob=.999,threshold=1.5/np.mean([K[0,0],K[1,1]]),maxIters=10000)
    if essential is None or essential.shape!=(3,3):raise RuntimeError('One essential matrix not found.')
    count,R,t,mask=cv2.recoverPose(essential,norm(a),norm(b),np.eye(3),mask=mask)
    rt.save(out/'seed.json',dict(pair=[0,seed_j],appearance_matches=seed_matches,cheirality=int(count)))
    if count<80:raise RuntimeError(f'Seed cheirality {count}<80; no threshold retry.')
    E=np.eye(4);E[:3,:3]=R;E[:3,3]=t[:,0];matrices={0:np.eye(4),seed_j:E};points={}
    def triangulate():
        centers={i:-e[:3,:3].T@e[:3,3] for i,e in matrices.items()};groups={}
        for ti,tr in enumerate(tracks):
            if ti in points:continue
            ids=sorted(set(tr)&matrices.keys())
            if len(ids)<2:continue
            _,i,j=max((np.linalg.norm(centers[i]-centers[j]),i,j) for z,i in enumerate(ids) for j in ids[z+1:])
            groups.setdefault((i,j),[]).append(ti)
        for (i,j),ids in groups.items():
            u=np.array([xy[i][tracks[k][i]] for k in ids]);v=np.array([xy[j][tracks[k][j]] for k in ids])
            X=cv2.triangulatePoints(K@matrices[i][:3],K@matrices[j][:3],u.T,v.T);X=(X[:3]/X[3]).T
            p,zi=project(matrices[i],X);q,zj=project(matrices[j],X)
            ra=X-centers[i];rb=X-centers[j];angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(ra,rb),axis=1),np.einsum('ij,ij->i',ra,rb)))
            good=np.isfinite(X).all(1)&(zi>0)&(zj>0)&(np.linalg.norm(p-u,axis=1)<3)&(np.linalg.norm(q-v,axis=1)<3)&(angle>1)
            for ti,x in zip(np.asarray(ids)[good],X[good]):points[int(ti)]=x
    triangulate();status('fresh_camera_seed',registered=2,points=len(points))
    while len(matrices)<n:
        candidates=[]
        for i in range(n):
            if i not in matrices:
                ids=[k for k in points if i in tracks[k]];candidates.append((len(ids),i,ids))
        done=0
        for _,i,ids in sorted(candidates,reverse=True):
            if len(ids)<50:continue
            X=np.array([points[k] for k in ids]);u=np.array([xy[i][tracks[k][i]] for k in ids])
            ok,r,t,ii=cv2.solvePnPRansac(X,u,K,None,iterationsCount=2000,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if not ok or ii is None or len(ii)<50:continue
            ii=ii[:,0];r,t=cv2.solvePnPRefineLM(X[ii],u[ii],K,None,r,t)
            e=np.eye(4);e[:3,:3]=cv2.Rodrigues(r)[0];e[:3,3]=t[:,0];matrices[i]=e;done+=1
        triangulate();status('fresh_PnP_registration',registered=len(matrices),points=len(points))
        if not done:raise RuntimeError('Missing real camera poses: '+str(sorted(set(range(n))-matrices.keys())))
    verified={}
    for k,X in points.items():
        kept=[]
        for i,j in tracks[k].items():
            p,z=project(matrices[i],X[None])
            if np.isfinite(p).all() and z[0]>0 and np.linalg.norm(p[0]-xy[i][j])<3:kept.append((i,j))
        if len(kept)>=3:verified[k]=kept
    selected=sorted(verified,key=lambda k:(-len(verified[k]),k))[:5000]
    ci=[];pi=[];uv=[]
    for p,k in enumerate(selected):
        for i,j in verified[k]:ci.append(i);pi.append(p);uv.append(xy[i][j])
    ci,pi,uv=np.array(ci),np.array(pi),np.array(uv)
    if len(uv)==0 or np.bincount(ci,minlength=n).min()<50:raise RuntimeError('Insufficient verified fitting observations per real camera.')
    cp=np.array([np.r_[Rotation.from_matrix(matrices[i][:3,:3]).as_rotvec(),matrices[i][:3,3]] for i in range(n)])
    free=np.ones((n,6),bool);free[0]=False;free[seed_j,3+np.argmax(abs(cp[seed_j,3:]))]=False
    cols=np.full((n,6),-1,int);cols[free]=np.arange(free.sum());nc=int(free.sum())
    initial=np.r_[cp[free],np.array([points[k] for k in selected]).ravel()]
    sparsity=lil_matrix((2*len(uv),len(initial)),dtype=np.int8);rr=np.arange(len(uv))
    for k in range(6):
        valid=cols[ci,k]>=0;sparsity[2*rr[valid],cols[ci[valid],k]]=1;sparsity[2*rr[valid]+1,cols[ci[valid],k]]=1
    for k in range(3):sparsity[2*rr,nc+3*pi+k]=1;sparsity[2*rr+1,nc+3*pi+k]=1
    calls=0
    def residual(x):
        nonlocal calls
        calls+=1
        if time.monotonic()-start>1800:raise RuntimeError('Camera BA time budget exceeded.')
        p=cp.copy();p[free]=x[:nc];R=Rotation.from_rotvec(p[:,:3]).as_matrix();X=x[nc:].reshape(-1,3)
        q=np.einsum('nij,nj->ni',R[ci],X[pi])+p[ci,3:];z=np.maximum(q[:,2],1e-6)
        pred=np.c_[K[0,0]*q[:,0]/z+K[0,2],K[1,1]*q[:,1]/z+K[1,2]]
        if calls%200==0:status('fixed_K_BA',residual_calls=calls)
        return (pred-uv).ravel()
    before=summary(np.linalg.norm(residual(initial).reshape(-1,2),axis=1))
    sol=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=60,ftol=1e-5)
    p=cp.copy();p[free]=sol.x[:nc];Es=np.tile(np.eye(4),(n,1,1));Es[:,:3,:3]=Rotation.from_rotvec(p[:,:3]).as_matrix();Es[:,:3,3]=p[:,3:]
    if not np.isfinite(Es).all():raise RuntimeError('Nonfinite camera candidate.')
    hold_results=[]
    # Fit 3D on alternate reserved observations, evaluate remaining observations.
    # No camera or K optimization; no held observations excluded for error.
    for tr in held:
        nodes=tr['nodes'];anchors=nodes[::2];checks=nodes[1::2]
        if len(anchors)<2:continue
        A=[]
        for i,j in anchors:
            P=K@Es[i,:3];u,v=xy[i][j];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
        _,_,v=np.linalg.svd(A);X=v[-1,:3]/v[-1,3]
        errors=[];depths=[]
        for i,j in checks:
            q,z=project(Es[i],X[None]);errors.append(float(np.linalg.norm(q[0]-xy[i][j])));depths.append(float(z[0]))
        hold_results.append(dict(id=tr['id'],nodes=nodes,anchor_frames=[i for i,j in anchors],test_frames=[i for i,j in checks],
            errors=errors,positive_depth=all(z>0 for z in depths),point=X.tolist()))
    rt.save(out/'held-evaluation.json',dict(records=hold_results,all_errors=summary([e for t in hold_results for e in t['errors']]),
        whole_tracks_reserved=True,landmarks_are_not_material_truth=True,physical_identity_reviewed=False,formal_calibration_accepted=False))
    records=[];Kc=K.copy();Kc[:2,2]+=.5
    for i,d in enumerate(decoded):
        records.append(dict(**d,image=str(root/'raw'/d['name']),K=Kc.tolist(),K_integer_centers=K.tolist(),w2c=Es[i].tolist(),
            mask=None,mask_sha256=None,official_prediction_slot=i,camera_kind='fixed_factory_K_opencv_research',
            camera_registration=dict(interpolation_used=False),observation_weight=1.))
    manifest=dict(records=records,pixel_domain='original decoded exported video, no lens remap',
        coordinate_contract='K stores pixel corners; detector uses integer centers; CUDA subtracts0.5',
        formal_calibration_accepted=False,from_scratch=True,camera_source='Fresh fixed-K OpenCV/scipy; no COLMAP pose estimation')
    rt.save(root/'data/inputs.json',manifest);rt.save(out/'candidate-cameras.json',manifest)
    np.savez_compressed(out/'solver-output.npz',K=K,w2c=Es,points=sol.x[nc:].reshape(-1,3),obs_cam=ci,obs_point=pi,observations=uv,track_ids=np.array([train[k]['id'] for k in selected]))
    rt.save(out/'result.json',dict(state='candidate_review_pending',frames=n,before=before,
        fitting_errors=summary(np.linalg.norm(residual(sol.x).reshape(-1,2),axis=1)),optimizer_success=bool(sol.success),nfev=sol.nfev,
        held_track_count=len(held),held_test_count=len(hold_results),independent_physical_accuracy_proven=False,formal_training_admitted=False))
    status('camera_candidate_ready_for_review',registered=n)

