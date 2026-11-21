"""One bounded fixed-factory-K/OpenCV camera candidate; no old 3-D/pose initialization."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sqlite3, time, traceback
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

BASE=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
OUT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-fixedk-opencv-seedrepair')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
START=time.monotonic()
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def progress(stage,**kw):
    if time.monotonic()-START>900:raise RuntimeError('900 second camera candidate budget reached.')
    record=dict(updated=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,elapsed_seconds=time.monotonic()-START,**kw)
    save(OUT/'progress.json',record);print(json.dumps(record),flush=True)
def summary(x):
    a=np.asarray(x,float)
    return dict(n=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if a.size else dict(n=0)
def project(E,X,K):
    q=X@E[:3,:3].T+E[:3,3];uv=q@K.T
    return uv[:,:2]/uv[:,2,None],q[:,2]

def main():
    OUT.mkdir(exist_ok=False)
    decoded=read(BASE/'decoded.json')['records'];n=len(decoded);assert n==89
    init=read(BASE/'camera-initialization.json');factory=read(REC/'protocol.json')['calibration']
    K=np.asarray(factory['K'],float);x0,y0,cw,ch=init['center_crop_xywh'];w,h=decoded[0]['wh']
    sx,sy=w/cw,h/ch
    A=np.array([[sx,0,sx*(.5-x0)-.5],[0,sy,sy*(.5-y0)-.5],[0,0,1.]])
    K=A@K
    plan=dict(created=datetime.now(timezone.utc).isoformat(),kind='one_bounded_calibration_candidate',
      hypothesis='Exported video is rectified/pinhole with a fixed centered crop and half-pixel resize from factory image. This is a testable hypothesis, NOT verified FISHEYE_8 semantics.',
      factory_K=factory['K'],K_integer_centers=K.tolist(),pixel_map=A.tolist(),factory_D8_applied=False,
      dimensions=[w,h],frames=n,reference='Original full-frame COLMAP cameras remain unchanged.',
      solver='OpenCV five-point Essential/RANSAC + PnP/RANSAC + SciPy fixed-K bundle adjustment',
      old_camera_or_structure_initialization=False,old_subject_or_Gaussian_loaded=False,
      matching='Fresh mutual SIFT descriptor ratio matches from pre-extracted raw features after frozen holdout exclusion; no old two-view geometry or fitted scene points.',
      max_features_per_frame=2500,temporal_hops=[1,2,4,8],ratio=.75,seed_pair_rule='Largest temporal hop from frame 0 in [1,2,4,8] with >=160 direct mutual appearance matches; select once before geometric estimation.',
      initialization_repair_of='/opt/gs-slam-lab/shuolee-new-capture-118-fixedk-opencv',
      repair_reason='Original fixed pair 0/8 had 23 direct appearance matches and failed the unchanged 80-cheirality-inlier gate. This was insufficient seed support, not a calibration rejection.',
      changed_variable='Initialization-pair selection only; same K, match budget, matcher, triangulation and error thresholds.',
      thresholds=dict(seed_inliers=80,pnp_inliers=50,ransac_px=1.5,pnp_px=3.,triangulation_px=3.,parallax_degrees=1.),
      max_ba_nfev=60,max_ba_points=5000,time_budget_seconds=900,
      stop_conditions=['Any missing registered frame','Nonfinite or nonrigid pose','Failed source identity or held-measurement exclusion','Held face, body or background degradation','Unresolved physical lens/export mapping prevents formal-training acceptance'],
      no_threshold_retry=True,training_admitted=False,quality_accepted=False,
      source_hashes={str(p):sha(p) for p in [BASE/'decoded.json',BASE/'sfm/raw-features.db',BASE/'sfm/holdout/split.json']},
      implementation_sha256=sha(Path(__file__)))
    save(OUT/'plan.json',plan);progress('matching-original-image-features',frames=n)
    db=sqlite3.connect((BASE/'sfm/raw-features.db').as_uri()+'?mode=ro',uri=True)
    images=dict(db.execute('select name,image_id from images'))
    xy=[];desc=[];original=[];offset=[0]
    for row in decoded:
        iid=images[row['name']]
        def arr(table,dtype):
            a,b,data=db.execute(f'select rows,cols,data from {table} where image_id=?',(iid,)).fetchone()
            return np.frombuffer(data,dtype).reshape(a,b)
        key=arr('keypoints','<f4')[:,:2]-.5;d=arr('descriptors','uint8')
        kept=np.load(BASE/f'sfm/holdout/row-maps/{iid}.npz')['kept'][:2500]
        xy.append(np.ascontiguousarray(key[kept],np.float64));desc.append(np.ascontiguousarray(d[kept],np.float32));original.append(kept)
        offset.append(offset[-1]+len(kept))
    db.close();cv2.setNumThreads(6);cv2.setRNGSeed(118)
    parent=np.arange(offset[-1],dtype=np.int64);size=np.ones(offset[-1],np.int64)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    matcher=cv2.BFMatcher(cv2.NORM_L2);pairs=[]
    for i in range(n):
        for hop in [1,2,4,8]:
            j=i+hop
            if j>=n:continue
            forward=matcher.knnMatch(desc[i],desc[j],k=2);backward=matcher.knnMatch(desc[j],desc[i],k=2)
            back={a.queryIdx:a.trainIdx for a,b in backward if a.distance<.75*b.distance}
            m=[(a.queryIdx,a.trainIdx) for a,b in forward if a.distance<.75*b.distance and back.get(a.trainIdx)==a.queryIdx]
            for a,b in m:union(offset[i]+a,offset[j]+b)
            pairs.append(dict(i=i,j=j,count=len(m)))
        if i%10==0:progress('matching-original-image-features',completed_frames=i+1,frames=n)
    groups={}
    for i in range(n):
        for k in range(len(xy[i])):
            a=offset[i]+k;r=find(a)
            if size[r]>=3:groups.setdefault(r,[]).append((i,k))
    tracks=[dict(nodes) for nodes in groups.values() if len({i for i,k in nodes})==len(nodes)]
    tracks.sort(key=lambda t:tuple(t.items()))
    save(OUT/'matching.json',dict(tracks=len(tracks),ambiguous_components=len(groups)-len(tracks),pairs=pairs,physical_identity_verified=False))
    tri_index=[[] for _ in range(n)]
    for ti,t in enumerate(tracks):
        for i in t:tri_index[i].append(ti)
    eligible=[p for p in pairs if p['i']==0 and p['count']>=160]
    if not eligible:raise RuntimeError('No seed pair meets the pre-geometry direct-appearance support gate.')
    chosen=max(eligible,key=lambda p:p['j']);seed_j=chosen['j']
    save(OUT/'seed-selection.json',dict(rule=plan['seed_pair_rule'],selected=chosen,selection_before_geometric_estimation=True))
    matrices={0:np.eye(4)};points={}
    seed=[ti for ti,t in enumerate(tracks) if 0 in t and seed_j in t]
    a=np.array([xy[0][tracks[t][0]] for t in seed]);b=np.array([xy[seed_j][tracks[t][seed_j]] for t in seed])
    normalized=lambda v:cv2.undistortPoints(v.reshape(-1,1,2),K,None).reshape(-1,2)
    essential,mask=cv2.findEssentialMat(normalized(a),normalized(b),np.eye(3),method=cv2.RANSAC,prob=.999,threshold=1.5/np.mean([K[0,0],K[1,1]]),maxIters=10000)
    if essential is None or essential.shape!=(3,3):raise RuntimeError('Seed pair did not produce one essential matrix.')
    count,R,t,mask=cv2.recoverPose(essential,normalized(a),normalized(b),np.eye(3),mask=mask)
    if count<80:raise RuntimeError(f'Seed cheirality inliers {count}<80.')
    E=np.eye(4);E[:3,:3]=R;E[:3,3]=t[:,0];matrices[seed_j]=E
    def triangulate_new():
        grouped={}
        centers={i:-E[:3,:3].T@E[:3,3] for i,E in matrices.items()}
        for ti,track in enumerate(tracks):
            if ti in points:continue
            ids=sorted(set(track)&matrices.keys())
            if len(ids)<2:continue
            # Choose maximum baseline without looking at the residual; every candidate is still checked.
            _,i,j=max((np.linalg.norm(centers[i]-centers[j]),i,j) for k,i in enumerate(ids) for j in ids[k+1:])
            grouped.setdefault((i,j),[]).append(ti)
        for (i,j),ids in grouped.items():
            u=np.array([xy[i][tracks[k][i]] for k in ids]);v=np.array([xy[j][tracks[k][j]] for k in ids])
            X=cv2.triangulatePoints(K@matrices[i][:3],K@matrices[j][:3],u.T,v.T);X=(X[:3]/X[3]).T
            p,zi=project(matrices[i],X,K);q,zj=project(matrices[j],X,K)
            ra=X-centers[i];rb=X-centers[j]
            angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(ra,rb),axis=1),np.einsum('ij,ij->i',ra,rb)))
            good=np.isfinite(X).all(1)&(zi>0)&(zj>0)&(np.linalg.norm(p-u,axis=1)<3)&(np.linalg.norm(q-v,axis=1)<3)&(angle>1)
            for k,x in zip(np.asarray(ids)[good],X[good]):points[int(k)]=x
    triangulate_new();progress('seed-initialized',seed_inliers=int(count),points=len(points),cameras=2)
    while len(matrices)<n:
        candidates=[]
        for i in range(n):
            if i not in matrices:
                common=[ti for ti in tri_index[i] if ti in points]
                candidates.append((len(common),i,common))
        done=0
        for _,i,common in sorted(candidates,reverse=True):
            if len(common)<50:continue
            X=np.array([points[k] for k in common]);u=np.array([xy[i][tracks[k][i]] for k in common])
            ok,r,t,inliers=cv2.solvePnPRansac(X,u,K,None,iterationsCount=2000,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers)<50:continue
            idx=inliers[:,0];r,t=cv2.solvePnPRefineLM(X[idx],u[idx],K,None,r,t)
            E=np.eye(4);E[:3,:3]=cv2.Rodrigues(r)[0];E[:3,3]=t[:,0]
            matrices[i]=E;done+=1
        triangulate_new();progress('pose-registration',cameras=len(matrices),points=len(points))
        if not done:raise RuntimeError('Missing real poses: '+str(sorted(set(range(n))-matrices.keys())))
    # Point budget is ordered by visibility, then track ID; no held tracks enter fitting.
    selected=sorted(points,key=lambda k:(-len(tracks[k]),k))[:5000]
    obs_cam=[];obs_point=[];uv=[]
    for p,ti in enumerate(selected):
        for i,k in tracks[ti].items():obs_cam.append(i);obs_point.append(p);uv.append(xy[i][k])
    ci=np.array(obs_cam);pi=np.array(obs_point);uv=np.array(uv)
    source_rows=np.array([int(original[i][k]) for ti in selected for i,k in tracks[ti].items()])
    hold=read(BASE/'sfm/holdout/split.json')
    reserved={(o['image_id'],o['feature_index']) for t in hold['tracks'] for o in t['observations']}
    conflicts=[(int(i),int(k)) for i,k in zip(ci,source_rows) if (images[decoded[i]['name']],int(k)) in reserved]
    if conflicts:raise RuntimeError('Held observations leaked into solve: '+str(conflicts[:5]))
    save(OUT/'observation-provenance.json',dict(fitting_observations=len(uv),held_tracks=len(hold['tracks']),direct_reserved_row_conflicts=len(conflicts),alias_exclusion='Frozen pre-SfM kept-row maps, unchanged',raw_feature_sha256=plan['source_hashes'][str(BASE/'sfm/raw-features.db')],held_split_sha256=plan['source_hashes'][str(BASE/'sfm/holdout/split.json')],no_old_camera_or_structure=True))
    camera=np.array([np.r_[Rotation.from_matrix(matrices[i][:3,:3]).as_rotvec(),matrices[i][:3,3]] for i in range(n)])
    camera_mask=np.ones((n,6),bool);camera_mask[0]=False;camera_mask[seed_j,3+np.argmax(abs(camera[seed_j,3:]))]=False
    cols=-np.ones((n,6),int);cols[camera_mask]=np.arange(camera_mask.sum());nc=int(camera_mask.sum());np3=len(selected)
    initial=np.r_[camera[camera_mask],np.array([points[k] for k in selected]).ravel()]
    sparsity=lil_matrix((2*len(uv),len(initial)),dtype=np.int8)
    rr=np.arange(len(uv))
    for k in range(6):
        valid=cols[ci,k]>=0
        sparsity[2*rr[valid],cols[ci[valid],k]]=1;sparsity[2*rr[valid]+1,cols[ci[valid],k]]=1
    for k in range(3):sparsity[2*rr,nc+3*pi+k]=1;sparsity[2*rr+1,nc+3*pi+k]=1
    calls=[0]
    def residual(state):
        calls[0]+=1
        if time.monotonic()-START>900:raise RuntimeError('900 second candidate budget reached during BA.')
        cp=camera.copy();cp[camera_mask]=state[:nc];X=state[nc:].reshape(-1,3)
        R=Rotation.from_rotvec(cp[:,:3]).as_matrix();q=np.einsum('nij,nj->ni',R[ci],X[pi])+cp[ci,3:]
        z=np.maximum(q[:,2],1e-6);pred=np.c_[K[0,0]*q[:,0]/z+K[0,2],K[1,1]*q[:,1]/z+K[1,2]]
        if calls[0]%100==0:progress('fixed-K-bundle-adjustment',residual_calls=calls[0],cameras=n,points=np3)
        return (pred-uv).ravel()
    progress('fixed-K-bundle-adjustment',cameras=n,points=np3,observations=len(uv))
    before=summary(np.linalg.norm(residual(initial).reshape(-1,2),axis=1))
    sol=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=60,ftol=1e-5)
    cp=camera.copy();cp[camera_mask]=sol.x[:nc];Es=np.tile(np.eye(4),(n,1,1));Es[:,:3,:3]=Rotation.from_rotvec(cp[:,:3]).as_matrix();Es[:,:3,3]=cp[:,3:]
    final_errors=np.linalg.norm(residual(sol.x).reshape(-1,2),axis=1)
    records=[dict(frame=r['frame'],image=str(BASE/'raw'/r['name']),source_sha256=r['source_sha256'],wh=r['wh'],K_integer_centers=K.tolist(),w2c=Es[i].tolist()) for i,r in enumerate(decoded)]
    save(OUT/'candidate-cameras.json',dict(records=records,pixel_domain='raw_decoded_integer_centers',factory_model_semantics_verified=False,not_for_training_yet=True))
    np.savez_compressed(OUT/'solver-output.npz',K=K,w2c=Es,points=sol.x[nc:].reshape(-1,3),obs_cam=ci,obs_point=pi,observations=uv,source_feature_row=source_rows,track_indices=np.array(selected)[pi],errors=final_errors)
    save(OUT/'solver-result.json',dict(state='camera_candidate_complete_validation_pending',cameras=n,points=np3,before=before,after=summary(final_errors),nfev=sol.nfev,success=bool(sol.success),message=sol.message,training_admitted=False,elapsed_seconds=time.monotonic()-START))
    progress('candidate-complete-validation-pending',cameras=n,training_started=False)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.is_dir():save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START,training_started=False))
        raise
