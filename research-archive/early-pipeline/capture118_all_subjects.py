"""Registered 2026-10-27 multi-subject research intake. Never resumes a fit/run.

Original images, explicit fixed factory-K hypothesis, fresh OpenCV geometry.
This module only prepares observations and priors; it cannot launch Gaussians.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, shutil, subprocess, sys, time, traceback
import numpy as np

PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
MASTER = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
REC = MASTER/'all-persons-20261027'
ROOT = Path('/opt/gs-slam-lab/shuolee-capture118-all-persons-20261027')
SOURCE = Path('/mnt/d/00/工作/数据集')
ASSETS = '/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import runtime as rt

SUBJECTS = [
    dict(id='grass', label='草地白衣人物', video='2f1aac19-8ae1_1791971027322.mp4',
         review='Front and both sides visible; back coverage incomplete. Legs visible. Small head image; breeze/hair/hand motion possible.'),
    dict(id='sofa', label='沙发坐姿人物', video='819080de-8b00.mp4',
         review='Seated pose; sofa hides pelvis/thighs and table sometimes hides feet. Missing person mask behind furniture is not proof of empty 3D space.'),
    dict(id='portrait', label='竖屏蓝衣黄裙人物', video='online4.mp4',
         review='No confirmed camera provenance; 9:16 crop is a research hypothesis only. Visible head turning, hair/skirt motion; static model assumptions require separate review.'),
    dict(id='indoor118', label='室内白衣人物118', video='df776c03-91a4_1792572380952.mp4',
         review='Previous fresh18k failed face and lower-leg quality. Retain it; audit furniture occlusion/negative-alpha supervision before one justified new training.'),
]

def now(): return datetime.now(timezone.utc).isoformat()
def state(subject, **kw):
    rt.save(REC/f'{subject}-state.json', dict(updated=now(), subject=subject, pid=os.getpid(),
        command=sys.argv, quality_accepted=False, physical_calibration_accepted=False, **kw))
    print(json.dumps(dict(subject=subject, **kw), ensure_ascii=False), flush=True)

def init():
    if (REC/'protocol.json').exists(): return
    REC.mkdir(parents=True, exist_ok=True); ROOT.mkdir(exist_ok=False)
    calibration = rt.read(MASTER/'protocol.json')['calibration']
    assert rt.sha(SOURCE/'mono_main_calib.yaml') == calibration['sha256']
    for subject in SUBJECTS:
        subject.update(video_sha256=rt.sha(SOURCE/subject['video']), run_root=str(ROOT/subject['id']))
    protocol = dict(created=now(), authorization='User: 做长期任务，持续推进训练；每个人物都做',
        subjects=SUBJECTS, calibration=calibration, goal='Four separately recorded static portrait reconstructions; no shared personalized geometry or optimizer.',
        budgets=dict(max_new_camera_candidates_per_untrained_subject=1, max_fit_steps=1200,
            initial_points=200000, max_gaussian_steps=18000, max_new_training_per_subject=1,
            camera_seconds=1800, decode_seconds=300, observations_seconds=1800, prior_seconds=900),
        input_contract=dict(frame_stride=5, max_image_size=1600, no_pose_interpolation=True,
            no_silent_dropped_cameras=True, fixed_K='Factory K transformed by explicit centered aspect crop and half-pixel resize; unverified video applicability.',
            unknown_D8_applied=False, old_camera_or_3D_initialization=False, new_SIFT_from_raw=True,
            formal_calibration_accepted=False, exact_camera_source='OpenCV essential/PnP and fixed-K scipy sparse BA, not COLMAP pose estimation'),
        stops=['No new training before actual raw-image/camera/fit review.',
            'One bounded fresh research training per subject only when finite geometry and visible support permit; calibration remains uncertain until independently proven.',
            'Failed subject does not block independent preparation of other subjects.',
            'If a code defect is found, preserve failure evidence and fix only that defect; no threshold sweeps.',
            'Review 1000-step checkpoint before remaining same-run budget; finish final native/PLY/UI quality review.',
            'Only one GPU task; no changes to running source; no new downloads or company training.'],
        white118_next_variable='Audit negative alpha in furniture-occluded body regions with original images. Do not repeat rejected self-visibility weighting or repeat the same failed18k.',
        delivery='Keep old outputs. Export actual trained PLY including covariance and degree3SH, native parity and local viewer OpenFile Gaussian/points; mark quality separately.',
        end='After four reviewed results or concrete failed admission evidence; no automatic extra experiments.')
    rt.save(REC/'protocol.json', protocol)
    rt.save(REC/'queue-state.json', dict(updated=now(), state='active', active_subject=None,
        subjects=[dict(id=s['id'], label=s['label'], state='pending_occlusion_diagnosis' if s['id']=='indoor118' else 'pending_fresh_intake') for s in SUBJECTS],
        prior_phase='completed_bounded_run_quality_failed', prior_model_preserved='new-capture-118-fixedk'))
    for s in SUBJECTS: state(s['id'], state='pending_occlusion_diagnosis' if s['id']=='indoor118' else 'pending_fresh_intake', training_running=False)

def prepare(subject):
    root = ROOT/subject['id']; root.mkdir(exist_ok=False)
    for d in ['data', 'logs', 'camera']: (root/d).mkdir()
    cfg = rt.read(PROJECT/'configs/server.json')
    cfg.update(frame_stride=5, max_image_size=1600, max_frames=300, fit_steps=1200,
        training_height=899, seed=6060, initial_points=200000, iterations=18000,
        pixel_center_convention='renderer_corner_to_detector_integer', landmark_image_support='mask_margin_4px',
        camera_estimator='OpenCV essential/PnP with fixed factory-crop K', physical_calibration_validated=False)
    rt.save(root/'config.json',cfg)
    rt.save(root/'run.json', dict(created=now(), video=str(SOURCE/subject['video']), video_sha256=subject['video_sha256'],
        subject=subject, old_subject_loaded=False, old_fitted_person_loaded=False, old_gaussian_loaded=False, old_optimizer_loaded=False))
    state(subject['id'], state='extracting_raw_frames', run_root=str(root), training_running=False)
    from shuolee_portrait.video_sfm import extract
    extract(SOURCE/subject['video'],root,cfg)
    decoded=rt.read(root/'decoded.json')['records']; w,h=decoded[0]['wh']
    factory=rt.read(REC/'protocol.json')['calibration'];fw,fh=factory['image_wh']
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(SOURCE/subject['video'])]))['streams'][0]
    if stream.get('tags',{}).get('rotate','0')!='0' or any(abs(float(s.get('rotation',0)))>.01 for s in stream.get('side_data_list',[])):
        raise RuntimeError('Rotation requires an explicit sensor-to-export mapping; not silently assumed.')
    aspect=stream['width']/stream['height']
    if aspect>=fw/fh: cw,ch=fw,fw/aspect
    else: cw,ch=fh*aspect,fh
    x0,y0=(fw-cw)/2,(fh-ch)/2
    A=np.array([[w/cw,0,w/cw*(.5-x0)-.5],[0,h/ch,h/ch*(.5-y0)-.5],[0,0,1.]])
    K=A@np.asarray(factory['K'])
    rt.save(root/'intrinsic-contract.json',dict(factory_sha256=factory['sha256'],factory_K=factory['K'],
        crop_xywh=[x0,y0,cw,ch],crop_is_assumption=True,raw_K_integer=K.tolist(),pixel_map=A.tolist(),
        wh=[w,h],source='New raw exported video pixels; no lens remap',D8_applied=False,
        calibration_applicability_confirmed=False,principal_center_convention='integer; renderer receives +0.5'))
    cfg['training_height']=h;rt.save(root/'config.json',cfg)
    return root

def camera(root):
    """Fresh track split precedes any geometry. Holdout never affects fitting."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.sparse import lil_matrix
    start=time.monotonic(); out=root/'camera';subject=root.name
    decoded=rt.read(root/'decoded.json')['records'];n=len(decoded)
    K=np.asarray(rt.read(root/'intrinsic-contract.json')['raw_K_integer'])
    def status(stage,**kw):
        if time.monotonic()-start>1800: raise RuntimeError('1800-second camera budget exceeded.')
        state(subject,state=stage,run_root=str(root),training_running=False,cameras_total=n,seconds=time.monotonic()-start,**kw)
    def project(E,X):
        q=X@E[:3,:3].T+E[:3,3];uv=q@K.T
        return uv[:,:2]/uv[:,2,None],q[:,2]
    def summary(e):
        a=np.asarray(e);return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(count=0)
    rt.save(out/'plan.json',dict(created=now(),source_sha256=rt.sha(Path(__file__)),pixels='integer full raw decoded pixels',
        frame_count=n,features=5000,ratio=.75,temporal_hops=[1,2,4,8,16],first_last_edges=8,
        split='SHA256 of complete raw appearance component; one fifth whole-track reserved before any geometry, no spatial duplicates within1.5px/frame.',
        minimum_seed_direct_matches=160,seed_cheirality=80,min_pnp_inliers=50,min_BA_obs_per_camera=50,
        ransac_px=1.5,pnp_px=3,triangulation_px=3,min_parallax_degrees=1,max_BA_points=5000,max_BA_nfev=60,
        time_budget_seconds=1800,K_fixed=True,old_poses_loaded=False,training_started=False))
    cv2.setNumThreads(6);cv2.setRNGSeed(118)
    sift=cv2.SIFT_create(nfeatures=7000);xy=[];ds=[];offset=[0]
    for i,r in enumerate(decoded):
        key,desc=sift.detectAndCompute(cv2.imread(str(root/'raw'/r['name']),cv2.IMREAD_GRAYSCALE),None)
        order=sorted(range(len(key)),key=lambda j:(-key[j].response,key[j].pt[0],key[j].pt[1]))
        # Keep the strongest descriptor per small pixel neighborhood, including scale duplicates.
        cells={};kept=[]
        for j in order:
            p=np.asarray(key[j].pt);cell=tuple(np.floor(p/1.5).astype(int))
            neighbours=[v for dx in [-1,0,1] for dy in [-1,0,1] for v in cells.get((cell[0]+dx,cell[1]+dy),[])]
            if any(np.linalg.norm(p-v)<1.5 for v in neighbours):continue
            kept.append(j);cells.setdefault(cell,[]).append(p)
            if len(kept)==5000:break
        if len(kept)<100:raise RuntimeError(f'Insufficient fresh raw texture at {r["frame"]}.')
        xy.append(np.asarray([key[j].pt for j in kept],np.float64));ds.append(np.ascontiguousarray(desc[kept],np.float32));offset.append(offset[-1]+len(kept))
        np.savez_compressed(out/f'features-{i:03d}.npz',xy=xy[-1],descriptors=ds[-1])
        if i%10==0:status('fresh_SIFT',done=i+1)
    parent=np.arange(offset[-1]);sz=np.ones(offset[-1],int)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if sz[a]<sz[b]:a,b=b,a
        parent[b]=a;sz[a]+=sz[b]
    pairs=[];matcher=cv2.BFMatcher(cv2.NORM_L2)
    for i in range(n):
        js=sorted(set([i+d for d in [1,2,4,8,16] if i+d<n]+(list(range(max(i+1,n-8),n)) if i<8 else [])))
        for j in js:
            fw=matcher.knnMatch(ds[i],ds[j],k=2);bw=matcher.knnMatch(ds[j],ds[i],k=2)
            back={a.queryIdx:a.trainIdx for a,b in bw if a.distance<.75*b.distance}
            m=[(a.queryIdx,a.trainIdx) for a,b in fw if a.distance<.75*b.distance and back.get(a.trainIdx)==a.queryIdx]
            for a,b in m:union(offset[i]+a,offset[j]+b)
            pairs.append(dict(i=i,j=j,count=len(m)))
        if i%10==0:status('mutual_material_candidates',done=i+1)
    groups={}
    for i in range(n):
        for j in range(len(xy[i])):
            node=offset[i]+j;p=find(node)
            if sz[p]>=3:groups.setdefault(p,[]).append((i,j))
    components=sorted((v for v in groups.values() if len({i for i,j in v})==len(v)),key=lambda v:tuple(v))
    records=[]
    for nodes in components:
        ident=hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()
        records.append(dict(id=ident,held=int(ident[:8],16)%5==0,nodes=nodes))
    rt.save(out/'appearance-tracks.json',dict(records=records,pairs=pairs,rejected_ambiguous_components=len(groups)-len(components),physical_identity_verified=False))
    train=[r for r in records if not r['held']];held=[r for r in records if r['held']]
    tracks=[dict(r['nodes']) for r in train]
    if len(held)<30:raise RuntimeError('Fewer than30 whole-track holdouts; no geometry solve.')
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

def run(subject_id, inputs):
    subject=next(s for s in rt.read(REC/'protocol.json')['subjects'] if s['id']==subject_id)
    if subject_id=='indoor118':raise RuntimeError('Existing118 requires registered causal change, not rerunning intake.')
    root=ROOT/subject_id
    try:
        if inputs:
            assert (root/'camera/result.json').exists() and not (root/'data/observations.npz').exists()
        else:
            root=prepare(subject);camera(root)
            return
        from shuolee_portrait.cli import gpu_lock
        with gpu_lock():
            for stage,timeout in [('observations',1800),('prior',900)]:
                command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',stage,'--output',str(root),'--assets',ASSETS]
                env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
                with (root/'logs'/f'{stage}.log').open('x') as log:
                    proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state(subject_id,state='fresh_'+stage,run_root=str(root),child_pid=proc.pid,child_command=command,training_running=False)
                    try:rc=proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                        raise RuntimeError(stage+' finite time budget exceeded')
                    if rc:raise RuntimeError(stage+' failed: '+str(rc))
        state(subject_id,state='fresh_inputs_ready_actual_review_required',run_root=str(root),training_running=False)
    except BaseException as exc:
        state(subject_id,state='failed_preserved_for_review',run_root=str(root),training_running=False,error=repr(exc))
        rt.save(REC/f'{subject_id}-failure.json',dict(created=now(),error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--init',action='store_true');ap.add_argument('--subject',choices=[s['id'] for s in SUBJECTS]);ap.add_argument('--inputs',action='store_true');args=ap.parse_args()
    if args.init:init()
    elif args.subject:run(args.subject,args.inputs)
    else:ap.error('Choose --init or --subject')
