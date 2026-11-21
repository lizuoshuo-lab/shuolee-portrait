"""Freeze image-only three-view material probes and body visibility before P3 fit."""
from pathlib import Path
import collections, datetime as dt, hashlib, json, os, time, traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/portrait-alignment-open-source-116'; H=O/'hsfm'; J=H/'joint-diagnostic-reserved'
LABELS=['background','face_lower','face_upper','torso','legs_shoes']
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def patch(im,p):
    a=cv2.getRectSubPix(im,(13,13),tuple(map(float,p))).astype(float);a-=a.mean()
    return a/max(np.linalg.norm(a),1e-8)
def main():
    assert not J.exists(),'Never refreeze validation after candidate fitting'
    J.mkdir(); t0=time.monotonic(); cv2.setNumThreads(4)
    rec=read(H/'input-pilot/inputs.json')['records']; ids=sorted(r['raw_frame'] for r in rec)
    rows={r['source_frame_index']:r for r in read(B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json')['records']}
    kp=np.load(H/'input-pilot/observations.npz'); kr={int(f):i for i,f in enumerate(kp['raw_frame_ids'])}
    K=np.array(rec[0]['K_from_existing_manifest']); intr=np.array(read(B/'current/full-scene-localization-074/cameras.json')['intrinsics'])
    # Reserve all measurements before HMR inference or camera fitting. No geometry residual gates.
    reg=dict(created=now(),state='registered_before_fitting',candidate='P3',repair_of='joint-diagnostic/validation-result.json',repair_reason='Apply fixed prior-bank exclusion before the finite corner budget. Post-filtering the top60 corners can leave no new background seeds. No changed LK thresholds, fitting or camera-residual selection.',raw_frame_ids=[r['raw_frame'] for r in rec],
        same_dense_predictions_for_control_and_joint=True,full_final_required=175,
        hypotheses=['Additional shared-body reprojection may reduce scene-camera ambiguity',
                    'Anatomical priors may instead improve fitted joints while degrading real material correspondence',
                    'Sequential person motion and view-dependent observations may violate shared-static assumptions'],
        control='Native DUSt3R scene consistency; same initial cameras, depth, gauge, frozen observations, budget',
        treatment='Same native scene plus upstream HSfM shared SMPL human loss with explicit local interface adaptations',
        adaptations=['HMR2.0a local approved checkpoint instead of HSfM default2.0b',
                     'RTMW body23 instead of ViTPose; no SMPL-X hand/face substitute',
                     'Actual portrait K and PIL pixel-center transforms in depth unprojection and human projection',
                     'Same-name weighted root translation; correct xyxy bbox area',
                     'Only pre-reviewed front/three-quarter visible body observations; no hidden joints forced as truth'],
        visibility=dict(eligible_frames=[50,63,75,88,100,113],
                        joint_indices=[0,1,2,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22],
                        raw_score_min=.5,reason='Freeze before fit. Exclude hips under loose clothes, ears and ambiguous far-side joints in side views.',
                        anatomical_joints_are_material_GT=False),
        validation=dict(coordinates='Original rectified1514x2694 zero-based raster centers',
                        blocks=[[ids[i-1],ids[i],ids[i+1]] for i in range(1,len(ids)-1)],
                        held='Middle view; triangulate endpoints only. Report all fixed tracks, including invalid candidate geometry.',
                        exclusion_previous_banks_rect_px=6,max_corners=[60,35,35,50,50],
                        lk='Half-resolution adjacent-frame LK; FB<=0.5 halfpx, NCC>=0.85, direct cycle<=0.75 halfpx each4steps and at endpoint',
                        source='New image-selected corners, exclude P1 and101 at seed. No candidate/camera residual used for admission.',
                        acceptance='At least10 lower-face probes across3 blocks, plus >=10 per other region; then original-image review',
                        limitations='Same video reused; KLT/cycle consistency is not physical GT, reflection/motion can survive. Dense network receptive fields may overlap.',
                        fit_exclusion='Zero dense alignment weights within radius3 processed pixels of every reserved observation; identical both arms.'),
        planned_fit=dict(initialization='Native MST, first-camera rigid gauge, one shared scale from fresh HMR priors; never load an old subject fit',
                         iterations_per_arm=500,lr_start=.015,lr_end=.0001,
                         shared_stage='Fresh shared body root/shape registration before both arm snapshots; no per-frame deformation',
                         human_weight=5.,shape_prior_weight=0.,freeze_first_camera_pose=True,
                         source='HSfM shared-SMPL loss and native DUSt3R PointCloudOptimizer; headless local adapter, not full official reproduction'),
        budgets=dict(cpu_validation_seconds=900,gpu_total_seconds=1800,setup_deadline_utc=read(O/'queue-state.json')['setup_deadline_utc']),
        stop_on=['Invalid identity/hash/coordinate mapping','Nonfinite values','GPU conflict or OOM','Insufficient frozen validation coverage','Budget reached'],
        no_gaussian_training=True,no_parameter_sweep=True,no_overwrite074=True,
        input_sha256=sha(H/'input-pilot/inputs.json'),observations_sha256=sha(H/'input-pilot/observations.npz'),script_sha256=sha(__file__))
    save(J/'preregistration.json',reg)
    scores=kp['scores'][:,:23].copy(); use=np.zeros_like(scores,bool)
    for f in reg['visibility']['eligible_frames']:
        use[kr[f],reg['visibility']['joint_indices']]=True
    use &= scores>=.5
    np.savez_compressed(J/'visibility.npz',raw_frame_ids=kp['raw_frame_ids'],visible=use,scores=np.where(use,scores,0))
    # Previous observations are exclusion-only; convert original distorted P1 bank once.
    p1=np.load(O/'pixsfm/independent-validation/tracks.npz')['observations'].copy()
    Kr=np.array([[intr[0],0,intr[1]-.5],[0,intr[0],intr[2]-.5],[0,0,1.]])
    Kc=K.copy();Kc[:2,2]-=.5
    p1[:,1:]=cv2.undistortPoints(p1[:,1:].reshape(-1,1,2),Kr,np.array([intr[3],0,0,0,0.]),P=Kc).reshape(-1,2)
    old=np.load(B/'current/shared-point-bundle-convergence-101/input.npz')['observations'].copy()
    trees={f:cKDTree(np.concatenate([p1[p1[:,0]==f-1,1:],old[old[:,0]==f-1,1:]],axis=0)) for f in ids}
    images={}; regions={}; hashes={}
    def get(f):
        if f not in images:
            p=Path(rows[f]['image']);m=Path(rows[f]['mask'])
            assert sha(p)==rows[f]['source_sha256'] and sha(m)==rows[f]['mask_sha256']
            img=cv2.imread(str(p),0);assert img.shape==(2694,1514)
            images[f]=cv2.resize(img,(757,1347),interpolation=cv2.INTER_AREA)
            hashes[str(f)]=rows[f]['source_sha256']
        return images[f]
    for r in rec:
        f=r['raw_frame'];get(f);mask=cv2.imread(r['mask'],0)>127
        small=cv2.resize(mask.astype(np.uint8),(757,1347),interpolation=cv2.INTER_NEAREST)
        fg=cv2.erode(small,np.ones((7,7),np.uint8))>0; bg=cv2.erode(1-small,np.ones((11,11),np.uint8))>0
        yy,xx=np.mgrid[:1347,:757]; yy=yy*2+.5;xx=xx*2+.5
        top=r['bbox_xyxy'][1];ht=r['bbox_xyxy'][3]-top
        a=np.full(fg.shape,-1,np.int8);a[bg]=0;a[fg&(yy>top+.23*ht)&(yy<top+.64*ht)]=3;a[fg&(yy>=top+.64*ht)]=4
        face=kp['keypoints'][kr[f],23:91];lo=face.min(0);hi=face.max(0);split=float(np.median(face[27:36,1]))
        fm=fg&(xx>=lo[0]-5)&(xx<=hi[0]+5)&(yy>=lo[1]-5)&(yy<=hi[1]+5)
        a[fm&(yy>=split)]=1;a[fm&(yy<split)]=2;regions[f]=a
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),minEigThreshold=1e-5)
    tracks=[]; rejects=collections.Counter()
    for left,seed,right in reg['validation']['blocks']:
        for group in range(5):
            admission_mask=(regions[seed]==group).astype(np.uint8)*255
            for p in trees[seed].data:
                q=np.rint((p-.5)/2).astype(int);cv2.circle(admission_mask,tuple(q),4,0,-1)
            cand=cv2.goodFeaturesToTrack(get(seed),maxCorners=reg['validation']['max_corners'][group],qualityLevel=.01,minDistance=5,mask=admission_mask,blockSize=5)
            if cand is None:continue
            keep=trees[seed].query(cand[:,0]*2+.5)[0]>6;cand=cand[keep]
            if not len(cand):continue
            histories=[{seed:p[0].copy()} for p in cand]
            for end,direction in [(left,-1),(right,1)]:
                active=np.arange(len(cand));prev=cand.copy()
                for step,f in enumerate(range(seed+direction,end+direction,direction),1):
                    nxt,st,_=cv2.calcOpticalFlowPyrLK(get(f-direction),get(f),prev,None,**lk)
                    back,sb,_=cv2.calcOpticalFlowPyrLK(get(f),get(f-direction),nxt,None,**lk)
                    valid=(st[:,0]>0)&(sb[:,0]>0)&(np.linalg.norm(back[:,0]-prev[:,0],axis=1)<=.5)
                    rejects['adjacent_fb']+=int((~valid).sum())
                    if step%4==0 or f==end:
                        direct,sd,_=cv2.calcOpticalFlowPyrLK(get(seed),get(f),cand[active],None,**lk)
                        rev,sr,_=cv2.calcOpticalFlowPyrLK(get(f),get(seed),direct,None,**lk)
                        cyc=(sd[:,0]>0)&(sr[:,0]>0)&(np.linalg.norm(direct[:,0]-nxt[:,0],axis=1)<=.75)&(np.linalg.norm(rev[:,0]-cand[active,0],axis=1)<=.75)
                        rejects['direct_cycle']+=int((valid&~cyc).sum());valid &= cyc
                    for j in np.flatnonzero(valid):
                        x,y=np.rint(nxt[j,0]).astype(int)
                        if not(7<=x<750 and 7<=y<1340) or (f==end and regions[f][y,x]!=group):valid[j]=False;rejects['region']+=1;continue
                        if (patch(get(f-direction),prev[j,0])*patch(get(f),nxt[j,0])).sum()<.85:valid[j]=False;rejects['ncc']+=1
                    if f==end:
                        for j in np.flatnonzero(valid):histories[active[j]][f]=nxt[j,0].copy()
                    active=active[valid];prev=nxt[valid]
                    if not len(active):break
            for hist in histories:
                if len(hist)!=3:continue
                tracks.append(dict(frames=[left,seed,right],xy=np.array([hist[f]*2+.5 for f in [left,seed,right]]),group=group))
        save(J/'validation-progress.json',dict(updated=now(),pid=os.getpid(),state='running',seed=seed,tracks=len(tracks),seconds=time.monotonic()-t0))
        print(json.dumps(dict(seed=seed,tracks=len(tracks))),flush=True)
        assert time.monotonic()-t0<900
    assert tracks,'No new independent validation tracks'
    np.savez_compressed(J/'validation-tracks.npz',frames=np.array([x['frames'] for x in tracks]),xy=np.stack([x['xy'] for x in tracks]),groups=[x['group'] for x in tracks])
    support={name:dict(tracks=sum(t['group']==g for t in tracks),blocks=sorted(set(t['frames'][1] for t in tracks if t['group']==g))) for g,name in enumerate(LABELS)}
    selection=[]
    for g in range(5):
        blocks=support[LABELS[g]]['blocks']
        for b in sorted(set(blocks[i] for i in [0,len(blocks)//2,len(blocks)-1])) if blocks else []:
            selection.append(next(i for i,t in enumerate(tracks) if t['group']==g and t['frames'][1]==b))
    panels=[]
    for tid in selection:
        t=tracks[tid];panel=Image.new('RGB',(720,266),'white');d=ImageDraw.Draw(panel)
        d.text((8,4),f"probe {tid} | {LABELS[t['group']]} | center view held",fill='black')
        for col,(f,xy) in enumerate(zip(t['frames'],t['xy'])):
            box=np.round(np.r_[xy-70,xy+70]).astype(int);crop=Image.open(rows[f]['image']).convert('RGB').crop(tuple(box)).resize((224,224),Image.Resampling.LANCZOS)
            panel.paste(crop,(240*col+8,36));d.text((240*col+8,19),f'raw {f:06d}',fill='black')
            q=(xy-box[:2])*1.6+[240*col+8,36];d.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),outline='#00ccff',width=2)
        panels.append(panel)
    for b in range(0,len(panels),5):
        ps=panels[b:b+5];im=Image.new('RGB',(720,266*len(ps)),'white')
        for j,p in enumerate(ps):im.paste(p,(0,j*266))
        im.save(J/f'validation-review-{b//5+1}.jpg',quality=95)
    result=dict(updated=now(),state='frozen_visual_review_pending',tracks=len(tracks),groups=support,
                coverage_passed=all(x['tracks']>=10 and len(x['blocks'])>=3 for x in support.values()),
                validation_sha256=sha(J/'validation-tracks.npz'),visibility_sha256=sha(J/'visibility.npz'),
                images_sha256=hashes,visual_sample_track_ids=selection,rejects=dict(rejects),seconds=time.monotonic()-t0,
                camera_fit_started=False,admission_uses_camera_residual=False)
    save(J/'validation-result.json',result);save(J/'validation-progress.json',dict(updated=now(),state='completed',pid=os.getpid()))
    print(json.dumps({k:result[k] for k in ['tracks','groups','coverage_passed','seconds']}),flush=True)
try:main()
except Exception as e:
    if J.exists():save(J/'validation-failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
    raise
