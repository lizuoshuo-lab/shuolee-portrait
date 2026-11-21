"""Native training-image patch tracks, separate from anatomical keypoints."""
from pathlib import Path
import hashlib, json, time, shutil
import cv2
import numpy as np

L=Path(__file__).resolve().parents[2]
R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075'
I=L/'outputs/dynamic-human-enerf-articulated-control-064'
M=L/'outputs/dynamic-human-enerf-actor-supervision-070'
P=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
R.mkdir(exist_ok=True); (R/'tracks').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'preparation.json').exists()
cams=read(M/'mask-visual-review.json')['accepted_cameras']
times=[744,746,748,750,752,754,756]
save(R/'protocol.json',dict(status='fixed_before_tracks_and_model_motion_comparison',
    cameras=cams,frames=times,reference_frame=750,heldout_views_accessed=False,
    source='Original064 native1920x1080 images;13 visually accepted070 conservative actor interiors at750. Original foreground mattes, no photo enhancement.',
    feature_detection=dict(method='Shi-Tomasi',body_per_actor=300,body_quality=.01,minimum_distance_px=5,additional_right_hand_per_actor=60,hand_quality=.005,hand_minimum_distance_px=3),
    hand_regions='Bounding rectangle of corrected074 whole-body right-hand predictions112:133 plus8px. Used for coverage labels and additional seeds only; no anatomical GT claim.',
    tracking=dict(method='Pyramidal Lucas-Kanade on native grayscale',window=21,levels=3,iterations=40,epsilon=.01,minimum_eigenvalue=.0001,
        max_forward_backward_px=.5,max_sequential_direct_disagreement_px=1.0,min_patch_NCC=.97,max_patch_MAE_u8=12,max_adjacent_motion_px=40,patch_size=11,destination_matte_min=.90),
    rule='Track from750 through adjacent even frames in each direction; stop a track after failure. Also require direct750-to-target consistency. Destination foreground mask contains both actors; identity/occlusion is not certified by these gates.',
    purpose='Compare reliable-looking native image patch motion with frozen Gaussian visibility-weighted projected-center motion before adding any flow training loss.',
    model_updates=0,training_labels_accepted=False,
    limits='Texture-biased correspondence hypotheses, not optical-flow or finger ground truth. Smooth skin and occluded fingers may have no reliable tracks. No new degree of freedom or model acceptance from this diagnostic.',goal_complete=False))
manifest={(x['camera'],x['frame']):x for x in read(I/'input-audit.json')['records'] if x['split']=='train'}
poses={(x['camera'],x['frame']):x for x in read(P/'observations.json')['records']}
LK=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01),minEigThreshold=.0001)
cv2.setNumThreads(4)
began=time.monotonic();records=[]

def image_values(a,xy):
    return cv2.remap(a.astype(np.float32),xy[:,0].astype(np.float32).reshape(-1,1),xy[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1)

def lk(a,b,xy):
    y,s,_=cv2.calcOpticalFlowPyrLK(a,b,xy.astype(np.float32).reshape(-1,1,2),None,**LK)
    y=y.reshape(-1,2)
    return y,s.reshape(-1).astype(bool)&np.isfinite(y).all(1)

def patches(a,b,x,y):
    ncc=np.zeros(len(x),np.float32);mae=np.full(len(x),255,np.float32)
    for i,(u,v) in enumerate(zip(x,y)):
        if not np.isfinite(v).all():continue
        p=cv2.getRectSubPix(a,(11,11),tuple(map(float,u))).astype(np.float64).ravel()
        q=cv2.getRectSubPix(b,(11,11),tuple(map(float,v))).astype(np.float64).ravel()
        pp=p-p.mean();qq=q-q.mean();den=np.linalg.norm(pp)*np.linalg.norm(qq)
        ncc[i]=pp@qq/den if den>1e-8 else -1
        mae[i]=np.abs(p-q).mean()
    return ncc,mae

for c in cams:
    inputs={};sources=[]
    for t in times:
        f=I/'native-input'/f'camera{c}-frame{t:06d}.npz'
        h=sha(f);assert h==manifest[c,t]['input_sha256']
        z=np.load(f);inputs[t]={k:z[k] for k in ['rgb','valid','matte','bbox','K','w2c']};z.close()
        inputs[t]['gray']=cv2.cvtColor(inputs[t]['rgb'],cv2.COLOR_RGB2GRAY)
        sources.append(dict(frame=t,sha256=h))
    label_file=M/'masks'/f'camera{c}-frame000750.npz'
    labels=np.load(label_file)['labels'];assert labels.shape==(1080,1920)
    xy=[];actors=[];regions=[];handboxes=[]
    for actor in [0,1]:
        hand=np.array(poses[c,750]['detections'][actor]['keypoints'])[112:133]
        hb=np.r_[hand.min(0)-8,hand.max(0)+8];handboxes.append(hb.tolist())
        core=(labels==actor).astype(np.uint8)*255
        points=cv2.goodFeaturesToTrack(inputs[750]['gray'],300,.01,5,mask=core,blockSize=5)
        points=np.empty((0,2),np.float32) if points is None else points.reshape(-1,2)
        handmask=np.zeros_like(core);x0,y0,x1,y1=np.rint(hb).astype(int)
        handmask[max(0,y0):min(1080,y1),max(0,x0):min(1920,x1)]=255
        extra=cv2.goodFeaturesToTrack(inputs[750]['gray'],60,.005,3,mask=core&handmask,blockSize=3)
        if extra is not None:
            for e in extra.reshape(-1,2):
                if not len(points) or np.linalg.norm(points-e,axis=1).min()>=2.5:points=np.vstack([points,e])
        for p in points:
            xy.append(p);actors.append(actor)
            regions.append('right_hand_box' if hb[0]<=p[0]<=hb[2] and hb[1]<=p[1]<=hb[3] else 'other_actor_interior')
    xy=np.array(xy,np.float32);actors=np.array(actors,np.int8);regions=np.array(regions)
    assert len(xy)>100
    tracks=[]
    for direction in [[748,746,744],[752,754,756]]:
        current=xy.copy();alive=np.ones(len(xy),bool);prev=750
        for t in direction:
            y,forward_ok=lk(inputs[prev]['gray'],inputs[t]['gray'],current)
            back,back_ok=lk(inputs[t]['gray'],inputs[prev]['gray'],y)
            direct,direct_ok=lk(inputs[750]['gray'],inputs[t]['gray'],xy)
            direct_back,direct_back_ok=lk(inputs[t]['gray'],inputs[750]['gray'],direct)
            fb=np.linalg.norm(back-current,axis=1);dfb=np.linalg.norm(direct_back-xy,axis=1)
            disagree=np.linalg.norm(y-direct,axis=1)
            ncc,mae=patches(inputs[750]['gray'],inputs[t]['gray'],xy,y)
            inside=(y[:,0]>=6)&(y[:,0]<1914)&(y[:,1]>=6)&(y[:,1]<1074)
            good=alive&forward_ok&back_ok&direct_ok&direct_back_ok&inside&(fb<.5)&(dfb<.5)&(disagree<1)&(ncc>=.97)&(mae<=12)
            good&=np.linalg.norm(y-current,axis=1)<=40
            good&=image_values(inputs[t]['matte'],y)>.90
            good&=image_values(inputs[t]['valid'],y)>.999
            # Keep rejected candidates in the local archive for independent gate checks.
            tracks.append(dict(frame=t,xy=y.copy(),accepted=good.copy(),fb=fb,dfb=dfb,disagreement=disagree,ncc=ncc,mae=mae))
            current=y;alive=good;prev=t
    tracks.sort(key=lambda x:x['frame'])
    np.savez_compressed(R/'tracks'/f'camera{c}.npz',source_xy=xy,actor=actors,region=regions,
        target_frame=np.array([v['frame'] for v in tracks]),target_xy=np.stack([v['xy'] for v in tracks]),
        accepted=np.stack([v['accepted'] for v in tracks]),fb=np.stack([v['fb'] for v in tracks]),
        direct_fb=np.stack([v['dfb'] for v in tracks]),direct_disagreement=np.stack([v['disagreement'] for v in tracks]),
        patch_ncc=np.stack([v['ncc'] for v in tracks]),patch_mae=np.stack([v['mae'] for v in tracks]),hand_boxes=np.array(handboxes))
    row=dict(camera=c,reference_seeds=len(xy),seeds_per_actor=[int((actors==a).sum()) for a in [0,1]],
        right_hand_box_seeds_per_actor=[int(((actors==a)&(regions=='right_hand_box')).sum()) for a in [0,1]],
        accepted_by_frame={str(v['frame']):int(v['accepted'].sum()) for v in tracks},
        accepted_right_hand_by_frame={str(v['frame']):[int((v['accepted']&(actors==a)&(regions=='right_hand_box')).sum()) for a in [0,1]] for v in tracks},
        sources=sources,actor_label_sha256=sha(label_file),tracks_sha256=sha(R/'tracks'/f'camera{c}.npz'))
    records.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['sources','actor_label_sha256','tracks_sha256']}),flush=True)
save(R/'preparation.json',dict(status='native_training_patch_candidates_ready_model_comparison_and_visual_review_pending',records=records,
    seconds=time.monotonic()-began,reference_seeds=sum(v['reference_seeds'] for v in records),
    accepted_observations=sum(sum(v['accepted_by_frame'].values()) for v in records),goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
