from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
from smplx.joint_names import JOINT_NAMES

A=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
B=A.parent/'shuolee-new-capture-118-native-visible'
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
OUT=REC/'native-fit-review'
OUT.mkdir(exist_ok=True)
rows=json.loads((A/'data/training-inputs.json').read_text())['records']
obs=dict(np.load(A/'data/observations.npz'))
a=dict(np.load(A/'fit/fitted.npz'));b=dict(np.load(B/'fit/fitted.npz'))
wa=dict(np.load(A/'fit/observation-weights.npz'));wb=dict(np.load(B/'fit/observation-weights.npz'))
assert np.array_equal(wa['face'],wb['face']) and np.array_equal(wa['feet'],wb['feet'])
footids=[JOINT_NAMES.index(x) for x in ['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']]
metrics={}
for key,original,pa,pb,weights in [
    ('body17',wa['body_target'],a['projected_coco17'],b['projected_coco17'],wa['body']),
    ('face51',obs['keypoints'][:,40:91],a['projected_face51'],b['projected_face51'],wa['face']),
    ('feet6',obs['keypoints'][:,17:23],a['projected_joints'][:,footids],b['projected_joints'][:,footids],wa['feet'])]:
    keep=weights>.25
    def stat(v):
        q=np.linalg.norm(v-original,axis=-1)[keep]
        return dict(n=len(q),median_px=float(np.median(q)),p90_px=float(np.quantile(q,.9)),mean_px=float(q.mean()))
    metrics[key]=dict(frozen_observation_subset='reference raw weights >0.25, identical target rows for both arms',reference=stat(pa),surface_visibility=stat(pb))
metrics['visibility_suppression']={name:dict(raw_supported=int((wa[name]>.25).sum()),variant_still_supported=int(((wa[name]>.25)&(wb[name+'_surface_visibility']>0)).sum())) for name in ['face','feet']}
selected=[0,17,35,52,70,88]
row_meta=[]
for kind in ['body','face','feet']:
    panels=[]
    for i in selected:
        row=rows[i];w,h=row['wh']
        if kind=='body':
            xy=obs['boxes'][i].copy();xy[:2]-=30;xy[2:]+=30
        elif kind=='face':
            pts=np.concatenate([obs['keypoints'][i,40:91],a['projected_face51'][i],b['projected_face51'][i]])
            xy=np.r_[pts.min(0)-30,pts.max(0)+30]
        else:
            pts=np.concatenate([obs['keypoints'][i,17:23],a['projected_joints'][i,footids],b['projected_joints'][i,footids]])
            xy=np.r_[pts.min(0)-35,pts.max(0)+35]
        x0,y0,x1,y1=map(int,[max(0,xy[0]),max(0,xy[1]),min(w,xy[2]),min(h,xy[3])])
        if x1<=x0 or y1<=y0:
            x0,y0,x1,y1=0,max(0,h-200),w,h
        ims=[Image.open(A/'reviews/fit'/f"{row['frame']}_input.png").convert('RGB'),
             Image.open(A/'reviews/fit'/f"{row['frame']}_mesh_overlay.png").convert('RGB'),
             Image.open(B/'reviews/fit'/f"{row['frame']}_mesh_overlay.png").convert('RGB')]
        # Fit-review images can be display scaled. Map the exact same crop to all three.
        triplet=Image.new('RGB',(900,400),'white')
        for col,im in enumerate(ims):
            sx,sy=im.width/w,im.height/h
            crop=im.crop((round(x0*sx),round(y0*sy),round(x1*sx),round(y1*sy)))
            crop.thumbnail((300,400),Image.Resampling.LANCZOS)
            triplet.paste(crop,(300*col+(300-crop.width)//2,(400-crop.height)//2))
        panels.append(triplet)
        row_meta.append(dict(kind=kind,frame=row['frame'],crop_xyxy=[x0,y0,x1,y1],columns=['original','reference','surface_visibility'],text_burned_in=False))
    sheet=Image.new('RGB',(900,400*len(panels)),'white')
    for j,im in enumerate(panels):sheet.paste(im,(0,j*400))
    sheet.save(OUT/(kind+'_original_reference_surface-visibility.jpg'),quality=95)
report=dict(metrics=metrics,selected_frames=[rows[i]['frame'] for i in selected],image_layout=row_meta,
    fit_sha256={name:hashlib.sha256((root/'fit/fitted.npz').read_bytes()).hexdigest() for name,root in [('reference',A),('surface_visibility',B)]},
    interpretation='Fixed detector observations are proxies, not calibrated ground truth; suppressed points are counted rather than silently removed.',
    candidate_camera_failed_independent_protection_retained=True,physical_calibration_accepted=False,actual_visual_review_pending=True)
(OUT/'measurements.json').write_text(json.dumps(report,indent=2))
print(json.dumps(metrics,indent=2))
