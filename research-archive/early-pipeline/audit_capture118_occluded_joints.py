"""Measure existing fit supervision against actual full-image foreground support."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
OUT=ROOT/'alignment/occluded-joint-audit';OUT.mkdir(exist_ok=False)
rows=json.loads((ROOT/'data/inputs.json').read_text())['records']
obs=dict(np.load(ROOT/'data/observations.npz'));weight=dict(np.load(ROOT/'fit/observation-weights.npz'))
inside=[];in_image=[]
for i,row in enumerate(rows):
    a=np.asarray(Image.open(row['mask']).convert('L'))>127
    xy=np.rint(obs['keypoints'][i]).astype(int);ok=(xy[:,0]>=0)&(xy[:,0]<a.shape[1])&(xy[:,1]>=0)&(xy[:,1]<a.shape[0])
    foreground=np.zeros(133,bool);foreground[ok]=a[xy[ok,1],xy[ok,0]]
    inside.append(foreground);in_image.append(ok)
inside=np.array(inside);in_image=np.array(in_image)
groups=[('feet',np.arange(17,23),weight['feet']),('ankles',np.array([15,16]),weight['body'][:,15:17]*weight['visibility'][:,15:17]),
        ('other_body',np.arange(15),weight['body'][:,:15]*weight['visibility'][:,:15]),
        ('hands',np.arange(91,133),weight['hands'])]
summary={}
for name,ids,w in groups:
    active=w>.25;outside=~inside[:,ids]
    scores=obs['scores'][:,ids]
    summary[name]=dict(observations=int(w.size),active=int(active.sum()),
        active_outside_person_mask=int((active&outside).sum()),active_outside_image=int((active&~in_image[:,ids]).sum()),
        active_outside_weight_fraction=float((w*outside).sum()/max(w.sum(),1e-12)),
        raw_score_quantiles=np.quantile(scores,[0,.1,.5,.9,1]).tolist(),
        normalized_weight_quantiles=np.quantile(w,[0,.1,.5,.9,1]).tolist())
selected=[]
for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
    r=rows[i];im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
    for j in list(range(15,23)):
        x,y=obs['keypoints'][i,j]
        if in_image[i,j]:
            color=(255,40,40) if not inside[i,j] else (30,220,90)
            draw.ellipse((float(x-5),float(y-5),float(x+5),float(y+5)),outline=color,width=2)
    name=f'{r["frame"]}_ankle_toe_observations_red_outside_mask_green_inside.png'
    im.save(OUT/name)
    selected.append(dict(frame=r['frame'],path=name,ankles_feet_scores=obs['scores'][i,15:23].tolist(),
        in_image=in_image[i,15:23].tolist(),inside_person_mask=inside[i,15:23].tolist(),
        foot_weights=weight['feet'][i].tolist()))
result=dict(updated=datetime.now(timezone.utc).isoformat(),groups=summary,selected=selected,
    interpretation='Mask membership only establishes foreground image support, not true 3D visibility or anatomical correctness. No points/weights/cameras were changed.',
    fit_gate_body_only=True,foot_visibility_explicitly_used_in_fit=False,
    existing_weight_rule='Per-joint q90 score normalization and horizontal flip consistency; feet do not use mesh visibility or image/mask membership.',
    training_started=False,quality_accepted=False)
for p in [OUT/'result.json',REC/'occluded-joint-audit.json']:
    p.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(summary))
