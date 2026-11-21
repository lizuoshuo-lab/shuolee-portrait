"""Native context only for visually ambiguous N11 material; no detection/fitting."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
B=T/'shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';O=B/'scene-support-supplement'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=O/'context-review-plan.json';assert not plan.exists()
idx=read(O/'visual-index.json');selected=[1,3,4,9,10,11,25,30,31,32]
save(plan,dict(registered=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),review_index_sha256=sha(O/'visual-index.json'),rows=selected,selection='After actual42rows review, these10rows have water/shore/reflection ambiguity at the cropcenter. Context clarifies material identity before any geometry; not selected by reprojection error.',operation='First/middle/last320x320 originalpixel context, no resampling, no annotations or imagewarps',newfeatures=0,newcameras=0))
images={};rows=[]
for i in selected:
    r=idx['rows'][i];canvas=Image.new('RGB',(960,320),(80,80,80));bounds=[]
    for j,ob in enumerate(r['observations']):
        f=ob['frame']
        if f not in images:images[f]=Image.open(B/f'frame_{f:06d}.png').convert('RGB')
        x,y=np.rint(ob['uv']).astype(int);box=[int(x-160),int(y-160),int(x+160),int(y+160)]
        canvas.paste(images[f].crop(tuple(box)),(320*j,0));bounds.append(box)
    name=f'context_row_{i:02d}_family_{r["family"]}.png';canvas.save(O/name)
    rows.append(dict(row=i,family=r['family'],split=r['split'],observations=r['observations'],bounds=bounds,file=name,point_is_crop_center=True))
for k in range(0,len(rows),2):
    subset=rows[k:k+2];canvas=Image.new('RGB',(960,320*len(subset)))
    for j,r in enumerate(subset):canvas.paste(Image.open(O/r['file']),(0,320*j))
    canvas.save(O/f'context_page_{k//2:02d}.png')
save(O/'context-index.json',dict(rows=rows,pages=[f'context_page_{k:02d}.png' for k in range(5)]))
print(json.dumps(dict(context_rows=len(rows),new_features=0,new_cameras=0)))
