"""Prepare the preregistered24 native comparisons, with no filtering or image labels."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,itertools
import numpy as np
from PIL import Image
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'video10-correspondence-contrast/head-roi-contrast';V=D/'visual-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
p=read(D/'plan.json');q=read(D/'queries.json');alloc=read(D/'review-allocation.json')['rows'];assert not V.exists();V.mkdir()
images={int(f):Image.open(c['source']).convert('RGB') for f,c in p['crops'].items()}
def crop(f,uv):
    if not np.isfinite(uv).all():return Image.new('RGB',(192,192),(128,128,128)),None
    x,y=np.floor(np.array(uv)+.5).astype(int)-96;box=[int(x),int(y),int(x+192),int(y+192)];return images[f].crop(box),box
records=[];sheets=[]
for rowid,r in enumerate(alloc):
    a=r['frame'];b,c=[f for f in p['frames'] if f!=a];i=next(i for i,t in enumerate(q[str(a)]) if t['feature']==r['feature'])
    vals={}
    for kind in ['baseline','roi']:
        with np.load(D/'run'/f'{kind}_cycle_{a:06d}_{b:06d}_{c:06d}.npz') as z:
            vals[kind]={k:z[k][i].tolist() for k in z.files}
    pieces=[(a,r['uv']),(b,vals['baseline']['b_uv']),(b,vals['roi']['b_uv']),(c,vals['baseline']['c_uv']),(c,vals['roi']['c_uv'])]
    strips=[];boxes=[]
    for f,uv in pieces:
        im,box=crop(f,uv);strips.append(im);boxes.append(dict(frame=f,uv=uv,box=box))
    if rowid%4==0:sheets.append(Image.new('RGB',(960,768),(235,235,235)))
    for col,im in enumerate(strips):sheets[-1].paste(im,(col*192,(rowid%4)*192))
    records.append(dict(row=rowid,**r,cycle=[a,b,c,a],baseline=vals['baseline'],roi=vals['roi'],crops=boxes,sheet=f'comparison_{rowid//4+1:02d}.png',sheet_row=rowid%4))
for i,im in enumerate(sheets):im.save(V/f'comparison_{i+1:02d}.png')
result=dict(updated=datetime.now(timezone.utc).isoformat(),columns=['source','baseline targetB','ROI targetB','baseline targetC','ROI targetC'],native_crop=192,images_resized=False,text_drawn=False,rows=records)
(V/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(row=x['row'],source=x['frame'],family=x['family'],split=x['split'],old_gate=x['baseline']['eligible'],new_gate=x['roi']['eligible'],old_cycle=x['baseline']['closure'],new_cycle=x['roi']['closure']) for x in records]))
