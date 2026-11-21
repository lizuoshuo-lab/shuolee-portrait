import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
R=B/'current/person-static-human-surface-init-fresh-083'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def local(p):return Path('C:'+str(p)[6:]) if str(p).startswith('/mnt/c') else Path(p)
with np.load(R/'clothing-init/rejected-face-candidate.npz') as z:a={k:z[k] for k in z.files}
with np.load(R/'head-fit/observations.npz') as z:o={k:z[k] for k in z.files}
rows=read(R/'data/training-inputs.json')['records'];out=R/'clothing-init/rejected-review';out.mkdir(exist_ok=True)
rank=read(R/'clothing-init/rejected-face-preflight.json')['worst_frames']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
for i in [rank[0]['raw']-1,rank[5]['raw']-1,69,135]:
    source=Image.open(local(rows[i]['image'])).convert('RGB');q=o['uv'][i]
    center=(np.quantile(q,.05,axis=0)+np.quantile(q,.95,axis=0))/2;origin=np.rint(center-[220,240]).astype(int)
    photo=source.crop((*origin,*(origin+[440,480])));panel=photo.copy();d=ImageDraw.Draw(panel)
    order=np.argsort(~a['mask_support'][i])
    for j in order:
        p=a['projected'][i,j]-origin;ok=a['mask_support'][i,j]
        color=(30,220,220) if ok else (255,30,40);radius=1 if ok else 2
        d.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),fill=color)
    board=Image.new('RGB',(880,540),'white');board.paste(photo,(0,60));board.paste(panel,(440,60));d=ImageDraw.Draw(board)
    d.text((5,8),f'raw {i+1:06d} | original source',font=font,fill='black')
    d.text((445,8),'Cyan=inside mask; red=outside mask',font=font,fill='black')
    d.text((445,32),'Neutral surface candidates;0Gaussian updates',font=font,fill='black')
    board.save(out/f'raw-{i+1:06d}.jpg',quality=96)
print(json.dumps(dict(review=str(out),frames=[rank[0]['raw'],rank[5]['raw'],70,136])))
