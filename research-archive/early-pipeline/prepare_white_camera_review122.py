"""Prepare source-native evidence for N12h; no training/camera mutation."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
O=I/'clip03-camera-observability';D=I/'clip03-fresh-geometry';F=I/'clip03-roof-review';M=I/'clip03-scene-support'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'figure-index.json').exists()
frames=[200,232,280];support=read(F/'reviewed-support.json')['tracks']
bg=next(t for t in sorted(support,key=lambda t:t['family']) if t['split']=='heldout' and set(frames)<=set(o['frame'] for o in t['observations']))
idx=read(D/'visual-index.json');face=idx['sets']['face']['rows'][4];feet=idx['sets']['feet']['rows'][5]
tracks=[('background',bg),('face_all_diagnostic',face),('feet_diagnostic',feet)]
vals=read(D/'validation.json')['final'];canvas=Image.new('RGB',(384,384));records=[]
for row,(group,t) in enumerate(tracks):
    look={(a['family'],a['target']):a for a in vals[group]['observations']}
    for col,f in enumerate(frames):
        obs=next(o for o in t['observations'] if o['frame']==f);u,v=obs['uv'];x,y=np.rint([u,v]).astype(int)
        patch=Image.open(M/f'frame_{f:06d}.png').convert('RGB').crop((x-64,y-64,x+64,y+64))
        draw=ImageDraw.Draw(patch);xx,yy=u-x+64,v-y+64;draw.ellipse((xx-2,yy-2,xx+2,yy+2),outline=(0,220,255),width=1)
        pred=look[t['family'],f];assert pred['valid_projection'];px,py=np.array(pred['predicted'])-[x-64,y-64]
        draw.line((px-3,py,px+3,py),fill=(255,0,210),width=1);draw.line((px,py-3,px,py+3),fill=(255,0,210),width=1)
        canvas.paste(patch,(128*col,128*row))
    records.append(dict(group=group,family=t['family'],split=t['split'],frames=frames))
canvas.save(O/'background_face_feet.png')
save(O/'figure-index.json',dict(rows=records,native_cell_pixels=128,no_image_text=True,no_warp=True,selection='Background first eligible frozen heldout family by numericID, face/feet same N12g frozen illustration. No residual-based selection.',cyan='originalSIFT',magenta='existing N12g leave-target-out prediction'))
print(json.dumps(records))
