from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/static-observation-consensus-081'
J=O/'joint-bundle'
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
with np.load(J/'cross-bank-evaluation.npz') as z:
    a={k:z[k] for k in z.files}
with np.load(B/'current/camera-local-inconsistency-audit-080/geometry.npz') as z:
    boxes=z['boxes_raw']
frame=99
obs=a['observations']; ids=np.flatnonzero(a['held'] & (obs[:,0]==frame) & np.isin(a['groups'],[1,2]))
center=(boxes[frame,:2]+boxes[frame,2:])/2
origin=np.rint(center-[190,220]).astype(int)
src=Image.open(B/f'current/full-scene-localization-074/images/raw_{frame+1:06d}.png').convert('RGB')
board=Image.new('RGB',(1140,525),'white'); d=ImageDraw.Draw(board)
for col,name in enumerate(['Original observations','Original SfM projection','Joint pose/point BA projection']):
    crop=src.crop((*origin,*(origin+[380,440]))); dr=ImageDraw.Draw(crop)
    for i in ids:
        p=obs[i,1:]-origin
        dr.ellipse((*list(p-3),*list(p+3)),outline=(0,220,255),width=1)
        if col:
            q=a['prediction' if col==1 else 'joint_prediction'][i]-origin
            dr.line((q[0]-3,q[1],q[0]+3,q[1]),fill=(255,35,30),width=1)
            dr.line((q[0],q[1]-3,q[0],q[1]+3),fill=(255,35,30),width=1)
    board.paste(crop,(380*col,85))
    d.text((380*col+5,6),name,font=font,fill='black')
    d.text((380*col+5,32),f'raw 000100 | {len(ids)} held head features',font=small,fill='black')
    d.text((380*col+5,57),'Cyan=observed; red=projected; 1:1px',font=small,fill='black')
out=J/'cross-bank-frame100.jpg'
assert not out.exists()
board.save(out,quality=98)
receipt={'file':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
         'frame':100,'held_features':len(ids),'source_image_warped':False,
         'training_render':False,'visually_reviewed':False}
(J/'visual-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
