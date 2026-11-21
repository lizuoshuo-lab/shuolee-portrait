from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';Q=L/'outputs/dynamic-human-enerf-mano-surface-099';O=L/'outputs/dynamic-human-enerf-hand-visibility-100';read=lambda p:json.loads(p.read_text(encoding='utf8'))
c,f='03',732;r=next(r for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records'] if r['camera']==c and r['frame']==f);x0,y0,x1,y1=r['crop'];stem=f'camera{c}-frame{f:06d}';z=np.load(S/'native-input'/f'{stem}.npz');arrays=[z['rgb'][y0:y1,x0:x1]]+[np.array(Image.open(root/'native-evaluation'/f'{stem}-trained.png'))[y0:y1,x0:x1] for root in [Q/'photo_pose',O/'narrow_reset',O/'context_reset']];im=Image.new('RGB',(640,192),'white');d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
for i,(a,label) in enumerate(zip(arrays,['Original 03/732','Prior surface','1937 reset','6330 reset'])):im.paste(Image.fromarray(a),(160*i,32));d.text((160*i+3,8),label,font=font,fill='black')
im.save(O/'early-fixed-view.png')
