from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-time-scope-088';records=json.loads((L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json').read_text())['records'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
for c in ['01','04']:
 r=next(r for r in records if r['camera']==c and r['frame']==732);x0,y0,x1,y1=r['crop'];stem=f'camera{c}-frame000732';source=np.load(S/'native-input'/f'{stem}.npz')['rgb'];images=[source]
 for root in [S/'local_bspline',O/'single_time']:
  z=np.load(root/'native-evaluation'/f'{stem}-prediction.npz');images.append(np.rint(np.clip(z['rgb']+.5*(1-z['alpha'][...,None]),0,1)*255).astype(np.uint8))
 im=Image.new('RGB',(480,192),'white');d=ImageDraw.Draw(im)
 for i,(a,label) in enumerate(zip(images,['Original','Parent','Fit frame732'])):im.paste(Image.fromarray(a[y0:y1,x0:x1]),(i*160,32));d.text((i*160+4,8),label,font=font,fill='black')
 im.save(O/f'early-camera{c}-frame732.png')
