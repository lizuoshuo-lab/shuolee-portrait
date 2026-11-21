from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-support-correction-081';D=O/'comparison';D.mkdir(exist_ok=True)
plan=json.loads((S/'review-plan/protocol.json').read_text());font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);count=0
for row in plan['full_person']+plan['details']:
 c,t=row['camera'],row['frame'];name=row.get('name','full-person');stem=f'camera{c}-frame{t:06d}';dest=D/f'{stem}-{name}.png'
 if dest.exists():continue
 f=O/'local_bspline/native-evaluation'/f'{stem}-prediction.npz'
 if not f.exists() or not f.with_name(f'{stem}-person-comparison.png').exists():continue
 z=np.load(S/'native-input'/f'{stem}.npz');arrays=[z['rgb']]
 for p in [S/'local_bspline/native-evaluation'/f.name,f]:
  q=np.load(p);arrays.append(np.rint(np.clip(q['rgb']+.5*(1-q['alpha'][...,None]),0,1)*255).astype(np.uint8))
 x0,y0,x1,y1=row.get('bbox',z['bbox']);w=x1-x0;h=y1-y0;panel=Image.new('RGB',(w*3,h+32),'white');draw=ImageDraw.Draw(panel)
 for j,(arr,label) in enumerate(zip(arrays,['Original photograph','Original training support','Conservative corrected support'])):
  panel.paste(Image.fromarray(arr[y0:y1,x0:x1]),(w*j,32));draw.text((w*j+4,7),label,font=font,fill='black')
 panel.save(dest);print(dest.name);count+=1
print('New ready panels',count)
