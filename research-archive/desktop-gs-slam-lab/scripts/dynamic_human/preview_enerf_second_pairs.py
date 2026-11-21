from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'early-review';O.mkdir(exist_ok=True)
plan=json.loads((R/'review-plan/protocol.json').read_text());font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);records=[]
for c,f in [('00',721),('04',721),('06',721),('07',731)]:
 name=f'camera{c}-frame{f:06d}';files=[R/b/'native-evaluation'/f'{name}-prediction.npz' for b in ['global_cubic','local_bspline']]
 if not all(p.exists() for p in files):continue
 z=np.load(R/'native-input'/f'{name}.npz');arrays=[z['rgb']]
 for path in files:
  p=np.load(path);arrays.append(np.rint(np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1)*255).astype(np.uint8))
 regions=[dict(name='full-person',bbox=z['bbox'].tolist())]+[r for r in plan['details'] if r['camera']==c and r['frame']==f]
 for r in regions:
  x0,y0,x1,y1=r['bbox'];w=x1-x0;h=y1-y0;im=Image.new('RGB',(w*3,h+32),'white');draw=ImageDraw.Draw(im)
  for k,(arr,label) in enumerate(zip(arrays,['Original photograph','Global cubic','Cubic + local B-splines'])):
   im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(k*w,32));draw.text((k*w+4,7),label,font=font,fill='black')
  path=O/f'{name}-{r["name"]}.png';im.save(path);out=np.array(Image.open(path))
  for k,arr in enumerate(arrays):assert np.array_equal(out[32:,k*w:(k+1)*w],arr[y0:y1,x0:x1])
  records.append(str(path))
print(json.dumps(records))
