from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-contour-surface-103';S=L/'outputs/dynamic-human-enerf-second-interval-079'
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
plan=[r for r in read(L/'outputs/dynamic-human-enerf-hand-fine-085/target-hand-review-plan.json')['records'] if r['frame']==731];assert len(plan)==2
panels=read(O/'comparison/summary.json')['panels'];im=Image.new('RGB',(640,412),'white');d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);sources=[]
for j,r in enumerate(plan):
 c,f=r['camera'],r['frame'];rec=next(p for p in panels if p['camera']==c and p['frame']==f and p['name']=='target_right_hand');a=np.array(Image.open(O/'comparison'/rec['file']));y=j*206
 d.text((4,y+5),f'Camera {c} | held-out time {f}',font=font,fill='black')
 for k,(col,label) in enumerate(zip([0,1,3,4],['Original','Parent','Prior control','Contour prior'])):
  d.text((k*160+4,y+25),label,font=font,fill='black');im.paste(Image.fromarray(a[32:,col*160:(col+1)*160]),(k*160,y+46))
 sources.append(rec)
path=O/'native-articulated-hand-comparison.png';im.save(path);arr=np.array(Image.open(path))
for j,r in enumerate(sources):
 src=np.array(Image.open(O/'comparison'/r['file']))
 for k,col in enumerate([0,1,3,4]):assert np.array_equal(arr[j*206+46:j*206+206,k*160:(k+1)*160],src[32:,col*160:(col+1)*160])
(O/'public-figure-audit.json').write_text(json.dumps(dict(file=path.name,sha256=sha(path),size=[640,412],native_pixel_tiles=True,resampled=False,source_panels=sources,model_overlay=False),indent=2)+'\n');print(path)
