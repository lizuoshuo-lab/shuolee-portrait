"""Read-only native crops and all-target summaries for the completed N12g result."""
from pathlib import Path
from PIL import Image,ImageDraw
import json,hashlib
import numpy as np
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027');D=I/'clip03-fresh-geometry';G=I/'clip03-grid-coverage/array-cache-repair';M=I/'clip03-scene-support'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (D/'visual-index.json').exists()
v=read(G/'visual-index.json');rv=read(G/'visual-review.json');results=read(D/'validation.json')['final'];frames=v['columns'];ims={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in frames}
sets={'face':[r['row'] for r in rv['rows'] if r['reviewed_material']=='face_surface_coarse'],'feet':[r['row'] for r in rv['rows'] if r['reviewed_material'] in ['shoe','sock']]}
index={};summaries={}
for name,nums in sets.items():
 er=results['face_all_diagnostic' if name=='face' else 'feet_diagnostic']['observations'];look={(e['family'],e['target']):e for e in er};rows=[];pages=[]
 for i,n in enumerate(nums):
  t=v['rows'][n];strip=Image.new('RGB',(896,128),(80,80,80));overlay=strip.copy();context=Image.new('RGB',(1792,256),(80,80,80))
  for obs in t['observations']:
   f=obs['frame'];c=frames.index(f);u,w=obs['uv'];x,y=np.rint([u,w]).astype(int);patch=ims[f].crop((x-64,y-64,x+64,y+64));strip.paste(patch,(c*128,0));context.paste(ims[f].crop((x-128,y-128,x+128,y+128)),(c*256,0));marked=patch.copy();draw=ImageDraw.Draw(marked)
   xx,yy=u-x+64,w-y+64;draw.ellipse((xx-2,yy-2,xx+2,yy+2),outline=(0,220,255),width=1)
   e=look[t['family'],f]
   if e['valid_projection']:
    px,py=np.array(e['predicted'])-[x-64,y-64];draw.line((px-3,py,px+3,py),fill=(255,0,210),width=1);draw.line((px,py-3,px,py+3),fill=(255,0,210),width=1)
   overlay.paste(marked,(c*128,0))
  sn=f'{name}_{i:02d}_family_{t["family"]}_native.png';on=sn.replace('_native','_projection');cn=sn.replace('_native','_context');strip.save(D/sn);overlay.save(D/on);context.save(D/cn)
  rows.append(dict(row=n,family=t['family'],split=t['split'],file=sn,projection_file=on,context_file=cn,observations=t['observations']))
  if i%3==0:pg=Image.new('RGB',(896,780),(235,235,235));cx=Image.new('RGB',(1792,780),(235,235,235))
  pg.paste(strip,(0,(i%3)*260));pg.paste(overlay,(0,(i%3)*260+130));cx.paste(context,(0,(i%3)*260))
  if i%3==2 or i==len(nums)-1:
   pn=f'{name}_review_page_{i//3:02d}.png';qn=f'{name}_context_page_{i//3:02d}.png';pg.save(D/pn);cx.save(D/qn);pages.extend([pn,qn])
 index[name]=dict(rows=rows,pages=pages)
for name,data in results.items():
 rows=[]
 for f in frames:
  es=[x for x in data['observations'] if x['target']==f];valid=[x for x in es if x['valid_projection']];rr=np.array([x['error_px'] for x in valid]);delta=np.array([x['delta'] for x in valid])
  rows.append(dict(frame=f,total=len(es),valid=len(valid),under1=int(sum(rr<1)),median=float(np.median(rr)) if len(rr) else None,max=float(max(rr)) if len(rr) else None,median_vector=np.median(delta,axis=0).tolist() if len(rr) else None))
 summaries[name]=dict(per_frame=rows,under1=sum(x['valid_projection'] and x['error_px']<1 for x in data['observations']),denominator=len(data['observations']))
save(D/'visual-index.json',dict(sets=index,frames=frames,native_pixels=True,spatial_warp=False,originals_untouched=True,annotations='No text. Native rows unchanged; separate diagnostic overlays: cyan=originalSIFT; magenta=target-excluded3Dprojection.',hashes={n:sha(D/n) for n in ['result.json','validation.json']}))
save(D/'regional-review-statistics.json',summaries)
print(json.dumps(summaries,ensure_ascii=False))
