"""Read actual compositing weights in the predefined hand contexts."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-mano-surface-099';S=L/'outputs/dynamic-human-enerf-second-interval-079';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';A=O/'contribution';A.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
inputs={(r['camera'],r['frame']):r for r in read(U/'evaluation-protocol.json')['records']};details=[dict(r,name='target_right_hand') for r in read(L/'outputs/dynamic-human-enerf-hand-fine-085/target-hand-review-plan.json')['records']]+[dict(camera=r['camera'],frame=r['frame'],name='same_time_hand',bbox=r['crop']) for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records'] if r['frame']==732];rows=[];panels=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
for r in details:
 if r['name'] not in ['same_time_hand','target_right_hand']:continue
 c,f=r['camera'],r['frame'];stem=f'camera{c}-frame{f:06d}';z=np.load((U if inputs[c,f].get('input_location')=='088' else S)/'native-input'/f'{stem}.npz');x0,y0,x1,y1=r['bbox'];mask=(z['valid']&(z['matte']>.5))[y0:y1,x0:x1];raw=z['rgb'][y0:y1,x0:x1];arrays=[raw];values={}
 for b in ['fixed_pose','photo_pose']:
  p=np.load(O/b/'native-evaluation'/f'{stem}-prediction.npz');w=p['surface_weight'][y0:y1,x0:x1];alpha=p['alpha'][y0:y1,x0:x1];rgb=p['rgb'][y0:y1,x0:x1];assert np.isfinite(w).all() and (w>=-1e-6).all() and (w<=alpha+1e-5).all();values[b]=dict(foreground_context_pixels=int(mask.sum()),mean_visible_weight=float(w[mask].mean()),context_pixels_weight_gt_0_1=int(((w>.1)&mask).sum()),context_pixels_weight_gt_0_5=int(((w>.5)&mask).sum()),new_fraction_of_total_opacity=float(w[mask].sum()/max(float(alpha[mask].sum()),1e-9)))
  arrays.extend([u8(np.clip(rgb+.5*(1-alpha[...,None]),0,1)),np.repeat(u8(w)[...,None],3,-1)])
 rows.append(dict(camera=c,frame=f,name=r['name'],bbox=r['bbox'],metrics=values))
 if (c,f) in [('04',731),('04',732),('13',731),('03',732)]:
  h,w=raw.shape[:2];im=Image.new('RGB',(5*w,h+32),'white');d=ImageDraw.Draw(im)
  for i,(a,label) in enumerate(zip(arrays,['Original','Fixed pose RGB','New weight [0,1]','Photo pose RGB','New weight [0,1]'])):im.paste(Image.fromarray(a),(i*w,32));d.text((i*w+3,8),label,font=font,fill='black')
  path=A/f'{stem}-{r["name"]}.png';im.save(path);panels.append(dict(file=path.name,camera=c,frame=f,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(rows)==21
save(A/'summary.json',dict(status='actual_joint_depth_compositing_weights_in21native_hand_contexts',records=rows,panels=panels,aggregate={b:{k:float(np.mean([r['metrics'][b][k] for r in rows])) for k in ['mean_visible_weight','new_fraction_of_total_opacity','context_pixels_weight_gt_0_1','context_pixels_weight_gt_0_5']} for b in ['fixed_pose','photo_pose']},scope='Source-definedhandcontext intersects original foreground; includes arm/toy and is not a semantic hand mask. Weight is sum alpha_i*T_i of new Gaussians inside actual joint rasterization. Brightness is not image sharpness or geometric accuracy.',goal_complete=False))
shutil.copy2(__file__,A/Path(__file__).name);print(json.dumps(read(A/'summary.json')['aggregate']))
