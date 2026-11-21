"""Independent18image checks plus exactly the same068fixed regions."""
from pathlib import Path
import json,hashlib,shutil,numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-detail-weight-076';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068';E=R/'heldout/native-evaluation';O=R/'heldout/paired-comparison';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout'
assert read(B/'native-evaluation/independent-image-audit.json')['images']==18
reports={(r['camera'],r['frame']):r for r in read(E/'metrics.json')['records']};inputs=[r for r in read(I/'input-audit.json')['records'] if r['frame']%2];protocol=read(I/'review-protocol.json');rows=[];details=[]
for r in inputs:
 c,t=r['camera'],r['frame'];name=f'camera{c}-frame{t:06d}';source=I/'native-input'/f'{name}.npz';assert sha(source)==r['input_sha256'];z=np.load(source);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);images=[]
 for root in [B/'native-evaluation',E]:
  p=np.load(root/f'{name}-prediction.npz');images.append(np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1))
 before,after=images;assert np.array_equal(np.array(Image.open(E/f'{name}-trained.png')),u8(after));valid=z['valid']&(z['matte']>.5)
 def psnr(image,ref,mask):return float(-10*np.log10(np.mean((image[mask].astype(np.float64)-ref[mask].astype(np.float64))**2)))
 v=[psnr(im,ref,valid) for im in images];assert abs(v[1]-reports[c,t]['foreground_psnr_db'])<1e-9
 baseline=next(x for x in read(B/'native-evaluation/metrics.json')['records'] if x['camera']==c and x['frame']==t)['foreground_psnr_db'];assert abs(v[0]-baseline)<1e-9
 for region in [dict(name='full-person',bbox=r['bbox'])]+[r for r in protocol['regions'] if r['camera']==c and r['frame']==t]:
  x0,y0,x1,y1=region['bbox'];w=x1-x0;h=y1-y0;cw=max(w,180);panel=Image.new('RGB',(3*cw,h+32),'white');draw=ImageDraw.Draw(panel)
  ims=[z['rgb'],u8(before),u8(after)]
  for i,(label,im) in enumerate(zip(['Original photograph','Corrected RGB pose','Detail-weighted RGB'],ims)):panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*cw,32));draw.text((i*cw+4,10),label,fill='black')
  file=f"{name}-{region['name']}.png";panel.save(O/file);actual=np.array(Image.open(O/file))
  for i,im in enumerate(ims):assert np.array_equal(actual[32:,i*cw:i*cw+w],im[y0:y1,x0:x1])
  if region['name']!='full-person':
   sl=(slice(y0,y1),slice(x0,x1));vv=[psnr(im[sl],ref[sl],z['valid'][sl]) for im in images];details.append(dict(camera=c,frame=t,region=region['name'],bbox=region['bbox'],before=vv[0],after=vv[1],delta_db=vv[1]-vv[0],panel=file))
 full=[psnr(im,ref,z['valid']) for im in images]
 rows.append(dict(camera=c,frame=t,before=v[0],after=v[1],foreground_delta_db=v[1]-v[0],full_valid_before=full[0],full_valid_after=full[1],full_valid_delta_db=full[1]-full[0],prediction_sha256=sha(E/f'{name}-prediction.npz')))
summary=dict(status='18newcamera_native_predictions36pairedpanels_independently_checked',records=rows,details=details,
 validation_foreground=dict(before_mean_psnr_db=float(np.mean([r['before'] for r in rows])),after_mean_psnr_db=float(np.mean([r['after'] for r in rows])),mean_delta_db=float(np.mean([r['foreground_delta_db'] for r in rows])),improved_images=sum(r['foreground_delta_db']>0 for r in rows),total_images=18),
 per_camera={c:float(np.mean([r['foreground_delta_db'] for r in rows if r['camera']==c])) for c in ['00','06','12']},worst_change=min(rows,key=lambda r:r['foreground_delta_db']),goal_complete=False)
save(O/'summary.json',summary);save(E/'independent-image-audit.json',dict(status=summary['status'],images=18,records=rows,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps({k:v for k,v in summary.items() if k not in ['records','details']}))
