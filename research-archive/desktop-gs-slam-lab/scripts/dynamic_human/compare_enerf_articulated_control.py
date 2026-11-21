from pathlib import Path
import json,hashlib,shutil,argparse
import numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';ap=argparse.ArgumentParser();ap.add_argument('--before-root',type=Path,default=ROOT/'fixed_offsets');ap.add_argument('--after-root',type=Path,default=ROOT/'estimated_offsets');ap.add_argument('--output-root',type=Path,default=ROOT);ap.add_argument('--before-label',default='Same frame-index model');ap.add_argument('--after-label',default='Capture-time model');args=ap.parse_args();A=args.before_root;B=args.after_root;OUT=args.output_root/'paired-comparison';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
for branch in [A,B]:assert read(branch/'native-evaluation/independent-image-audit.json')['images']==102
assert read(A/'initial-and-sampling-audit.json')==read(B/'initial-and-sampling-audit.json')
xa=np.load(A/'sampling-schedule.npz');xb=np.load(B/'sampling-schedule.npz');assert all(np.array_equal(xa[k],xb[k]) for k in xa.files)
protocol=read(ROOT/'review-protocol.json');regions={}
for r in protocol['regions']:regions.setdefault((r['camera'],r['frame']),[]).append(r)
rows=[];detail=[]
for a,b in zip(read(A/'native-evaluation/metrics.json')['records'],read(B/'native-evaluation/metrics.json')['records']):
 for k in ['camera','frame','split','input_sha256','bbox']:assert a[k]==b[k]
 c,t=a['camera'],a['frame'];name=f'camera{c}-frame{t:06d}';archive=np.load(ROOT/'native-input'/f'{name}.npz');z={k:archive[k] for k in archive.files};archive.close();pa=np.load(A/'native-evaluation'/f'{name}-prediction.npz');pb=np.load(B/'native-evaluation'/f'{name}-prediction.npz');ia=np.clip(pa['rgb']+.5*(1-pa['alpha'][...,None]),0,1);ib=np.clip(pb['rgb']+.5*(1-pb['alpha'][...,None]),0,1);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None])
 boxes=[dict(name='full-person',bbox=a['bbox'])]+regions.get((c,t),[])
 for region in boxes:
  x0,y0,x1,y1=region['bbox'];w=x1-x0;h=y1-y0;cw=max(w,175);panel=Image.new('RGB',(3*cw,h+32),'white');d=ImageDraw.Draw(panel)
  for i,(label,im) in enumerate([('Original photograph',z['rgb']),(args.before_label,u8(ia)),(args.after_label,u8(ib))]):panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*cw,32));d.text((i*cw+4,10),label,fill='black')
  output=f"{name}-{region['name']}.png";panel.save(OUT/output);actual=np.array(Image.open(OUT/output))
  for i,im in enumerate([z['rgb'],u8(ia),u8(ib)]):assert np.array_equal(actual[32:,i*cw:i*cw+w],im[y0:y1,x0:x1])
  if region['name']!='full-person':
   m=z['valid'][y0:y1,x0:x1];r=ref[y0:y1,x0:x1];scores={label:float(-10*np.log10(np.mean((p[y0:y1,x0:x1][m].astype(np.float64)-r[m].astype(np.float64))**2))) for label,p in [('before',ia),('after',ib)]};detail.append(dict(camera=c,frame=t,region=region['name'],bbox=region['bbox'],**scores,delta_db=scores['after']-scores['before'],panel=output))
 rows.append(dict(camera=c,frame=t,split=a['split'],before=a['metrics'],after=b['metrics'],foreground_delta_db=b['metrics']['foreground']['psnr_db']-a['metrics']['foreground']['psnr_db']))
val=[r for r in rows if r['split']=='validation_time'];train=[r for r in rows if r['split']=='train'];summary=dict(status='102paired_native_panels24fixed_details_and_same_initial_schedule_verified_visual_review_pending',records=rows,details=detail,
 validation_foreground=dict(before_mean_psnr_db=float(np.mean([r['before']['foreground']['psnr_db'] for r in val])),after_mean_psnr_db=float(np.mean([r['after']['foreground']['psnr_db'] for r in val])),mean_delta_db=float(np.mean([r['foreground_delta_db'] for r in val])),improved_images=sum(r['foreground_delta_db']>0 for r in val),total_images=90),
 training_foreground_mean_delta_db=float(np.mean([r['foreground_delta_db'] for r in train])),per_camera_validation_delta_db={c:float(np.mean([r['foreground_delta_db'] for r in val if r['camera']==c])) for c in sorted({r['camera'] for r in val})},
 worst_change=min(val,key=lambda r:r['foreground_delta_db']),goal_complete=False)
summary['before_root']=str(A);summary['after_root']=str(B);save(OUT/'summary.json',summary);shutil.copy2(__file__,args.output_root/Path(__file__).name);print(json.dumps({k:v for k,v in summary.items() if k not in ['records','details','worst_change']}))
