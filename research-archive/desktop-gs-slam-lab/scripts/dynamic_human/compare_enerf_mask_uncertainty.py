from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/dynamic-human-enerf-mask-uncertainty-053'
A=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame';B=ROOT;OUT=ROOT/'paired-comparison';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(ROOT/'paired-training-state-audit.json')['status'].startswith('paired6000updates_exact')
for p in [A,B]:assert read(p/'native-evaluation/independent-image-audit.json')['full_pngs']==24
regions=read(LAB/'outputs/dynamic-human-enerf-camera10-time-050/fixed-detail-regions.json')['regions'];rows=[];details=[]
for a,b in zip(read(A/'native-evaluation/metrics.json')['records'],read(B/'native-evaluation/metrics.json')['records']):
 for k in ['camera','source_rgb_sha256','source_matte_sha256','bbox']:assert a[k]==b[k]
 c=a['camera'];prefix=f'camera{c}-frame000750'
 za=np.load(A/'native-evaluation'/f'{prefix}-raw.npz');zb=np.load(B/'native-evaluation'/f'{prefix}-raw.npz')
 for k in ['source','matte','valid','K','w2c']:assert np.array_equal(za[k],zb[k])
 ref=za['source'].astype(np.float32)/255*za['matte'][...,None]+.5*(1-za['matte'][...,None])
 pa=np.clip(za['rgb']+.5*(1-za['alpha'][...,None]),0,1);pb=np.clip(zb['rgb']+.5*(1-zb['alpha'][...,None]),0,1)
 u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
 for label,box in [('full-person',a['bbox'])]+list(regions.get(c,{}).items()):
  x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;cw=max(w,140)
  panel=Image.new('RGB',(3*cw,h+32),'white');d=ImageDraw.Draw(panel)
  for i,(title,im) in enumerate([('Photo',za['source']),('Full supervision',u8(pa)),('Uncertainty ignored',u8(pb))]):
   panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*cw,32));d.text((i*cw+3,10),title,fill='black')
  fn=f'{prefix}-{label}.png';panel.save(OUT/fn);actual=np.array(Image.open(OUT/fn))
  for i,im in enumerate([za['source'],u8(pa),u8(pb)]):assert np.array_equal(actual[32:,i*cw:i*cw+w],im[y0:y1,x0:x1])
  if label!='full-person':
   mask=za['valid'][y0:y1,x0:x1];r=ref[y0:y1,x0:x1]
   scores={k:float(-10*np.log10(np.mean((p[y0:y1,x0:x1][mask].astype(np.float64)-r[mask].astype(np.float64))**2))) for k,p in [('before',pa),('after',pb)]}
   details.append(dict(camera=c,region=label,bbox=box,**scores,delta_db=scores['after']-scores['before'],panel=fn))
 rows.append(dict(camera=c,split='heldout' if c in ['00','06','12'] else 'training_diagnostic',before=a['metrics'],after=b['metrics'],
  foreground_delta_db=b['metrics']['foreground_matte_gt0_5']['psnr_db']-a['metrics']['foreground_matte_gt0_5']['psnr_db']))
report=dict(status='six_native_person_and17fixed_detail_panels_verified_visual_review_pending',records=rows,details=details,goal_complete=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(foreground_deltas={r['camera']:r['foreground_delta_db'] for r in rows},detail_deltas={r['camera']+':'+r['region']:r['delta_db'] for r in details})))
