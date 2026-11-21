"""CPU recomputation of all120pairs plus native full/detail panels and temporal residuals."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'comparison';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (O/'summary.json').exists()
branches=['global_cubic','local_bspline'];labels=['Original photograph','Global cubic','Cubic + local B-splines'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
inputs={(r['camera'],r['frame']):r for r in read(R/'input-audit.json')['records']}
reports={b:{(r['camera'],r['frame']):r for r in read(R/b/'native-evaluation/metrics.json')['records']} for b in branches}
assert len(reports[branches[0]])==120 and reports[branches[0]].keys()==reports[branches[1]].keys()
plan=read(R/'review-plan/protocol.json');full={(r['camera'],r['frame']) for r in plan['full_person']};detail={}
for r in plan['details']:detail.setdefault((r['camera'],r['frame']),[]).append(r)
before={b:sha(R/b/'resume.pt') for b in branches};rows=[];panels=[];transitions=[];previous=None;maxerr=0.;imagechecks=0
def panel(camera,frame,name,bb,arrays):
 x0,y0,x1,y1=bb;w=x1-x0;h=y1-y0;im=Image.new('RGB',(w*3,h+32),'white');draw=ImageDraw.Draw(im)
 for k,(arr,label) in enumerate(zip(arrays,labels)):
  crop=arr[y0:y1,x0:x1];im.paste(Image.fromarray(crop),(w*k,32));draw.text((w*k+4,7),label,font=font,fill='black')
 path=O/f'camera{camera}-frame{frame:06d}-{name}.png';im.save(path);actual=np.array(Image.open(path))
 for k,arr in enumerate(arrays):assert np.array_equal(actual[32:,w*k:w*(k+1)],arr[y0:y1,x0:x1])
 panels.append(dict(camera=camera,frame=frame,name=name,bbox=bb,file=path.name,sha256=sha(path)))
for c,f in sorted(reports[branches[0]]):
 prefix=f'camera{c}-frame{f:06d}';ip=R/'native-input'/f'{prefix}.npz';assert sha(ip)==inputs[c,f]['input_sha256'];z=np.load(ip);rgb=z['rgb'];matte=z['matte'];valid=z['valid'];ref=rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);x0,y0,x1,y1=z['bbox'];box=np.zeros_like(valid);box[y0:y1,x0:x1]=True
 masks={'full_valid':valid,'foreground':valid&(matte>.5),'foreground_bbox':valid&box};predictions={};metrics={}
 for b in branches:
  ep=R/b/'native-evaluation';p=np.load(ep/f'{prefix}-prediction.npz');assert np.isfinite(p['rgb']).all() and np.isfinite(p['alpha']).all();pred=np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1);predictions[b]=pred;metrics[b]={}
  for name,m in masks.items():
   d=pred.astype(np.float64)[m]-ref.astype(np.float64)[m];psnr=float(-10*np.log10(np.mean(d*d)));mae=float(abs(d).mean());old=reports[b][c,f]['metrics'][name]
   er=max(abs(psnr-old['psnr_db']),abs(mae-old['mae']));maxerr=max(maxerr,er);assert er<1e-9 and int(m.sum())==old['pixels']
   metrics[b][name]=dict(pixels=int(m.sum()),psnr_db=psnr,mae=mae)
  for suffix,arr in [('trained',u8(pred)),('source',rgb),('reference',u8(ref)),('alpha',u8(p['alpha']))]:
   assert np.array_equal(np.array(Image.open(ep/f'{prefix}-{suffix}.png')),arr);imagechecks+=1
 row=dict(camera=c,frame=f,split=inputs[c,f]['split'],metrics=metrics,foreground_change_db=metrics[branches[1]]['foreground']['psnr_db']-metrics[branches[0]]['foreground']['psnr_db']);rows.append(row)
 arrays=[rgb]+[u8(predictions[b]) for b in branches]
 if (c,f) in full:panel(c,f,'full-person',z['bbox'].tolist(),arrays)
 for r in detail.get((c,f),[]):panel(c,f,r['name'],r['bbox'],arrays)
 if f%2:
  current=dict(camera=c,frame=f,reference=ref,valid=valid,matte=matte,predictions=predictions)
  if previous is not None and previous['camera']==c:
   assert f-previous['frame']==2;m=valid&previous['valid']&((matte>.5)|(previous['matte']>.5));target=(ref-previous['reference']).astype(np.float64)
   values={b:float(abs((predictions[b]-previous['predictions'][b]).astype(np.float64)[m]-target[m]).mean()) for b in branches};transitions.append(dict(camera=c,frames=[previous['frame'],f],split=inputs[c,f]['split'],foreground_union_pixels=int(m.sum()),mean_absolute_temporal_difference_residual=values))
  previous=current
 save(O/'progress.json',dict(pairs=len(rows),PNG_checks=imagechecks,panels=len(panels),temporal_transitions=len(transitions)))
def aggregate(split):
 rr=[r for r in rows if r['split']==split];changes=np.array([r['foreground_change_db'] for r in rr]);return dict(images=len(rr),foreground_psnr_db={b:float(np.mean([r['metrics'][b]['foreground']['psnr_db'] for r in rr])) for b in branches},mean_change_db=float(changes.mean()),improved=int((changes>1e-6).sum()),regressed=int((changes< -1e-6).sum()),worst=sorted([dict(camera=r['camera'],frame=r['frame'],change_db=r['foreground_change_db']) for r in rr],key=lambda r:r['change_db'])[:3])
summary={s:aggregate(s) for s in ['train','validation_time','validation_camera']}
for s in ['validation_time','validation_camera']:
 for r in summary[s]['worst']:
  c,f=r['camera'],r['frame'];prefix=f'camera{c}-frame{f:06d}';z=np.load(R/'native-input'/f'{prefix}.npz');arrays=[z['rgb']]
  for b in branches:
   p=np.load(R/b/'native-evaluation'/f'{prefix}-prediction.npz');arrays.append(u8(np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1)))
  if (c,f) not in full:panel(c,f,'worst-full-person',z['bbox'].tolist(),arrays)
assert len(rows)==120 and imagechecks==960 and len(transitions)==90
assert all(sha(R/b/'resume.pt')==before[b] for b in branches)
for name in ['initial-and-sampling-audit.json']:
 a,b=[read(R/br/name) for br in branches];assert a==b
out=dict(status='120paired_native_predictions_960PNGs_independently_verified_visual_review_pending',splits=summary,temporal_residuals={s:{b:float(np.mean([r['mean_absolute_temporal_difference_residual'][b] for r in transitions if r['split']==s])) for b in branches} for s in ['validation_time','validation_camera']},
 checkpoints=before,identical_initialization_and_sampling=True,max_recomputed_metric_difference=maxerr,PNG_checks=imagechecks,panels=panels,
 qualification='Same-scene new motion interval, one seed; all120pairs evaluated, no upscaling/sharpening. PSNR or temporal residual improvement is not clearance of hands/shoes/occlusion failures. Fullbackground integration still incomplete.',goal_complete=False)
save(O/'records.json',dict(records=rows));save(O/'temporal-records.json',dict(records=transitions));save(O/'summary.json',out);shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(summary),flush=True)
