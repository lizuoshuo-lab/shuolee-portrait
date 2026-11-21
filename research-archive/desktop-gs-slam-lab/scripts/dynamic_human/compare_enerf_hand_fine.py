"""Independent original-target comparison of parent and both matched local joint controls."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';T=L/'outputs/dynamic-human-enerf-hand-fine-085';O=T/'comparison';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (O/'summary.json').exists();branches=['parent','fine_photo','fine_flow'];roots={'parent':S/'local_bspline',**{b:T/b for b in branches[1:]}};labels=['Original','Parent model','Fine photo','Fine + motion'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
inputs={(r['camera'],r['frame']):r for r in read(S/'input-audit.json')['records']};reports={b:{(r['camera'],r['frame']):r for r in read(roots[b]/'native-evaluation/metrics.json')['records']} for b in branches};assert len(reports['parent'])==120 and all(reports[b].keys()==reports['parent'].keys() for b in branches)
plan=read(S/'review-plan/protocol.json');full={(r['camera'],r['frame']) for r in plan['full_person']};detail={}
for r in plan['details']:detail.setdefault((r['camera'],r['frame']),[]).append(r)
for r in read(T/'target-hand-review-plan.json')['records']:detail.setdefault((r['camera'],r['frame']),[]).append(dict(r,name='target_right_hand'))
before={b:sha(roots[b]/'resume.pt') for b in branches};rows=[];panels=[];details=[];transitions=[];previous=None;maxerr=0.;pngchecks=0
def panel(c,f,name,bb,arrays):
 x0,y0,x1,y1=bb;w=x1-x0;h=y1-y0;im=Image.new('RGB',(4*w,h+32),'white');draw=ImageDraw.Draw(im)
 for k,(arr,label) in enumerate(zip(arrays,labels)):
  im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(w*k,32));draw.text((w*k+4,7),label,font=font,fill='black')
 path=O/f'camera{c}-frame{f:06d}-{name}.png';im.save(path);actual=np.array(Image.open(path))
 for k,arr in enumerate(arrays):assert np.array_equal(actual[32:,w*k:w*(k+1)],arr[y0:y1,x0:x1])
 panels.append(dict(camera=c,frame=f,name=name,bbox=bb,file=path.name,sha256=sha(path)))
for c,f in sorted(reports['parent']):
 stem=f'camera{c}-frame{f:06d}';ip=S/'native-input'/f'{stem}.npz';assert sha(ip)==inputs[c,f]['input_sha256'];z=np.load(ip);rgb=z['rgb'];matte=z['matte'];valid=z['valid'];target=rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);x0,y0,x1,y1=z['bbox'];box=np.zeros_like(valid);box[y0:y1,x0:x1]=True;masks={'full_valid':valid,'foreground':valid&(matte>.5),'foreground_bbox':valid&box};preds={};metrics={}
 for b in branches:
  ep=roots[b]/'native-evaluation';p=np.load(ep/f'{stem}-prediction.npz');assert np.isfinite(p['rgb']).all() and np.isfinite(p['alpha']).all();preds[b]=np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1);metrics[b]={}
  for name,m in masks.items():
   d=preds[b].astype(np.float64)[m]-target.astype(np.float64)[m];psnr=float(-10*np.log10(np.mean(d*d)));mae=float(abs(d).mean());reported=reports[b][c,f]['metrics'][name];error=max(abs(psnr-reported['psnr_db']),abs(mae-reported['mae']));maxerr=max(maxerr,error);assert error<1e-9 and int(m.sum())==reported['pixels'];metrics[b][name]=dict(pixels=int(m.sum()),psnr_db=psnr,mae=mae)
  for suffix,arr in [('trained',u8(preds[b])),('alpha',u8(p['alpha']))]:assert np.array_equal(np.array(Image.open(ep/f'{stem}-{suffix}.png')),arr);pngchecks+=1
 rows.append(dict(camera=c,frame=f,split=inputs[c,f]['split'],metrics=metrics,changes_db={b:metrics[b]['foreground']['psnr_db']-metrics['parent']['foreground']['psnr_db'] for b in branches[1:]},flow_vs_photo_db=metrics['fine_flow']['foreground']['psnr_db']-metrics['fine_photo']['foreground']['psnr_db']))
 arrays=[rgb]+[u8(preds[b]) for b in branches]
 if (c,f) in full:panel(c,f,'full-person',z['bbox'].tolist(),arrays)
 for r in detail.get((c,f),[]):
  panel(c,f,r['name'],r['bbox'],arrays);xx0,yy0,xx1,yy1=r['bbox'];roi=np.zeros_like(valid);roi[yy0:yy1,xx0:xx1]=True;m=roi&valid&(matte>.5);vals={}
  for b in branches:
   d=preds[b].astype(np.float64)[m]-target.astype(np.float64)[m];vals[b]=dict(psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
  details.append(dict(camera=c,frame=f,name=r['name'],bbox=r['bbox'],split=inputs[c,f]['split'],pixels=int(m.sum()),metrics=vals,changes_db={b:vals[b]['psnr_db']-vals['parent']['psnr_db'] for b in branches[1:]},flow_vs_photo_db=vals['fine_flow']['psnr_db']-vals['fine_photo']['psnr_db']))
 if f%2:
  now=dict(camera=c,frame=f,target=target,valid=valid,matte=matte,preds=preds)
  if previous is not None and previous['camera']==c:
   assert f-previous['frame']==2;m=valid&previous['valid']&((matte>.5)|(previous['matte']>.5));dt=(target-previous['target']).astype(np.float64);vals={b:float(abs((preds[b]-previous['preds'][b]).astype(np.float64)[m]-dt[m]).mean()) for b in branches};transitions.append(dict(camera=c,frames=[previous['frame'],f],split=inputs[c,f]['split'],foreground_union_pixels=int(m.sum()),mean_absolute_temporal_difference_residual=vals))
  previous=now
 save(O/'progress.json',dict(cases=len(rows),PNG_checks=pngchecks,panels=len(panels),transitions=len(transitions)))
summary={}
for split in ['train','validation_time','validation_camera']:
 rr=[r for r in rows if r['split']==split];summary[split]=dict(images=len(rr),foreground_psnr_db={b:float(np.mean([r['metrics'][b]['foreground']['psnr_db'] for r in rr])) for b in branches},versus_parent={})
 for b in branches[1:]:
  ds=np.array([r['changes_db'][b] for r in rr]);summary[split]['versus_parent'][b]=dict(mean_change_db=float(ds.mean()),improved=int((ds>1e-6).sum()),regressed=int((ds< -1e-6).sum()),worst=sorted([dict(camera=r['camera'],frame=r['frame'],change_db=r['changes_db'][b]) for r in rr],key=lambda r:r['change_db'])[:3])
for split in ['validation_time','validation_camera']:
 for b in branches[1:]:
  for r in summary[split]['versus_parent'][b]['worst']:
   c,f=r['camera'],r['frame'];name='worst-full-person'
   if (c,f) in full or any(x['camera']==c and x['frame']==f and x['name']==name for x in panels):continue
   stem=f'camera{c}-frame{f:06d}';z=np.load(S/'native-input'/f'{stem}.npz');arrays=[z['rgb']]
   for br in branches:
    p=np.load(roots[br]/'native-evaluation'/f'{stem}-prediction.npz');arrays.append(u8(np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1)))
   panel(c,f,name,z['bbox'].tolist(),arrays)
assert len(rows)==120 and pngchecks==720 and len(transitions)==90;assert all(sha(roots[b]/'resume.pt')==before[b] for b in branches);a,b=[dict(np.load(T/br/'schedule.npz')) for br in branches[1:]];assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
detail_summary={name:dict(count=sum(r['name']==name for r in details),changes_db={b:float(np.mean([r['changes_db'][b] for r in details if r['name']==name])) for b in branches[1:]},flow_vs_photo_db=float(np.mean([r['flow_vs_photo_db'] for r in details if r['name']==name]))) for name in sorted({r['name'] for r in details})}
out=dict(status='120contexts_3models_720PNGs_independently_verified_visual_review_pending',splits=summary,detail_summary=detail_summary,temporal_residuals={s:{b:float(np.mean([r['mean_absolute_temporal_difference_residual'][b] for r in transitions if r['split']==s])) for b in branches} for s in ['validation_time','validation_camera']},checkpoints=before,identical_initialization_and_sampling=True,max_recomputed_metric_difference=maxerr,PNG_checks=pngchecks,panels=panels,scope='Same-scene development. Original079eval RGB/mattes; targethandadditionalwindows reused unchanged from084. Nojointstaticbackground ornewsceneacceptance.',goal_complete=False)
save(O/'detail-metrics.json',dict(records=details,scope='Source-definedcontext windows intersectoriginalmatte;notexactsemanticbodypartmasks.'));save(O/'records.json',dict(records=rows));save(O/'temporal-records.json',dict(records=transitions));save(O/'summary.json',out);shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(dict(splits=summary,details=detail_summary)),flush=True)
