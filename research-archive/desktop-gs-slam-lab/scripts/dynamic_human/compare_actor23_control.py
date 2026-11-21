"""Recompute native metrics and assemble unchanged-pixel comparisons."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';D=O/'comparison';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
B=['static','dynamic_nominal','dynamic_clock'];labels=['Original','Static GS','Dynamic GS','Dynamic GS + time offset'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
assert not (D/'summary.json').exists()
plan=read(O/'evaluation-protocol.json');inputs={(r['camera'],r['frame']):r for r in plan['records']};reviews=read(O/'review-plan/protocol.json')
reports={b:{(r['camera'],r['frame']):r for r in read(O/b/'native-evaluation/metrics.json')['records']} for b in B}
assert len(inputs)==264 and all(reports[b].keys()==inputs.keys() for b in B)
before={b:sha(O/b/'resume.pt') for b in B};initials={b:read(O/b/'initial-and-sampling-audit.json') for b in B}
for b in B:
    assert before[b]==read(O/b/'completion.json')['checkpoint_sha256']
    assert initials[b]['initial']==initials['static']['initial'] and initials[b]['initial_graph']==initials['static']['initial_graph']
    assert initials[b]['draw_sha256']==initials['static']['draw_sha256']
    sample=np.load(O/b/'sampling-schedule.npz');base=np.load(O/'static/sampling-schedule.npz')
    assert all(np.array_equal(sample[k],base[k]) for k in base.files)
full={(r['camera'],r['frame']) for r in reviews['full_person']};details={}
for r in reviews['details']:details.setdefault((r['camera'],r['frame']),[]).append(r)
rows=[];drows=[];panels=[];transitions=[];previous=None;pngchecks=0;maxerror=0.
def panel(c,f,name,bb,arrays):
    x0,y0,x1,y1=map(int,bb);w=x1-x0;h=y1-y0;im=Image.new('RGB',(4*w,h+32),'white');draw=ImageDraw.Draw(im)
    for k,(arr,label) in enumerate(zip(arrays,labels)):
        im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(k*w,32));draw.text((k*w+4,7),label,font=font,fill='black')
    path=D/f'camera{c}-frame{f:06d}-{name}.png';im.save(path);check=np.array(Image.open(path))
    for k,arr in enumerate(arrays):assert np.array_equal(check[32:,k*w:(k+1)*w],arr[y0:y1,x0:x1])
    panels.append(dict(camera=c,frame=f,name=name,bbox=list(map(int,bb)),file=path.name,sha256=sha(path)))
def pred(b,c,f):
    root=O/b/'native-evaluation';stem=f'camera{c}-frame{f:06d}';r=reports[b][c,f]
    for filename,digest in r['output_sha256'].items():assert sha(root/filename)==digest
    z=np.load(root/f'{stem}-prediction.npz');assert z['rgb'].shape==(1080,1920,3) and z['alpha'].shape==(1080,1920)
    assert np.isfinite(z['rgb']).all() and np.isfinite(z['alpha']).all()
    p=np.clip(z['rgb']+.5*(1-z['alpha'][...,None]),0,1)
    for suffix,array in [('trained',u8(p)),('alpha',u8(z['alpha']))]:assert np.array_equal(np.array(Image.open(root/f'{stem}-{suffix}.png')),array)
    return p
for c,f in sorted(inputs):
    r=inputs[c,f];path=O/r['input'];assert sha(path)==r['input_sha256'];z=np.load(path);rgb=z['rgb'];matte=z['matte'];valid=z['valid'];target=rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);x0,y0,x1,y1=z['bbox'];box=np.zeros_like(valid);box[y0:y1,x0:x1]=True
    preds={b:pred(b,c,f) for b in B};pngchecks+=6;values={b:{} for b in B}
    for b in B:
        assert reports[b][c,f]['input_sha256']==r['input_sha256']
        for name,mask in dict(full_valid=valid,foreground=valid&(matte>.5),foreground_bbox=valid&box).items():
            e=preds[b].astype(np.float64)[mask]-target.astype(np.float64)[mask];m=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(e*e))),mae=float(abs(e).mean()));old=reports[b][c,f]['metrics'][name];err=max(abs(m['psnr_db']-old['psnr_db']),abs(m['mae']-old['mae']));maxerror=max(maxerror,err);assert err<1e-9 and m['pixels']==old['pixels'];values[b][name]=m
    rows.append(dict(camera=c,frame=f,split=r['split'],metrics=values,dynamic_vs_static_db={b:values[b]['foreground']['psnr_db']-values['static']['foreground']['psnr_db'] for b in B[1:]},clock_vs_nominal_db=values['dynamic_clock']['foreground']['psnr_db']-values['dynamic_nominal']['foreground']['psnr_db']))
    arrays=[rgb]+[u8(preds[b]) for b in B]
    if (c,f) in full:panel(c,f,'full-person',z['bbox'].tolist(),arrays)
    for d in details.get((c,f),[]):
        panel(c,f,d['name'],d['bbox'],arrays);xx0,yy0,xx1,yy1=d['bbox'];mask=np.zeros_like(valid);mask[yy0:yy1,xx0:xx1]=True;mask&=valid&(matte>.5);assert mask.sum()>0;v={}
        for b in B:
            e=preds[b].astype(np.float64)[mask]-target.astype(np.float64)[mask];v[b]=dict(psnr_db=float(-10*np.log10(np.mean(e*e))),mae=float(abs(e).mean()))
        drows.append(dict(camera=c,frame=f,name=d['name'],split=r['split'],bbox=d['bbox'],window_basis=d['window_basis'],pixels=int(mask.sum()),metrics=v))
    if f%2:
        if previous is not None and previous['camera']==c:
            assert f-previous['frame']==2;mask=valid&previous['valid']&((matte>.5)|(previous['matte']>.5));dt=(target-previous['target']).astype(np.float64)
            transitions.append(dict(camera=c,frames=[previous['frame'],f],split=r['split'],pixels=int(mask.sum()),mean_absolute_difference_residual={b:float(abs((preds[b]-previous['preds'][b]).astype(np.float64)[mask]-dt[mask]).mean()) for b in B}))
        previous=dict(camera=c,frame=f,valid=valid,matte=matte,target=target,preds=preds)
    save(D/'progress.json',dict(images=len(rows),PNG_checks=pngchecks,panels=len(panels),odd_frame_transitions=len(transitions)))
    if len(rows)%24==0:print('ACTOR23_COMPARE',len(rows),flush=True)
assert len(rows)==264 and pngchecks==1584 and len(drows)==72 and len(transitions)==198
splits={}
for split in sorted(plan['split_counts']):
    rr=[r for r in rows if r['split']==split]
    splits[split]=dict(images=len(rr),foreground_psnr_db={b:float(np.mean([r['metrics'][b]['foreground']['psnr_db'] for r in rr])) for b in B},clock_vs_nominal=dict(mean_change_db=float(np.mean([r['clock_vs_nominal_db'] for r in rr])),improved=sum(r['clock_vs_nominal_db']>1e-6 for r in rr),regressed=sum(r['clock_vs_nominal_db']< -1e-6 for r in rr),worst=sorted([dict(camera=r['camera'],frame=r['frame'],change_db=r['clock_vs_nominal_db']) for r in rr],key=lambda r:r['change_db'])[:3]))
    if split!='train_check':
        for worst in splits[split]['clock_vs_nominal']['worst']:
            c,f=worst['camera'],worst['frame']
            if (c,f) in full or any(p['camera']==c and p['frame']==f and p['name']=='worst-full-person' for p in panels):continue
            z=np.load(O/inputs[c,f]['input']);panel(c,f,'worst-full-person',z['bbox'].tolist(),[z['rgb']]+[u8(pred(b,c,f)) for b in B])
assert all(sha(O/b/'resume.pt')==before[b] for b in B)
summary=dict(status='264contexts_1584PNGs_verified_native_visual_review_required',splits=splits,detail_summary={n:{b:float(np.mean([r['metrics'][b]['psnr_db'] for r in drows if r['name']==n])) for b in B} for n in sorted({r['name'] for r in drows})},temporal_difference_residual={s:{b:float(np.mean([r['mean_absolute_difference_residual'][b] for r in transitions if r['split']==s])) for b in B} for s in ['validation_time','validation_camera_time']},checkpoints=before,max_metric_recomputation_error=maxerror,PNG_checks=pngchecks,panels=panels,identical_initial_parameters_and_sampling=True,scope='Independent actors, same geographic recording environment. Fixed15000updates/195photos per model; compute and converged Gaussian counts may differ. Calibration-visible other cameras are appearance/geometry-heldout. Metrics use uncertain author matte, include props, and are image-wise averages. Not full-background reconstruction or official4K4D/OmniRe result.',goal_complete=False)
save(D/'records.json',dict(records=rows));save(D/'detail-metrics.json',dict(records=drows));save(D/'temporal-records.json',dict(records=transitions));save(D/'summary.json',summary);shutil.copy2(__file__,D/Path(__file__).name);print(json.dumps(dict(splits=splits,details=summary['detail_summary'])),flush=True)
