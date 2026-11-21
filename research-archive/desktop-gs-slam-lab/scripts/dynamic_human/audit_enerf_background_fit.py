"""Frozen state, original source, PNG metrics and native comparison checks."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from enerf_background_data import load_background
from audit_enerf_input import digest
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run);out=run/'evaluation';root=Path(a.root);protocol=json.loads((run/'protocol.json').read_text());metric=json.loads((out/'metrics.json').read_text())
cp=torch.load(run/'resume.pt',map_location='cpu',weights_only=False);parent=torch.load(run.parent/'resume.pt',map_location='cpu',weights_only=False)
assert cp['step']==6000 and parent['step']==2401
assert digest(run.parent/'resume.pt')==protocol['recovery']['parent_sha256']
assert digest(run/'resume.pt')==metric['checkpoint_sha256']
params=[]
for name,v in cp['params'].items():
    assert len(v)==532803 and v.shape==parent['params'][name].shape and torch.isfinite(v).all()
    delta=(v-parent['params'][name]).abs();params.append(dict(name=name,shape=list(v.shape),finite=True,mean_abs_parent_change=float(delta.mean()),max_abs_parent_change=float(delta.max())))
cache_checks=[]
for o in load_background(root,protocol['train_cameras']):
    cache=np.load(run/f'train-camera-{o["camera"]}.npz')
    for k in ['rgb','valid','K','w2c']:assert np.array_equal(cache[k],o[k]),(o['camera'],k)
    assert o['sha256']==next(s['sha256'] for s in protocol['sources'] if s['camera']==o['camera'])
    cache_checks.append(o['camera'])
checks=[];font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18);panels=[]
for o in load_background(root,['00','06','12','01','08','17']):
    c=o['camera'];reference=np.array(Image.open(out/f'camera{c}-reference.png'));assert np.array_equal(reference,o['rgb']);valid=o['valid'];assert np.array_equal(np.array(Image.open(out/f'camera{c}-valid.png'))>0,valid)
    for model in ['initial','trained']:
        pred=np.array(Image.open(out/f'camera{c}-{model}.png'));al=np.array(Image.open(out/f'camera{c}-{model}-alpha.png'));row=next(r for r in metric['rows'] if r['camera']==c and r['model']==model);assert row['reference_sha256']==o['sha256']
        for name,m in [('full_valid',valid),('top_rows0_800',valid&(np.arange(1080)[:,None]<800)),('bottom_rows900_1080',valid&(np.arange(1080)[:,None]>=900))]:
            squared=(pred.astype('float64')-reference.astype('float64'))**2;psnr=10*np.log10(255**2/np.mean(squared[m]));assert abs(psnr-row['scores'][name]['psnr_db'])<1e-9
            assert abs(float((al[m]<243).mean())-row['scores'][name]['alpha_below_point95_fraction'])<1e-12
        checks.append(dict(camera=c,model=model,reference_exact=True,metrics_verified=True,rgb_sha256=digest(out/f'camera{c}-{model}.png'),alpha_sha256=digest(out/f'camera{c}-{model}-alpha.png')))
    images=[Image.open(out/f'camera{c}-{g}.png').convert('RGB') for g in ['reference','initial','trained']]
    for name,box in [('wall',[400,250,800,650]),('bottom',[800,880,1400,1080])]:
        w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(3*w,h+44),'white');d=ImageDraw.Draw(panel)
        for j,(im,label) in enumerate(zip(images,['Reference','Initial sparse points','Trained background'])):
            panel.paste(im.crop(box),(j*w,44));d.text((j*w+7,12),label,font=font,fill='black')
        path=out/f'camera{c}-{name}-comparison.png';panel.save(path);panels.append(dict(file=path.name,box=box,resized=False))
result=dict(status='background_state15source_caches12RGB12alpha_and_metrics_verified_visual_review_separate',parameters=params,train_source_caches=cache_checks,checks=checks,panels=panels,
    parent_stopped_step=2401,final_step=6000,points=532803,scope='Shared source rectification routine; saved source pixels/hashes and independent double-precision image metrics checked. Empty background only.')
(run/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
(run.parent/'stopped-run.json').write_text(json.dumps(dict(status='stopped_at_density_guard_not_completed',step=2401,points=532803,reason='DefaultStrategy growth exceeded500000 configured engineering budget. Terminal process exit1. No OOM or nonfinite evidence; parent checkpoint retained.',continuation='density-capped',continuation_changes='No further growth/pruning/opacity resets; optimizer and random state continue to6000. Not a clean original-protocol6000run.'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=result['status'],images=len(checks),points=532803)))
