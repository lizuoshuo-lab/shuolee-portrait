"""Frozen 044 geometry/color: measure native point radii and render radius controls."""
from pathlib import Path
from types import MethodType
import json,hashlib,random,torch,numpy as np
from PIL import Image,ImageDraw
from easyvolcap.engine import cfg,MODELS,DATASETS
from easyvolcap.scripts.main import preflight
from easyvolcap.utils.data_utils import default_collate,to_cuda,add_iter
from fourkd_source_support import enable_source_foreground_support
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044/source_support_blending'
OUT=LAB/'outputs/dynamic-human-4k4d-point-footprints-046';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
assert not (OUT/'summary.json').exists()
assert read(BASE/'independent-state-audit.json')['optimizer_max_step']==3000
checkpoint=BASE/'model/latest.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
protocol=dict(status='fixed_before_readonly_probe',parent_sha256=sha,history_updates=3000,
    variants={'original':1.0,'radius075':.75,'radius050':.5},cameras=['00','06','12','01','08','17'],
    intervention='Only multiply learned world-space point radius at rasterization. Same geometry, source-support color and opacity. Each radius needs its own rasterization and alpha.',
    exclusion='No targetRGB/matte in forward, no training or image postprocessing. Radius controls are diagnostic, not a validated model.',goal_complete=False)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
torch.set_num_threads(8);random.seed(0);torch.manual_seed(0)
cfg.model_cfg.sampler_cfg.skip_loading_points=True
model=MODELS.build(cfg.model_cfg).cuda();cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
model.load_state_dict(cp['model'],strict=True);model.eval();del cp
enable_source_foreground_support(model.sampler)
dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg);captured={}
def quant(x):return np.quantile(x,[0,.1,.5,.9,.99,1]).tolist()
def render(self,xyz,rgb,rad,occ,batch):
    full=self.render_pytorch(xyz,rgb,rad,occ,batch,return_full=True)
    original=full[:3];idx,a=full[3:]
    # Positive contributors only; zeroed padding indices do not count.
    used=torch.unique(idx[a>1e-4]);captured['used']=used.cpu().numpy()
    depth=(xyz@batch.R.mT+batch.T.mT)[...,2]
    projected=batch.K[:,1,1,None]*rad[...,0]/depth
    captured['stats']=dict(world_radius_quantiles=quant(rad.detach().cpu().numpy()),
        opacity_quantiles=quant(occ.detach().cpu().numpy()),
        contributing_native_radius_px_quantiles=quant(projected[0,used].cpu().numpy()),
        contributing_points=len(used),points=xyz.shape[1])
    captured['variants']={}
    for key,scale in [('radius075',.75),('radius050',.5)]:
        color,alpha,_=self.render_pytorch(xyz,rgb,rad*scale,occ,batch)
        captured['variants'][key]=(color.cpu().reshape(1080,1920,3).numpy(),alpha.cpu().reshape(1080,1920).numpy())
    if not (OUT/'geometry.npz').exists():
        np.savez_compressed(OUT/'geometry.npz',xyz=xyz[0].cpu().numpy(),radius=rad[0,:,0].cpu().numpy(),opacity=occ[0,:,0].cpu().numpy())
    captured['radii']=projected[0].cpu().numpy()
    return original
model.sampler.render_points=MethodType(render,model.sampler)
rows=[];u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
for row in read(BASE/'native-evaluation/metrics.json')['records']:
    name=row['camera'];c=int(name);prefix=f'camera{name}-frame000750'
    z=np.load(BASE/'native-evaluation'/f'{prefix}-raw.npz')
    dataset.append_gt_prob=1. if c in [0,6,12] else 0.;item=dataset[c]
    sources=[int(dataset.view_inds[int(i)]) for i in item.src_inds]
    assert sources==row['source_cameras'] and c not in sources and not set(sources)&{0,6,12}
    item.K=torch.tensor(z['K'],dtype=torch.float32);item.meta.K=item.K;item.tar_ixt=item.K
    item.H=1080;item.W=1920;item.meta.H=1080;item.meta.W=1920
    for key in ['rgb','msk','wet','dpt','bkg','norm','ray_o','ray_d','near','far','coords']:item.pop(key,None);item.meta.pop(key,None)
    batch=to_cuda(add_iter(default_collate([item]),3000,3000))
    with torch.no_grad():result=model(batch)
    rgb=result.rgb_map.cpu().reshape(1080,1920,3).numpy();alpha=result.acc_map.cpu().reshape(1080,1920).numpy()
    delta=np.abs(rgb-z['rgb']);assert delta.mean()<2e-7 and (delta>1/255).sum()<100 and np.max(abs(alpha-z['alpha']))<2e-5
    variants={'original':(rgb,alpha),**captured['variants']};pred={k:np.clip(v+.5*(1-a[...,None]),0,1) for k,(v,a) in variants.items()}
    ref=z['source'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None])
    metrics={};raw=dict(projected_radius=captured['radii'],contributing_indices=captured['used'])
    for key,(v,a) in variants.items():
        raw[key+'_rgb']=v;raw[key+'_alpha']=a
        metrics[key]={label:float(-10*np.log10(np.mean((pred[key][m].astype(np.float64)-ref[m].astype(np.float64))**2)))
                      for label,m in [('foreground',z['valid']&(z['matte']>.5)),('full_valid',z['valid'])]}
        Image.fromarray(u8(pred[key])).save(OUT/f'{prefix}-{key}.png')
    np.savez_compressed(OUT/f'{prefix}-raw.npz',**raw)
    x0,y0,x1,y1=row['bbox'];w=x1-x0;h=y1-y0;panel=Image.new('RGB',(4*w,h+32),'white');d=ImageDraw.Draw(panel)
    for i,(label,im) in enumerate([('Original photograph',z['source']),('Learned radius',u8(pred['original'])),('Radius x0.75',u8(pred['radius075'])),('Radius x0.50',u8(pred['radius050']))]):
        panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*w,32));d.text((i*w+4,10),label,fill='black')
    panel.save(OUT/f'{prefix}-comparison.png')
    rows.append(dict(camera=name,split=row['split'],sources=sources,bbox=row['bbox'],metrics=metrics,statistics=captured['stats'],
                     replay_rgb_meanerr=float(delta.mean()),replay_channels_gt_u8=int((delta>1/255).sum())))
    print('FOOTPRINT',json.dumps(rows[-1]),flush=True)
    del result,batch,item;torch.cuda.empty_cache()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha
(OUT/'summary.json').write_text(json.dumps(dict(status='six_readonly_radius_controls_complete_audit_review_pending',records=rows,checkpoint_unchanged=True,goal_complete=False),indent=2)+'\n')
