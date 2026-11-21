"""Read-only real-model alpha diagnostic; never changes weights or source images."""
from pathlib import Path
from types import MethodType
import json,hashlib,random,torch,numpy as np
from PIL import Image,ImageDraw
from easyvolcap.engine import cfg,MODELS,DATASETS
from easyvolcap.scripts.main import preflight
from easyvolcap.utils.data_utils import default_collate,to_cuda,add_iter
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/dynamic-human-4k4d-projection-control-040/original_projection'
OUT=LAB/'outputs/dynamic-human-4k4d-alpha-composition-041';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
assert read(OUT/'cpu-semantic-check.json')['raw_rgb_unchanged']
assert read(BASE/'training-completion.json')['history_updates']==2000
assert not (OUT/'real-model-summary.json').exists()
torch.set_num_threads(8);random.seed(0);torch.manual_seed(0)
checkpoint=BASE/'model/latest.pt';checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
(OUT/'protocol.json').write_text(json.dumps(dict(status='frozen_before_readonly_probe',parent=str(checkpoint),parent_sha256=checkpoint_hash,
    cameras=['00','06','12','01','08','17'],frame=750,intervention='Same frozen model and rasterizer fragments. Replace accumulated opacity-valued feature by accumulated ones for background composition only.',
    scope='Read-only diagnostic, no training. Does not claim improved geometry or texture.',goal_complete=False),indent=2)+'\n')
cfg.model_cfg.sampler_cfg.skip_loading_points=True
model=MODELS.build(cfg.model_cfg).cuda();cp=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval();del cp
dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg)
captured={}
def render(self,pcd,rgb,rad,occ,batch):
    old_rgb,old_acc,dpt,idx,pix_occ=self.render_pytorch(pcd,rgb,rad,occ,batch,return_full=True)
    # This expression uses exact actual depth-ordered per-pixel alphas.
    correct_acc=1-torch.prod(1-pix_occ,dim=-1,keepdim=True)
    assert (correct_acc>=old_acc-2e-5).all() and correct_acc.max()<=1.00001
    yy,xx=torch.meshgrid(torch.arange(0,1080,29,device=pcd.device),torch.arange(0,1920,31,device=pcd.device),indexing='ij')
    samples=pix_occ[0,yy,xx];sample_opacities=occ.flatten()[idx[0,yy,xx]]
    captured.update(correct_acc=correct_acc.detach().cpu().reshape(1080,1920).numpy(),
        sample_y=yy.cpu().numpy(),sample_x=xx.cpu().numpy(),sample_pixel_alpha=samples.cpu().numpy(),sample_point_opacity=sample_opacities.cpu().numpy(),
        opacity_quantiles=torch.quantile(occ.float().flatten(),torch.tensor([0.,.1,.5,.9,1.],device=occ.device)).cpu().tolist())
    return old_rgb,old_acc,dpt
model.sampler.render_points=MethodType(render,model.sampler)
rows=[];u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
for r in read(BASE/'native-evaluation/metrics.json')['records']:
    name=r['camera'];c=int(name);prefix=f'camera{name}-frame000750';reference=np.load(BASE/'native-evaluation'/f'{prefix}-raw.npz')
    dataset.append_gt_prob=1. if c in [0,6,12] else 0.;item=dataset[c]
    sources=[int(dataset.view_inds[int(i)]) for i in item.src_inds];assert sources==r['source_cameras']
    assert c not in sources and not set(sources)&{0,6,12}
    item.K=torch.tensor(reference['K'],dtype=torch.float32);item.meta.K=item.K;item.tar_ixt=item.K
    item.H=1080;item.W=1920;item.meta.H=1080;item.meta.W=1920
    for key in ['rgb','msk','wet','dpt','bkg','norm','ray_o','ray_d','near','far','coords']:
        item.pop(key,None);item.meta.pop(key,None)
    batch=to_cuda(add_iter(default_collate([item]),2000,2000))
    with torch.no_grad():result=model(batch)
    rgb=result.rgb_map.detach().cpu().reshape(1080,1920,3).numpy();old_alpha=result.acc_map.detach().cpu().reshape(1080,1920).numpy();alpha=captured['correct_acc']
    rgb_err=float(np.max(np.abs(rgb-reference['rgb'])));alpha_err=float(np.max(np.abs(old_alpha-reference['alpha'])))
    delta=np.abs(rgb-reference['rgb'])
    diagnostic=dict(camera=name,rgb_maxerr=rgb_err,rgb_meanerr=float(delta.mean()),rgb_rmserr=float(np.sqrt(np.mean(delta**2))),channels_above_one_u8=int((delta>1/255).sum()),alpha_maxerr=alpha_err)
    (OUT/f'{prefix}-reproduction-check.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    if rgb_err>=2e-5:np.savez_compressed(OUT/f'{prefix}-reproduction-difference.npz',current_rgb=rgb,previous_rgb=reference['rgb'],current_alpha=old_alpha,previous_alpha=reference['alpha'])
    # Separate-process rasterization has sparse RGB differences (camera00: six
    # channels above one u8, mean1.35e-8). Compare both alpha variants within the
    # SAME fragments; do not claim bit-exact replay of the earlier RGB image.
    assert diagnostic['rgb_meanerr']<2e-7 and diagnostic['channels_above_one_u8']<100 and alpha_err<2e-5,diagnostic
    before=np.clip(rgb+.5*(1-old_alpha[...,None]),0,1);after=np.clip(rgb+.5*(1-alpha[...,None]),0,1)
    source=reference['source'];matte=reference['matte'];valid=reference['valid'];ref=source.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None])
    metrics={}
    for label,mask in [('valid',valid),('foreground',valid&(matte>.5)),('render_boundary',valid&(alpha>.01)&(alpha<.99))]:
        diff=alpha[mask]-old_alpha[mask]
        metrics[label]=dict(pixels=int(mask.sum()),mean_alpha_delta=float(diff.mean()),max_alpha_delta=float(diff.max()),
            before_psnr=float(-10*np.log10(np.mean((before[mask].astype(np.float64)-ref[mask].astype(np.float64))**2))),
            after_psnr=float(-10*np.log10(np.mean((after[mask].astype(np.float64)-ref[mask].astype(np.float64))**2))))
    for suffix,array in [('source',source),('legacy',u8(before)),('standard-alpha',u8(after)),('legacy-alpha',u8(old_alpha)),('standard-alpha-map',u8(alpha))]:
        Image.fromarray(array).save(OUT/f'{prefix}-{suffix}.png')
    np.savez_compressed(OUT/f'{prefix}-raw.npz',rgb=rgb,legacy_alpha=old_alpha,standard_alpha=alpha,
        **{k:captured[k] for k in ['sample_y','sample_x','sample_pixel_alpha','sample_point_opacity']})
    x0,y0,x1,y1=r['bbox'];w=x1-x0;h=y1-y0;im=Image.new('RGB',(3*w,h+32),'white');draw=ImageDraw.Draw(im)
    for i,(label,a) in enumerate([('Original photo',source),('Legacy alpha',u8(before)),('Accumulated alpha',u8(after))]):
        im.paste(Image.fromarray(a[y0:y1,x0:x1]),(i*w,32));draw.text((i*w+5,10),label,fill='black')
    im.save(OUT/f'{prefix}-comparison.png')
    rows.append(dict(camera=name,split=r['split'],sources=sources,bbox=r['bbox'],metrics=metrics,opacity_quantiles=captured['opacity_quantiles'],baseline_reproduction=diagnostic))
    print('ALPHA_CHECK',json.dumps(rows[-1]),flush=True)
    del batch,item,result;torch.cuda.empty_cache()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_hash
(OUT/'real-model-summary.json').write_text(json.dumps(dict(status='six_readonly_views_complete_independent_audit_and_review_pending',records=rows,checkpoint_unchanged=True,goal_complete=False),indent=2)+'\n')
