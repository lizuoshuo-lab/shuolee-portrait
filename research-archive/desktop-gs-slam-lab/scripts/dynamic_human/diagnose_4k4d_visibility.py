"""Frozen-geometry controls for premultiplied source colors outside foreground support."""
from pathlib import Path
from types import MethodType
import json,hashlib,random,torch,numpy as np
from PIL import Image,ImageDraw
from easyvolcap.engine import cfg,MODELS,DATASETS
from easyvolcap.scripts.main import preflight
from easyvolcap.utils.data_utils import default_collate,to_cuda,add_iter
from easyvolcap.utils.ibr_utils import compute_src_inps,sample_geometry_feature_image
from easyvolcap.utils.image_utils import pad_image
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.base_utils import dotdict
from audit_enerf_input import read_cameras
from fourkd_source_support import enable_source_foreground_support
import torch.nn.functional as F
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044/source_support_blending'
OUT=LAB/'outputs/dynamic-human-4k4d-visibility-047';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
assert read(BASE/'independent-state-audit.json')['optimizer_max_step']==3000
assert not (OUT/'summary.json').exists()
torch.set_num_threads(8);random.seed(0);torch.manual_seed(0)
checkpoint=BASE/'model/latest.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
(OUT/'protocol.json').write_text(json.dumps(dict(status='frozen_before_readonly_probe',parent_sha256=sha,frame=750,cameras=['00','06','12','01','08','17'],
    variants=['original','source_visibility1','source_visibility2'],
    intervention='Reweight existing mask-supported blend weights by model-derived source visibility. Native zbuffer first point with radial alpha>0.25. Point visible if z<=frontdepth+tau*worldradius, tau1 or2. Nearest pixel sampling. No visible source falls back to original supported blend. Original SH residual retained. Unsupported points fall back to original color.',
    fixed='All model parameters, point geometry/radii/opacities, original projection, eight training sources and legacy renderer alpha. No target RGB/matte in forward.',
    qualification='Foreground support is not occlusion visibility; support masks are imperfect. Inference-only counterfactual, not retrained model or completed task.',goal_complete=False),indent=2)+'\n')
cfg.model_cfg.sampler_cfg.skip_loading_points=True
model=MODELS.build(cfg.model_cfg).cuda();cp=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval();del cp
enable_source_foreground_support(model.sampler)
dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg);captured={}
cameras=read_cameras(Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6/optimized'))
geo=np.load(LAB/'outputs/dynamic-human-4k4d-point-footprints-046/geometry.npz')
X=torch.tensor(geo['xyz'],device='cuda')[None];radius=torch.tensor(geo['radius'],device='cuda')[None,:,None];opacity=torch.tensor(geo['opacity'],device='cuda')[None,:,None]
visibility={};zbuffer_audit=[]
with torch.no_grad():
    for c in range(18):
        if c in [0,6,12]:continue
        cam=cameras[f'{c:02d}'];K=torch.tensor(cam['K'],dtype=torch.float32,device='cuda')[None];w2c=torch.tensor(cam['w2c'],dtype=torch.float32,device='cuda')
        b=dotdict(K=K,R=w2c[:3,:3][None],T=w2c[:3,3][None,:,None],meta=dotdict(H=torch.tensor([1080]),W=torch.tensor([1920])))
        idx,alpha=model.sampler.render_pytorch(X,torch.zeros_like(X),radius,opacity,b,return_frags=True)
        valid=alpha>.25;first=valid.to(torch.int32).argmax(-1);has=valid.any(-1)
        pidx=idx.gather(-1,first[...,None])[...,0]
        pc=X@b.R.mT+b.T.mT;depth=pc[...,2]
        front=depth[0,pidx];front=torch.where(has,front,0)
        uv=pc@K.mT;uv=uv[...,:2]/uv[...,2:]
        grid=(uv+.5)/torch.tensor([1920,1080],device='cuda')*2-1
        front_at=F.grid_sample(front[:,None],grid[:,None],align_corners=False,mode='nearest',padding_mode='zeros')[0,0,0]
        inside=(uv[0,:,0]>=0)&(uv[0,:,0]<1920)&(uv[0,:,1]>=0)&(uv[0,:,1]<1080)&(depth[0]>0)&(front_at>0)
        visibility[c]=torch.stack([inside&(depth[0]<=front_at+tau*radius[0,:,0]) for tau in [1.,2.]])
        np.savez_compressed(OUT/f'source{c:02d}-visibility.npz',front_depth=front[0].cpu().numpy(),point_depth=depth[0].cpu().numpy(),point_uv=uv[0].cpu().numpy(),sampled_front=front_at.cpu().numpy(),inside=inside.cpu().numpy(),visible=visibility[c].cpu().numpy())
        zbuffer_audit.append(dict(camera=c,occupied_pixels=int(has.sum()),visible_points=[int(v.sum()) for v in visibility[c]]))
        print('SOURCE_DEPTH',c,zbuffer_audit[-1],flush=True)
        del idx,alpha,valid,first,has,pidx,front,pc,depth,uv,grid,front_at,b
(OUT/'source-depth-summary.json').write_text(json.dumps(zbuffer_audit,indent=2)+'\n')
del X,radius,opacity;torch.cuda.empty_cache()

def reg_hook(module,args):captured['features']=args[0]
model.sampler.ibr_regressor.register_forward_pre_hook(reg_hook)
def render(self,pcd,rgb,rad,occ,batch):
    z=captured.pop('features');xyz_feat,direction=z[...,:-3],z[...,-3:];feat=batch.output.src_feat_rgb
    reg=self.ibr_regressor;assert not reg.blend_shs and not reg.skip_shs and not reg.skip_eval_shs
    expanded=xyz_feat[:,None].expand(feat.shape[:-1]+(xyz_feat.shape[-1],))
    logits=reg.rgb_mlp(torch.cat([expanded,feat],dim=-1));weights=logits.softmax(-3)
    colors=feat[...,-3:];sh=reg.sh_mlp(xyz_feat).reshape(*xyz_feat.shape[:-1],3,reg.sh_dim//3)
    residual=eval_sh(reg.sh_deg,sh,direction).tanh()*reg.resd_limit
    masks=compute_src_inps(batch,'src_msks')
    masks=pad_image(masks,size=batch.persistent.src_feat_rgb.shape[-2:])
    support=sample_geometry_feature_image(pcd,masks,batch.src_exts,batch.src_ixts,batch.persistent.src_scale,padding_mode='zeros')
    supported=support>=.5;gated=weights*supported;denom=gated.sum(-3,keepdim=True);usable=denom>1e-7
    norm=torch.where(usable,gated/denom.clamp_min(1e-7),weights)
    original=((norm*colors).sum(-3)+residual).clamp(0,1)
    assert (original-rgb).abs().max()<2e-6
    assert np.allclose(pcd[0].cpu().numpy(),geo['xyz'],atol=1e-7)
    variants=[];stats={};captured['visibility_masks']={}
    for n,key in enumerate(['source_visibility1','source_visibility2']):
        vis=torch.stack([visibility[c][n] for c in sources])[None,...,None]
        candidate=norm*vis;total=candidate.sum(-3,keepdim=True)
        vweights=torch.where(total>1e-7,candidate/total.clamp_min(1e-7),norm)
        variants.append((key,((vweights*colors).sum(-3)+residual).clamp(0,1)))
        stats[key]=dict(no_visible_source=int((total<=1e-7).sum()),mean_rejected_weight=float((norm*(~vis)).sum(-3).mean()))
        captured['visibility_masks']['sample_'+key]=vis[0,:,::257,0].cpu().numpy()
    full=self.render_pytorch(pcd,rgb,rad,occ,batch,return_full=True)
    result=full[:3];idx,pix_alpha=full[3:]
    # Reuse one depth ordering and exactly one set of alpha weights for all
    # variants, avoiding sparse tie-order changes from repeated rasterization.
    def composite(color):
        return self.compositor(idx.permute(0,3,1,2),pix_alpha.permute(0,3,1,2),color.reshape(-1,3).T).permute(0,2,3,1)
    assert (composite(rgb)-result[0]).abs().max()<2e-6
    captured['variants']={}
    for key,color in variants:
        captured['variants'][key]=composite(color).cpu().reshape(1080,1920,3).numpy()
    captured['statistics']=dict(mean_weight_on_unsupported_sources=float((weights*(~supported)).sum(-3).mean()),
        points_with_no_supported_source=int((~usable).sum()),points=int(pcd.shape[1]),
        mean_blend_entropy=float((-(weights*weights.clamp_min(1e-10).log()).sum(-3)).mean()),
        same_alpha_all_variants=True,original_point_rgb_maxerr=float((original-rgb).abs().max()))
    captured['statistics']['visibility']=stats
    captured['sample_weights']=norm[0,:,::257,0].cpu().numpy();captured['sample_support']=support[0,:,::257,0].cpu().numpy()
    return result
model.sampler.render_points=MethodType(render,model.sampler)
rows=[];u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
for r in read(BASE/'native-evaluation/metrics.json')['records']:
    name=r['camera'];c=int(name);prefix=f'camera{name}-frame000750';z=np.load(BASE/'native-evaluation'/f'{prefix}-raw.npz')
    dataset.append_gt_prob=1. if c in [0,6,12] else 0.;item=dataset[c];sources=[int(dataset.view_inds[int(i)]) for i in item.src_inds]
    assert sources==r['source_cameras'] and c not in sources and not set(sources)&{0,6,12}
    item.K=torch.tensor(z['K'],dtype=torch.float32);item.meta.K=item.K;item.tar_ixt=item.K
    item.H=1080;item.W=1920;item.meta.H=1080;item.meta.W=1920
    for key in ['rgb','msk','wet','dpt','bkg','norm','ray_o','ray_d','near','far','coords']:item.pop(key,None);item.meta.pop(key,None)
    batch=to_cuda(add_iter(default_collate([item]),3000,3000))
    with torch.no_grad():result=model(batch)
    original=result.rgb_map.cpu().reshape(1080,1920,3).numpy();alpha=result.acc_map.cpu().reshape(1080,1920).numpy()
    delta=np.abs(original-z['rgb']);assert delta.mean()<2e-7 and (delta>1/255).sum()<100 and np.max(np.abs(alpha-z['alpha']))<2e-5
    captured['statistics']['baseline_reproduction']=dict(rgb_maxerr=float(delta.max()),rgb_meanerr=float(delta.mean()),channels_above_one_u8=int((delta>1/255).sum()))
    variants=dict(original=original,**captured['variants']);pred={key:np.clip(rgb+.5*(1-alpha[...,None]),0,1) for key,rgb in variants.items()}
    ref=z['source'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);metrics={}
    for key,array in pred.items():
        metrics[key]={label:float(-10*np.log10(np.mean((array[mask].astype(np.float64)-ref[mask].astype(np.float64))**2))) for label,mask in [('full_valid',z['valid']),('foreground',z['valid']&(z['matte']>.5))]}
        Image.fromarray(u8(array)).save(OUT/f'{prefix}-{key}.png')
    np.savez_compressed(OUT/f'{prefix}-raw.npz',alpha=alpha,**variants,sample_weights=captured['sample_weights'],sample_support=captured['sample_support'],**captured['visibility_masks'])
    x0,y0,x1,y1=r['bbox'];w=x1-x0;h=y1-y0;panel=Image.new('RGB',(4*w,h+32),'white');d=ImageDraw.Draw(panel)
    for i,(label,array) in enumerate([('Original photograph',z['source']),('Support blend 3000',u8(pred['original'])),('Source visibility 1r',u8(pred['source_visibility1'])),('Source visibility 2r',u8(pred['source_visibility2']))]):
        panel.paste(Image.fromarray(array[y0:y1,x0:x1]),(i*w,32));d.text((i*w+4,10),label,fill='black')
    panel.save(OUT/f'{prefix}-comparison.png')
    rows.append(dict(camera=name,split=r['split'],sources=sources,bbox=r['bbox'],metrics=metrics,statistics=captured['statistics']))
    print('SOURCE_SUPPORT',json.dumps(rows[-1]),flush=True)
    del result,batch,item;torch.cuda.empty_cache()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha
(OUT/'summary.json').write_text(json.dumps(dict(status='six_readonly_source_visibility_controls_complete_audit_and_review_pending',records=rows,checkpoint_unchanged=True,goal_complete=False),indent=2)+'\n')
