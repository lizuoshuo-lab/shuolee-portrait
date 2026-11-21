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
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/dynamic-human-4k4d-projection-control-040/original_projection'
OUT=LAB/'outputs/dynamic-human-4k4d-source-support-042';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
assert read(BASE/'independent-state-audit.json')['optimizer_max_step']==2000
assert not (OUT/'summary.json').exists()
torch.set_num_threads(8);random.seed(0);torch.manual_seed(0)
checkpoint=BASE/'model/latest.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
(OUT/'protocol.json').write_text(json.dumps(dict(status='frozen_before_readonly_probe',parent_sha256=sha,frame=750,cameras=['00','06','12','01','08','17'],
    variants=['original','source_foreground_gate','source_foreground_gate_unpremultiply'],
    intervention='Reweight original learned blending weights by source-mask support>=0.5; second variant also unpremultiplies sampled sourceRGB by sampled mask. Original SH residual retained. Unsupported points fall back to original color.',
    fixed='All model parameters, point geometry/radii/opacities, original projection, eight training sources and legacy renderer alpha. No target RGB/matte in forward.',
    qualification='Foreground support is not occlusion visibility; support masks are imperfect. Inference-only counterfactual, not retrained model or completed task.',goal_complete=False),indent=2)+'\n')
cfg.model_cfg.sampler_cfg.skip_loading_points=True
model=MODELS.build(cfg.model_cfg).cuda();cp=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval();del cp
dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg);captured={}
def reg_hook(module,args):captured['features']=args[0]
model.sampler.ibr_regressor.register_forward_pre_hook(reg_hook)
def render(self,pcd,rgb,rad,occ,batch):
    z=captured.pop('features');xyz_feat,direction=z[...,:-3],z[...,-3:];feat=batch.output.src_feat_rgb
    reg=self.ibr_regressor;assert not reg.blend_shs and not reg.skip_shs and not reg.skip_eval_shs
    expanded=xyz_feat[:,None].expand(feat.shape[:-1]+(xyz_feat.shape[-1],))
    logits=reg.rgb_mlp(torch.cat([expanded,feat],dim=-1));weights=logits.softmax(-3)
    colors=feat[...,-3:];sh=reg.sh_mlp(xyz_feat).reshape(*xyz_feat.shape[:-1],3,reg.sh_dim//3)
    residual=eval_sh(reg.sh_deg,sh,direction).tanh()*reg.resd_limit
    original=((weights*colors).sum(-3)+residual).clamp(0,1)
    assert (original-rgb).abs().max()<2e-6
    masks=compute_src_inps(batch,'src_msks')
    masks=pad_image(masks,size=batch.persistent.src_feat_rgb.shape[-2:])
    support=sample_geometry_feature_image(pcd,masks,batch.src_exts,batch.src_ixts,batch.persistent.src_scale,padding_mode='zeros')
    assert support.shape==weights.shape and support.min()>=0 and support.max()<=1.00001
    supported=support>=.5;gated=weights*supported;denom=gated.sum(-3,keepdim=True);usable=denom>1e-7
    norm=torch.where(usable,gated/denom.clamp_min(1e-7),weights)
    restored=colors/support.clamp_min(.5)
    variants=[('source_foreground_gate',((norm*colors).sum(-3)+residual).clamp(0,1)),
              ('source_foreground_gate_unpremultiply',((norm*torch.where(usable,restored,colors)).sum(-3)+residual).clamp(0,1))]
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
    captured['sample_weights']=weights[0,:,::257,0].cpu().numpy();captured['sample_support']=support[0,:,::257,0].cpu().numpy()
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
    batch=to_cuda(add_iter(default_collate([item]),2000,2000))
    with torch.no_grad():result=model(batch)
    original=result.rgb_map.cpu().reshape(1080,1920,3).numpy();alpha=result.acc_map.cpu().reshape(1080,1920).numpy()
    delta=np.abs(original-z['rgb']);assert delta.mean()<2e-7 and (delta>1/255).sum()<100 and np.max(np.abs(alpha-z['alpha']))<2e-5
    captured['statistics']['baseline_reproduction']=dict(rgb_maxerr=float(delta.max()),rgb_meanerr=float(delta.mean()),channels_above_one_u8=int((delta>1/255).sum()))
    variants=dict(original=original,**captured['variants']);pred={key:np.clip(rgb+.5*(1-alpha[...,None]),0,1) for key,rgb in variants.items()}
    ref=z['source'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);metrics={}
    for key,array in pred.items():
        metrics[key]={label:float(-10*np.log10(np.mean((array[mask].astype(np.float64)-ref[mask].astype(np.float64))**2))) for label,mask in [('full_valid',z['valid']),('foreground',z['valid']&(z['matte']>.5))]}
        Image.fromarray(u8(array)).save(OUT/f'{prefix}-{key}.png')
    np.savez_compressed(OUT/f'{prefix}-raw.npz',alpha=alpha,**variants,sample_weights=captured['sample_weights'],sample_support=captured['sample_support'])
    x0,y0,x1,y1=r['bbox'];w=x1-x0;h=y1-y0;panel=Image.new('RGB',(4*w,h+32),'white');d=ImageDraw.Draw(panel)
    for i,(label,array) in enumerate([('Original photograph',z['source']),('Original rendering',u8(pred['original'])),('Source-mask support',u8(pred['source_foreground_gate'])),('Support + unpremultiply',u8(pred['source_foreground_gate_unpremultiply']))]):
        panel.paste(Image.fromarray(array[y0:y1,x0:x1]),(i*w,32));d.text((i*w+4,10),label,fill='black')
    panel.save(OUT/f'{prefix}-comparison.png')
    rows.append(dict(camera=name,split=r['split'],sources=sources,bbox=r['bbox'],metrics=metrics,statistics=captured['statistics']))
    print('SOURCE_SUPPORT',json.dumps(rows[-1]),flush=True)
    del result,batch,item;torch.cuda.empty_cache()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha
(OUT/'summary.json').write_text(json.dumps(dict(status='six_readonly_source_support_controls_complete_audit_and_review_pending',records=rows,checkpoint_unchanged=True,goal_complete=False),indent=2)+'\n')
