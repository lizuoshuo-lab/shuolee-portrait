"""Read-only mask/geometry gradient audit. Never calls an optimizer."""
import fcntl, os, time
import numpy as np
import torch
from PIL import Image
import fit_demo_flame_rgb123 as g
from run_official_gaussians123 import save, sha, now

OUT=g.OUT/'mask-audit'
CHECKPOINT=g.OUT/'epoch-30/tracked-flame-with-surface.npz'

def main():
    assert not OUT.exists(), 'Do not repeat this audit.'
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic();digest=sha(CHECKPOINT)
    save(OUT/'plan.json',dict(updated=now(),mode='read_only_no_optimizer',pid=os.getpid(),
        checkpoint_sha256=digest,frames=[0,12,23,28,32,35,46,58],
        hypothesis='Original target background and absent silhouette loss do not penalize all missing foreground; hair boundary detach prevents mask gradient at that region.',
        interventions='In-memory gradient probes only: same alpha L1, original boundary exclusions versus removing hair alone. No training or checkpoint change.',
        budget_seconds=120))
    t,rows=g.setup()
    with np.load(CHECKPOINT,allow_pickle=False) as z,torch.no_grad():
        for key in ['shape',*g.f.PER_FRAME,'scene_scale','tex_extra','lights','static_offset']:
            getattr(t,key).copy_(torch.as_tensor(z[key],device='cuda'))
    hair=t.flame.mask.get_vid_by_region(['hair']); faces=t.flame.faces
    offset=t.static_offset.detach().clone();regions={}
    for name in ['hair','face','ears','neck']:
        ids=t.flame.mask.get_vid_by_region([name]);n=offset[0,ids].norm(dim=-1)
        regions[name]=dict(vertices=len(ids),median=float(n.median()),p90=float(torch.quantile(n,.9)),max=float(n.max()))
    probe=[];geometry=[]
    for i in [0,12,23,28,32,35,46,58]:
        assert time.monotonic()-start<120
        s=t.dataset.one(i);alpha=s['alpha_map'].cuda();gt=s['rgb'].cuda()
        one=dict(frame=rows[i]['frame'],device_overlap=i in g.OCCLUDED)
        for variant,excluded in [('original',['bottomline','hair']),('hair_boundary_enabled',['bottomline'])]:
            t.clear_cache();v,c,l,tex=t.forward_flame(s['timestep_index'])
            rast=t.rasterize_flame(s,v,faces,train_mode=True)
            render=t.render_rgba(rast,v,faces,tex,t.lights[None],gt.permute(0,2,3,1),
                t.flame.mask.get_fid_by_region(['hair']),t.flame.mask.get_vid_by_region(excluded),False)
            pred=render['rgba'][:,3:];mask_loss=(pred-alpha).abs().mean()
            grad=torch.autograd.grad(mask_loss,v,retain_graph=True)[0]
            hair_grad=grad[0,hair].norm(dim=-1)
            missing=(alpha>.5)&(pred.detach()<.01)
            rgb_error=(render['rgba'][:,:3]-gt).abs().mean(1,keepdim=True)
            one[variant]=dict(mask_L1=float(mask_loss),hair_vertex_grad_max=float(hair_grad.max()),
                hair_vertex_grad_nonzero=int((hair_grad>1e-12).sum()),
                missing_foreground_pixels=int(missing.sum()),
                missing_foreground_RGB_L1_no_random_disturbance=float(rgb_error[missing].mean()) if missing.any() else None)
            if variant=='original':
                xy=t.dataset.crops[i];side=t.dataset.side
                raw=Image.open(rows[i]['image']).convert('RGB').crop((*xy,*(xy+side)))
                target=Image.fromarray((alpha[0,0].detach().cpu().numpy()*255).astype('uint8')).convert('RGB')
                prediction=Image.fromarray(np.clip(pred[0,0].detach().cpu().numpy()*255,0,255).astype('uint8')).convert('RGB')
                im=Image.new('RGB',(side*3,side),'white')
                for j,panel in enumerate([raw,target,prediction]):im.paste(panel,(j*side,0))
                im.save(OUT/(rows[i]['frame']+'_原图_SAM2遮罩_当前网格遮罩.png'))
        probe.append(one)
        with torch.no_grad():
            t.clear_cache();v,c,l,tex=t.forward_flame(s['timestep_index']);with_offset=v.clone()
            t.static_offset.zero_();zero,c,l,tex=t.forward_flame(s['timestep_index']);t.static_offset.copy_(offset)
            K=torch.tensor(rows[i]['K_integer_centers'],device='cuda',dtype=v.dtype)
            project=lambda x:(x@K.T)[...,:2]/(x@K.T)[...,2:]
            displacement=(project(with_offset)-project(zero)).norm(dim=-1)
            geometry.append(dict(frame=rows[i]['frame'],shared_offset_projection_median_px=float(displacement.median()),
                shared_offset_projection_p90_px=float(torch.quantile(displacement,.9)),
                shared_offset_projection_max_px=float(displacement.max())))
            oldlight=t.render.lighting_type;t.render.lighting_type='front-range'
            plain=t.render.render_rgba_vis(with_offset,faces,s['extrinsic'].cuda(),s['intrinsic'].cuda(),t.image_size,background_color=[1,1,1])
            t.render.lighting_type=oldlight
            rgb=(plain['rgba'][0,...,:3].clamp(0,1).cpu().numpy()*255).astype('uint8')
            im=Image.new('RGB',(side*2,side),'white');im.paste(raw);im.paste(Image.fromarray(rgb),(side,0))
            im.save(OUT/(rows[i]['frame']+'_原图_纯几何独立渲染.png'))
    assert sha(CHECKPOINT)==digest
    save(OUT/'result.json',dict(updated=now(),state='completed_read_only',elapsed_seconds=time.monotonic()-start,
        optimizer_updates=0,checkpoint_unchanged=True,offset_regions_model_units=regions,
        probes=probe,projected_shared_offset=geometry,independent_3D_truth=False,
        silhouette_loss_active_in_original_training=False,background='target',
        geometry_gradient_note='Texture UV-gradient and boundary-gradient exclusions do not freeze all shading gradients.',
        numerical_mask_diagnostic_not_quality_acceptance=True))
    print('MASK_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
