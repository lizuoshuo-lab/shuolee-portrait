"""Frozen SH0 model: one RGB-gamut render control and exact loss-gradient audit.

No optimizer is created or stepped. Original PLYs, poses, cameras and fitting
are immutable. A display-only counterfactual is not a newly trained model.
"""
import argparse
import ast
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from PIL import Image
import replay_smplify_pose_source123 as lease
from run_official_gaussians123 import PROJECT, ASSETS, read, save, sha, now
from train_demo_body_upstream123 import DEMO, UP, module

ROOT=DEMO/'body-gaussian-smplify-eight-dc-reviewedfit'
SNAP=ROOT/'gaussian-run/step-018000'
OUT=ROOT/'frozen-rgb-gradient-diagnostic'
HERE=Path(__file__).resolve().parent


def checked():
    p=read(OUT/'plan.json')
    for f,h in p['files'].items(): assert sha(f)==h,f
    return p


def prepare():
    assert not OUT.exists(),'Preserve the bounded diagnostic.'
    assert read(SNAP/'result.json')['active_sh_degree']==0
    assert read(SNAP/'visual-review.json')['actual_visual_review']
    assert read(ROOT/'progress.json')['state']=='bounded_control_completed'
    p=read(ROOT/'plan.json')
    paths=[Path(__file__),ROOT/'plan.json',ROOT/'fit/fitted.npz',ROOT/'fit/surface-cache.npz',
        ROOT/'smplify-gaussian-interface-receipt.json',ROOT/'selection.json',ROOT/'data/training-inputs.json',
        SNAP/'bound-local.ply',SNAP/'world.ply',SNAP/'binding.npz',SNAP/'checkpoint.pt',
        SNAP/'current-meshes.npz',SNAP/'review/result.json',SNAP/'result.json',
        ROOT/'gaussian-run/upstream-training-function.py',Path(p['laplacian_cache']),
        HERE/'smplify_direct_surface123.py',HERE/'facial_geometry_boundary123.py',
        HERE/'body_root_sh_colors123.py',UP/'scene/gaussian_model.py',UP/'train.py',
        PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',
        PROJECT/'src/avatar_core/gaussian_renderer/__init__.py',PROJECT/'src/shuolee_portrait/training.py']
    for row in read(ROOT/'data/training-inputs.json')['records']:
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        paths += [Path(row['image']),Path(row['mask']),SNAP/(row['frame']+'_gaussian.png')]
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered_before_execution',
        hypothesis_rgb='In fixed geometry, out-of-range DC colours may contribute to bright patchwork. A single clamp to physical-display0..1 tests their image contribution, not whether retraining will fix geometry.',
        changed_variable='Render override_color only: existing SH0 RGB versus clamp(RGB,0,1). No coefficient or PLY rewrite.',
        hypothesis_mesh='At the exact saved state, measure native RGB, original xyz/scale and shared Laplacian gradients to distinguish a broken derivative from loss competition and small Adam displacement.',
        gradient_scope='All8 frozen training frames; original loss block, no fitting or gradient update. Gradient norms across different parameter units are not directly comparable.',
        frames=59,orbit_views=0,optimizer_created=False,optimization_updates=0,
        gpu_timeout_seconds=240,CPU_timeout_seconds=180,frame_reselection=False,coefficient_sweep=False,
        stop=['Any input/source hash change','Nonfinite rendering/gradient','Original snapshot PNG mean absolute replay discrepancy >=2e-5','Nonzero protected-face gradient','Changed Gaussian tensor or shared offset','Original SH0 versus explicit rawRGB override discrepancy >=2e-6 mean'],
        unchanged=['all Gaussian geometry/opacity/DC coefficients','shared clothing offset','SMPLify-X fit','59 cameras/masks/images','fixed8 roster','native rasterizer'],
        quality_accepted=False,model_promoted=False,goal_complete=False,
        files={str(f):sha(f) for f in paths}))
    print('FROZEN_RGB_GRADIENT_REGISTERED',flush=True)


def loss_function(torch):
    import torch.nn.functional as F
    from utils.loss_utils import l1_loss,ssim
    source=(UP/'train.py').read_text()
    fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='training')
    body=ast.get_source_segment(source,fn)
    start=body.index('        losses = {}')
    end=body.index("        losses['total'].backward()")
    block=body[start:end]
    code='def losses_for(gaussians,image,visibility_filter,gt_image,opt):\n'+''.join('    '+line[8:]+'\n' for line in block.splitlines())+'    return losses\n'
    ns=dict(torch=torch,F=F,l1_loss=l1_loss,ssim=ssim)
    exec(compile(code,'<unchanged-original-loss-block>','exec'),ns)
    (OUT/'original-loss-block.py').write_text(code)
    return ns['losses_for']


def replay(pid):
    checked();assert lease.process(pid)[1]=='T' and not (OUT/'result.json').exists()
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.training import camera,image
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from smplify_direct_surface123 import enable,vertices
    from facial_geometry_boundary123 import clothing_only_offset
    from gaussian_renderer import render
    official=module(UP/'scene/gaussian_model.py','frozen_rgb_original_model123')
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:binding=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:meshes=dict(z)
    p=read(ROOT/'plan.json')
    with np.load(p['laplacian_cache'],allow_pickle=False) as z:lap=dict(z)
    class Body(official.GaussianModel):
        select_mesh_by_timestep=SMPLXGaussianModel.select_mesh_by_timestep
        get_covariance=SMPLXGaussianModel.get_covariance
        shared_offset_vertices=vertices
        def compute_laplacian_loss(self):
            safe=clothing_only_offset(self.static_offset,self.surface_clothing,self.surface_protected)
            return torch.sparse.mm(self.shared_laplacian,safe).square().sum(-1).mean()
    model=Body(0);model.static_offset=None;model.num_timesteps=59
    model.vertices_sequence=torch.as_tensor(fit['vertices_world'],device='cuda')
    model.faces=torch.as_tensor(fit['faces'],device='cuda',dtype=torch.long)
    model.load_ply(str(SNAP/'bound-local.ply'))
    assert np.array_equal(model.binding.cpu().numpy(),binding['triangle_ids'])
    enable(model,ROOT/'fit/fitted.npz',ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with torch.no_grad():model.static_offset.copy_(torch.as_tensor(binding['static_offset'],device='cuda'))
    model.shared_laplacian=torch.sparse_coo_tensor(torch.tensor(lap['indices'],device='cuda'),torch.tensor(lap['values'],device='cuda'),tuple(lap['shape'])).coalesce()
    parameters=[model.static_offset,model._xyz,model._features_dc,model._scaling,model._rotation,model._opacity]
    names=['mesh_offset_m','Gaussian_xyz_face_units','DC_SH_coefficient','log_scale','quaternion','opacity_logit']
    frozen=[v.detach().clone() for v in parameters]
    rows=read(ROOT/'data/training-inputs.json')['records'];selected=set(read(ROOT/'selection.json')['selected_frames'])
    assert [r['frame'] for r in rows]==list(fit['frame_ids'])==list(meshes['frame_ids'])
    opt=SimpleNamespace(**p['options']);losses_for=loss_function(torch)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');records=[];gradients=[];started=time.monotonic()
    raw_rgb=(.28209479177387814*model._features_dc[:,0,:]+.5).clamp_min(0)
    clamped=raw_rgb.clamp(0,1).detach()
    # Load moments as read-only numbers. Never restore/create an optimizer.
    cp=torch.load(SNAP/'checkpoint.pt',map_location='cpu',weights_only=False)
    optim=cp['capture'][12]
    group=next(g for g in optim['param_groups'] if g['name']=='static_offset')
    state=optim['state'][group['params'][0]];assert group['lr']==1e-6
    def stats(g):
        if g is None:return dict(norm=0.,max_abs=0.,nonzero=0)
        return dict(norm=float(g.norm()),max_abs=float(g.abs().max()),nonzero=int((g!=0).sum()))
    for i,row in enumerate(rows):
        cam,_=camera(row,i,pose_index=i)
        with torch.no_grad():
            model.select_mesh_by_timestep(i)
            error=float((model.shared_offset_vertices(i)-torch.as_tensor(meshes['vertices_camera'][i],device='cuda')).abs().max())
            assert error<2e-6
            original=render(cam,model,pipe,bg)['render']
            if i==0:
                explicit=render(cam,model,pipe,bg,override_color=raw_rgb.detach())['render']
                parity=float((original-explicit).abs().mean());assert parity<2e-6
            limited=render(cam,model,pipe,bg,override_color=clamped)['render']
            assert torch.isfinite(original).all() and torch.isfinite(limited).all()
            original_png=np.asarray(image(original)).astype(np.int16)
            previous=np.asarray(Image.open(SNAP/(row['frame']+'_gaussian.png')).convert('RGB')).astype(np.int16)
            replay_error=float(np.abs(original_png-previous).mean()/255);assert replay_error<2e-5
            image(limited).save(OUT/(row['frame']+'_clamped.png'),compress_level=1)
            records.append(dict(frame=row['frame'],train8=row['frame'] in selected,original_replay_MAE=replay_error,mesh_replay_max_m=error,raw_render_pixels_above1=float((original>1).any(0).float().mean()),limited_render_max=float(limited.max()),mean_render_change=float((original-limited).abs().mean())))
            del original,limited
        if row['frame'] in selected:
            model.select_mesh_by_timestep(i)
            rendered=render(cam,model,pipe,bg)
            source=torch.from_numpy(np.asarray(Image.open(row['image']).convert('RGB')).copy()).to('cuda').permute(2,0,1).float()/255
            mask=torch.from_numpy(np.asarray(Image.open(row['mask']).convert('L')).copy()).to('cuda')[None].float()/255
            losses=losses_for(model,rendered['render'],rendered['visibility_filter'],source*mask,opt)
            components={'photometric':losses['l1']+losses['ssim'],**{k:v for k,v in losses.items() if k not in ('l1','ssim','total')}}
            parts={};mesh_grads={};total=[torch.zeros_like(x) for x in parameters]
            for name,value in components.items():
                gs=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                assert all(g is None or torch.isfinite(g).all() for g in gs)
                parts[name]=dict(loss=float(value),parameters={k:stats(g) for k,g in zip(names,gs)})
                mesh_grads[name]=gs[0].detach() if gs[0] is not None else torch.zeros_like(model.static_offset)
                for j,g in enumerate(gs):
                    if g is not None:total[j]+=g.detach()
            assert float(total[0][model.surface_protected].abs().max())==0
            photo=mesh_grads['photometric'];smooth=mesh_grads['lap']
            cosine=float((photo*smooth).sum()/(photo.norm()*smooth.norm()).clamp_min(1e-30))
            g=total[0].cpu();beta1,beta2=group['betas'];step=float(state['step'])+1
            m=state['exp_avg']*beta1+g*(1-beta1);v=state['exp_avg_sq']*beta2+g.square()*(1-beta2)
            delta=-group['lr']*(m/(1-beta1**step))/(v.sqrt()/math.sqrt(1-beta2**step)+group['eps'])
            length=delta.norm(dim=-1);learn=model.surface_clothing.cpu()
            gradients.append(dict(frame=row['frame'],components=parts,lap_to_photo_mesh_gradient_norm=float(smooth.norm()/photo.norm().clamp_min(1e-30)),lap_photo_cosine=cosine,mesh_total_gradient=stats(total[0]),protected_face_gradient=0,adam_prediction=dict(scope='Hypothetical next single-frame delta using unchanged saved moments; not applied',saved_step=float(state['step']),lr=group['lr'],clothing_delta_median_m=float(length[learn].median()),clothing_delta_p99_m=float(torch.quantile(length[learn],.99)),maximum_m=float(length.max()))))
            del rendered,source,mask,losses,components,parts,mesh_grads,total,gs,g,m,v,delta
        save(OUT/'progress.json',dict(updated=now(),state='read_only_render_and_gradient',pid=os.getpid(),frames=i+1,gradient_frames=len(gradients),optimization_updates=0))
    assert len(gradients)==8 and all(torch.equal(x,y) for x,y in zip(parameters,frozen))
    assert model._features_rest.numel()==0
    checked();assert lease.process(pid)[1]=='T'
    save(OUT/'result.json',dict(updated=now(),state='read_only_diagnostic_complete',frames=59,records=records,gradients=gradients,raw_override_parity_MAE=parity,parameter_tensors_bitwise_unchanged=True,optimizer_created=False,optimization_updates=0,seconds=time.monotonic()-started,quality_accepted=False,model_promoted=False,actual_visual_review=False))
    print('FROZEN_RGB_GRADIENT_COMPLETE',len(gradients),flush=True)


def review():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=read(OUT/'result.json');original=read(SNAP/'review/result.json')
    lookup={r['frame']:r for r in original['records']}
    rows=read(ROOT/'data/training-inputs.json')['records'];records=[];dest=Path('/mnt/d/00/记录/demo')
    for row in rows:
        frame=row['frame'];new=Image.open(OUT/(frame+'_clamped.png')).convert('RGB')
        source=np.asarray(Image.open(row['image']).convert('RGB'),dtype=np.float32)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        old=np.asarray(Image.open(SNAP/(frame+'_gaussian.png')).convert('RGB'),dtype=np.float32)
        errs={k:np.abs(x-source).mean(-1)/255 for k,x in [('original',old),('clamped',np.asarray(new,dtype=np.float32))]}
        items=[];values={}
        for item in lookup[frame]['images']:
            x0,y0,x1,y1=item['crop_integer_xyxy'];region=item['region']
            oldpanels=Image.open(item['path']).convert('RGB');assert oldpanels.size==((x1-x0)*3,y1-y0)
            panels=Image.new('RGB',((x1-x0)*4,y1-y0));panels.paste(oldpanels,(0,0));panels.paste(new.crop((x0,y0,x1,y1)),((x1-x0)*3,0))
            path=dest/f'高斯_固定几何RGB范围诊断_{frame}_{region}_原图_当前mesh_原始SH0_限幅SH0.png';panels.save(path,compress_level=1)
            m=mask[y0:y1,x0:x1];values[region]={k:float(e[y0:y1,x0:x1][m].mean()) for k,e in errs.items()} if m.any() else None
            items.append(dict(region=region,path=str(path),sha256=sha(path),crop_integer_xyxy=item['crop_integer_xyxy'],panels=['source','current_mesh','original_SH0','RGB_clamp_counterfactual'],resampling=False,text_added=False))
        records.append(dict(frame=frame,images=items,regions=values))
    records.sort(key=lambda r:r['frame']);pages=[]
    for region in ('body','face','left_hand','right_hand','feet'):
        count=6 if region=='body' else 12
        for start in range(0,59,count):
            group=records[start:start+count];pictures=[]
            for row in group:
                im=Image.open(next(x['path'] for x in row['images'] if x['region']==region)).convert('RGB')
                if region=='body':
                    scale=min(1.,360/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
                pictures.append(im)
            w=max(x.width for x in pictures);h=max(x.height for x in pictures)
            page=Image.new('RGB',(w*2,h*math.ceil(len(pictures)/2)),'white')
            for j,im in enumerate(pictures):page.paste(im,((j%2)*w,(j//2)*h))
            path=OUT/f'{region}-{start//count+1:02d}.png';page.save(path)
            pages.append(dict(region=region,path=str(path),sha256=sha(path),frames=[x['frame'] for x in group]))
    selected=set(read(ROOT/'selection.json')['selected_frames']);groups={}
    for name,subset in [('all59',records),('train8',[r for r in records if r['frame'] in selected]),('excluded51',[r for r in records if r['frame'] not in selected])]:
        groups[name]={}
        for region in ('body','face','left_hand','right_hand','feet'):
            values=[r['regions'][region] for r in subset if r['regions'][region] is not None]
            old=np.array([x['original'] for x in values]);new=np.array([x['clamped'] for x in values])
            groups[name][region]=dict(frames=len(values),improved=int((new<old).sum()),original_mean=float(old.mean()),clamped_mean=float(new.mean()),relative_median=float(np.median(new/old-1)))
    save(OUT/'review-result.json',dict(updated=now(),frames=59,records=records,pages=pages,groups=groups,actual_visual_review=False,quality_accepted=False,model_promoted=False))
    print('RGB_DIAGNOSTIC_REVIEW_READY',groups['all59'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','replay','review']);p.add_argument('--pid',type=int);a=p.parse_args()
    try:
        if a.mode=='run':
            lease.OUT=OUT;lease.__file__=__file__;lease.run(a.pid)
        elif a.mode=='replay':replay(a.pid)
        else:globals()[a.mode]()
    except BaseException as e:
        if OUT.exists():save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
