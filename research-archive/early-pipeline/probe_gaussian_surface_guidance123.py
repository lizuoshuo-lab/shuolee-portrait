"""All59-frame zero-update gradient audit of two independent local loss hooks."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import run_official_gaussians123 as io

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset')
OUT = ROOT/'gaussian-mesh-guidance-audit'
SNAP = ROOT/'gaussian-run/step-120000'
UP = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX')
HELPER = Path(__file__).with_name('gaussian_surface_guidance123.py')


def process(pid):
    raw = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, state=raw[0], start_identity=raw[19],
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())


def prepare():
    OUT.mkdir(exist_ok=False)
    files = [Path(__file__), HELPER, ROOT/'fit/fitted.npz', ROOT/'data/training-inputs.json',
        SNAP/'bound-local.ply', SNAP/'binding.npz', SNAP/'current-meshes.npz', SNAP/'checkpoint.pt']
    files += [Path(p) for p in io.read(ROOT/'plan.json')['source_hashes']]
    io.save(OUT/'plan.json', dict(created=io.now(), frames=59, saved_step=120000,
        hashes={str(p):io.sha(p) for p in files}, max_seconds=240, optimizer_updates=0,
        hypotheses=['Original nonmetric local-coordinate regularizers have no direct mesh gradient.',
            'Spherical SH energy has gradients only to non-DC SH, not mesh or covariance.',
            'Detached saved Gaussian targets can supply a direct metric gradient to shared mesh offsets.'],
        controls=['All59 saved mesh and native render replay', 'Original L1/DSSIM offset gradient',
            'Original xyz/scale direct-gradient isolation', 'SH DC and geometry isolation',
            'Teacher Gaussian position/opacity stop-gradient isolation', 'Double-precision finite differences'],
        support_policy='Diagnostic proxy only: original raster frustum visibility, positive depth, native mask at center, detached sigmoid opacity. This is NOT transmittance visibility or verified material correspondence.',
        guidance_beta_m=.005, candidate_loss_weight=None, no_training_parameter_selected=True,
        refusal='Do not enable a teacher loss in production or use these Gaussians as clothing truth solely because its gradient is nonzero.',
        no_old_candidate_training_resumed=True, no_new_mesh_or_ply=True, quality_accepted=False))
    print('GUIDANCE_AUDIT_REGISTERED', flush=True)


def verify():
    plan=io.read(OUT/'plan.json')
    for path,digest in plan['hashes'].items(): assert io.sha(path)==digest,path
    return plan


def synthetic_checks(torch, helper):
    t=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=torch.float64)
    q=torch.tensor([[.2,.3,.01]],dtype=torch.float64,requires_grad=True)
    bary,valid,inside=helper.closest_bound_barycentrics(t,q)
    assert bool(valid.all() and inside.all())
    assert torch.allclose(bary,torch.tensor([[.5,.2,.3]],dtype=torch.float64))
    v=t[0].clone().requires_grad_();face=torch.tensor([[0,1,2]]);binding=torch.tensor([0])
    weight=torch.ones(1,dtype=torch.float64,requires_grad=True)
    loss=helper.gaussian_to_mesh_loss(v,face,binding,bary,q,weight)
    gv,gq,gw=torch.autograd.grad(loss,[v,q,weight],allow_unused=True)
    assert gq is None and gw is None and bool((gv[:,2]<0).all())
    records=[]
    for idx in ((0,2),(1,0),(2,2)):
        plus=v.detach().clone();minus=plus.clone();plus[idx]+=1e-6;minus[idx]-=1e-6
        fn=lambda z:helper.gaussian_to_mesh_loss(z,face,binding,bary,q,weight)
        fd=float((fn(plus)-fn(minus))/(2e-6));auto=float(gv[idx])
        assert abs(fd-auto)<1e-8
        records.append(dict(parameter=list(idx),autograd=auto,finite_difference=fd))
    sh=torch.arange(45,dtype=torch.float64).reshape(1,15,3).div(100).requires_grad_()
    derivative=torch.autograd.grad(helper.directional_color_energy(sh),sh)[0]
    expected=sh/(6*np.pi)
    assert torch.allclose(derivative,expected,atol=1e-12,rtol=1e-12)
    return dict(passed=True,guide_finite_difference=records,teacher_gradients_none=True,sh_analytic_gradient_equal=True)


def replay(pid):
    plan=verify();assert process(pid)['state']=='T'
    start=time.monotonic()
    import torch
    from PIL import Image
    import train_demo_body_upstream123 as base
    import gaussian_surface_guidance123 as helper
    sys.path.insert(0,str(io.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(io.ASSETS)
    from shuolee_portrait.training import camera,named_pose_layout
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss,ssim
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    checks=synthetic_checks(torch,helper)
    official=base.module(UP/'scene/gaussian_model.py','guidance_original_model123')
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as f:fit=dict(f)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as f:saved=dict(f)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as f:saved_binding=dict(f)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];layout=named_pose_layout(fit,rows)
    class Body(official.GaussianModel):
        select_mesh_by_timestep=SMPLXGaussianModel.select_mesh_by_timestep
        enable_shared_static_offset=SMPLXGaussianModel.enable_shared_static_offset
        shared_offset_vertices=SMPLXGaussianModel.shared_offset_vertices
    model=Body(3);model.static_offset=None
    model.vertices_sequence=torch.as_tensor(fit['vertices_world'],device='cuda')
    model.faces=torch.as_tensor(fit['faces'],device='cuda',dtype=torch.long);model.num_timesteps=59
    model.enable_shared_static_offset(ROOT/'fit/fitted.npz',io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    model.load_ply(str(SNAP/'bound-local.ply'))
    assert np.array_equal(model.binding.cpu().numpy(),saved_binding['triangle_ids'])
    with torch.no_grad():model.static_offset.copy_(torch.as_tensor(saved_binding['static_offset'],device='cuda').reshape_as(model.static_offset))
    model.active_sh_degree=3
    names=['static_offset','local_xyz','SH_DC','SH_nonDC','opacity','local_scale','local_rotation']
    parameters=[model.static_offset,model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation]
    initial=[x.detach().clone() for x in parameters]
    sh=helper.directional_color_energy(model._features_rest)
    gsh=torch.autograd.grad(sh,parameters,allow_unused=True)
    assert gsh[3] is not None and all(g is None for i,g in enumerate(gsh) if i!=3)
    color=dict(value=float(sh),unclamped_linear_RGB_directional_RMS=float(sh.sqrt()),gradient_only_nonDC_SH=True)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');records=[];gphoto_all=[];gguide_all=[]
    for i,row in enumerate(rows):
        assert time.monotonic()-start<plan['max_seconds'],'Registered time budget'
        assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
        assert str(saved['frame_ids'][i])==row['frame']
        pose=layout['pose_indices'][i];model.select_mesh_by_timestep(pose)
        vertices=model.shared_offset_vertices(pose)
        current=torch.as_tensor(saved['vertices_camera'][i],device='cuda')
        parity=float((vertices-current).abs().max());assert parity<2e-6
        cam,_=camera(row,i,pose_index=pose)
        mask=torch.as_tensor(np.asarray(Image.open(row['mask']).convert('L')).copy(),device='cuda').float()/255
        target=torch.as_tensor(np.asarray(Image.open(row['image']).convert('RGB')).copy(),device='cuda').permute(2,0,1).float()/255*mask[None]
        pkg=render(cam,model,pipe,bg);rgb=pkg['render'];visible=pkg['visibility_filter']
        original=torch.as_tensor(np.asarray(Image.open(SNAP/(row['frame']+'_gaussian.png')).convert('RGB')).copy(),device='cuda').permute(2,0,1).float()/255
        quantized=(rgb.detach().clamp(0,1)*255).byte();pngdiff=(quantized.float()-original*255).abs()
        assert float(pngdiff.mean())<.02 and float(pngdiff.max())<=2
        photo=.8*l1_loss(rgb,target)+.2*(1-ssim(rgb,target))
        gp=torch.autograd.grad(photo,parameters,retain_graph=True,allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gp)
        xyz=.01*torch.relu(model._xyz[visible].norm(dim=1)-1).mean()
        scale=torch.relu(model._scaling[visible].exp()-.6).norm(dim=1).mean()
        glocal=torch.autograd.grad(xyz+scale,parameters,allow_unused=True)
        assert glocal[0] is None
        with torch.no_grad():
            teacher=model.get_xyz.detach().clone();tri=vertices[model.faces[model.binding]].detach()
            bary,valid,inside=helper.closest_bound_barycentrics(tri,teacher)
            E=torch.tensor(row['w2c'],device='cuda',dtype=teacher.dtype);K=torch.tensor(row['K'],device='cuda',dtype=teacher.dtype)
            camspace=teacher@E[:3,:3].T+E[:3,3];proj=camspace@K.T
            uv=proj[:,:2]/proj[:,2:].clamp_min(1e-8)
            # row K is pixel-corner convention: subtract .5 for integer centers.
            # Nearest-neighbor index floor((u-.5)+.5) therefore equals floor(u).
            ix=uv[:,0].floor().long();iy=uv[:,1].floor().long()
            h,w=mask.shape;in_image=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(camspace[:,2]>0)
            support=valid&visible&in_image
            sample=mask[iy.clamp(0,h-1),ix.clamp(0,w-1)]
            weights=model.get_opacity[:,0].detach()*support*sample
            dist=(teacher-(tri*bary[:,:,None]).sum(1)).norm(dim=1)
        # Deliberately pass differentiable teacher/opacity to prove stop-gradient.
        teacher_input=teacher.clone().requires_grad_();weight_input=weights.clone().requires_grad_()
        guide=helper.gaussian_to_mesh_loss(vertices,model.faces,model.binding,bary,teacher_input,weight_input,plan['guidance_beta_m'])
        gg=torch.autograd.grad(guide,[*parameters,teacher_input,weight_input],allow_unused=True)
        assert gg[0] is not None and torch.isfinite(gg[0]).all()
        assert all(x is None for x in gg[1:])
        pnorm=float(gp[0].norm());gnorm=float(gg[0].norm())
        cosine=float((gp[0]*gg[0]).sum()/(gp[0].norm()*gg[0].norm()).clamp_min(1e-20))
        gphoto_all.append(gp[0].detach().cpu().numpy());gguide_all.append(gg[0].detach().cpu().numpy())
        valid_distance=dist[weights>0].cpu().numpy()
        records.append(dict(frame=row['frame'],pose_index=pose,mesh_replay_max_abs_m=parity,
            native_png_parity_mean_u8=float(pngdiff.mean()),native_png_parity_max_u8=float(pngdiff.max()),
            original_photo_loss=float(photo),original_xyz_loss=float(xyz),original_scale_loss=float(scale),
            original_regularizers_direct_mesh_gradient=False,
            photometric_gradient_L2_by_parameter={n:float(g.norm()) for n,g in zip(names,gp)},
            guide_loss_m=float(guide),guide_offset_gradient_L2=gnorm,photo_offset_gradient_L2=pnorm,
            photo_vs_guide_offset_gradient_cosine=cosine,guide_gradients_only_offset=True,
            proxy_supported_points=int((weights>0).sum()),bound_plane_interior_supported=int(((weights>0)&inside).sum()),
            proxy_supported_distance_mm=dict(zip(['median','p90','max'],map(float,np.quantile(valid_distance*1000,[.5,.9,1]))))))
        io.save(OUT/'progress.json',dict(state='diagnostic_no_updates',pid=os.getpid(),frames=len(records),elapsed_seconds=time.monotonic()-start))
        print('GUIDANCE_FRAME',len(records),row['frame'],flush=True)
        del gp,gg,pkg,rgb,vertices,photo,guide,tri,teacher,target,original
    assert all(torch.equal(x,y) for x,y in zip(parameters,initial)) and model.optimizer is None
    p=np.stack(gphoto_all);g=np.stack(gguide_all)
    aggregate=lambda a:dict(l2_of_mean=float(np.linalg.norm(a.mean(0))),mean_l2_per_frame=float(np.linalg.norm(a.reshape(59,-1),axis=1).mean()),
        coherence=float(np.linalg.norm(a.sum(0))/np.maximum(np.linalg.norm(a.reshape(59,-1),axis=1).sum(),1e-30)))
    verify()
    result=dict(state='complete_no_updates',updated=io.now(),frames=59,points=len(model._xyz),source_step=120000,
        mathematical_checks=checks,color=color,frames_result=records,
        gradient_aggregate=dict(photo=aggregate(p),guide=aggregate(g)),
        optimizer_created=False,parameter_updates=0,learned_parameters_unchanged=True,
        production_source_changed=False,new_mesh_or_ply=False,teacher_surface_accepted=False,quality_accepted=False,
        elapsed_seconds=time.monotonic()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        caveat='Guide has units of fitted-model meters; photo/color use image units. Raw gradient magnitudes alone cannot select a learning rate. Proxy support does not establish occlusion visibility or correct clothing geometry.')
    np.savez_compressed(OUT/'offset-gradient-only.npz',photometric=p,guide=g)
    io.save(OUT/'result.json',result)
    io.save(OUT/'progress.json',dict(state='complete_no_updates',pid=None,frames=59,elapsed_seconds=result['elapsed_seconds']))
    print('GUIDANCE_AUDIT_COMPLETE',result['elapsed_seconds'],flush=True)


def run(pid):
    verify();assert not (OUT/'result.json').exists()
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in before['command'] and before['state'] not in ('T','Z')
    io.save(OUT/'gpu-handoff.json',dict(started=io.now(),baseline=before,state='suspending'))
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)['state']=='T':break
            time.sleep(.05)
        assert process(pid)['state']=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=300)
    except BaseException as e:
        io.save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_training_started=True));raise
    finally:
        after=process(pid)
        assert after['start_identity']==before['start_identity']
        os.kill(pid,signal.SIGCONT)
        io.save(OUT/'gpu-handoff.json',dict(started_identity=before,finished=io.now(),state='resumed',baseline_after=process(pid)))
        print('BASELINE_RESUMED',pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args.pid)
