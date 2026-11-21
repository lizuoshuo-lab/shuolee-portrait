"""Fixed-input VJP isolation; no fit, optimizer, parameter update or source patch."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import os
import sys
import time
import traceback
import numpy as np
import torch
import run_official_gaussians123 as io


def difference(actual, reference):
    return {name: dict(max_abs=float((x-y).abs().max()),
        relative_l2=float((x-y).norm()/y.norm().clamp_min(1.e-12)),
        reference_l2=float(y.norm())) for name,x,y in zip(actual[0],actual[1],reference)}


def run(plan_path):
    plan=io.read(plan_path);root=Path(plan['source_root']);out=Path(plan['output_root'])
    out.mkdir(exist_ok=False);start=time.monotonic()
    def progress(state, **values):
        io.save(out/'progress.json',dict(state=state,pid=None if state.startswith('complete') or state.startswith('failed') else os.getpid(),
            updated=io.now(),training_started=False,elapsed_seconds=time.monotonic()-start,**values))
    def verify():
        assert io.sha(__file__)==plan['probe_sha256']
        for path,digest in plan['source_hashes'].items():assert io.sha(io.PROJECT/path)==digest,path
        for path,digest in plan['input_hashes'].items():assert io.sha(root/path)==digest,path
    def budget():
        if time.monotonic()-start>plan['max_seconds']:raise TimeoutError('Boundary diagnostic time limit')
    progress('checking')
    try:
        verify()
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
            sys.path.insert(0,str(io.PROJECT/'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(io.ASSETS)
            from shuolee_portrait.training import Views,named_pose_layout,render_pose,foreground_loss
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            from gaussian_renderer import render
            import diff_gaussian_rasterization as raster
            with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
            rows=io.read(root/'data/training-inputs.json')['records'];layout=named_pose_layout(fit,rows)
            views=Views(rows,1520,1,pose_indices=layout['pose_indices'])
            model=SMPLXGaussianModel(3);model.load_fresh_meshes(root/'fit/fitted.npz')
            model.enable_shared_static_offset(root/'fit/fitted.npz',io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            snap=root/'gaussian-run/step-018000';model.load_ply(str(snap/'bound-local.ply'))
            with np.load(snap/'shared-static-offset.npz',allow_pickle=False) as z:
                with torch.no_grad():model.static_offset.copy_(torch.as_tensor(z['static_offset'],device='cuda'))
                model.active_sh_degree=int(z['active_sh_degree'])
            idx=plan['frame_index'];pose=layout['pose_indices'][idx];camera,gt,alpha=views.get(idx)
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            bg=torch.zeros(3,device='cuda')
            local_names=('static_offset','local_xyz','sh_dc','sh_rest','opacity_logit','local_logscale','local_rotation')
            local=(model.static_offset,model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation)
            initial=[p.detach().clone() for p in local]
            parents=model.offset_body.parents.detach().clone();host=parents.cpu()
            world_names=('world_xyz','features','opacity','world_scale','world_rotation')
            def outputs():
                return (model.get_xyz,model.get_features,model.get_opacity,model.get_scaling,model.get_rotation)
            with torch.no_grad():
                model.select_mesh_by_timestep(pose)
                frozen=[x.detach().clone().requires_grad_(True) for x in outputs()]
            # Render detached world tensors: there is no SMPL-X or bound-mesh graph here.
            view=SimpleNamespace(get_xyz=frozen[0],get_features=frozen[1],get_opacity=frozen[2],
                get_scaling=frozen[3],get_rotation=frozen[4],active_sh_degree=model.active_sh_degree,max_sh_degree=3)
            ones=torch.ones_like(frozen[0])
            rgb=render(camera,view,pipe,bg)['render']
            maskrgb=render(camera,view,pipe,bg,override_color=ones)['render']
            photo=foreground_loss(rgb,gt,alpha)[0];silhouette=.1*(maskrgb[:1]-alpha).abs().mean()
            pixel_grad=torch.autograd.grad(photo,rgb)[0].detach().clone()
            mask_pixel_grad=torch.autograd.grad(silhouette,maskrgb)[0].detach().clone()
            rgb_reference=rgb.detach().clone();mask_reference=maskrgb.detach().clone()
            del rgb,maskrgb,photo,silhouette
            native_reference=None;native_records=[]
            for rep in range(plan['repeats']):
                budget()
                rgb=render(camera,view,pipe,bg)['render']
                maskrgb=render(camera,view,pipe,bg,override_color=ones)['render']
                grads=torch.autograd.grad((rgb,maskrgb),frozen,grad_outputs=(pixel_grad,mask_pixel_grad))
                assert all(torch.isfinite(g).all() for g in grads)
                if native_reference is None:native_reference=[g.detach().clone() for g in grads]
                record=dict(repetition=rep,rgb_max_abs=float((rgb-rgb_reference).abs().max()),
                    mask_max_abs=float((maskrgb-mask_reference).abs().max()),
                    gradients=difference((world_names,grads),native_reference))
                native_records.append(record)
                io.save(out/'native-raster-repeats.json',native_records)
                assert record['rgb_max_abs']==record['mask_max_abs']==0
                del grads,rgb,maskrgb
            progress('raster_boundary_complete')
            # Feed one identical tensor-gradient to the geometric graph every time.
            # This cannot be contaminated by native pixel-gradient accumulation order.
            geometry_reference=None;geometry_records=[]
            for mode in ('cuda_parents','cuda_parents_repeat','host_parents','host_parents_repeat'):
                budget();model.offset_body.parents=parents if mode.startswith('cuda') else host
                model.select_mesh_by_timestep(pose);world=outputs()
                output_delta=max(float((a-b).abs().max()) for a,b in zip(world,frozen))
                grads=torch.autograd.grad(world,local,grad_outputs=native_reference)
                assert all(torch.isfinite(g).all() for g in grads)
                if geometry_reference is None:geometry_reference=[g.detach().clone() for g in grads]
                record=dict(mode=mode,world_output_max_abs=output_delta,
                    gradients=difference((local_names,grads),geometry_reference))
                geometry_records.append(record);io.save(out/'geometric-vjp-repeats.json',geometry_records)
                assert output_delta==0
                del grads,world
            model.offset_body.parents=parents
            assert all(torch.equal(p,b) for p,b in zip(local,initial))
            verify()
            raster_max=max(x['relative_l2'] for r in native_records[1:] for x in r['gradients'].values())
            geometric_max=max(x['relative_l2'] for r in geometry_records[1:] for x in r['gradients'].values())
            sources={str(Path(raster.__file__)):io.sha(raster.__file__),str(Path(raster._C.__file__)):io.sha(raster._C.__file__)}
            result=dict(state='completed_no_updates',updated=io.now(),frame=rows[idx]['frame'],
                plan_sha256=io.sha(plan_path),native_raster_max_relative_gradient_difference=raster_max,
                geometric_VJP_max_relative_gradient_difference=geometric_max,
                tolerance=plan['gradient_relative_l2_tolerance'],
                geometric_VJP_within_registered_tolerance=geometric_max<=plan['gradient_relative_l2_tolerance'],
                raster_repeats=native_records,geometry_repeats=geometry_records,loaded_backend_sources=sources,
                elapsed_seconds=time.monotonic()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                optimizer_created=False,optimizer_updates=0,learned_parameters_unchanged=True,
                production_source_changed=False,quality_accepted=False,
                limitation='One completed18k state/view, fixed upstream gradients. Localizes repeat variation; does not prove visible artifact cause, full training equivalence, or15-minute reconstruction.')
            io.save(out/'result.json',result);progress('completed_no_updates')
            print(__import__('json').dumps({k:result[k] for k in ('state','elapsed_seconds','native_raster_max_relative_gradient_difference','geometric_VJP_max_relative_gradient_difference','geometric_VJP_within_registered_tolerance')}),flush=True)
    except BaseException as error:
        io.save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),training_started=False))
        progress('failed_preserved');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)
