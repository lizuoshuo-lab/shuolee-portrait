"""Read-only, bounded gradients on the completed18k model; no optimizer."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
import run_official_gaussians123 as io


def stats(x):
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return dict(median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                maximum=float(np.max(a)), minimum=float(np.min(a)))


def gradient_stats(g):
    if g is None:
        return dict(connected=False)
    assert torch.isfinite(g).all(), 'Nonfinite gradient'
    length = g.detach().norm(dim=-1)
    return dict(connected=True, finite=True, nonzero=int((length > 0).sum()),
                total=len(length), l2=float(length.norm()), norm=stats(length))


def run(plan_path):
    plan = io.read(plan_path)
    source, out = Path(plan['source_root']), Path(plan['output_root'])
    assert not out.exists(), 'Never repeat/overwrite a completed diagnostic.'
    out.mkdir()
    start = time.monotonic()
    def progress(stage, **kw):
        io.save(out/'progress.json', dict(state=stage, pid=os.getpid(), updated=io.now(),
            elapsed_seconds=time.monotonic()-start, training_started=False, **kw))
    progress('checking')
    try:
        for p, digest in plan['source_hashes'].items():
            assert io.sha(io.PROJECT/p) == digest, p
        for p, digest in plan['input_hashes'].items():
            assert io.sha(source/p) == digest, p
        with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            sys.path.insert(0, str(io.PROJECT/'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(io.ASSETS)
            from shuolee_portrait.training import Views, named_pose_layout, render_pose, foreground_loss, image
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            from scene.gaussian_model import GaussianModel
            from gaussian_renderer import render
            from arguments import OptimizationParams
            with np.load(source/'fit/fitted.npz', allow_pickle=False) as data:
                fit = dict(data)
            rows = io.read(source/'data/training-inputs.json')['records']
            layout = named_pose_layout(fit, rows)
            indices = plan['observation_indices']
            views = Views(rows, 1520, len(indices), pose_indices=layout['pose_indices'])
            model = SMPLXGaussianModel(3)
            model.load_fresh_meshes(source/'fit/fitted.npz')
            model.enable_shared_static_offset(source/'fit/fitted.npz', io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            snap = source/'gaussian-run/step-018000'
            model.load_ply(str(snap/'bound-local.ply'))
            with np.load(snap/'shared-static-offset.npz', allow_pickle=False) as data:
                final = torch.as_tensor(data['static_offset'], device='cuda').clone()
            with torch.no_grad(): model.static_offset.copy_(final)
            model.binding_counter = torch.bincount(model.binding.long(), minlength=len(model.faces))
            world = GaussianModel(3); world.load_ply(str(snap/'world.ply'))
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            bg = torch.zeros(3, device='cuda')
            ap = argparse.ArgumentParser(); opt = OptimizationParams(ap).extract(ap.parse_args([]))
            grads, observations = [], []
            parameters = (model.static_offset, model._xyz, model._scaling)
            names = ('static_offset', 'local_xyz', 'local_logscale')
            progress('gradient_diagnostic')
            for i in indices:
                if time.monotonic()-start > plan['wall_budget_seconds']:
                    raise TimeoutError('Fixed diagnostic wall budget exceeded')
                cam, gt, alpha = views.get(i)
                pkg = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][i])
                photo, l1, ssim = foreground_loss(pkg['render'], gt, alpha)
                visible = pkg['visibility_filter']
                gphoto = torch.autograd.grad(photo, parameters)
                aa = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][i],
                    override_color=torch.ones_like(model._xyz))['render'][:1]
                silhouette = .1*(aa-alpha).abs().mean()
                gmask = torch.autograd.grad(silhouette, parameters)
                reg = torch.relu(model._xyz[visible].norm(dim=-1)-opt.threshold_xyz).mean()*opt.lambda_xyz
                reg += torch.relu(model._scaling[visible].exp()-opt.threshold_scale).norm(dim=-1).mean()*opt.lambda_scale
                greg = torch.autograd.grad(reg, parameters, allow_unused=True)
                record = dict(frame=rows[i]['frame'], photo=float(photo), l1=float(l1),
                    ssim=float(ssim), weighted_silhouette=float(silhouette), local_regularizer=float(reg),
                    visible=int(visible.sum()), gradients={name:{'photo':gradient_stats(a),
                    'weighted_silhouette':gradient_stats(b),'local_regularizer':gradient_stats(c)}
                    for name,a,b,c in zip(names,gphoto,gmask,greg)})
                go = gphoto[0] + gmask[0]
                grads.append(go.detach().cpu().numpy())
                if i == 50:
                    with torch.no_grad():
                        restored = render_pose(render, cam, model, pipe, bg, 50)['render']
                        reference = render(cam, world, pipe, bg)['render']
                        delta = (restored-reference).abs()
                        record['bound_restore_parity'] = dict(mean_abs=float(delta.mean()),max_abs=float(delta.max()))
                        assert delta.mean() < 1.e-6, 'Bound snapshot restoration differs'
                        image(gt).save(out/'reference_input.png')
                        image(restored).save(out/'reference_final_offset.png')
                        final_vertices = model.shared_offset_vertices(50).clone()
                        model.static_offset.zero_()
                        zero = render_pose(render, cam, model, pipe, bg, 50)['render']
                        zphoto, zl1, zssim = foreground_loss(zero, gt, alpha)
                        image(zero).save(out/'reference_zero_offset.png')
                        zero_vertices = model.shared_offset_vertices(50)
                        E = model.offset_w2c[50]
                        K = torch.as_tensor(fit['K_integer_centers'][50],device='cuda',dtype=torch.float32)
                        def project(p):
                            q=(p @ E[:3,:3].T+E[:3,3]) @ K.T
                            return q[:,:2]/q[:,2:]
                        motion=(project(final_vertices)-project(zero_vertices)).norm(dim=-1)
                        record['zero_offset_counterfactual'] = dict(photo=float(zphoto),l1=float(zl1),ssim=float(zssim),
                            final_vertex_motion_native_px=stats(motion),fraction_vertices_motion_under1px=float((motion<1).float().mean()),
                            limitation='Frozen final Gaussian parameters; counterfactual sensitivity, not new optimization or quality proof.')
                        model.static_offset.copy_(final)
                    direction = go.detach()/go.detach().norm(dim=-1).max().clamp_min(1.e-20)
                    analytic=float((go*direction).sum())
                    finite=[]
                    with torch.no_grad():
                        for eps in plan['finite_difference_eps_m']:
                            values=[]
                            for sign in (-1,1):
                                model.static_offset.copy_(final+sign*eps*direction)
                                rgb=render_pose(render,cam,model,pipe,bg,50)['render']
                                a=render_pose(render,cam,model,pipe,bg,50,override_color=torch.ones_like(model._xyz))['render'][:1]
                                values.append(float(foreground_loss(rgb,gt,alpha)[0]+.1*(a-alpha).abs().mean()))
                            numeric=(values[1]-values[0])/(2*eps)
                            finite.append(dict(max_vertex_perturbation_m=eps,analytic=analytic,numeric=numeric,
                                relative_error=abs(numeric-analytic)/max(abs(analytic),1.e-12)))
                        model.static_offset.copy_(final)
                    record['render_loss_directional_derivative'] = finite
                observations.append(record)
                progress('gradient_diagnostic', completed_frames=len(observations), last_frame=rows[i]['frame'])
            g=np.stack(grads);length=np.linalg.norm(g,axis=-1)
            support=(length>1.e-12).sum(axis=0)
            cancellation=np.linalg.norm(g.sum(axis=0),axis=-1)/np.maximum(length.sum(axis=0),1.e-20)
            active=support>=2
            result=dict(state='completed_read_only',created=io.now(),plan_sha256=io.sha(plan_path),observations=observations,
                steps_updated=0,optimizer_constructed=False,training_started=False,sources_unchanged=True,
                points=len(model._xyz),vertices=len(final),same_checkpoint=True,
                shared_gradient_agreement=dict(vertices_seen_in_multiple_views=int(active.sum()),ratio=stats(cancellation[active]),
                    definition='norm(sum per-view gradients)/sum(norm per-view gradients), at the saved18k solution only',
                    limitations='View-dependent gradient disagreement alone does not identify wrong cameras, micromotion, or material correspondence.'),
                local_scale_world_m=stats(model.get_scaling.detach().max(dim=-1).values),
                local_center_norm=stats(model._xyz.detach().norm(dim=-1)),
                local_center_above_original_threshold_fraction=float((model._xyz.detach().norm(dim=-1)>opt.threshold_xyz).float().mean()),
                final_offset_m=stats(final.norm(dim=-1)),quality_accepted=False,
                elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            np.savez_compressed(out/'per-view-offset-gradients.npz',gradients=g,indices=np.asarray(indices),agreement=cancellation,support=support)
        for p,digest in plan['input_hashes'].items(): assert io.sha(source/p)==digest,p
        for p,digest in plan['source_hashes'].items(): assert io.sha(io.PROJECT/p)==digest,p
        io.save(out/'result.json',result);progress('completed_read_only')
        print(json.dumps({k:v for k,v in result.items() if k!='observations'}),flush=True)
    except Exception as exc:
        progress('failed',error=str(exc));(out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    run(Path(parser.parse_args().plan))
