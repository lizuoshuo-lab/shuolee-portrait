"""Bounded no-update timing/parity check for shared-offset parent indices."""
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


def run(plan_path):
    plan = io.read(plan_path)
    root, out = Path(plan['source_root']), Path(plan['output_root'])
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    def progress(state, **details):
        io.save(out/'progress.json', dict(state=state, pid=None if state in ('completed_no_updates','failed_preserved') else os.getpid(), updated=io.now(),
            elapsed_seconds=time.monotonic()-start, training_started=False, **details))
    def check_budget():
        if time.monotonic()-start > plan['max_seconds']:
            raise TimeoutError('No-update performance probe budget exceeded')
    def check_hashes():
        assert io.sha(__file__) == plan['probe_sha256'], 'Probe source changed'
        for path, digest in plan['source_hashes'].items():
            assert io.sha(io.PROJECT/path) == digest, path
        for path, digest in plan['input_hashes'].items():
            assert io.sha(root/path) == digest, path
    progress('checking')
    try:
        check_hashes()
        with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            sys.path.insert(0, str(io.PROJECT/'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(io.ASSETS)
            from shuolee_portrait.training import Views, named_pose_layout, render_pose, foreground_loss
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            from gaussian_renderer import render
            from arguments import OptimizationParams
            with np.load(root/'fit/fitted.npz', allow_pickle=False) as z:
                fit = dict(z)
            rows = io.read(root/'data/training-inputs.json')['records']
            layout = named_pose_layout(fit, rows)
            views = Views(rows, 1520, len(plan['frame_indices']), pose_indices=layout['pose_indices'])
            model = SMPLXGaussianModel(3)
            model.load_fresh_meshes(root/'fit/fitted.npz')
            model.enable_shared_static_offset(root/'fit/fitted.npz', io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            snap = root/'gaussian-run/step-018000'
            model.load_ply(str(snap/'bound-local.ply'))
            with np.load(snap/'shared-static-offset.npz', allow_pickle=False) as z:
                with torch.no_grad(): model.static_offset.copy_(torch.as_tensor(z['static_offset'],device='cuda'))
                model.active_sh_degree = int(z['active_sh_degree'])
            parents = model.offset_body.parents.detach().clone()
            assert parents.device.type == 'cuda' and parents.dtype == torch.int64
            assert parents.shape == (55,) and int(parents[0]) == -1
            host = parents.cpu()
            ids = host.tolist()
            assert all(0 <= parent < child for child, parent in enumerate(ids[1:], 1))
            params = (model.static_offset, model._xyz, model._features_dc, model._features_rest,
                      model._opacity, model._scaling, model._rotation)
            names = ('static_offset', 'local_xyz', 'sh_dc', 'sh_rest', 'opacity', 'local_logscale', 'local_rotation')
            before = [p.detach().clone() for p in params]
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            bg = torch.zeros(3,device='cuda')
            parser = argparse.ArgumentParser()
            options = OptimizationParams(parser).extract(parser.parse_args([]))
            records = []
            for frame_index in plan['frame_indices']:
                check_budget()
                camera, gt, alpha = views.get(frame_index)
                pose = layout['pose_indices'][frame_index]
                def closure():
                    pkg = render_pose(render,camera,model,pipe,bg,pose)
                    photo, _, _ = foreground_loss(pkg['render'],gt,alpha)
                    aa = render_pose(render,camera,model,pipe,bg,pose,
                        override_color=torch.ones_like(model.get_xyz))['render'][:1]
                    visible = pkg['visibility_filter']
                    reg = torch.relu(model._xyz[visible].norm(dim=-1)-options.threshold_xyz).mean()*options.lambda_xyz
                    reg += torch.relu(model._scaling[visible].exp()-options.threshold_scale).norm(dim=-1).mean()*options.lambda_scale
                    loss = photo+.1*(aa-alpha).abs().mean()+reg
                    grad = torch.autograd.grad(loss,params)
                    return loss.detach(), pkg['render'].detach(), grad
                reference = None
                modes = []
                # ABBA reduces order/thermal effects without changing any learned parameter.
                for mode in ('cuda_indices_A','cpu_indices_B','cpu_indices_B_repeat','cuda_indices_A_repeat'):
                    model.offset_body.parents = parents if mode.startswith('cuda') else host
                    for _ in range(plan['warmup_repeats']):
                        unused = closure(); del unused
                    torch.cuda.synchronize()
                    seconds = []
                    for _ in range(plan['timed_repeats']):
                        check_budget(); begin = time.perf_counter()
                        loss, rgb, gradients = closure()
                        torch.cuda.synchronize()
                        seconds.append(time.perf_counter()-begin)
                    with torch.no_grad(): verts = model.shared_offset_vertices(pose).clone()
                    assert all(torch.isfinite(g).all() for g in gradients)
                    if reference is None:
                        reference = dict(loss=loss.clone(), rgb=rgb.clone(), vertices=verts,
                            gradients=[g.detach().clone() for g in gradients])
                    gradient_delta = {}
                    for name,g,base in zip(names,gradients,reference['gradients']):
                        difference = g-base
                        gradient_delta[name] = dict(max_abs=float(difference.abs().max()),
                            relative_l2=float(difference.norm()/base.norm().clamp_min(1.e-12)))
                    record = dict(mode=mode,seconds=seconds,median_seconds=float(np.median(seconds)),
                        loss=float(loss),loss_abs_delta=float((loss-reference['loss']).abs()),
                        render_max_abs_delta=float((rgb-reference['rgb']).abs().max()),
                        vertex_max_abs_delta=float((verts-reference['vertices']).abs().max()),
                        gradient_delta=gradient_delta)
                    record['gradient_within_registered_tolerance'] = all(
                        d['relative_l2'] <= plan['gradient_relative_l2_tolerance'] for d in gradient_delta.values())
                    modes.append(record)
                    io.save(out/'partial-evidence.json',dict(frame=rows[frame_index]['frame'],modes=modes,
                        completed_records=records,threshold_unchanged=True,no_parameter_updates=True))
                    assert record['loss_abs_delta'] == 0 and record['render_max_abs_delta'] == 0
                    assert record['vertex_max_abs_delta'] == 0
                    del loss,rgb,gradients,verts
                # Profile only the unchanged real closure; do not time profiler overhead.
                profiles = []
                if frame_index == plan['frame_indices'][0]:
                    from torch.profiler import profile,ProfilerActivity
                    for mode,tensor in [('cuda_indices',parents),('cpu_indices',host)]:
                        model.offset_body.parents = tensor
                        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                            value = closure();torch.cuda.synchronize();del value
                        events = prof.key_averages()
                        selected = [dict(name=e.key,count=e.count,self_cpu_time_us=e.self_cpu_time_total,
                                         self_device_time_us=e.self_device_time_total)
                            for e in events if 'item' in e.key or 'local_scalar' in e.key or 'Synchronize' in e.key]
                        profiles.append(dict(mode=mode,synchronization_events=selected))
                records.append(dict(frame=rows[frame_index]['frame'],modes=modes,profiles=profiles))
                io.save(out/'partial-evidence.json',dict(completed_records=records,
                    threshold_unchanged=True,no_parameter_updates=True))
                progress('checked_frame',frames_completed=len(records))
                del reference
            model.offset_body.parents = parents
            assert all(torch.equal(a,b) for a,b in zip(params,before)), 'A learned parameter changed'
            assert all(p.device.type == 'cuda' for p in model.offset_body.parameters())
            check_hashes()
            parity_pass = all(m['gradient_within_registered_tolerance'] for r in records for m in r['modes'])
            result = dict(state='completed_no_updates',gradient_parity_passed=parity_pass,
                production_change_admitted=False,updated=io.now(),plan_sha256=io.sha(plan_path),
                records=records,points=len(model._xyz),active_sh_degree=model.active_sh_degree,
                elapsed_seconds=time.monotonic()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                optimizer_created=False,optimizer_updates=0,source_code_changed=False,
                all_learned_parameters_unchanged=True,inputs_unchanged=True,
                limitation='Fixed completed18k model, two cached views, render/loss/backward only. Excludes Adam, loading, ADC, snapshots, and fit; not end-to-end training speed or quality acceptance.')
            io.save(out/'result.json',result)
            progress('completed_no_updates')
            print(__import__('json').dumps(dict(state=result['state'],seconds=result['elapsed_seconds'],
                frame_medians=[[(r['mode'],r['median_seconds']) for r in f['modes']] for f in records])),flush=True)
    except BaseException as error:
        io.save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),training_started=False))
        progress('failed_preserved');raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    run(p.parse_args().plan)
