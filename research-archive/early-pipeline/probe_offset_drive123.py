"""No-update shared-offset gradient audit on fixed inputs and original init."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import json
import os
import sys
import time
import numpy as np
import run_official_gaussians123 as io


def stats(x):
    return dict(zip(('median', 'p90', 'max'), map(float, np.quantile(x, [.5, .9, 1]))))


def main(plan_path):
    plan = io.read(plan_path)
    root, out = Path(plan['source_root']), Path(plan['output_root'])
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    sources = {str(io.PROJECT/p): h for p, h in plan['source_hashes'].items()}
    sources.update({str(root/p): h for p, h in plan['input_hashes'].items()})
    sources[str(Path(__file__))] = plan['probe_sha256']
    for p, h in sources.items():
        assert io.sha(p) == h, p
    sys.path.insert(0, str(io.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(io.ASSETS)
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from shuolee_portrait.training import Views, named_pose_layout, render_pose, foreground_loss
    from gaussian_renderer import render
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as f:
        fit = dict(f)
    rows = io.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    groups, all_cases = None, []
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        views = Views(rows, 1520, 1, pose_indices=layout['pose_indices'])
        for mode in plan['cases']:
            torch.manual_seed(6060); torch.cuda.manual_seed_all(6060); np.random.seed(6060)
            model = SMPLXGaussianModel(3)
            model.load_fresh_meshes(root/'fit/fitted.npz')
            if mode != 'saved_ADC1000':
                count = 20908 if mode == 'original_centroid_init' else 200000
                initialization = model.initialize_surface(count, 6060)
                if count == 20908:
                    assert torch.equal(model.binding, torch.arange(count, device='cuda'))
                    assert torch.equal(model.binding_counter, torch.ones(count, dtype=torch.int32, device='cuda'))
                    assert model._xyz.count_nonzero() == model._scaling.count_nonzero() == 0
                    assert torch.equal(model.get_xyz, model.face_center)
            model.enable_shared_static_offset(root/'fit/fitted.npz', io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            if mode == 'saved_ADC1000':
                snap = root/'gaussian-run/step-001000'
                model.load_ply(str(snap/'bound-local.ply'))
                with np.load(snap/'shared-static-offset.npz', allow_pickle=False) as f:
                    with torch.no_grad():
                        model.static_offset.copy_(torch.as_tensor(f['static_offset'], device='cuda'))
                    model.active_sh_degree = int(f['active_sh_degree'])
                initialization = dict(saved_state=True, optimizer_loaded=False)
            with torch.no_grad():
                canonical = model.offset_body.v_template + torch.einsum('bl,vcl->bvc',
                    model.offset_parameters['betas'], model.offset_body.shapedirs[:, :, :10])[0]
                tri = canonical[model.faces]
                fn = torch.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], dim=-1)
                normal = torch.zeros_like(canonical)
                for j in range(3): normal.index_add_(0, model.faces[:, j], fn)
                normal = torch.nn.functional.normalize(normal, dim=-1)
                dominant = model.offset_body.lbs_weights.argmax(1).cpu().numpy()
                groups = dict(all=np.ones(len(dominant), bool),
                    torso_upper_limbs=np.isin(dominant, [0, 3, 6, 9, 13, 14, 16, 17, 18, 19]),
                    legs_feet=np.isin(dominant, [1, 2, 4, 5, 7, 8, 10, 11]),
                    head_neck=np.isin(dominant, [12, 15, 22, 23, 24]),
                    hands=(dominant >= 25) | np.isin(dominant, [20, 21]))
                before = [p.detach().clone() for p in (model.static_offset, model._xyz, model._scaling)]
            gradients, records = [], []
            for idx in plan['frame_indices']:
                if time.monotonic()-started > plan['max_seconds']:
                    raise TimeoutError('Registered no-update audit budget exhausted')
                cam, gt, alpha = views.get(idx)
                pkg = render_pose(render, cam, model, pipe, torch.zeros(3, device='cuda'), layout['pose_indices'][idx])
                photo = foreground_loss(pkg['render'], gt, alpha)[0]
                mask = render(cam, model, pipe, torch.zeros(3, device='cuda'),
                    override_color=torch.ones_like(model.get_xyz))['render'][:1]
                silhouette = .1*(mask-alpha).abs().mean()
                gp = torch.autograd.grad(photo, model.static_offset, retain_graph=True)[0]
                gs = torch.autograd.grad(silhouette, model.static_offset)[0]
                assert torch.isfinite(gp).all() and torch.isfinite(gs).all()
                g = gp+gs; gradients.append(g.detach().cpu().numpy())
                rec = dict(frame=rows[idx]['frame'], photo=float(photo), weighted_alpha=float(silhouette),
                    photo_gradient_l2=float(gp.norm()), alpha_gradient_l2=float(gs.norm()),
                    regions={name: dict(vertices=int(sel.sum()),
                        uniform_outward_1mm_first_order_photo=float((gp[sel]*normal[sel]).sum()*.001),
                        uniform_outward_1mm_first_order_alpha=float((gs[sel]*normal[sel]).sum()*.001))
                        for name, sel in groups.items()})
                records.append(rec)
                del gp, gs, g, pkg, mask, photo, silhouette
            arr = np.stack(gradients)
            coherence = np.linalg.norm(arr.sum(0), axis=1)/np.maximum(np.linalg.norm(arr, axis=2).sum(0), 1e-30)
            normal_np = normal.cpu().numpy(); summed = arr.sum(0)
            active = np.linalg.norm(arr, axis=2).sum(0) > 0
            summaries = {}
            for name, sel in groups.items():
                valid = sel & active
                projection = -(summed[valid]*normal_np[valid]).sum(1)
                summaries[name] = dict(active_vertices=int(valid.sum()),
                    gradient_coherence=stats(coherence[valid]),
                    outward_descent_fraction=float((projection > 0).mean()),
                    inward_descent_fraction=float((projection < 0).mean()))
            assert all(torch.equal(p, q) for p, q in zip((model.static_offset, model._xyz, model._scaling), before))
            result = dict(case=mode, initialization=initialization, records=records, regions=summaries,
                points=len(model._xyz), learned_parameters_unchanged=True)
            all_cases.append(result)
            np.savez_compressed(out/(mode+'-gradients.npz'), gradients=arr, normals=normal_np, dominant_joint=dominant)
            io.save(out/'progress.json', dict(state='no_update_diagnostic', completed_cases=len(all_cases),
                pid=os.getpid(), elapsed_seconds=time.monotonic()-started))
            print(json.dumps(dict(case=mode, regions=summaries)), flush=True)
            del model, before
        for p, h in sources.items(): assert io.sha(p) == h, p
    io.save(out/'result.json', dict(state='complete_no_updates', updated=io.now(),
        plan_sha256=io.sha(plan_path), cases=all_cases, optimizer_created=False, optimizer_updates=0,
        learned_parameters_unchanged=True, elapsed_seconds=time.monotonic()-started,
        interpretation='Signed first-order image-loss derivatives, not optimized displacement or clothing truth. '
            'Coherence only covers seven registered views; dominant skinning groups are anatomical, not clothing labels. '
            'Positive uniform-outward loss derivative means indiscriminate inflation locally increases that loss.',
        quality_accepted=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    main(Path(p.parse_args().plan))
