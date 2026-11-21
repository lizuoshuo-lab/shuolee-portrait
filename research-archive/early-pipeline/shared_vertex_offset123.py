"""Bounded original-static-offset bridge check and fresh Gaussian run."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import shutil
import sys
import time

import run_official_gaussians123 as runner


def preflight(plan, plan_path):
    import fcntl
    import numpy as np
    import torch
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(runner.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(runner.ASSETS)
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    source = Path(plan['source_root'])
    model = SMPLXGaussianModel(3)
    model.load_fresh_meshes(source/'fit/fitted.npz')
    start = time.monotonic()
    result = model.enable_shared_static_offset(source/'fit/fitted.npz', runner.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with np.load(source/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    max_px = 0.
    with torch.no_grad():
        for i in range(model.num_timesteps):
            new = model.shared_offset_vertices(i).cpu().numpy().astype(float)
            E, K = fit['w2c'][i], fit['K_integer_centers'][i]
            project = lambda p: (p @ E[:3, :3].T + E[:3, 3]) @ K.T
            a, b = project(new), project(fit['vertices_world'][i])
            assert np.all(a[:, 2] > 0) and np.all(b[:, 2] > 0)
            max_px = max(max_px, float(abs(a[:, :2]/a[:, 2:]-b[:, :2]/b[:, 2:]).max()))
    assert max_px < .01, max_px
    # Different poses share one parameter; derivative checked against central FD.
    direction = torch.randn(model.static_offset.shape, generator=torch.Generator(device='cuda').manual_seed(123), device='cuda')
    direction /= direction.norm()
    checks = []
    for i in (0, 50, 93):
        weights = torch.arange(10475*3, device='cuda').reshape(10475, 3).float().sin()
        value = (model.shared_offset_vertices(i)*weights).sum()
        grad, = torch.autograd.grad(value, model.static_offset)
        analytic = float((grad*direction).sum())
        eps = .001
        with torch.no_grad():
            model.static_offset.copy_(eps*direction)
            plus = float((model.shared_offset_vertices(i)*weights).sum())
            model.static_offset.copy_(-eps*direction)
            minus = float((model.shared_offset_vertices(i)*weights).sum())
            model.static_offset.zero_()
        numeric = (plus-minus)/(2*eps)
        assert torch.isfinite(grad).all() and grad.abs().max() > 0
        checks.append(dict(frame=str(fit['frame_ids'][i]), analytic=analytic, finite_difference=numeric,
                           absolute_error=abs(analytic-numeric)))
        assert abs(analytic-numeric) < .03, checks[-1]
    result.update(state='passed', plan_sha256=runner.sha(plan_path), native_projection_max_abs_px=max_px,
                  gradient_checks=checks, elapsed_seconds=time.monotonic()-start,
                  model_source_sha256=runner.sha(runner.PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py'),
                  installed_smplx_forward_reused=True, training_started=False, quality_accepted=False)
    runner.save(plan_path.with_name('shared-vertex-offset-preflight.json'), result)
    print(json.dumps(result), flush=True)


def run(plan, plan_path):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source, target = Path(plan['source_root']), Path(plan['target_root'])
    check = runner.read(plan_path.with_name('shared-vertex-offset-preflight.json'))
    assert check['state'] == 'passed' and check['plan_sha256'] == runner.sha(plan_path)
    assert check['model_source_sha256'] == runner.sha(runner.PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py')
    for file, digest in plan['input_hashes'].items():
        assert runner.sha(source/file) == digest, file
    for file, digest in plan['source_hashes'].items():
        assert runner.sha(runner.PROJECT/file) == digest, file
    assert runner.sha(runner.__file__) == plan['existing_launcher_sha256']
    assert not target.exists(), 'Fresh run only; never overwrite or resume.'
    target.mkdir(exist_ok=False)
    for file in ('fit/official-fit-result.json', 'data/training-inputs.json', 'progress.json'):
        out = target/file; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/file, out)
    # Only reference metadata changes to the already reviewed open-eye frame51.
    with np.load(source/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = {key: archive[key] for key in archive.files}
    old_reference = str(fit['reference_frame_id'].item())
    assert plan['reference_frame_id'] in fit['frame_ids']
    fit['reference_frame_id'] = np.asarray(plan['reference_frame_id'])
    np.savez_compressed(target/'fit/fitted.npz', **fit)
    receipt = runner.read(target/'fit/official-fit-result.json')
    receipt['reference_metadata_derivation'] = dict(source_fit_sha256=plan['input_hashes']['fit/fitted.npz'],
        source_reference=old_reference, selected_reference=plan['reference_frame_id'],
        all_geometry_pose_camera_observation_arrays_unchanged=True, source_root=str(source), fit_reused_not_refitted=True)
    receipt['fit_sha256'] = runner.sha(target/'fit/fitted.npz')
    runner.save(target/'fit/official-fit-result.json', receipt)
    runner.save(target/'control-input.json', dict(plan=plan, plan_sha256=runner.sha(plan_path),
        fit_reused_not_refitted=True, Gaussian_from_scratch=True, end_to_end_benchmark=False,
        human_direction='shared vertexoffset; use original source mechanism first',
        upstream_static_offset_optimizer_default='commented out; enabled explicitly in local SMPL-X bridge'))
    runner.SOURCES = {key: plan['source_hashes']['src/shuolee_portrait/'+key] for key in ('training.py', 'export.py')}
    runner.CFG = plan['config']
    # First control is capped at1000. Later extension requires a reviewed decision.
    original_review = runner.review_decision
    def stop_only(path, request):
        decision = original_review(path, request)
        if decision['decision'] != 'stop':
            raise ValueError('This first original-static-offset comparison is capped at1000.')
        return decision
    runner.review_decision = stop_only
    runner.run(SimpleNamespace(root=str(target), project_root=str(runner.PROJECT), assets_root=str(runner.ASSETS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    assert os.name == 'posix', 'Existing WSL CUDA environment required.'
    path = Path(args.plan); plan = runner.read(path)
    (preflight if args.preflight else run)(plan, path)
