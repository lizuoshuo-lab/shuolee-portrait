"""Width/height 3x demo control using the frozen static sofa training code."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import os
import shutil
import time

import numpy as np
from PIL import Image

ROOT = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo')
RUN = ROOT / 'upsampled3x'
spec = importlib.util.spec_from_file_location('frozen_static_experiment', ROOT / 'experiment.py')
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
ex.select_subject('demo')
ex.ROOT = RUN


def verify_source():
    manifest = ex.read(ROOT / 'source-manifest.json')
    checked = 0
    for name, digest in manifest.items():
        if name.endswith('.py'):
            assert ex.sha(ROOT / 'source' / name) == digest, name
            checked += 1
    assert manifest['src/shuolee_portrait/clothed_training.py'] == '12b830120f1733ba6bc44a764d9b5e810bbb94a272f7b1a3067887151b40cdaf'
    if (RUN / 'source-adjustment.json').exists():
        adjustment = ex.read(RUN / 'source-adjustment.json')
        for name, digest in adjustment['local_package_sha256'].items():
            assert ex.sha(RUN / 'source/src/shuolee_portrait' / name) == digest, name
    return checked


def prepare():
    checked = verify_source()
    RUN.mkdir(exist_ok=False)
    for folder in ['data/images', 'data/masks', 'fit', 'clothing-init']:
        (RUN / folder).mkdir(parents=True)
    original = ROOT / 'original'
    rows = ex.read(original / 'data/training-inputs.json')['records']
    assert len(rows) == 59 and all(r['wh'] == [476, 960] for r in rows)
    copied = {}
    for name in ['fit/fitted.npz', 'clothing-init/points.npz', 'clothing-init/motion.npz', 'clothing-init/audit.json']:
        shutil.copy2(original / name, RUN / name)
        assert ex.sha(original / name) == ex.sha(RUN / name)
        copied[name] = ex.sha(RUN / name)
    cfg = ex.read(original / 'config.json')
    cfg.update(training_height=2880, max_image_size=8112, image_cache_frames=59,
               iterations=18000, seed=6060, pre_training_enlargement=3,
               native_video_input=True, no_rgb_downsampling=True)
    ex.save(RUN / 'config.json', cfg)
    result = []
    max_error = 0.0
    for index, row in enumerate(rows):
        assert ex.sha(row['image']) == row['source_sha256']
        assert ex.sha(row['mask']) == row['mask_sha256']
        image_path = RUN / 'data/images' / row['name']
        mask_path = RUN / 'data/masks' / row['name']
        with Image.open(row['image']) as im:
            assert im.size == (476, 960)
            im.convert('RGB').resize((1428, 2880), Image.Resampling.LANCZOS).save(image_path)
        with Image.open(row['mask']) as im:
            assert im.size == (476, 960)
            im.convert('L').resize((1428, 2880), Image.Resampling.BILINEAR).save(mask_path)
        rr = copy.deepcopy(row)
        K = np.diag([3., 3., 1.]) @ np.asarray(row['K'])
        Ki = K.copy(); Ki[:2, 2] -= .5
        rr.update(image=str(image_path), mask=str(mask_path), wh=[1428, 2880],
                  K=K.tolist(), K_integer_centers=Ki.tolist(), upsampling_scale=3,
                  parent_image_sha256=row['source_sha256'], parent_mask_sha256=row['mask_sha256'],
                  source_sha256=ex.sha(image_path), mask_sha256=ex.sha(mask_path))
        for key in ['face_box_xyxy', 'initial_box_xyxy']:
            if key in rr:
                rr[key] = ((np.asarray(row[key]) + .5) * 3 - .5).tolist()
        cp = np.array([[.1, .2, 2.], [-.4, -.2, 4.], [.3, .1, 3.]])
        q = cp @ np.asarray(row['K']).T
        expected = (q[:, :2] / q[:, 2:]) * 3 - .5
        actual = cp @ K.T; actual = actual[:, :2] / actual[:, 2:] - .5
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        assert rr['w2c'] == row['w2c'] and max_error < 1e-9
        result.append(rr)
        if index % 10 == 0 or index == len(rows) - 1:
            print('PREPARED', index + 1, '/', len(rows), '1428x2880', flush=True)
    ex.save(RUN / 'data/training-inputs.json', dict(records=result, coordinate_contract='pixel_corners',
            source=str(original / 'data/training-inputs.json'), transform='native crop scaled directly 3x',
            fresh_gaussians=True, cached_fit_and_surface=True, quality_accepted=False))
    plan = dict(created=ex.now(), subject='demo', width_height_scale=3, pixel_count_scale=9,
                original_wh=[476, 960], actual_training_wh=[1428, 2880], frames=59,
                interpolation='Lanczos RGB; bilinear mask', no_learned_superresolution=True,
                initial_points=200000, iterations=18000, seed=6060,
                fresh_gaussians_and_optimizer=True, cached_fit_and_surface_reused=True,
                frozen_training_source=str(ROOT / 'source'), source_python_files_verified=checked,
                source_manifest_sha256=ex.sha(ROOT / 'source-manifest.json'),
                fixed_extrinsics=True, full_intrinsics_scaled_by=3,
                maximum_projection_scale_error_px=max_error, copied_source_hashes=copied,
                source_video=ex.read(ROOT / 'plan.json')['source_video'],
                source_video_sha256=ex.read(ROOT / 'plan.json')['source_video_sha256'],
                memory_cap_gib=6, original_and_2x_outputs_preserved=True,
                quality_accepted=False, deployment_changed=False)
    ex.save(RUN / 'plan.json', plan)
    shutil.copy2(__file__, RUN / 'experiment.py')
    print(json.dumps(plan), flush=True)


def precision_contract():
    verify_source()
    diagnosis = ex.read(RUN / 'projection-precision-diagnostic.json')
    assert all(v == 0 for v in diagnosis['max_full_projection_matrix_difference'].values())
    assert diagnosis['max_errors_px']['upsampled3x'] / 3 < .001
    package = RUN / 'source/src/shuolee_portrait'
    shutil.copytree(ROOT / 'source/src/shuolee_portrait', package,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (RUN / 'source/src/avatar_core').symlink_to(ROOT / 'source/src/avatar_core', target_is_directory=True)
    (RUN / 'source/third_party').symlink_to(ROOT / 'source/third_party', target_is_directory=True)
    p = package / 'training.py'
    original = p.read_text(encoding='utf-8')
    old = '    if error > .001:\n'
    assert original.count(old) == 1
    changed = original.replace(old,
        '    # Preserve the original-resolution tolerance when inputs are enlarged.\n'
        '    if error > .001 * float(row.get(\'upsampling_scale\', 1.0)):\n')
    compile(changed, str(p), 'exec')
    p.write_text(changed, encoding='utf-8')
    adjustment = dict(created=ex.now(), reason='Unchanged normalized projection matrices have pixel errors proportional to input scale.',
        original_resolution_tolerance_px=.001, actual_3x_tolerance_px=.003,
        actual_max_3x_error_px=diagnosis['max_errors_px']['upsampled3x'],
        actual_max_original_equivalent_error_px=diagnosis['max_errors_px']['upsampled3x'] / 3,
        all_59_normalized_projection_matrices_bitwise_identical_to_baseline=True,
        camera_matrices_changed=False, training_strategy_changed=False,
        original_training_helper_sha256=ex.sha(ROOT / 'source/src/shuolee_portrait/training.py'),
        local_package_sha256={str(p.relative_to(package)): ex.sha(p) for p in package.rglob('*') if p.is_file()})
    ex.save(RUN / 'source-adjustment.json', adjustment)
    plan = ex.read(RUN / 'plan.json')
    plan['runtime_source_directory'] = str(RUN / 'source')
    plan['projection_precision_contract'] = {k: v for k, v in adjustment.items() if k != 'local_package_sha256'}
    ex.save(RUN / 'plan.json', plan)
    shutil.copy2(__file__, RUN / 'experiment.py')
    print(json.dumps({k: v for k, v in adjustment.items() if k != 'local_package_sha256'}), flush=True)


def preflight():
    verify_source()
    rt, torch = ex.runtime()
    from types import SimpleNamespace
    from shuolee_portrait.training import Views, foreground_loss
    from shuolee_portrait.clothed_surface import make_model
    from gaussian_renderer import render
    rows = ex.read(RUN / 'data/training-inputs.json')['records']
    views = Views(rows, 2880, 1)
    model = make_model(RUN)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    checks = []
    for idx in [0, len(rows) // 2, len(rows) - 1]:
        torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
        cam, gt, alpha = views.get(idx)
        assert list(gt.shape[1:]) == [2880, 1428]
        pkg = render(cam, model, pipe, bg)
        photo, l1, ssim = foreground_loss(pkg['render'], gt, alpha)
        aa = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
        loss = photo + .1 * (aa - alpha).abs().mean()
        loss.backward(); torch.cuda.synchronize()
        assert torch.isfinite(loss)
        for name in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
            assert torch.isfinite(getattr(model, name).grad).all()
        checks.append(dict(frame=rows[idx]['frame'], wh=rows[idx]['wh'], loss=float(loss),
                           nonzero_xyz_gradients=int((model._xyz.grad.norm(dim=1) > 0).sum()),
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                           seconds=time.monotonic() - start))
        for name in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
            getattr(model, name).grad = None
        del pkg, photo, l1, ssim, aa, loss, cam, gt, alpha
    ex.save(RUN / 'preflight.json', dict(created=ex.now(), passed=True, optimizer_updates=0, checks=checks,
            max_projection_error_px=max(x['projection_error_px'] for x in views.records), other_processes_interrupted=False))
    print(json.dumps(ex.read(RUN / 'preflight.json')), flush=True)


def train():
    verify_source()
    assert ex.read(RUN / 'preflight.json')['passed']
    rt, torch = ex.runtime()
    from shuolee_portrait.clothed_training import train_clothed
    state = dict(created=ex.now(), pid=os.getpid(), arm='upsampled3x', state='training', quality_accepted=False)
    ex.save(RUN / 'execution.json', state)
    start = time.monotonic()
    try:
        train_clothed(RUN, ex.read(RUN / 'config.json'))
        state.update(state='completed_pending_visual_comparison', seconds=time.monotonic() - start,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3, finished=ex.now())
        ex.save(RUN / 'execution.json', state)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=ex.now())
        ex.save(RUN / 'execution.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'precision', 'preflight', 'train'])
    args = parser.parse_args()
    {'prepare': prepare, 'precision': precision_contract, 'preflight': preflight, 'train': train}[args.action]()
