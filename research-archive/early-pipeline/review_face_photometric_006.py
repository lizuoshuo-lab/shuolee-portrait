"""Compare the registered 006 face-loss endpoint at its unchanged 1.0 cameras."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import audit_unified_face_opacity as original
import launch_face_photometric_006 as launch
from review_bound_alpha_ab import delivered, detector_rois, pin, read, require, sha
from review_bound_scale_coverage import face_alpha, scale_pixels
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, save


PLAN = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/'
            'shuolee/current/robust-portrait-122/smpl-face-photometric-preregistration.json')
PLAN_SHA = 'e62aa89be98ebec30efc7cae01585cd3bfeb842d018c3606a00016a203e22d02'
STAGED_SHA = '1f20ef14f2e40ddf25f4939446bbd60fa2525f7a587958907b460a43e035f080'
CANDIDATE = LAB/'shuolee-smpl-face-photometric-006'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-photometric')
COMMIT = '187988f49cd31966e28278374a42cf1778fe8e2e'
OUT = LAB/'shuolee-smpl-face-photometric-006-endpoint-face'


def eye_regions(saved, row, source, mask):
    official = read(saved.snapshot.parents[1]/'fit/official-fit-result.json')
    observed, = [(Path(path), digest) for path, digest in official['source']['input_files'].items()
                 if Path(path).name == 'observations.npz']
    path, digest = observed
    pin(path, digest)
    with np.load(path, allow_pickle=False) as archive:
        index = archive['frame_ids'].tolist().index(row['frame'])
        keypoints = archive['keypoints'][index]
    union, _, _ = detector_rois(source, mask, keypoints)
    result = {}
    for name, points in (('eye_36_41', keypoints[59:65]), ('eye_42_47', keypoints[65:71])):
        xy = points[:, :2]
        require(np.isfinite(points).all() and ((xy >= 0) & (xy < row['wh'])).all(), 'Eye polygon leaves native raster.')
        polygon = Image.new('1', tuple(row['wh']))
        ImageDraw.Draw(polygon).polygon([tuple(map(float, point)) for point in xy], fill=1)
        region = np.asarray(polygon) & (mask > .99)
        require(region.sum() == np.asarray(polygon).sum() and region.any(), 'Whole eye polygon is not confident foreground.')
        result[name] = region
    require(np.array_equal(union, result['eye_36_41'] | result['eye_42_47']), 'Eye detector masks differ.')
    return result, digest


def candidate_info(plan):
    root = CANDIDATE/'run'
    launch.verify_source()
    pin(CANDIDATE/'preregistration.json', STAGED_SHA)
    if not (CANDIDATE/'trial-terminal.json').is_file():
        return None
    terminal = read(root/'result.json')
    wrapper = read(CANDIDATE/'wrapper-terminal.json')
    trial = read(CANDIDATE/'trial-terminal.json')
    digest = pin(root/'result.json', wrapper['result_sha256'])
    require(trial['state'] == 'delivered_pending_quality_review' and trial['acceptance_plan_sha256'] == PLAN_SHA
            and trial['result_sha256'] == digest and trial['preregistration_sha256'] == STAGED_SHA
            and trial['verified_camera_sha256'] == launch.CAMERA_PINS
            and wrapper['candidate_commit'] == COMMIT and wrapper['state'] == 'completed'
            and terminal['state'] == 'completed' and terminal['format_passed'] is True
            and terminal['reference_frame_id'] == plan['baseline_notes']['reference_frame']
            and terminal['fit_reused'] is True and terminal['fit_calls'] == 0
            and terminal['gaussian_from_scratch'] is True and terminal['from_scratch'] is False
            and terminal['steps'] == plan['budget']['gaussian_iterations'] and terminal['sh_degree'] == 0,
            'The candidate terminal is outside the registered scope.')
    pin(CANDIDATE/'preregistration.json', terminal['reused_fit_preregistration_sha256'])
    for path, digest_camera in trial['verified_camera_sha256'].items():
        pin(root/path, digest_camera)
    for path, key in (('fit/fitted.npz', 'fixed_fit'), ('data/training-inputs.json', 'fixed_rows'),
                      ('fit/selection.json', 'fixed_selection')):
        pin(root/path, plan[key]['sha256'])
    profile = read(root/'gaussian-run/config.json')
    require(profile['face_photometric'] is True and profile['face_photometric_weight'] == plan['change']['applied_weight']
            and profile['seed'] == plan['budget']['seed'] and profile['optimizer_initial_state_entries'] == 0
            and profile['face_photometric_roi_pixels'] == trial['actual_face_roi_pixels']
            and profile['optimization']['position_lr_max_steps'] == plan['budget']['schedule_iterations'],
            'Wrong isolated face-loss profile.')
    return dict(run=str(root), project=str(PROJECT), terminal_sha256=digest,
                trial_terminal_sha256=sha(CANDIDATE/'trial-terminal.json'))


def preflight():
    pin(PLAN, PLAN_SHA)
    plan = read(PLAN)
    require(plan['state'] == 'registered_before_candidate_training' and plan['subject'] == '006', 'Wrong acceptance plan.')
    for key in ('baseline_terminal', 'fixed_fit', 'fixed_rows', 'fixed_selection', 'fixed_face_cameras_and_regions',
                'fixed_spill_bands', 'fixed_formal_holdouts'):
        pin(Path(plan[key]['path']), plan[key]['sha256'])
    saved, pose, row, near, native_roi, near_roi, native_crop, dolly_crop, _, identity = original.preflight('006')
    previous = read(Path(plan['fixed_face_cameras_and_regions']['path']))
    require(previous['plan_code_sha256'] == sha(Path(original.__file__)) and previous['restored_exact'] is True
            and previous['native_K'] == row['K'] and previous['native_w2c'] == row['w2c']
            and previous['near_w2c'] == near['w2c'] and previous['scale'] == 1.
            and previous['reference'] == identity['reference'], 'Saved face-camera contract changed.')
    counts = {item['camera']: item['stock']['pixels'] for item in previous['rows']}
    require(counts == {'native': int(native_roi.sum()), 'dolly': int(near_roi.sum())}, 'Rebuilt face masks differ.')
    with Image.open(row['image']) as im, Image.open(row['mask']) as target:
        source = np.asarray(im.convert('RGB'))
        mask = np.asarray(target.convert('L'), dtype=np.float32)/255
    eyes, detector_sha = eye_regions(saved, row, source, mask)
    baseline = dict(run=str(Path(plan['baseline_terminal']['path']).parent), project=str(original.PROJECT),
                    terminal_sha256=plan['baseline_terminal']['sha256'])
    spec = dict(acceptance_plan_sha256=PLAN_SHA, frame=row['frame'], pose_index=pose, source_row=row,
                near_row=near, native_crop=native_crop, dolly_crop=dolly_crop,
                roi_pixels={**counts, **{key: int(roi.sum()) for key, roi in eyes.items()}}, detector_sha256=detector_sha,
                baseline_face_audit_root=str(Path(plan['fixed_face_cameras_and_regions']['path']).parent),
                baseline_face_audit_sha256=plan['fixed_face_cameras_and_regions']['sha256'],
                baseline_face_audit_rows=previous['rows'], baseline=baseline, candidate=candidate_info(plan),
                face_gates={key: value for key, value in plan['gates'].items() if key.startswith('inner_face_')},
                scale=1., optimizer_updates=0, refits=0)
    return spec, dict(native=native_roi, dolly=near_roi, **eyes)


def appearance(rgb, source, roi):
    actual, target = rgb[roi], source[roi]
    weights = np.array([.2126, .7152, .0722])

    def luminance(pixels):
        y = pixels @ weights
        q10, q90 = np.quantile(y, [.10, .90])
        return dict(mean=float(y.mean()), contrast_p90_p10=float(q90-q10))

    return dict(pixels=int(roi.sum()), black_rgb_mae=float(np.abs(actual-target).mean()),
                black_render_luma=luminance(actual), source_luma=luminance(target))


def capture(output, side, lease_fd):
    require(os.path.samestat(os.fstat(lease_fd), GPU_LEASE.stat()), 'GPU lease not inherited.')
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = read(output/'plan.json')
    arm = spec[side]
    root = Path(arm['run'])
    pin(root/'result.json', arm['terminal_sha256'])
    sys.path.insert(0, str(Path(arm['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    from gaussian_renderer import render

    rows = read(root/'data/training-inputs.json')['records']
    saved = delivered(root, ASSET, rows, read(root/'fit/selection.json')['retained_holdouts'])
    require(rows[0] == spec['source_row'] and saved.facts['reference_pose_index'] == spec['pose_index'] == 0
            and saved.facts['source_files_sha256'] == saved.bound['source_files_sha256'], 'Reference rows or renderer changed.')
    pin(output/'rois.npz', spec['roi_archive_sha256'])
    with np.load(output/'rois.npz', allow_pickle=False) as masks:
        rois = dict(masks)
    pin(Path(rows[0]['image']), rows[0]['source_sha256'])
    with Image.open(rows[0]['image']) as im:
        source = np.asarray(im.convert('RGB'), dtype=np.float32)/255
    rt.configure(ASSET.parents[1])
    panels, receipt = {}, {}
    with gpu_lock(), torch.no_grad():
        require(torch.cuda.device_count() == 1, 'Expected one visible CUDA GPU.')
        model, parity = _load_model(saved.snapshot/'bound-local.ply', saved.snapshot, saved.fit,
                                    saved.mesh, saved.facts, ASSET)
        black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        model.select_mesh_by_timestep(0)
        centers, opacity = model.get_xyz.clone(), model._opacity.clone()
        for name, view, crop in (('native', rows[0], spec['native_crop']), ('dolly', spec['near_row'], spec['dolly_crop'])):
            cam, camera_receipt = camera(view, 0, pose_index=0)
            prior = next(row for row in spec['baseline_face_audit_rows'] if row['camera'] == name)
            require(camera_receipt == prior['camera_receipt'], 'Replay camera differs from the frozen face camera.')
            depth_row = centers.new_tensor(view['w2c'][2])
            min_depth = float((centers @ depth_row[:3] + depth_row[3]).min())
            require(min_depth > .2, f'{name} camera intersects the renderer cutoff: {min_depth} m.')
            pixels, composition = scale_pixels(render, render_pose, cam, model, pipe, black, white, 0, 1.)
            alpha_metrics = face_alpha(pixels[2], rois[name])
            if name == 'native':
                saved_row = saved.replay['records'][0]
                path = saved.snapshot/'posed-replay'/saved_row['image']
                pin(path, saved_row['image_sha256'])
                with Image.open(path) as im:
                    difference = abs(original.uint8(pixels[0]).astype(np.int16)-np.asarray(im.convert('RGB')))
                require(difference.max() <= 1, 'Native replay differs from the delivered Gaussian.')
            if side == 'baseline':
                require(abs(alpha_metrics['fraction_below_095']-prior['stock']['fraction_below_095']) <= 1/rois[name].sum(),
                        'The baseline face coverage does not reproduce.')
            x0, y0, x1, y1 = crop
            color_tiles = [original.uint8(a[y0:y1, x0:x1]) for a in pixels[:2]]
            alpha_tile = original.uint8(pixels[2][y0:y1, x0:x1])
            panels[name] = np.stack([*color_tiles, np.repeat(alpha_tile[..., None], 3, axis=2)])
            if side == 'baseline':
                path = Path(spec['baseline_face_audit_root'])/prior['sheet']['file']
                pin(path, prior['sheet']['sha256'])
                h, w = alpha_tile.shape
                with Image.open(path) as im:
                    expected = np.array(im.crop((0, 21, 3*w, 21+h)).convert('RGB'))
                actual = np.concatenate(panels[name], axis=1).astype(np.int16)
                require(abs(actual-expected).max() <= 1, 'Baseline cropped RGBA differs from the prior control.')
            receipt[name] = dict(camera=camera_receipt, alpha=alpha_metrics, composition_error=composition, min_center_depth_m=min_depth,
                                 appearance=appearance(pixels[0], source, rois[name]) if name == 'native' else None)
            if name == 'native':
                receipt[name]['eyes'] = {key: appearance(pixels[0], source, roi)
                                          for key, roi in rois.items() if key.startswith('eye_')}
        require(model.optimizer is None and torch.equal(model.get_xyz, centers) and torch.equal(model._opacity, opacity),
                'Read-only replay changed the saved centers or opacity.')
    np.savez_compressed(output/f'{side}-panels.npz', **panels)
    save(output/f'{side}.json', dict(views=receipt, model_parity=parity, optimizer_updates=0,
                                    panels_sha256=sha(output/f'{side}-panels.npz')))


def gates(base, candidate, spec):
    improvement = base['dolly']['alpha']['fraction_below_095']-candidate['dolly']['alpha']['fraction_below_095']
    regression = candidate['native']['appearance']['black_rgb_mae']-base['native']['appearance']['black_rgb_mae']
    registered = spec['face_gates']
    return dict(dolly_below_095_improvement=improvement, native_black_rgb_mae_regression=regression,
                dolly_coverage_passed=improvement >= registered['inner_face_dolly_alpha_below_095_absolute_improvement_min'],
                native_appearance_passed=regression <= registered['inner_face_native_black_RGB_MAE_absolute_regression_max'])


def merge(output, spec):
    arms = ('baseline', 'candidate')
    reports, tiles, sheets = {}, {}, {}
    for arm in arms:
        reports[arm] = read(output/f'{arm}.json')
        path = output/f'{arm}-panels.npz'
        pin(path, reports[arm]['panels_sha256'])
        with np.load(path, allow_pickle=False) as archive:
            tiles[arm] = dict(archive)
    for name in ('native', 'dolly'):
        _, h, w, _ = tiles['baseline'][name].shape
        panel = Image.new('RGB', ((4 if name == 'native' else 3)*w, 2*(h+21)), '#181818')
        draw = ImageDraw.Draw(panel)
        if name == 'native':
            pin(Path(spec['source_row']['image']), spec['source_row']['source_sha256'])
            with Image.open(spec['source_row']['image']) as im:
                source = im.convert('RGB').crop(tuple(spec['native_crop']))
        for i, arm in enumerate(arms):
            if name == 'native':
                x, y = 3*w, i*(h+21)
                panel.paste(source, (x, y+21))
                draw.text((x+4, y+4), 'Observed native RGB', fill='white')
            for j, (layer, pixels) in enumerate(zip(('black', 'white', 'alpha'), tiles[arm][name], strict=True)):
                x, y = j*w, i*(h+21)
                panel.paste(Image.fromarray(pixels), (x, y+21))
                draw.text((x+4, y+4), f'{arm} / {layer}', fill='white')
        path = output/f'{name}.png'
        panel.save(path)
        sheets[name] = dict(file=path.name, sha256=sha(path))
    report = gates(*(reports[arm]['views'] for arm in arms), spec)
    result = output/'result.json'
    save(result, dict(state='face_endpoint_review_complete', plan_sha256=sha(output/'plan.json'), gates=report,
                      reports=reports, panels=sheets, visual_gray_occlusion_review='pending',
                      protected_holdout_gate='separate', dolly_rgb_truth_available=False,
                      quality_accepted=False, optimizer_updates=0, refits=0))
    print(json.dumps(dict(result=str(result), sha256=sha(result), **report)), flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        capture(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    spec, rois = preflight()
    if args.action == 'check':
        print(json.dumps(dict(roi_pixels=spec['roi_pixels'], candidate=spec['candidate'], gpu_calls=0)))
        return
    require(spec['candidate'] is not None, 'Wait for the registered candidate delivery.')
    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        np.savez_compressed(OUT/'rois.npz', **rois)
        spec.update(roi_archive_sha256=sha(OUT/'rois.npz'), code_sha256=sha(Path(__file__)))
        save(OUT/'plan.json', spec)
        for side in ('baseline', 'candidate'):
            subprocess.run([sys.executable, '-B', __file__, '--worker', str(OUT), side, str(lease.fileno())],
                           pass_fds=(lease.fileno(),), check=True)
    merge(OUT, spec)


if __name__ == '__main__':
    main()
