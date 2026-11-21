"""Replay three 001 opacity-fork terminals under the same archived native and dolly cameras."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from review_bound_alpha_ab import detector_rois, mask_metrics, native_box, pin, read, require, sha
from review_bound_scale_coverage import face_alpha, interior, scale_pixels, write_dolly
from run_opacity_fork001 import ARMS, BACKWARD, BASELINE_RESULT_SHA, RESET_STEPS, STEP, STOCK_RESETS, TRAINING


FRAMES = ('frame_000029', 'frame_000048', 'frame_000056')
REFERENCE_AUDIT_SHA = 'd96b3ce7f8e1b3202f9486600e12d6df8f3bd73961f72e91100d7fec05f79923'


def frozen_frame(row, identity, saved, fitted, keypoints):
    frame, pose = identity['frame'], identity['pose_index']
    require(all(saved[key] == value for key, value in identity.items()), f'Archive identity differs: {frame}')
    pin(Path(row['image']), identity['source_sha256'])
    pin(Path(row['mask']), identity['mask_sha256'])
    with Image.open(row['image']) as source, Image.open(row['mask']) as silhouette:
        require(list(source.size) == list(silhouette.size) == row['wh'], f'Native size differs: {frame}')
        rgb = np.asarray(source.convert('RGB'))
        mask = np.asarray(silhouette.convert('L'), dtype=np.float32) / 255.

    _, dark, _ = detector_rois(rgb, mask, keypoints)
    observed = fitted['observed_native_keypoints'][pose, 67:118]
    joints = fitted['joints_camera'][pose, 67:118]
    native_region = interior(observed[observed[:, 2] >= .5, :2], row['wh'])
    require(native_region is not None, f'No native face region: {frame}')
    native_region &= mask > .99

    archived = saved['closeup_1x']
    near = archived['camera']
    K, E = np.asarray(row['K']), np.asarray(row['w2c'])
    movement = np.asarray(archived['dolly_camera_space_m'], dtype=joints.dtype)
    expected_E = E.copy()
    expected_E[:3, 3] -= movement
    require(near['frame'] == frame and near['pose_index'] == pose and near['K'] == row['K']
            and near['wh'] == row['wh'] and np.array_equal(near['w2c'], expected_E)
            and near['pixel_convention'] == 'pixel_corners' and archived['scale'] == 1.,
            f'Dolly camera differs: {frame}')

    eligible = (observed[:, 2] >= .5) & (joints[:, 2] > .2)
    projected = joints[eligible] @ K.T
    residual = np.linalg.norm(projected[:, :2]/projected[:, 2:] - .5 - observed[eligible, :2], axis=1)
    matched = joints[eligible][residual <= archived['native_landmark_match_max_px']]
    require(len(matched) == archived['retained_landmarks'] and len(matched) >= 8,
            f'Archived landmark support differs: {frame}')
    projected_near = (matched - movement) @ K.T
    xy = projected_near[:, :2]/projected_near[:, 2:] - .5
    near_region = interior(xy, row['wh'])
    crop = native_box([*(xy.min(0)-24), *(xy.max(0)+24)], row['wh'])
    archived_panel, = [panel for panel in saved['panels'] if panel['file'] == frame+'-dolly.png']
    require(near_region is not None and native_region.sum() == saved['metrics']['1.0']['interior_face_alpha']['pixels']
            and near_region.sum() == archived['interior_face_alpha']['pixels']
            and dark.sum() == saved['metrics']['1.0']['dark_eye_foreground_pixels']
            and list(crop) == archived_panel['crop_xyxy'], f'Frozen region or crop differs: {frame}')
    return SimpleNamespace(row=row, identity=identity, mask=mask, dark=dark, native_region=native_region,
                           near_region=near_region, near_row=dict(row, w2c=near['w2c']),
                           near_camera=near, crop=crop)


def reference_frames(baseline, audit_path):
    pin(baseline/'result.json', BASELINE_RESULT_SHA)
    pin(audit_path, REFERENCE_AUDIT_SHA)
    audit = read(audit_path)
    saved = {item['frame']: item for item in audit['records']}
    require(audit['state'] == 'completed' and audit['terminal_sha256'] == BASELINE_RESULT_SHA
            and audit['optimizer_updates'] == 0 and set(saved) == set(FRAMES), 'Different reference alpha audit.')
    official = read(baseline/'fit/official-fit-result.json')
    fit_path, rows_path = baseline/'fit/fitted.npz', baseline/'data/training-inputs.json'
    pin(fit_path, official['fit_sha256'])
    pin(rows_path, official['training_inputs_sha256'])
    rows = read(rows_path)['records']
    selection = read(baseline/'fit/selection.json')
    require(len(rows) == 24 and set(selection['selected_frames']) == {row['frame'] for row in rows}
            and not set(FRAMES) & set(selection['retained_holdouts']), 'Different frozen training roster.')
    sources = official['source']['input_files']
    observation, = [Path(name) for name in sources if Path(name).name == 'observations.npz']
    pin(observation, sources[str(observation)])
    pin(observation, audit['observations_sha256'])
    with np.load(fit_path, allow_pickle=False) as fit, np.load(observation, allow_pickle=False) as data:
        landmarks = dict(zip(data['frame_ids'].tolist(), data['keypoints'], strict=True))
        frames = {}
        for i, row in enumerate(rows):
            frame = row['frame']
            if frame not in saved:
                continue
            pose = fit['frame_ids'].tolist().index(frame)
            identity = {key: row[key] for key in ('frame', 'source_frame_index', 'source_sha256', 'mask_sha256')}
            identity['pose_index'] = pose
            frames[frame] = (i, frozen_frame(row, identity, saved[frame], fit, landmarks[frame]))
    require(set(frames) == set(FRAMES), 'Missing matched evaluation frame.')
    return audit, frames


def recovery_roots(fork, result):
    roots = {name: fork/name/'run' for name in ARMS}
    if 'recovery_receipt' not in result:
        return roots, None

    import resume_opacity_fork001_suffix as resume

    recovered = fork.with_name(fork.name+'-recovered')
    path = recovered/'recovery.json'
    require(Path(result['recovery_receipt']).resolve(strict=True) == path,
            'Recovered runs must use the registered sibling directory.')
    pin(path, result['recovery_receipt_sha256'])
    receipt = read(path)
    require(receipt['state'] == 'suffixes_recovered'
            and Path(receipt['original_fork']).resolve(strict=True) == fork
            and Path(receipt['recovered_root']).resolve(strict=True) == recovered
            and receipt['checkpoint_sha256'] == result['checkpoint_sha256'] == resume.CHECKPOINT_SHA
            and receipt['original_pins_sha256'] == resume.PINS
            and receipt['allowed_numpy_globals'] == sorted(resume.UNSAFE_GLOBALS)
            and receipt['allowed_numpy_dtype'] == 'float64' and receipt['weights_only'] is True
            and receipt['expected_optimizer_updates_per_suffix'] == 1000
            and receipt['reused_stock_prefix'] is True and receipt['original_failure_preserved'] is True
            and receipt['recovered_arms'] == list(resume.ARMS)
            and receipt['arms'] == result['arms'][1:],
            'Recovery changed the checkpoint, scalar decoder or suffix scope.')
    pin(Path(resume.__file__), receipt['recovery_driver_sha256'])
    for name, digest in resume.PINS.items():
        pin(fork/name, digest)
    for arm in receipt['arms']:
        name = arm['arm']
        root = recovered/name/'run'
        pin(root/'status.json', receipt['recovered_arm_status_sha256'][name])
        status = read(root/'status.json')
        require(all(status[k] == v for k, v in arm.items()), f'Recovered status differs: {name}')
        roots[name] = root
    return roots, result['recovery_receipt_sha256']


def validate_fork(fork, project, audit_path):
    result_path, plan_path = fork/'result.json', fork/'preregistration.json'
    result, plan = read(result_path), read(plan_path)
    require(result['state'] == 'completed_pending_controlled_render_review'
            and result['quality_accepted'] is False and plan['baseline_result_sha256'] == BASELINE_RESULT_SHA
            and plan['step'] == STEP and plan['terminal'] == 10000 and plan['arms'] == ARMS
            and plan['seam'] == 'after_densify_before_opacity_reset_and_optimizer_step'
            and plan['stock_reset_steps'] == STOCK_RESETS and plan['omit_reset_steps'] == RESET_STEPS,
            'Different or incomplete registered opacity fork.')
    pin(fork/'shared-after-densify-step9000.pt', result['checkpoint_sha256'])
    pin(project/TRAINING, plan['diagnostic_training_sha256'])
    pin(project/BACKWARD, plan['backward_sha256'])
    pin(Path(__file__).with_name('run_opacity_fork001.py'), plan['driver_sha256'])
    baseline = Path(plan['baseline']).resolve(strict=True)
    audit, frames = reference_frames(baseline, audit_path)
    pin(baseline/'gaussian-run/training-cameras.json', result['training_camera_sha256'])
    require([a['arm'] for a in result['arms']] == list(ARMS), 'Missing or duplicated opacity arm.')
    roots, recovery_sha = recovery_roots(fork, result)
    assets = Path(read(baseline/'run.json')['assets']).resolve(strict=True)
    asset = assets/'smplx/SMPLX_NEUTRAL.npz'

    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs

    arm_data = {}
    for a in result['arms']:
        name = a['arm']
        root = roots[name]
        require(Path(a['root']).resolve() == root and a['state'] == 'completed'
                and a['checkpoint_sha256'] == result['checkpoint_sha256']
                and a['restored_exact'] is (name != 'stock')
                and a['new_optimizer_updates'] == (9999 if name == 'stock' else 1000)
                and a['training_camera_sha256'] == result['training_camera_sha256'],
                f'Arm lacks the common-state proof: {name}')
        train = root/'gaussian-run'
        training = read(train/'result.json')
        config = read(train/'config.json')
        require(training['state'] == 'completed' and training['steps'] == 10000
                and training['from_scratch'] is (name == 'stock')
                and training['gaussian_profile'] == 'smpl_sh0' and training['sh_degree'] == 0
                and training['points'] == a['points'] and training['final_snapshot'] == 'step-010000'
                and config['opacity_reset_steps'] == (STOCK_RESETS if ARMS[name] else RESET_STEPS),
                f'Wrong terminal budget, SH degree or reset schedule: {name}')
        for path in ('fit/fitted.npz', 'fit/official-fit-result.json', 'fit/shared-body.json',
                     'fit/selection.json', 'data/training-inputs.json'):
            pin(root/path, plan['input_sha256'][path])
        pin(train/'training-cameras.json', result['training_camera_sha256'])
        snap = train/'step-010000'
        pin(snap/'world.ply', a['world_ply_sha256'])
        pin(snap/'bound-local.ply', a['bound_ply_sha256'])
        pin(snap/'reference-render.png', a['reference_render_sha256'])
        manifest_path = snap/'bound-replay.json'
        manifest = read(manifest_path)
        require(manifest['schema'] == 'shuolee-smplx-bound-v1' and manifest['bound_ply'] == 'bound-local.ply'
                and manifest['bound_ply_sha256'] == a['bound_ply_sha256']
                and manifest['points'] == a['points'] and manifest['optimizer_updates'] == 0
                and manifest['sh_degree'] == 0 and manifest['reference_frame_id'] == 'frame_000056',
                f'Wrong terminal replay contract: {name}')
        renderer = manifest['source_files_sha256']
        require(set(renderer) == set(audit['renderer_source_sha256'])
                and all(digest == audit['renderer_source_sha256'][path]
                        for path, digest in renderer.items() if path != TRAINING),
                f'Renderer or mesh code differs from archived audit: {name}')
        fitted, mesh, facts = _inputs(root, snap, asset, manifest)
        require(all(facts['frames'][i] == frame.identity for i, frame in frames.values()),
                f'Named poses differ: {name}')
        arm_data[name] = SimpleNamespace(root=root, snap=snap, fitted=fitted, mesh=mesh, facts=facts,
                                         manifest_sha256=sha(manifest_path), terminal=a)
    evidence = dict(fork_result_sha256=sha(result_path), preregistration_sha256=sha(plan_path),
                    reference_audit_sha256=REFERENCE_AUDIT_SHA, checkpoint_sha256=result['checkpoint_sha256'],
                    camera_sha256=result['training_camera_sha256'], script_sha256=sha(Path(__file__)),
                    recovery_receipt_sha256=recovery_sha)
    return rt, asset, frames, arm_data, evidence


def replay(rt, asset, frames, arm_data, output):
    import torch
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            rt.configure(asset.parent.parent)
            from gaussian_renderer import render
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            output.mkdir(parents=True, exist_ok=False)
            rendered = {}
            for name, arm in arm_data.items():
                folder = output/name
                folder.mkdir()
                model, parity = _load_model(arm.snap/'bound-local.ply', arm.snap, arm.fitted,
                                            arm.mesh, arm.facts, asset)
                records = {}
                for frame in FRAMES:
                    i, inputs = frames[frame]
                    pose = inputs.identity['pose_index']
                    native, _ = camera(inputs.row, i, pose_index=pose)
                    dolly, camera_record = camera(inputs.near_row, i, pose_index=pose)
                    require(camera_record == inputs.near_camera, f'Actual dolly camera differs: {frame}')
                    model.select_mesh_by_timestep(pose)
                    E = torch.as_tensor(inputs.near_row['w2c'], dtype=model.get_xyz.dtype, device='cuda')
                    minimum_z = float((model.get_xyz @ E[2, :3] + E[2, 3]).min())
                    require(minimum_z > .2, f'Geometry crosses dolly cutoff: {name} {frame}')
                    native_pixels, native_error = scale_pixels(render, render_pose, native, model, pipe,
                                                                black, white, pose, 1.)
                    near_pixels, near_error = scale_pixels(render, render_pose, dolly, model, pipe,
                                                            black, white, pose, 1.)
                    rec = dict(**inputs.identity, **mask_metrics(native_pixels[2], inputs.mask, inputs.dark),
                               native_face_alpha=face_alpha(native_pixels[2], inputs.native_region),
                               dolly_face_alpha=face_alpha(near_pixels[2], inputs.near_region),
                               minimum_dolly_center_depth_m=minimum_z, native_composition_error=native_error,
                               dolly_composition_error=near_error)
                    if frame == arm.facts['reference_frame_id']:
                        with Image.open(arm.snap/'reference-render.png') as source:
                            saved = np.asarray(source.convert('RGB'), dtype=np.int16)
                        actual = np.rint(np.clip(native_pixels[0], 0, 1)*255).astype(np.int16)
                        difference = int(np.abs(actual-saved).max())
                        require(difference <= 1, f'Native reference render changed: {name}')
                        rec['reference_replay_max_abs_8bit'] = difference
                    rec['dolly_panel'] = write_dolly(folder/(frame+'-dolly.png'), inputs.crop, near_pixels)
                    require(model.optimizer is None, 'Read-only replay created an optimizer.')
                    records[frame] = rec
                rendered[name] = dict(terminal=arm.terminal, manifest_sha256=arm.manifest_sha256,
                                      load_parity=parity, records=records)
                del model
    return rendered


def compare(rendered):
    metrics = {
        'native_face_below_095': lambda row: row['native_face_alpha']['fraction_below_095'],
        'dolly_face_below_095': lambda row: row['dolly_face_alpha']['fraction_below_095'],
        'dark_eye_hole_at_half': lambda row: row['dark_eye_hole_fraction_at_half'],
        'exterior_spill_pixels_at_half': lambda row: row['exterior_spill_pixels_at_half'],
    }
    output = {}
    for frame in FRAMES:
        output[frame] = {}
        for metric, get in metrics.items():
            values = {name: get(rendered[name]['records'][frame]) for name in ARMS}
            stock, repeat, omit = (values[key] for key in ARMS)
            deltas = {} if None in values.values() else dict(stock_repeat_minus_stock=repeat-stock,
                omit_minus_stock=omit-stock, omit_minus_stock_repeat=omit-repeat,
                omit_below_both_controls=omit < min(stock, repeat),
                omit_smallest_improvement_exceeds_stock_range=min(stock, repeat)-omit > abs(repeat-stock))
            output[frame][metric] = dict(values=values, **deltas)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fork', 'project', 'reference-audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    fork, project, audit, output = (path.resolve() for path in (args.fork, args.project, args.reference_audit, args.output))
    require(not output.exists() and not output.is_relative_to(fork) and not output.is_relative_to(project),
            'Choose a new output separate from all fitted and source inputs.')
    rt, asset, frames, arms, evidence = validate_fork(fork, project, audit)
    if args.preflight:
        print(json.dumps(dict(state='fixed_fork_replay_inputs_verified', frames=list(FRAMES), arms=list(arms),
                              gpu_calls=0, optimizer_updates=0, **evidence)))
        return
    result = replay(rt, asset, frames, arms, output)
    rt.save(output/'result.json', dict(state='completed', optimizer_updates=0, quality_accepted=False,
        evaluation_scope='Three matched training views. Dolly uses the exact prior camera, with no observed RGB there.',
        comparison_scope='One stock-repeat control measures suffix variability for this single common checkpoint.',
        records=result, comparisons=compare(result), **evidence))
    print(json.dumps(dict(receipt=str(output/'result.json'), sha256=sha(output/'result.json'))))


if __name__ == '__main__':
    main()
