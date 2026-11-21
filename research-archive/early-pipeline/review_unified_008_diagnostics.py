"""Replay only the three archived 008 pose-screen failures on the unified model."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_006_cold_candidate_holdouts as prior
import review_capture_formal_holdouts as formal
import review_unified_queue_views as queue


views = queue.views
read, pin, require, sha = views.read, views.pin, views.require, views.sha
ARCHIVE = views.LAB/'shuolee-smpl-visibility-008-formal'
HELD = dict(frame_000015=112, frame_000043=336, frame_000071=561, frame_000098=777)
EVALUATED = ('frame_000015', 'frame_000043', 'frame_000098')
UNAVAILABLE = 'frame_000071'
PINS = {
    'evaluation-fitted.npz': 'ac20914babf47f05d8e8e20bbf0c94f2e8891a5b8b35468974c096366047b642',
    'fit-result.json': 'b7d48c7dc64b3be5b3679fb44b1ce107f1fbba9f438baeb1d7da3b9cdcd276f4',
    'evaluation-scope.json': 'c081d2a717ddc27e3326e7f72500811bad6b0d16f67e4bca211217cbb3382f44',
    'inputs.json': '1091f4a1a207f9ddd2e87f40a2326d9e54071e88528126a3eb3e27b9f6d63846',
    'renders/result.json': '8ae2cddf5b9dee8f9f2a58b826ccc669a70bd9302ec819704dddb4d56c6f519b',
}


def preflight():
    plan = queue.preflight('008')
    old = Path(plan['old']['run'])
    run = queue.common.root_for('008')/'run'
    pins = {ARCHIVE/name: pin(ARCHIVE/name, digest) for name, digest in PINS.items()}
    _, helper, scoring, baseline, unavailable, historical = formal.benchmark('008', HELD)
    pins.update({Path(path): digest for path, digest in historical.items()})
    scope, fit_receipt, pose_inputs, old_render = (
        read(ARCHIVE/name) for name in ('evaluation-scope.json', 'fit-result.json', 'inputs.json', 'renders/result.json'))
    require(scope['protected_frames'] == fit_receipt['protected_frames'] == old_render['protected_frames'] == HELD
            and scope['evaluated_frames'] == list(EVALUATED)
            and scope['unscored_outside_capture_arc'] == [UNAVAILABLE] and set(unavailable) == {UNAVAILABLE}
            and not (ARCHIVE/'frames'/UNAVAILABLE).exists()
            and fit_receipt['evaluation_only'] is True and old_render['model_changed'] is False
            and old_render['gaussian_optimizer_updates'] == 0
            and scope['fitted_sha256'] == fit_receipt['fitted_sha256'] == PINS['evaluation-fitted.npz']
            and fit_receipt['input_receipt_sha256'] == PINS['inputs.json']
            and old_render['fit_result_sha256'] == PINS['fit-result.json'],
            'Archived evaluation role or identities changed.')
    aggregate = read(run/'fit/official-fit-result.json')
    require(aggregate['selection']['retained_holdouts'] == list(HELD)
            and {r['shared_body_target_sha256'] for r in aggregate['frame_results']}
                == {fit_receipt['shared_target_sha256']}
            and aggregate['source']['confidence']['per_joint_q90'] == pose_inputs['frozen_q90'],
            'Frozen body target, confidence or holdouts differ.')
    native = old/'data/inputs.json'
    pins[native] = pin(native, aggregate['source']['input_files'][str(native)])
    rows = {row['frame']: row for row in read(native)['records']}
    frozen = ARCHIVE/'evaluation-fitted.npz'
    with np.load(frozen, allow_pickle=False) as data, np.load(run/'fit/fitted.npz', allow_pickle=False) as training:
        poses = dict(data)
        require(poses['frame_ids'].tolist() == list(EVALUATED)
                and poses['source_frames'].tolist() == [HELD[f] for f in EVALUATED]
                and np.array_equal(poses['shared_betas'], training['shared_betas'])
                and np.array_equal(poses['faces'], training['faces'])
                and set(HELD).isdisjoint(training['frame_ids'].tolist()), 'Frozen pose topology or shared beta differs.')

    sys.path.insert(0, str(queue.common.PROJECT/'src'))
    from shuolee_portrait import official_sequence as sequence, runtime as rt, stable_selection as stable
    from shuolee_portrait.training import camera

    checks = []
    for i, (frame, diagnostic, archived) in enumerate(zip(EVALUATED, fit_receipt['diagnostics'],
                                                           old_render['records'], strict=True)):
        row, historical_row = rows[frame], scoring[frame]
        require(frame == diagnostic['frame'] == archived['frame'] and row['source_frame_index'] == HELD[frame]
                and all(row[key] == historical_row[key] for key in formal.SOURCE_FIELDS)
                and poses['image_sha256'][i] == row['source_sha256'] == archived['native_source_sha256']
                and poses['mask_sha256'][i] == row['mask_sha256'] == archived['fit_mask_sha256']
                and np.array_equal(poses['K_integer_centers'][i], row['K_integer_centers'])
                and np.array_equal(poses['w2c'][i], row['w2c']),
                f'Frozen native frame, pose or camera changed: {frame}')
        winner = ARCHIVE/'frames'/frame/'winner-camera-space.npz'
        report = ARCHIVE/'frames'/frame/'result.json'
        pins[winner] = pin(winner, diagnostic['winner_sha256'])
        pins[report] = pin(report, diagnostic['report_sha256'])
        frame_report = read(report)
        sequence.check_result(frame_report, row)
        observed = poses['observed_native_keypoints'][i]
        with np.load(winner, allow_pickle=False) as original:
            _, flags, _, _ = stable._candidate(row, dict(original), observed,
                poses['K_integer_centers'][i], poses['shared_betas'], None, frame_report, True)
            require(np.array_equal(original['joints_projected_native'][0], poses['joints_projected_native'][i]),
                    f'Archived fitted projection changed: {frame}')
        hidden = scope['excluded_detector_slots'][frame]
        require(flags and flags == diagnostic['pose_screen_flags'] == archived['pose_screen_flags']
                and diagnostic['pose_screen_passed'] is archived['pose_screen_passed'] is False
                and not observed[hidden, 2].any(), f'Pose-screen or hidden observations changed: {frame}')
        image = Path(row['image'])
        mask = helper['relocate'](historical_row['mask'])
        pins[image] = pin(image, row['source_sha256'])
        pins[Path(row['mask'])] = pin(Path(row['mask']), row['mask_sha256'])
        pins[mask] = pin(mask, archived['scoring_mask_sha256'])
        with Image.open(image) as source, Image.open(mask) as silhouette:
            require(list(source.size) == list(silhouette.size) == row['wh'], f'Native raster differs: {frame}')
            crop = tuple(baseline[frame]['crop_xyxy'])
            rgb, scoring_mask = source.convert('RGB').crop(crop), silhouette.convert('L').crop(crop)
        picture = prior.archived_posed(ARCHIVE/'renders', archived, rgb, crop)
        path = ARCHIVE/'renders'/archived['image']
        pins[path] = archived['image_sha256']
        pose_index = len(plan['exact_native_common_training_frames'])+i
        expected = archived['posed_camera']
        require(camera(row, i, pose_index=pose_index)[1] == expected, f'Original native projection changed: {frame}')
        checks.append(SimpleNamespace(row=row, pose_index=pose_index, rgb=rgb, scoring_mask=scoring_mask,
                                      scoring_mask_sha256=archived['scoring_mask_sha256'], old=picture,
                                      crop=crop, camera=expected, screen_flags=flags))

    receipt = dict(state='frozen_diagnostic_inputs_exact', subject='008', protected_frames=HELD,
                   diagnostic_frames=list(EVALUATED), formal_frames=[], formal_score_computed=False,
                   unscored={UNAVAILABLE: 'Outside registered capture arcs, no frozen evaluation pose or scoring mask.'},
                   evaluation_fit_sha256=PINS['evaluation-fitted.npz'],
                   preregistration_sha256=plan['reused_fit_preregistration_sha256'],
                   shared_body_target_sha256=fit_receipt['shared_target_sha256'],
                   reconstruction_state=plan.get('reconstruction_state', 'completed'), optimizer_updates=0,
                   evaluation_refits=0, quality_accepted=False)
    if plan['state'] != 'ready':
        return dict(receipt, terminal_ready=False), None
    terminal_path = run/'result.json'
    pins[terminal_path] = pin(terminal_path, plan['new']['terminal_sha256'])
    terminal = read(terminal_path)
    bound_path = run/'gaussian-run/step-010000/bound-replay.json'
    replay_path = run/terminal['bound_replay']
    pins[replay_path] = pin(replay_path, terminal['bound_replay_sha256'])
    replay = read(replay_path)
    pins[bound_path] = pin(bound_path, replay['bound_replay_sha256'])
    manifest = read(bound_path)
    require(len(manifest['frames']) == terminal['frames'] == 19
            and manifest['points'] == terminal['points'] and replay['retained_holdouts'] == list(HELD),
            'Terminal has a different 19-frame scope.')
    return dict(receipt, terminal_ready=True, terminal_result_sha256=pins[terminal_path],
                terminal_bound_sha256=pins[bound_path]), SimpleNamespace(
                    rt=rt, run=run, snapshot=bound_path.parent, poses=poses, checks=checks, pins=pins)


def render(state, output):
    import torch
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import _frozen_smpl_surface, camera, render_pose

    with views.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            state.rt.configure(views.ASSET.parents[1])
            from gaussian_renderer import render as raster

            model, bound = load_bound_snapshot(state.run, state.snapshot, asset_path=views.ASSET)
            base, linear, clothing, _, _ = _frozen_smpl_surface(state.poses, views.ASSET)
            with np.load(state.snapshot/'reference-mesh.npz', allow_pickle=False) as mesh:
                delta = mesh['static_offset']
            require(not np.any(delta[~clothing]), 'Learned offset crossed protected skin.')
            offset = torch.from_numpy(delta).cuda()[..., None]
            poses = torch.stack([v+(a@offset).squeeze(-1) for v, a in zip(base.cuda(), linear.cuda(), strict=True)])
            model.vertices_sequence = torch.cat((model.vertices_sequence, poses))
            model.num_timesteps = len(model.vertices_sequence)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            output.mkdir(exist_ok=False)
            result = []
            for i, view in enumerate(state.checks):
                cam, actual = camera(view.row, i, pose_index=view.pose_index)
                require(actual == view.camera, 'Rendered pose camera changed.')
                pixels, composition = views.scale_pixels(raster, render_pose, cam, model, pipe,
                                                          black, white, view.pose_index, 1.)
                x0, y0, x1, y1 = view.crop
                color, _, alpha = pixels
                def crop(array):
                    return Image.fromarray(np.rint(np.clip(array[y0:y1,x0:x1],0,1)*255).astype(np.uint8)).convert('RGB')
                pictures = (view.rgb, view.scoring_mask.convert('RGB'), view.old, crop(color), crop(alpha))
                labels = ('Original RGB', 'Frozen mask', 'Source terminal / same pose',
                          'Unified / failed pose diagnostic', 'Unified alpha')
                w, h = view.rgb.size
                panel = Image.new('RGB', (5*w,h+23), '#171717')
                draw = ImageDraw.Draw(panel)
                for j, (label, picture) in enumerate(zip(labels, pictures, strict=True)):
                    draw.text((j*w+4,5), label, fill='white')
                    panel.paste(picture, (j*w,23))
                path = output/(view.row['frame']+'-pose-screen-diagnostic.png')
                panel.save(path)
                result.append(dict(frame=view.row['frame'], source_frame_index=view.row['source_frame_index'],
                    native_source_sha256=view.row['source_sha256'], camera=actual, crop_xyxy=view.crop,
                    fit_mask_sha256=view.row['mask_sha256'], scoring_mask_sha256=view.scoring_mask_sha256,
                    pose_screen_passed=False, pose_screen_flags=view.screen_flags, formal_score_computed=False,
                    composition_max_abs=composition, image=path.name, image_sha256=sha(path)))
            require(model.optimizer is None, 'Diagnostic replay created an optimizer.')
    formal.check_pins(state.pins)
    return result, bound['points']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    parser.add_argument('--output', type=Path, default=views.LAB/'shuolee-smpl-unified-008-protected-diagnostics')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    receipt, state = preflight()
    if args.action == 'check':
        print(json.dumps(dict(receipt, gpu_calls=0)))
        return
    require(state is not None, 'The unified 008 terminal is not ready.')
    output = args.output.resolve()
    require(not output.exists() and output.parent == views.LAB, 'Use a fresh separate lab output.')
    records, bound_points = render(state, output)
    path = output/'result.json'
    views.save(path, dict(receipt, state='unified_008_pose_screen_diagnostics_complete', records=records,
                          bound_point_count=bound_points, reviewer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(receipt=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
