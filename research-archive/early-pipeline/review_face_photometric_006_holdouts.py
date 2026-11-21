"""Candidate-only protected RGB/spill gate for the registered 006 face trial."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_unified_006_spatial_alpha as spatial
import launch_face_photometric_006 as trial
import review_unified_006_holdouts as unified


read, pin, require, sha = spatial.read, spatial.pin, spatial.require, spatial.sha
FRAMES = unified.PRIMARY
FAR_DISTANCE_PX = 24
ALPHA = .5
STAGED_SHA = '1f20ef14f2e40ddf25f4939446bbd60fa2525f7a587958907b460a43e035f080'


def baseline():
    trial.verify_acceptance()
    registered = read(trial.ACCEPTANCE)
    thresholds = registered['gates']
    limits = dict(rgb=thresholds['protected_full_heldout_RGB_MAE_absolute_regression_max'],
                  spill=thresholds['protected_each_holdout_relative_total_exterior_alpha_half_spill_increase_max'],
                  far_added=thresholds['protected_each_holdout_far_over_24px_new_spill_max_pixels'])
    require(limits == dict(rgb=.003, spill=.05, far_added=50)
            and registered['baseline_notes']['protected_primary'] == list(FRAMES), 'Acceptance gates changed.')
    result_path = Path(registered['fixed_spill_bands']['path'])
    evidence = read(result_path)
    root = result_path.parent
    spatial_plan_path = root/'preregistration.json'
    pin(spatial_plan_path, evidence['preregistration_sha256'])
    spatial_plan = read(spatial_plan_path)
    locked = spatial_plan['arms']['new']
    require(locked['terminal_result_sha256'] == trial.BASE_RESULT and spatial_plan['alpha_threshold'] == ALPHA
            and spatial_plan['distance_band_edges_native_px'][-1] == FAR_DISTANCE_PX,
            'Spatial reference does not bind the exact unified terminal and bands.')
    capture_path = root/'new-replay/capture.json'
    pin(capture_path, evidence['render_receipts_sha256']['new'])
    capture = read(capture_path)
    require(capture['preregistration_sha256'] == evidence['preregistration_sha256']
            and capture['optimizer_updates'] == capture['refits'] == 0, 'Saved native alpha replay changed.')
    alpha_path = capture_path.with_name('native-alpha.npz')
    pin(alpha_path, capture['native_alpha_sha256'])
    # Verify the archived heldout source/pose/mask independently in the baseline checkout.
    # The isolated child keeps its renderer imports out of the candidate process.
    command = [sys.executable, '-B', spatial.__file__, '_inspect', '--side', 'new']
    checked = subprocess.check_output(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True)
    require(json.loads(checked.splitlines()[-1]) == locked, 'Current frozen heldout inputs differ from spatial registration.')
    views = []
    with np.load(alpha_path, allow_pickle=False) as archived:
        require(tuple(archived.files) == FRAMES, 'Archived alpha roster differs.')
        for frame, spec, row, totals in zip(FRAMES, locked['records'], capture['records'], evidence['records'], strict=True):
            require(spec['frame'] == row['frame'] == totals['frame'] == frame
                    and row['pose_screen_passed'] is True
                    and row['posed_camera'] == spec['camera']
                    and row['scoring_mask_sha256'] == spec['score_mask_sha256']
                    and row['native_source_sha256'] == spec['source_sha256'], 'Archived heldout identity differs.')
            pin(Path(spec['score_mask']), spec['score_mask_sha256'])
            pin(Path(spec['image']), spec['source_sha256'])
            with Image.open(spec['score_mask']) as image, Image.open(spec['image']) as rgb:
                mask = np.asarray(image.convert('L')) >= 128
                source = np.asarray(rgb.convert('RGB'), dtype=np.float32)/255.
            alpha = archived[frame]
            require(alpha.dtype == np.float32 and alpha.shape == mask.shape == source.shape[:2]
                    and np.isfinite(alpha).all() and mask.any(), 'Invalid archived native raster.')
            exterior = ~mask
            far = distance_transform_edt(exterior) > FAR_DISTANCE_PX
            visible = alpha >= ALPHA
            spill = int((visible & exterior).sum())
            far_count = int((visible & far).sum())
            require(spill == row['metrics']['alpha_half_spill_pixels'] == totals['exterior']['new']
                    and far_count == totals['exterior_by_distance']['gt_24']['new'],
                    'Archived alpha plane does not reproduce the registered baseline counts.')
            views.append(SimpleNamespace(spec=spec, visible=visible, exterior=exterior,
                far=far, source=source, rgb_mae=row['metrics']['foreground_rgb_mae'], spill=spill, far_count=far_count))
    receipt = dict(acceptance_plan_sha256=trial.ACCEPTANCE_SHA, baseline_terminal_sha256=trial.BASE_RESULT,
                   baseline_native_alpha_sha256=capture['native_alpha_sha256'],
                   baseline_capture_sha256=evidence['render_receipts_sha256']['new'],
                   evaluation_fit_sha256=locked['frozen_evaluation_fit_sha256'], frames=list(FRAMES),
                   alpha_threshold=ALPHA, far_distance_strictly_over_px=FAR_DISTANCE_PX, limits=limits,
                   optimizer_updates=0, refits=0, quality_accepted=False)
    return receipt, views


def candidate(receipt, views):
    trial.verify_source()
    run = trial.ROOT/'run'
    plan_path = trial.ROOT/'preregistration.json'
    terminal_path = trial.ROOT/'trial-terminal.json'
    if not plan_path.is_file():
        return None
    pin(plan_path, STAGED_SHA)
    plan = read(plan_path)
    require(plan['schema'] == 'shuolee-006-face-photometric-frozen-fit-v1'
            and plan['candidate_commit'] == trial.COMMIT and plan['baseline_result_sha256'] == trial.BASE_RESULT
            and plan['acceptance_plan_sha256'] == trial.ACCEPTANCE_SHA and plan['fit_calls'] == 0,
            'Wrong staged face trial.')
    for name in trial.base.COPY:
        pin(run/name, plan['staged_sha256'][name])
        pin(trial.BASE/name, plan['staged_sha256'][name])
    if not terminal_path.is_file():
        return None
    terminal = read(terminal_path)
    require(terminal['state'] == 'delivered_pending_quality_review' and terminal['candidate_commit'] == trial.COMMIT
            and terminal['preregistration_sha256'] == sha(plan_path)
            and terminal['acceptance_plan_sha256'] == trial.ACCEPTANCE_SHA
            and terminal['fit_reused'] is True and terminal['fit_calls'] == 0
            and terminal['verified_camera_sha256'] == trial.CAMERA_PINS, 'Wrong completed candidate terminal.')
    result_path = run/'result.json'
    pin(result_path, terminal['result_sha256'])
    result = read(result_path)
    require(result['state'] == 'completed' and result['format_passed'] is True and result['steps'] == 10000
            and result['frames'] == 24 and result['sh_degree'] == 0
            and result['reused_fit_preregistration_sha256'] == sha(plan_path)
            and result['points'] == terminal['points'] and result['reference_frame_id'] == trial.REFERENCE
            and result['gaussian_from_scratch'] is True and result['fit_calls'] == 0,
            'Candidate did not complete the registered fixed-fit profile.')
    for name, digest in trial.CAMERA_PINS.items():
        pin(run/name, digest)
    replay_path = run/result['bound_replay']
    pin(replay_path, result['bound_replay_sha256'])
    replay = read(replay_path)
    snapshot = replay_path.parents[1]
    bound_path = snapshot/'bound-replay.json'
    pin(bound_path, replay['bound_replay_sha256'])
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs
    from shuolee_portrait.training import camera, verify_official_fit

    require(rt.PROJECT == trial.PROJECT and verify_official_fit(run)['fit_sha256'] == plan['staged_sha256']['fit/fitted.npz'],
            'Candidate runtime does not bind the original official fit.')
    asset = trial.common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit, mesh, facts = _inputs(run, snapshot, asset, read(bound_path))
    pose_path = spatial.cold.prior.POSE_ARCHIVE/'evaluation-fitted.npz'
    pin(pose_path, receipt['evaluation_fit_sha256'])
    with np.load(pose_path, allow_pickle=False) as frozen:
        poses = dict(frozen)
    require(poses['frame_ids'].tolist() == list(spatial.cold.prior.HELD)
            and np.array_equal(poses['shared_betas'], fit['shared_betas']) and np.array_equal(poses['faces'], fit['faces']),
            'Candidate and frozen heldout shape or topology differ.')
    for view in views:
        saved = view.spec['camera']
        view.row = dict(frame=view.spec['frame'], image=view.spec['image'], wh=saved['wh'], K=saved['K'], w2c=saved['w2c'])
        view.pose_index = saved['pose_index']
        # The original evaluator assigned camera ID by the complete four-frame archive order.
        view.camera_index = list(spatial.cold.prior.HELD).index(view.spec['frame'])
        slot = view.camera_index
        require(camera(view.row, view.camera_index, pose_index=view.pose_index)[1] == saved
                and view.pose_index == len(facts['frames'])+slot
                and poses['source_frames'][slot] == view.spec['source_frame_index']
                and poses['image_sha256'][slot] == view.spec['source_sha256']
                and np.array_equal(poses['w2c'][slot], saved['w2c']),
                'Candidate changed a protected native pose/camera.')
    receipt.update(candidate_terminal_sha256=terminal['result_sha256'], candidate_bound_sha256=sha(bound_path))
    return SimpleNamespace(rt=rt, run=run, snapshot=snapshot, asset=asset, poses=poses, views=views, mesh=mesh)


def compare(view, rgb, alpha, limits):
    require(alpha.shape == view.visible.shape and rgb.shape == view.source.shape
            and np.isfinite(rgb).all() and np.isfinite(alpha).all(), 'Invalid candidate native raster.')
    visible = alpha >= ALPHA
    mae = float(np.abs(rgb[~view.exterior]-view.source[~view.exterior]).mean())
    spill = int((visible & view.exterior).sum())
    far = int((visible & view.far).sum())
    added = int((visible & ~view.visible & view.far).sum())
    removed = int((~visible & view.visible & view.far).sum())
    require(far-view.far_count == added-removed, 'Far spill pixel accounting differs.')
    gates = dict(rgb=mae-view.rgb_mae <= limits['rgb'],
                 exterior=spill <= view.spill*(1+limits['spill']), far_added=added <= limits['far_added'])
    return dict(frame=view.spec['frame'], source_frame_index=view.spec['source_frame_index'], camera=view.spec['camera'],
                source_sha256=view.spec['source_sha256'], scoring_mask_sha256=view.spec['score_mask_sha256'],
                foreground_rgb_mae=dict(baseline=view.rgb_mae, candidate=mae, delta=mae-view.rgb_mae),
                exterior_pixels=dict(baseline=view.spill, candidate=spill, delta=spill-view.spill),
                far_pixels=dict(baseline=view.far_count, candidate=far, added=added, removed=removed, net=far-view.far_count),
                gates=gates, passed=all(gates.values()))


def render(state, limits):
    import torch
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import _frozen_smpl_surface, camera, render_pose

    with trial.common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            state.rt.configure(state.asset.parents[1])
            from gaussian_renderer import render as raster

            model, manifest = load_bound_snapshot(state.run, state.snapshot, asset_path=state.asset)
            base, linear, clothing, _, _ = _frozen_smpl_surface(state.poses, state.asset)
            delta = state.mesh['static_offset']
            require(not np.any(delta[~clothing]), 'Learned offset crossed protected skin.')
            offset = torch.from_numpy(delta).cuda()[..., None]
            posed = torch.stack([v+(a@offset).squeeze(-1) for v, a in zip(base.cuda(), linear.cuda(), strict=True)])
            model.vertices_sequence = torch.cat((model.vertices_sequence, posed))
            model.num_timesteps = len(model.vertices_sequence)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            records, arrays = [], {}
            for view in state.views:
                cam, actual = camera(view.row, view.camera_index, pose_index=view.pose_index)
                require(actual == view.spec['camera'], 'Candidate render camera changed.')
                pixels, error = spatial.cold.scale_pixels(raster, render_pose, cam, model, pipe,
                                                          black, white, view.pose_index, 1.)
                rgb, _, alpha = pixels
                record = compare(view, rgb, alpha, limits)
                records.append(dict(record, black_white_composition_error=error))
                arrays[view.spec['frame']] = alpha.copy()
            require(model.optimizer is None and len(model.binding) == manifest['points'], 'Replay changed bound model.')
    return records, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    parser.add_argument('--output', type=Path, default=trial.ROOT/'protected-holdouts')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    receipt, views = baseline()
    sys.path.insert(0, str(trial.PROJECT/'src'))
    state = candidate(receipt, views)
    if args.action == 'check':
        print(json.dumps(dict(receipt, state='protected_inputs_verified', terminal_ready=state is not None,
            baseline=[dict(frame=view.spec['frame'], rgb_mae=view.rgb_mae, exterior=view.spill, far=view.far_count)
                      for view in views], gpu_calls=0)))
        return
    require(state is not None, 'Wait for the registered 006 candidate terminal.')
    output = args.output.resolve()
    require(output.parent == trial.ROOT and not output.exists(), 'Use a fresh separate trial output directory.')
    records, arrays = render(state, receipt['limits'])
    output.mkdir(exist_ok=False)
    alpha_path = output/'candidate-native-alpha.npz'
    np.savez_compressed(alpha_path, **arrays)
    target = output/'result.json'
    spatial.save(target, dict(receipt, state='protected_gate_complete', records=records,
        protected_gate_passed=all(row['passed'] for row in records), candidate_alpha_sha256=sha(alpha_path),
        baseline_render_calls=0, candidate_rendered_frames=len(records), reviewer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(receipt=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
