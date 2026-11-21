"""CPU-only metadata completion of the saved127 fit, never an optimizer retry."""
import hashlib
import json
import os
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import run_officewoman_circle24_fit127 as fit
    import torch
    assert not torch.cuda.is_initialized()
    folder = fit.OUT
    original_names = ['selected-fit.npz', 'official-result.pkl', 'failure.json', 'process-exit.json',
        'fit.log', 'executed-runner.py.txt', 'plan.json', 'launch.json', 'parent-launch.json', 'progress.json']
    preserved = {name: sha(folder / name) for name in original_names}
    helper_hash = sha(Path(fit.__file__))
    assert helper_hash == '0dd3be080d32705bc4f6e8315eb0cacaeea66a4599e478676409abf7156bcac3'
    assert not (folder / 'result.json').exists() and not (folder / 'postprocess-recovery127.json').exists()
    plan = read(folder / 'plan.json')
    for path, digest in plan['source_hashes'].items():
        assert sha(path) == digest, path
    source = fit.source_plan()
    assert sha(fit.AUDIT / 'plan.json') == '96fae64878e672c42e6b8145f7850e319994bb2ed93327d6db6eb406871c6ded'
    failure = read(folder / 'failure.json')
    assert failure['error'] == 'ValueError'
    assert failure['message'] == 'zero-size array to reduction operation maximum which has no identity'
    assert 'line 41, in run' in failure['traceback']
    original_exit = read(folder / 'process-exit.json')
    assert original_exit['exit_code'] == 1 and read(folder / 'progress.json')['state'] == 'failed'
    runner = (folder / 'executed-runner.py.txt').read_text()
    assert "error[ix][w>0].max()" in runner.splitlines()[40]
    assert 'beta=np.zeros((1,10),dtype=np.float32)' in runner and 'freeze_betas=False' in runner
    log = (folder / 'fit.log').read_text(errors='replace')
    stages = re.findall(r'Stage (\d{3}) done after ([0-9.]+) seconds', log)
    assert [name for name, _ in stages] == ['000', '001', '002', '003', '004']
    body_times = re.findall(r'Body fitting Orientation 0 done after ([0-9.]+) seconds', log)
    losses = re.findall(r'Body final loss val = ([0-9.]+)', log)
    assert len(body_times) == len(losses) == 1
    with np.load(folder / 'selected-fit.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    rows, target = fit.data()
    frames = plan['frames']
    assert len(frames) == 24 and frames == source['training_frames'] == arrays['frame_ids'].astype(str).tolist()
    assert [r['frame'] for r in rows] == frames and np.array_equal(arrays['observed_native_keypoints'], target)
    for key, shape in dict(betas=(1, 10), pose_embedding=(1, 32), body_pose_axis_angle=(1, 63),
        vertices_camera=(24, 10475, 3), joints_camera=(24, 118, 3), joints_projected_native=(24, 118, 2),
        camera_translation=(24, 3), K_integer_centers=(3, 3)).items():
        assert arrays[key].shape == shape, key
    assert all(np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number))
    assert all(np.array_equal(row['w2c'], np.eye(4)) for row in rows)
    assert np.array_equal(arrays['K_integer_centers'], rows[0]['K_integer_centers'])
    assert arrays['image_sha256'].astype(str).tolist() == [r['source_sha256'] for r in rows]
    assert arrays['mask_sha256'].astype(str).tolist() == [r['mask_sha256'] for r in rows]
    assert arrays['source_frames'].tolist() == [r['source_frame_index'] for r in rows]
    with (folder / 'official-result.pkl').open('rb') as handle:
        winner = pickle.load(handle)  # Trusted self-generated local original fit, hash-bound above.
    parity = {}
    for key, value in winner.items():
        destination = 'pose_embedding' if key == 'body_pose' else key
        assert destination in arrays and np.array_equal(value, arrays[destination]), key
        parity[key] = True
    uv = arrays['joints_camera'] @ arrays['K_integer_centers'].T
    assert np.all(uv[:, :, 2] > 0)
    projection = float(np.max(np.abs(uv[:, :, :2] / uv[:, :, 2:] - arrays['joints_projected_native'])))
    assert projection < .001  # Same original executed export invariant, not a physical accuracy claim.
    shared = {'betas', 'faces', 'K_integer_centers', 'K_compute_float32', 'body_pose_axis_angle', 'pose_embedding'}
    prepared, metrics = [], []
    for i, row in enumerate(rows):
        path = folder / 'frames' / row['frame'] / 'winner-camera-space.npz'
        individual = {key: (value if key in shared else value[i:i+1]) for key, value in arrays.items() if key != 'frame_ids'}
        individual['frame_id'] = np.asarray(row['frame'])
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                assert set(old.files) == set(individual) and all(np.array_equal(old[k], individual[k]) and old[k].dtype == individual[k].dtype for k in old.files), path
        prepared.append((row['frame'], path, individual, path.exists()))
        error = np.linalg.norm(arrays['joints_projected_native'][i] - target[i, :, :2], axis=-1)
        metric = dict(frame=row['frame'], all118_error_px=error.tolist(), arm_error_px=error[[2,3,4,5,6,7]].tolist(),
            raw_unobserved_coordinates_are_not_ground_truth=True, source_face_is_observed=source['face_visibility'][row['frame']],
            source_joint_group_supported=source['joint_group_visibility'][row['frame']])
        for name, indices in [('body', fit.original.entry.COCO), ('face', np.arange(67,118)), ('feet', np.arange(19,25))]:
            weight = target[i, indices, 2] ** 2
            support = weight > 0
            count = int(support.sum())
            metric[name + '_supported_count'] = count
            metric[name + '_weight_sum'] = float(weight.sum())
            metric[name + '_weighted_rms_px'] = float(np.sqrt(np.sum(weight * error[indices] ** 2) / weight.sum())) if count else None
            metric[name + '_supported_max_px'] = float(error[indices][support].max()) if count else None
        metrics.append(metric)
    # Only after every existing package passes equality, complete missing exports.
    per_frame_files = []
    for frame, path, values, existed in prepared:
        if not existed:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
        with np.load(path, allow_pickle=False) as replay:
            assert set(replay.files) == set(values) and all(np.array_equal(replay[k], values[k]) and replay[k].dtype == values[k].dtype for k in replay.files)
        per_frame_files.append(dict(frame=frame, sha256=sha(path)))
    baseline = fit.original.entry.load_official_config(fit.original.OFFICIAL)
    candidate = fit.hand_control.candidate_config(baseline)
    assert candidate['hand_joints_weights'] == [0.] * 5 and candidate['use_joints_conf'] is True
    seconds = (datetime.fromisoformat(failure['updated']) - datetime.fromisoformat(read(folder / 'launch.json')['updated'])).total_seconds()
    result = dict(updated=datetime.now(timezone.utc).isoformat(), state='fitted_selected_views_review_pending',
        metadata_reconstructed=True, original_in_memory_receipt_recovered=False,
        reconstructed_metadata_scope='Saved arrays and verified frozen source/log evidence only; not the lost original in-memory receipt.',
        official_commit=fit.original.entry.COMMIT, source_hashes=fit.original.entry.PINNED,
        upstream_loss_monitor_optimizer_unchanged=True, five_stages=True, orientation_count=1,
        winning_loss=float(losses[0]), winning_loss_precision='Rounded original log value; full precision receipt was not recovered.',
        seconds=seconds, seconds_scope='UTC launch.json to failure.json; includes initialization/export and is not the lost monotonic timer.',
        body_orientation_seconds_from_log=float(body_times[0]), stage_seconds={k:float(v) for k,v in stages},
        frames=frames, frames_count=24, reference_frame=plan['reference_frame'],
        use_shared_betas=True, freeze_betas=False, shared_beta_parameter_count=10,
        beta_before=np.zeros((1,10)).tolist(), beta_after=arrays['betas'].tolist(),
        beta_delta_l2=float(np.linalg.norm(arrays['betas'])), shared_shape_updated=bool(np.any(arrays['betas'] != 0)),
        old_individual_shape_used=False, old_optimized_pose_used=False, original_zero_body_latent=True,
        single_shared_local_body_pose=True, per_view_root_and_placement=True, cross_frame_identity_policy_implemented=True,
        coordinate_system='opencv_camera_model_meters', translation_applied_once=True, full_K_integer_centers=True,
        source_camera_identity_placeholder=True, external_world_camera_used=False, independent_camera_calibration=False,
        physical_camera_calibration_accepted=False, gvhmr_used=False, interpenetration=False,
        projection_replay_max_abs_px=projection, metrics=metrics, fit_sha256=preserved['selected-fit.npz'],
        hand_data_control=dict(baseline=baseline['hand_joints_weights'], candidate=candidate['hand_joints_weights'], same_for_all_frames=True,
            body_wrists_and_hand_pose_prior_retained=True, reconstructed_from_frozen_configuration=True),
        heldout_excluded=source['heldout_frames'], historical_validation_reused=False,
        source_observation_visibility=plan['source_observation_visibility'],
        source_observation_visibility_policy_sha256=fit.visibility_sha256(source),
        empty_support_metrics='null with supported_count0; no physical face or foot claim for unobserved data.',
        optimizer_updates_added_by_recovery=0, original_process_exit_code=1, quality_accepted=False, gaussian_admitted=False)
    for name, digest in preserved.items(): assert sha(folder / name) == digest, name
    for path, digest in plan['source_hashes'].items(): assert sha(path) == digest, path
    write_new(folder / 'result.json', result)
    receipt = dict(updated=datetime.now(timezone.utc).isoformat(), cpu_exit_code=0, original_process_exit_code=1,
        metadata_only_recovery=True, original_failure_sha256=preserved['failure.json'],
        original_process_exit_sha256=preserved['process-exit.json'], fit_log_sha256=preserved['fit.log'],
        executed_runner_sha256=preserved['executed-runner.py.txt'], frozen_fit_helper_sha256=helper_hash,
        selected_fit_sha256_before=preserved['selected-fit.npz'], selected_fit_sha256_after=sha(folder / 'selected-fit.npz'),
        result_sha256=sha(folder / 'result.json'), source_hashes_verified=True, all5_stage_log_records_verified=True,
        no_optimizer_updates=True, no_GPU=True, all24_per_frame_arrays_verified=True, per_frame_files=per_frame_files,
        source_plan_sha256=sha(fit.AUDIT / 'plan.json'), plan_sha256=preserved['plan.json'],
        original_preserved_files=preserved, official_pickle_parameter_parity=parity,
        projection_replay_max_abs_px=projection, stage_seconds=result['stage_seconds'],
        body_orientation_seconds_from_log=float(body_times[0]), cpu_recovery_seconds=time.monotonic()-started,
        existing_per_frame_files_preserved=sum(existed for _,_,_,existed in prepared),
        missing_per_frame_files_completed=sum(not existed for _,_,_,existed in prepared),
        null_face_metric_frames=[r['frame'] for r in metrics if r['face_supported_count']==0],
        null_feet_metric_frames=[r['frame'] for r in metrics if r['feet_supported_count']==0],
        finalizer_sha256=sha(__file__), quality_accepted=False, gaussian_admitted=False)
    assert not torch.cuda.is_initialized()
    write_new(folder / 'postprocess-recovery127.json', receipt)
    print('OFFICEWOMAN_FIT127_CPU_METADATA_RECOVERED', sha(folder / 'postprocess-recovery127.json'), flush=True)


if __name__ == '__main__':
    main()
