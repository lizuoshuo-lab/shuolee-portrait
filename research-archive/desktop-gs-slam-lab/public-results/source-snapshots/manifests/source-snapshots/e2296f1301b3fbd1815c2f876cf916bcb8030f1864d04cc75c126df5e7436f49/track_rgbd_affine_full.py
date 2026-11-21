"""Prepare or explicitly execute complete-sequence frozen 6D/8D CPU odometry.

New orchestration only: the reviewed 313d last-accepted transition, bf219 affine
transition, 8016 6D solver, and 15c7 8D solver remain unchanged. Both model paths
use all associated frames, identity first camera/pair start, and their original
fixed parameters. GT fields are read by association; matrices enter evaluation
only after the complete estimated trajectory has been written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import track_rgbd_affine_photogeometric as frozen

ROOT = Path(__file__).resolve().parents[1]
original, recovery, affine = frozen.original, frozen.recovery, frozen.affine
require, sha, read, write, update = original.require, original.sha, original.read, original.write, original.update
DEADLINE = dt.datetime.fromisoformat('2026-10-09T08:00:00+08:00')
SHORT_PROTOCOL = ROOT/'outputs/rgbd-affine-photogeometric-real-cpu-preparation-001/protocol.json'
SHORT_PROTOCOL_SHA = '9d8e674827a83f2c29e7dc901f2bcb8e1203b5513d89692b5aa92b34db0d8a63'
PINNED_PIPELINE = 'bf219e04cdaa7f23806d03d22794f804d2329f3e1de98a3049edc17923e0a57e'
PINNED_DRIVER = 'd724672c6eae401f8db9feb5270143d04c151b460c3ad650ba001db6c840898f'
SCENES = {
    'desk': {'frames': 549, 'baseline': 'tum-icp-full549-001',
             'metrics_sha256': 'b440b3d2d117c81c69d217b9bc21e00ae216277488a74ddee8ff85c2f5328224',
             'validation_sha256': '59dc20518f1b9c828560fd7be7ad194a6ce127e10eb2513f1360a8b56d7b1da0'},
    'xyz': {'frames': 748, 'baseline': 'tum-xyz-icp-full748-001',
            'metrics_sha256': '5ec1fab9ecfd07d9c417033674326ab141bffceaeeb3a9c9a9212ce772eb4036',
            'validation_sha256': 'de2122f0083e19f9163172f4b07fb714fb088802eb5cfa5a50d68716273c6c00'}}
ROUNDS = [{'order': i+1, 'round_id': f'{scene}-lastaccepted-{dimension}d-full',
           'scene': scene, 'model_dimension': dimension,
           'run': f'tum-{scene}-lastaccepted-{dimension}d-full{SCENES[scene]["frames"]}-001'}
          for i, (scene, dimension) in enumerate((('desk', 6), ('desk', 8), ('xyz', 6), ('xyz', 8)))]


def source_checks():
    require(sha(SHORT_PROTOCOL) == SHORT_PROTOCOL_SHA, 'Frozen short affine protocol')
    previous = frozen.source_checks()
    require(previous == read(SHORT_PROTOCOL)['source_sha256'], 'Exact prior 15-source stack')
    require(previous['track_rgbd_affine_photogeometric.py'] == PINNED_PIPELINE and
            previous['run_rgbd_affine_photogeometric_validation.py'] == PINNED_DRIVER, 'Pinned bf219/d724 entrypoints')
    return {**previous, **{name: sha(ROOT/'scripts'/name) for name in (
        'track_rgbd_affine_full.py', 'run_rgbd_affine_full.py', 'check_rgbd_affine_full_preflight.py')}}


def before_deadline(now=None):
    stamp = now or dt.datetime.now(dt.timezone.utc)
    require(stamp.tzinfo is not None and stamp < DEADLINE, 'No new round at/after 2026-10-09 08:00 Asia/Shanghai')
    return stamp.isoformat()


def configuration_for(scene, dimension):
    require(scene in SCENES and dimension in (6, 8), 'Fixed full-sequence scene/model')
    base = copy.deepcopy(frozen.configuration_for(scene) if dimension == 8 else recovery.configuration_for(scene))
    base.update(frames=SCENES[scene]['frames'], start_frame=0, frame_ids=list(range(SCENES[scene]['frames'])))
    return base


def dataset_preflight(scene):
    from tum_rgbd import associate_frames, load_frames, scaled_intrinsics, safe_data_path
    spec = SCENES[scene]; dataset = ROOT/f'data/tum/rgbd_dataset_freiburg1_{scene}'
    frames, association = associate_frames(dataset, .02, .02, 32.)
    require(len(frames) == spec['frames'] and [f.frame_id for f in frames] == list(range(spec['frames'])), 'All associated frames without clipping')
    baseline_dir = ROOT/'outputs'/spec['baseline']
    require(sha(baseline_dir/'metrics.json') == spec['metrics_sha256'] and
            sha(baseline_dir/'validation.json') == spec['validation_sha256'], 'Frozen full ICP evidence')
    baseline, valid = read(baseline_dir/'metrics.json'), read(baseline_dir/'validation.json')
    require(baseline['frames_processed'] == spec['frames'] and valid['status'] == 'passed', 'Full ICP complete/validated')
    require(baseline['association'] == association, 'Same full association as ICP')
    require(baseline['configuration']['icp'] == json.loads(json.dumps(asdict(affine.frozen.icp.ICPConfig()))), 'Same fixed geometry base config')
    require(baseline['configuration']['source_sha256'] == {n: original.SOURCE_SHA[n] for n in
            ('track_rgbd_icp.py', 'tum_rgbd.py', 'audit_tum_trajectory.py')}, 'Frozen ICP core sources')
    metadata = [original.metadata(frame) for frame in frames]
    old_rows = read(baseline_dir/'selected_sensor_metadata.json')['selected_frames']
    require(len(old_rows) == len(metadata) and metadata == [{k: row[k] for k in metadata[0]} for row in old_rows], 'All sensor/GT index metadata agrees with full baseline')
    short = read(SHORT_PROTOCOL)['datasets'][scene]
    require(metadata[200:300] == short['selected_sensor_metadata'], 'Prior short slice metadata exact')
    pyramid, hashes, slice_hashes = [], {}, {}
    for level, (width, height) in enumerate(affine.frozen.icp.LEVELS):
        rgb, depth, mask = load_frames(dataset, frames, width, height, .2, 5.)
        gray = np.stack([affine.frozen.grayscale(image) for image in rgb])
        K = scaled_intrinsics(width, height)
        d = {'rgb': rgb, 'depth': depth, 'valid': mask, 'gray': gray, 'K': K}
        hashes[str(level)] = {k: original.array_record(v) for k, v in d.items()}
        slice_hashes[str(level)] = {k: original.array_record(v if k == 'K' else v[200:300]) for k, v in d.items()}
        require(slice_hashes[str(level)] == short['input_array_hashes'][str(level)], 'Full pyramid short slice is byte-identical to old input')
        pyramid.append(d)
    first_support = {}
    config = original.config_for(1.)
    for level, d in enumerate(pyramid):
        src = affine.frozen.make_frame(d['depth'][1], d['valid'][1], d['K'], d['rgb'][1], config)
        tgt = affine.frozen.make_frame(d['depth'][0], d['valid'][0], d['K'], d['rgb'][0], config)
        pair = affine.frozen.pair_correspondences(np.eye(4), src, tgt, config.geometry.distance_gates_m[level], config)
        first_support[str(level)] = {'counts': pair['counts'], 'ratio': pair['ratio'],
            'arrays': {key: original.array_record(pair[key]) for key in ('source', 'transformed', 'target', 'normals', 'source_intensity')}}
    raw_files = [{'frame_id': f.frame_id, **{k: {'path': getattr(f, k), 'sha256': sha(safe_data_path(dataset, getattr(f, k)))}
                 for k in ('rgb_path', 'depth_path')}} for f in frames]
    result = {'scene': scene, 'dataset': dataset.as_posix(), 'frame_count': len(frames), 'association': association,
        'selected_sensor_metadata': metadata, 'index_sha256': {n: sha(dataset/n) for n in ('rgb.txt', 'depth.txt', 'groundtruth.txt')},
        'raw_sensor_files': raw_files, 'input_array_hashes': hashes, 'short200_299_slice_array_hashes': slice_hashes,
        'short_slice_byte_identical_to_frozen_short_protocol': True, 'first_identity_support': first_support,
        'full_ICP_reference': {**spec, 'metadata_sha256': sha(baseline_dir/'selected_sensor_metadata.json')},
        'real_pair_registration_runs': 0,
        'ground_truth_scope': 'Frozen loader reads GT fields, timestamps and constructs matrices for association. No GT pose matrix is passed to geometry, identity support, initialization, registration, acceptance or parameters. Matrices are consumed only for evaluation after complete estimate export.'}
    return result, pyramid, frames


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--execute', action='store_true')
    p.add_argument('--protocol', type=Path); p.add_argument('--protocol-sha')
    p.add_argument('--round-id', choices=[r['round_id'] for r in ROUNDS]); p.add_argument('--run', required=True)
    a = p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', a.run): p.error('Simple exclusive run name')
    if any(a.execute != (v is not None) for v in (a.protocol, a.protocol_sha, a.round_id)):
        p.error('execute requires protocol/hash/round; prepare accepts none')
    if a.protocol_sha and not re.fullmatch(r'[a-f0-9]{64}', a.protocol_sha): p.error('Exact lowercase64 SHA')
    return a


def read_protocol(path, digest):
    raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == digest, 'Exact protocol bytes')
    protocol = json.loads(raw)
    require(protocol['status'] == 'ready_for_root_review_no_real_registration' and
            protocol['source_sha256'] == source_checks() and
            protocol['round_order'] == [r['round_id'] for r in ROUNDS] and len(protocol['rounds']) == 4 and
            protocol['deadline_beijing'] == DEADLINE.isoformat(), 'Fixed four round protocol/source/deadline')
    for row, spec in zip(protocol['rounds'], ROUNDS):
        require({k: row[k] for k in spec} == spec and row['configuration'] == configuration_for(row['scene'], row['model_dimension']), 'Every declared complete config exact')
    return raw, protocol


def prepare(output):
    sources = source_checks(); datasets = {}
    for scene in SCENES:
        record, _, _ = dataset_preflight(scene)
        path = output/f'{scene}-input-witness.json'; write(path, record)
        datasets[scene] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                           'frames': SCENES[scene]['frames'], 'start_frame': 0}
    rounds = []
    for spec in ROUNDS:
        require(not (ROOT/'outputs'/spec['run']).exists(), 'No old run or output overwritten')
        cfg = configuration_for(spec['scene'], spec['model_dimension'])
        require(cfg['odometry'] == original.config_record(1.) and cfg['reference_policy'] == recovery.POLICY, 'Frozen base parameters and causal reference')
        if spec['model_dimension'] == 8:
            require(cfg['joint_affine'] == frozen.config_record() and cfg['brightness_reset_for_every_pair'], 'Exact8D/reset configuration')
        rounds.append({**spec, 'configuration': cfg})
    write(output/'synthetic_policy_checks.json', frozen.synthetic_policy_controls())
    before_deadline(DEADLINE-dt.timedelta(microseconds=1)); denied = []
    for stamp in (DEADLINE, DEADLINE+dt.timedelta(microseconds=1)):
        try: before_deadline(stamp)
        except ValueError: denied.append(stamp.isoformat())
        else: raise AssertionError('cutoff equality/after allowed')
    protocol = {'status': 'ready_for_root_review_no_real_registration', 'source_sha256': sources,
        'short_protocol': {'path': SHORT_PROTOCOL.relative_to(ROOT).as_posix(), 'sha256': SHORT_PROTOCOL_SHA},
        'rounds': rounds, 'round_order': [r['round_id'] for r in ROUNDS], 'datasets': datasets,
        'deadline_beijing': DEADLINE.isoformat(), 'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'deadline_boundary_controls': {'one_microsecond_before_allowed': True, 'at_and_after_denied': denied},
        'synthetic_policy_sha256': sha(output/'synthetic_policy_checks.json'), 'CPU_registration_runs': 0, 'GPU_runs': 0,
        'limits': ['All 549/748 associated frames are selected before results, not all raw RGB images.',
                   'Fixed 6D and8D original model/config/lastaccepted paths; this is a whole-model acceptance-path comparison, not a per-step brightness-only causal ablation.',
                   'First camera and every relative pair start identity; 8D loggain0/bias0 reset each pair and are never accumulated across pairs.',
                   'All rejected and numeric-exception frames retain the preceding prefix and last accepted sensor reference, candidates/history preserved.',
                   'GT fields/timing association is read; no GT matrix used by any optimizer or selection. Complete trajectory is exported before final GT matrix evaluation.',
                   'No new clamp, brightness prior, damping, Schur solver, gate, retry or exposure calibration.',
                   'GT anchored final metrics reuse frozen audit_pose_arrays, not a new independent formula.',
                   'Actual full preflight rechecks all PNG/index/array hashes; immutable per-pair files and compact live progress change I/O only.',
                   'No new process at or after08 Beijing; first real registration also checks cutoff; already-started registration completes without per-frame cutoff.',
                   'No re-run on failure, no best-metric selection and no cross-sequence mean/FPS claim.']}
    require(source_checks() == sources and not any(k in sys.modules for k in ('torch', 'gsplat')), 'Sources/CPU preparation unchanged')
    path = output/'protocol.json'; write(path, protocol); digest = sha(path)
    read_protocol(path, digest); checks = []
    for row in rounds:
        argv = ['--execute', '--protocol', str(path), '--protocol-sha', digest, '--round-id', row['round_id'], '--run', row['run']]
        parsed = arguments(argv)
        require(parsed.round_id == row['round_id'] and parsed.run == row['run'], 'Actual4 child CLI')
        checks.append({'argv': argv, 'model_dimension': row['model_dimension'], 'frames': row['configuration']['frames'], 'start_frame': 0})
    write(output/'cli_checks.json', {'status': 'passed_no_execution', 'protocol_sha256': digest, 'rounds': checks,
                                    'registration_and_normal_equation_calls_forbidden': True, 'GPU_runs': 0})
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    for name in sources: (snapshot/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    write(output/'source_manifest.json', {'source_sha256': sources, 'protocol_sha256': digest})
    return {'status': protocol['status'], 'protocol_sha256': digest, 'rounds': 4, 'real_registration_runs': 0, 'GPU_runs': 0}


def compact_progress(state):
    # Full pair records stay immutable on disk and in final metrics, never silently dropped.
    return {k: v for k, v in state.items() if k != 'frame_metrics'}


def execute(args, output):
    raw, protocol = read_protocol(args.protocol, args.protocol_sha)
    row = next(r for r in protocol['rounds'] if r['round_id'] == args.round_id)
    require(row['run'] == args.run, 'Exact declared destination')
    witness = protocol['datasets'][row['scene']]; witness_path = ROOT/witness['path']
    require(sha(witness_path) == witness['sha256'], 'Input witness hash')
    record, data, frames = dataset_preflight(row['scene'])
    require(record == read(witness_path), 'All actual full sensor and reference evidence unchanged')
    started_utc = before_deadline(); output.mkdir(exist_ok=False); (output/'pairs').mkdir()
    (output/'protocol_used.json').write_bytes(raw)
    write(output/'configuration.json', {**row['configuration'], 'model_dimension': row['model_dimension'],
        'source_sha256': protocol['source_sha256'], 'protocol_sha256': args.protocol_sha, 'CPU_only': True})
    state = {'status': 'running', 'round_id': row['round_id'], 'model_dimension': row['model_dimension'],
        'configuration': row['configuration'], 'source_sha256': protocol['source_sha256'], 'protocol_sha256': args.protocol_sha,
        'started_utc': started_utc, 'frame_metrics': [], 'frames_processed': 0, 'GT_pose_used_in_optimizer': False,
        'ground_truth_scope': record['ground_truth_scope'], 'GPU_runs': 0}
    config = frozen.config_for() if row['model_dimension'] == 8 else original.config_for(1.)
    base = config.base if row['model_dimension'] == 8 else config
    poses = [np.eye(4)]; reference_index = 0; count = len(frames); rows = state['frame_metrics']; started = time.perf_counter()
    def geometry(index):
        return [affine.frozen.make_frame(d['depth'][index], d['valid'][index], d['K'], d['rgb'][index], base) for d in data]
    try:
        targets = geometry(0)
        anchor = {'local_index': 0, 'frame_id': 0, 'accepted': True, 'failure_flags': [], 'status': 'fixed_first_camera_identity',
            'estimated_c2w': poses[0].tolist(), 'reference_frame_id': 0, 'reference_gap_frames': 0,
            'last_accepted_frame_id_before': 0, 'last_accepted_frame_id_after': 0, 'pair_initialized_from_GT': False}
        write(output/'pairs/0000.json', anchor); rows.append(anchor); state['frames_processed'] = 1
        with (output/'iterations.jsonl').open('x', encoding='utf-8') as stream:
            for index in range(1, count):
                before = reference_index; sensors = geometry(index)
                if index == 1: state['first_pair_registration_start_utc'] = before_deadline()
                try:
                    result = affine.register_pair_affine(sensors, targets, config) if row['model_dimension'] == 8 else affine.frozen.register_pair(sensors, targets, config)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as error:
                    result = {'accepted': False, 'failure_flags': ['pair_numeric_exception'],
                        'candidate_reference_from_current': None, 'candidate_previous_from_current': None,
                        'candidate_log_relative_gain': None, 'candidate_relative_gain': None, 'candidate_bias_intensity': None,
                        'final_counts': {}, 'final_statistics': {}, 'iteration_history': [], 'levels': [],
                        'exception_type': type(error).__name__, 'exception_message': str(error), 'traceback': traceback.format_exc(),
                        'candidate_unknown_due_to_exception': True}
                if row['model_dimension'] == 8:
                    pose, reference_index = frozen.integrate_affine(index, before, poses, result)
                else:
                    pose, reference_index = recovery.integrate_reference(index, before, poses, result)
                    result = {**result, 'candidate_reference_from_current': result['candidate_previous_from_current']}
                poses.append(pose)
                pair = {**result, 'local_index': index, 'frame_id': index, 'current_frame_id': index,
                    'reference_frame_id': before, 'reference_gap_frames': index-before,
                    'last_accepted_frame_id_before': before, 'last_accepted_frame_id_after': reference_index,
                    'estimated_c2w': pose.tolist(), 'pair_initialized_from_GT': False, 'pair_initial_transform': np.eye(4).tolist(),
                    'candidate_direction': 'current -> last accepted sensor reference',
                    'source_sensor_metadata': record['selected_sensor_metadata'][index],
                    'reference_sensor_metadata': record['selected_sensor_metadata'][before], 'maximum_sensor_frame_used_for_this_pair': index}
                if row['model_dimension'] == 8:
                    pair.update(pair_initial_log_relative_gain=0., pair_initial_bias_intensity=0., brightness_accumulated_across_pairs=False)
                write(output/'pairs'/f'{index:04d}.json', pair); rows.append(pair)
                for item in result['iteration_history']:
                    stream.write(json.dumps({'local_index': index, 'frame_id': index, 'reference_frame_id': before, **item}, allow_nan=False)+'\n')
                stream.flush()
                if result['accepted']: targets = sensors
                state.update(frames_processed=len(rows), last_completed_frame_id=index, last_pair_accepted=result['accepted'],
                             last_pair_flags=result['failure_flags'], last_accepted_reference_id=reference_index)
                update(output/'progress.json', compact_progress(state))
                original.pose_text(output/'trajectory_estimated_partial.txt', frames[:len(poses)], poses)
                print(json.dumps({'round': row['round_id'], 'frame': index, 'reference': before, 'accepted': result['accepted'], 'flags': result['failure_flags']}), flush=True)
        original.pose_text(output/'trajectory_estimated.txt', frames, poses)
        # Evaluation matrix consumption only after all estimates (including rejects) are exported.
        absolute_gt = np.stack([frame.c2w for frame in frames]); anchored_gt = np.linalg.inv(absolute_gt[0])[None] @ absolute_gt
        from audit_tum_trajectory import audit_pose_arrays
        trajectory = audit_pose_arrays(np.asarray(poses), anchored_gt)
        original.pose_text(output/'trajectory_groundtruth_anchored.txt', frames, anchored_gt)
        np.savez_compressed(output/'trajectory_arrays.npz', estimated_c2w=np.asarray(poses),
            evaluation_only_anchored_GT_c2w=anchored_gt, rgb_timestamps=np.asarray([f.rgb_timestamp for f in frames]))
        failures = sum(bool(r['failure_flags']) for r in rows)
        state.update(status='completed' if not failures else 'completed_with_pair_rejections', frames_processed=count,
            failure_frame_count=failures, tracked_nonanchor_frame_denominator=count-1, all_frame_denominator=count,
            failure_flag_counts=dict(Counter(flag for r in rows for flag in r['failure_flags'])),
            reference_gap_counts=dict(Counter(r['reference_gap_frames'] for r in rows[1:])), trajectory=trajectory,
            selected_sensor_metadata=record['selected_sensor_metadata'], input_array_hashes=record['input_array_hashes'],
            observed_wall_seconds=time.perf_counter()-started, timing_scope='CPU geometry + pair/artifact I/O/export, not isolated FPS.',
            GT_pose_evaluation_after_all_estimates_saved=True, limits=protocol['limits'])
        require(source_checks() == protocol['source_sha256'] and sha(witness_path) == witness['sha256'], 'Final source/input guard')
        require(not any(k in sys.modules for k in ('torch', 'gsplat')), 'No GPU stack')
        write(output/'metrics.json', state); update(output/'progress.json', compact_progress(state))
        write(output/'pair_manifest.json', {str(p.relative_to(output)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted((output/'pairs').glob('*.json'))})
        return {'status': state['status'], 'frames_processed': count, 'failure_frame_count': failures, 'metrics_sha256': sha(output/'metrics.json')}
    except BaseException as error:
        state.update(status='failed_incomplete_frames_retained', exception_type=type(error).__name__, exception_message=str(error),
                     traceback=traceback.format_exc(), frames_processed=len(rows), GT_final_eval_completed=False)
        write(output/'failure.json', state); update(output/'progress.json', compact_progress(state)); raise


def main():
    args = arguments(); output = ROOT/'outputs'/args.run; require(not output.exists(), 'Unique output; no overwrites')
    if args.prepare:
        output.mkdir(); (output/'preparation_source.py').write_bytes(Path(__file__).read_bytes())
        targets = [(affine, 'register_pair_affine'), (affine, 'normalized_system'),
                   (affine.frozen, 'register_pair'), (affine.frozen, 'weighted_system'),
                   (affine.frozen.icp, 'register_pair')]
        previous = [getattr(module, key) for module, key in targets]
        def forbidden(*a, **k): raise AssertionError('Real pair registration/linear solve forbidden during full preparation')
        for module, key in targets: setattr(module, key, forbidden)
        try: result = prepare(output)
        except BaseException as error:
            write(output/'failure.json', {'status': 'preparation_failed_preserved', 'type': type(error).__name__,
                                          'message': str(error), 'traceback': traceback.format_exc(), 'GPU_runs': 0}); raise
        finally:
            for (module, key), function in zip(targets, previous): setattr(module, key, function)
    else: result = execute(args, output)
    print(json.dumps(result))


if __name__ == '__main__': main()
