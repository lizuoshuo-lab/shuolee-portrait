"""CPU preparation/explicit execution of last-accepted-sensor recovery.

Only the reference sensor/pose ledger changes. Frozen photo+geometry optimizer,
engineering scales, gates, identity pair initialization and sensor sequence
remain unchanged. Preparation forbids all real pair registration/linear solves.
GT matrices are accessed for evaluation only after all estimated poses are saved;
the frozen association loader honestly reads GT fields/timestamps beforehand.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.linalg import expm

import track_rgbd_photogeometric as original
import rgbd_photogeometric_odometry as odom

ROOT = original.ROOT
PREVIOUS_PROTOCOL = ROOT / 'outputs/rgbd-photogeometric-real-cpu-preparation-003/protocol.json'
PREVIOUS_PROTOCOL_SHA = '81325bceecd336b678e8cadda1af4bf926fd3c7e5788acbb0db128524ccc265b'
ORIGINAL_PIPELINE_SHA = '6c0ccf163db7c11613750ad54d3b1462e8419326eb7d092424b68a9ae8095194'
ORIGINAL_DRIVER_SHA = '475d09f7c514db6afbfb9317e56762d1f31b80f2e3dad98a4917af82ec7f86ff'
POLICY = 'reference last accepted SENSOR frame; compose that frame c2w @ reference_from_current; reject keeps preceding prefix'
ROUNDS = [{'order': i + 1, 'round_id': f'{scene}-lastaccepted', 'scene': scene,
           'lambda_photo': 1., 'run': f'tum-{scene}-photogeometric-lastaccepted-photo1-transfer100-001'}
          for i, scene in enumerate(('desk', 'xyz'))]
require, sha, read, write, update = original.require, original.sha, original.read, original.write, original.update


def source_checks():
    require(sha(PREVIOUS_PROTOCOL) == PREVIOUS_PROTOCOL_SHA, 'Original frozen input protocol')
    previous = read(PREVIOUS_PROTOCOL)
    for name, digest in previous['source_sha256'].items():
        require(sha(ROOT / 'scripts' / name) == digest, 'Frozen original source ' + name)
    require(previous['source_sha256']['track_rgbd_photogeometric.py'] == ORIGINAL_PIPELINE_SHA and
            previous['source_sha256']['run_rgbd_photogeometric_validation.py'] == ORIGINAL_DRIVER_SHA,
            'Original pipeline/driver exact versions')
    return {**previous['source_sha256'],
            'track_rgbd_photogeometric_last_accepted.py': sha(Path(__file__)),
            'run_rgbd_photogeometric_last_accepted.py': sha(ROOT / 'scripts/run_rgbd_photogeometric_last_accepted.py')}


def integrate_reference(index, reference_index, poses, result):
    """Pure causal ledger transition; no sensor/GT/optimization access."""
    require(index == len(poses) and 0 <= reference_index < index, 'Causal current/reference IDs')
    accepted = result['accepted']
    require(accepted == (not result['failure_flags']), 'Acceptance/flags agree')
    if accepted:
        transform = np.asarray(result['candidate_previous_from_current'], dtype=np.float64)
        require(transform.shape == (4, 4) and np.isfinite(transform).all(), 'Accepted finite candidate')
        require(np.max(np.abs(transform[3] - np.array([0., 0., 0., 1.]))) < 1e-12 and
                np.max(np.abs(transform[:3, :3].T @ transform[:3, :3] - np.eye(3))) < 1e-8 and
                abs(np.linalg.det(transform[:3, :3]) - 1) < 1e-8, 'Accepted rigid candidate')
        pose = np.asarray(poses[reference_index]) @ transform
        next_reference = index
    else:
        # The reference stays at the last successfully localized SENSOR frame;
        # its prefix equals the preceding propagated pose through all rejects.
        require(np.array_equal(poses[reference_index], poses[-1]), 'Rejected-prefix invariant')
        pose = np.asarray(poses[-1]).copy()
        next_reference = reference_index
    return pose, next_reference


def matrix_exp(delta):
    """Independent SciPy exponential used only for synthetic ledger controls."""
    v = np.asarray(delta)
    x, y, z = v[3:]
    A = np.zeros((4, 4))
    A[:3, :3] = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    A[:3, 3] = v[:3]
    return expm(A)


def synthetic_controls():
    T12 = matrix_exp([.04, -.02, .01, .13, -.08, .05])
    T23 = matrix_exp([-.01, .035, .015, -.06, .10, .04])
    T13 = T12 @ T23
    poses = [np.eye(4)]
    rejected = {'accepted': False, 'failure_flags': ['synthetic_rejection'], 'candidate_previous_from_current': T12.tolist()}
    pose2, reference = integrate_reference(1, 0, poses, rejected)
    poses.append(pose2)
    accepted = {'accepted': True, 'failure_flags': [], 'candidate_previous_from_current': T13.tolist()}
    pose3, reference3 = integrate_reference(2, reference, poses, accepted)
    correct_error = float(np.max(np.abs(pose3 - T13)))
    wrong_previous_error = float(np.max(np.abs(pose2 @ T23 - T13)))
    wrong_order_error = float(np.max(np.abs(T23 @ T12 - T13)))
    require(correct_error < 1e-14 and wrong_previous_error > .01 and wrong_order_error > .001 and reference3 == 2,
            'Noncommuting skip composition / wrong predecessor and order controls')
    patterns = {'all_rejected': [False] * 5, 'consecutive_then_recovery': [False, False, True, True, True],
                'reject_recover_reject_recover': [True, False, True, False, True], 'final_frame_rejected': [True, True, True, True, False]}
    ledgers = {}
    absolute_toy = [np.eye(4)]
    for i in range(1, 6):
        absolute_toy.append(absolute_toy[-1] @ matrix_exp([.01 * i, -.006, .003, .009, -.012 * i, .006]))
    for name, flags in patterns.items():
        poses = [np.eye(4)]; reference = 0; ledger = []
        for index, accepted_flag in enumerate(flags, 1):
            before_reference = reference
            candidate = np.linalg.inv(absolute_toy[reference]) @ absolute_toy[index]
            result = {'accepted': accepted_flag, 'failure_flags': [] if accepted_flag else ['synthetic_rejection'],
                      'candidate_previous_from_current': candidate.tolist()}
            pose, reference = integrate_reference(index, reference, poses, result)
            if accepted_flag:
                require(np.max(np.abs(pose - absolute_toy[index])) < 1e-13, 'Accepted toy pose at correct gauge')
            else:
                require(np.array_equal(pose, poses[-1]) and reference == before_reference, 'Consecutive reject ledger')
            poses.append(pose)
            ledger.append({'current_frame_id': 200 + index, 'reference_frame_id': 200 + before_reference,
                           'reference_gap_frames': index - before_reference, 'accepted': accepted_flag,
                           'last_accepted_frame_id_after': 200 + reference, 'estimated_c2w': pose.tolist(),
                           'candidate_reference_from_current': candidate.tolist()})
        require(len(ledger) == 5 and [x['current_frame_id'] for x in ledger] == list(range(201, 206)), 'Every synthetic frame retained')
        ledgers[name] = ledger
    # A numeric exception with no candidate follows the same rejection path.
    pose, ref = integrate_reference(1, 0, [np.eye(4)], {'accepted': False, 'failure_flags': ['pair_numeric_exception'],
                                                    'candidate_previous_from_current': None})
    require(np.array_equal(pose, np.eye(4)) and ref == 0, 'Unknown exception candidate retained safely')
    return {'status': 'passed', 'synthetic_anchor': {'frame_id': 200, 'accepted': True, 'reference_frame_id': 200,
            'last_accepted_frame_id_after': 200, 'estimated_c2w': np.eye(4).tolist()},
            'noncommuting_skip': {'T12': T12.tolist(), 'T23': T23.tolist(), 'T13_T12_times_T23': T13.tolist(),
            'correct_max_absolute_error': correct_error, 'wrong_previous_frame_recursion_max_absolute_error': wrong_previous_error,
            'wrong_reversed_product_max_absolute_error': wrong_order_error}, 'complete_synthetic_ledgers': ledgers,
            'unknown_candidate_exception_retains_reference': True, 'real_registration_or_linear_solves': 0,
            'scope': 'Toy poses construct exact relative candidates to test ledger algebra only; no real recovery success is claimed.'}


def original_reference(scene):
    folder = ROOT / f'outputs/tum-{scene}-photogeometric-photo1-transfer100-001'
    data = read(folder / 'metrics.json')
    require(data['status'] in ('completed', 'completed_with_pair_rejections') and data['frames_processed'] == 100,
            'Original previous-frame photo1 complete reference')
    require(data['configuration']['odometry'] == original.config_record(1.) and data['GT_pose_used_in_optimizer'] is False,
            'Same frozen photo1 objective/config')
    return {'run': folder.name, 'metrics_sha256': sha(folder / 'metrics.json'),
            'trajectory_sha256': sha(folder / 'trajectory_estimated.txt'), 'failure_frame_count': data['failure_frame_count'],
            'configuration': data['configuration'], 'scope': 'Read-only original previous-frame reference; no retuning or reexecution.'}


def configuration_for(scene, reference=None):
    previous = (reference or original_reference(scene))['configuration']
    return {**previous, 'pair_initialization': 'identity current-to-last-accepted-reference for every pair',
            'reference_policy': POLICY, 'candidate_direction': 'current -> chosen reference sensor frame'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--execute', action='store_true')
    p.add_argument('--protocol', type=Path); p.add_argument('--protocol-sha'); p.add_argument('--round-id', choices=[r['round_id'] for r in ROUNDS])
    p.add_argument('--run', required=True)
    return p


def arguments(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run): p.error('Simple unique run name required')
    if args.execute != (args.protocol is not None) or args.execute != (args.protocol_sha is not None) or args.execute != (args.round_id is not None):
        p.error('Execution requires protocol/hash/round-id; preparation accepts none')
    if args.protocol_sha and not re.fullmatch(r'[a-f0-9]{64}', args.protocol_sha): p.error('Exact lowercase SHA256 required')
    return args


def prepare(output):
    sources = source_checks(); old = read(PREVIOUS_PROTOCOL); datasets = {}; references = {}
    for scene in ('desk', 'xyz'):
        record, _, _ = original.dataset_preflight(scene)
        require(record == old['datasets'][scene], 'Exact original actual sensor/metadata/input/identity-support preflight')
        bundle = old['sensor_bundles'][scene]
        require(sha(ROOT / bundle['path']) == bundle['sha256'], 'Original GT-free prepared arrays unchanged')
        datasets[scene] = record; references[scene] = original_reference(scene)
    controls = synthetic_controls(); write(output / 'synthetic_reference_ledger_checks.json', controls)
    rounds = []
    for spec in ROUNDS:
        require(not (ROOT / 'outputs' / spec['run']).exists(), 'Predeclared unique run absent')
        cfg = configuration_for(spec['scene'], references[spec['scene']])
        require(cfg['odometry'] == original.config_record(1.), 'Scales/weights/gates unchanged')
        rounds.append({**spec, 'configuration': cfg})
    original.before_deadline(original.DEADLINE - dt.timedelta(microseconds=1))
    blocked = []
    for stamp in (original.DEADLINE, original.DEADLINE + dt.timedelta(microseconds=1)):
        try: original.before_deadline(stamp)
        except ValueError: blocked.append(stamp.isoformat())
        else: raise AssertionError('Deadline boundary accepted')
    protocol = {'status': 'ready_for_explicit_root_review_then_CPU_execution', 'source_sha256': sources,
                'original_protocol': {'path': PREVIOUS_PROTOCOL.relative_to(ROOT).as_posix(), 'sha256': PREVIOUS_PROTOCOL_SHA},
                'rounds': rounds, 'round_order': [r['round_id'] for r in ROUNDS], 'datasets': datasets,
                'sensor_bundles': old['sensor_bundles'], 'original_previous_frame_photo1_references': references,
                'reference_policy': POLICY, 'deadline_beijing': original.DEADLINE.isoformat(),
                'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'real_data_registration_runs': 0, 'GPU_runs': 0,
                'synthetic_control_sha256': sha(output / 'synthetic_reference_ledger_checks.json'),
                'deadline_controls': {'one_us_before_passed': True, 'at_and_after_rejected': blocked},
                'limits': ['Only recovery reference sensor/prefix selection differs from the two original photo1 controls.',
                           'Frozen photo1 weights/scales/support/gates/three levels/identity pair initialization remain exact.',
                           'Last accepted reference can have a larger frame gap; this policy cannot guarantee re-registration or recovery.',
                           'Reject keeps preceding pose and reference sensor; every frame/rejected candidate/unknown-exception flag remains.',
                           'GT fields/times are read by the frozen association loader; matrices are final-evaluation-only after all estimates are saved.',
                           'Final trajectory metrics reuse audit_pose_arrays, as do the original controls; no second independent metric formula is claimed.',
                           'No real registration or linear system solve occurs in preparation. Toy ledger success is only algebra evidence.',
                           'CPU host durations are observational and may overlap GPU GS work; no method speed comparison.']}
    require(source_checks() == sources and 'torch' not in sys.modules and 'gsplat' not in sys.modules, 'CPU/frozen preparation')
    write(output / 'protocol.json', protocol); digest = sha(output / 'protocol.json'); parsed = []
    for row in read(output / 'protocol.json')['rounds']:
        argv = ['--execute', '--protocol', str(output / 'protocol.json'), '--protocol-sha', digest,
                '--round-id', row['round_id'], '--run', row['run']]
        args = arguments(argv)
        require(args.round_id == row['round_id'] and args.run == row['run'] and row['configuration']['odometry'] == original.config_record(1.),
                'Actual parsed protocol/config')
        parsed.append({'round_id': row['round_id'], 'argv': argv, 'parsed_execute': args.execute,
                       'parsed_protocol_sha': args.protocol_sha, 'fixed_configuration': row['configuration']})
    write(output / 'cli_checks.json', {'status': 'passed', 'rounds': parsed, 'protocol_sha256': digest,
                                      'register_pair_and_weighted_system_forbidden': True, 'real_registration_runs': 0, 'GPU_runs': 0})
    write(output / 'source_manifest.json', {'source_sha256': sources, 'protocol_sha256': digest})
    for name in sources:
        shutil.copyfile(ROOT / 'scripts' / name, output / name)
    return {'status': protocol['status'], 'protocol_sha256': digest, 'source_sha256': sources,
            'real_data_registration_runs': 0, 'GPU_runs': 0}


def execute(args, output):
    protocol_bytes = args.protocol.read_bytes()
    require(hashlib.sha256(protocol_bytes).hexdigest() == args.protocol_sha, 'Exact reviewed protocol bytes')
    protocol = json.loads(protocol_bytes)
    require(protocol['status'] == 'ready_for_explicit_root_review_then_CPU_execution' and
            protocol['source_sha256'] == source_checks() and protocol['round_order'] == [r['round_id'] for r in ROUNDS], 'Reviewed source/order')
    rows = [r for r in protocol['rounds'] if r['round_id'] == args.round_id]
    require(len(rows) == 1 and rows[0]['run'] == args.run, 'Exact predeclared run')
    row = rows[0]
    spec = next(s for s in ROUNDS if s['round_id'] == args.round_id)
    require({k: row[k] for k in spec} == spec and row['configuration'] == configuration_for(row['scene']),
            'Complete declared row / runtime configuration exact')
    data_record, data, frames = original.dataset_preflight(row['scene'])
    require(data_record == protocol['datasets'][row['scene']] and original_reference(row['scene']) ==
            protocol['original_previous_frame_photo1_references'][row['scene']], 'Actual sources/sensors/references unchanged')
    bundle = protocol['sensor_bundles'][row['scene']]
    require(sha(ROOT / bundle['path']) == bundle['sha256'] and row['configuration']['odometry'] == original.config_record(1.) and
            row['configuration']['reference_policy'] == POLICY, 'Fixed inputs and engineering objective')
    started_utc = original.before_deadline(); output.mkdir(exist_ok=False)
    (output / 'protocol_used.json').write_bytes(protocol_bytes)
    write(output / 'configuration.json', {**row['configuration'], 'round_id': row['round_id'],
          'protocol_sha256': args.protocol_sha, 'source_sha256': protocol['source_sha256'], 'CPU_only': True})
    state = {'status': 'running', 'round_id': row['round_id'], 'configuration': row['configuration'],
             'source_sha256': protocol['source_sha256'], 'protocol_sha256': args.protocol_sha, 'started_utc': started_utc,
             'frame_metrics': [], 'frames_processed': 0, 'GT_pose_used_in_optimizer': False,
             'ground_truth_scope': data_record['ground_truth_scope'], 'GPU_runs': 0}
    config = original.config_for(1.); poses = [np.eye(4)]; reference_index = 0; started = time.perf_counter()
    records = state['frame_metrics']
    def geometry(index):
        return [odom.make_frame(d['depth'][index], d['valid'][index], d['K'], d['rgb'][index], config) for d in data]
    try:
        targets = geometry(0)
        records.append({'local_index': 0, 'frame_id': 200, 'accepted': True, 'failure_flags': [],
                        'status': 'fixed_first_camera_identity', 'estimated_c2w': poses[0].tolist(),
                        'reference_frame_id': 200, 'reference_gap_frames': 0, 'last_accepted_frame_id_before': 200,
                        'last_accepted_frame_id_after': 200, 'pair_initialized_from_GT': False})
        state['frames_processed'] = 1
        with (output / 'iterations.jsonl').open('x', encoding='utf-8') as stream:
            for index in range(1, 100):
                before_reference = reference_index; sources = geometry(index)
                if index == 1: state['first_pair_registration_start_utc'] = original.before_deadline()
                try: result = odom.register_pair(sources, targets, config)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                    result = {'accepted': False, 'failure_flags': ['pair_numeric_exception'], 'candidate_previous_from_current': None,
                              'final_counts': {}, 'final_statistics': {}, 'iteration_history': [], 'levels': [],
                              'exception_type': type(error).__name__, 'exception_message': str(error), 'traceback': traceback.format_exc(),
                              'candidate_unknown_due_to_exception': True}
                pose, reference_index = integrate_reference(index, before_reference, poses, result)
                poses.append(pose)
                record = {**result, 'local_index': index, 'frame_id': 200 + index, 'current_frame_id': 200 + index,
                          'reference_frame_id': 200 + before_reference, 'reference_gap_frames': index - before_reference,
                          'last_accepted_frame_id_before': 200 + before_reference, 'last_accepted_frame_id_after': 200 + reference_index,
                          'candidate_reference_from_current': result['candidate_previous_from_current'],
                          'pose_direction': 'current -> selected last accepted reference sensor',
                          'frozen_candidate_key_scope': 'candidate_previous_from_current names the chosen reference target, which may precede the temporal previous frame.',
                          'estimated_c2w': pose.tolist(), 'pair_initialized_from_GT': False,
                          'source_sensor_metadata': data_record['selected_sensor_metadata'][index],
                          'reference_sensor_metadata': data_record['selected_sensor_metadata'][before_reference],
                          'maximum_sensor_frame_used_for_this_pair': 200 + index}
                records.append(record)
                for item in result['iteration_history']:
                    stream.write(json.dumps({'local_index': index, 'frame_id': 200 + index,
                                             'reference_frame_id': 200 + before_reference, **item}, allow_nan=False) + '\n')
                stream.flush()
                if result['accepted']: targets = sources
                state['frames_processed'] = len(records)
                update(output / 'progress.json', state)
                original.pose_text(output / 'trajectory_estimated_partial.txt', frames[:len(poses)], poses)
                print(json.dumps({'round': row['round_id'], 'current': 200 + index, 'reference': 200 + before_reference,
                                  'gap': index - before_reference, 'accepted': result['accepted'], 'flags': result['failure_flags']}), flush=True)
        original.pose_text(output / 'trajectory_estimated.txt', frames, poses)
        # Evaluation starts only after complete estimated-prefix export.
        absolute_GT = np.stack([frame.c2w for frame in frames])
        anchored_GT = np.linalg.inv(absolute_GT[0])[None] @ absolute_GT
        from audit_tum_trajectory import audit_pose_arrays
        trajectory = audit_pose_arrays(np.asarray(poses), anchored_GT)
        original.pose_text(output / 'trajectory_groundtruth_anchored.txt', frames, anchored_GT)
        np.savez_compressed(output / 'trajectory_arrays.npz', estimated_c2w=np.asarray(poses),
                            evaluation_only_anchored_GT_c2w=anchored_GT, rgb_timestamps=np.asarray([f.rgb_timestamp for f in frames]))
        failures = sum(bool(r['failure_flags']) for r in records)
        state.update(status='completed' if not failures else 'completed_with_pair_rejections', frames_processed=100,
                     failure_frame_count=failures, tracked_nonanchor_frame_denominator=99, all_frame_denominator=100,
                     failure_flag_counts=dict(Counter(flag for r in records for flag in r['failure_flags'])),
                     reference_gap_counts=dict(Counter(r['reference_gap_frames'] for r in records[1:])),
                     trajectory=trajectory, selected_sensor_metadata=data_record['selected_sensor_metadata'],
                     input_array_hashes=data_record['input_array_hashes'], observed_wall_seconds=time.perf_counter() - started,
                     timing_scope='CPU host wall time includes geometry and causal I/O/final export; not isolated speed.',
                     GT_pose_evaluation_after_all_100_estimates_saved=True, limits=protocol['limits'])
        require(source_checks() == protocol['source_sha256'] and original_reference(row['scene']) ==
                protocol['original_previous_frame_photo1_references'][row['scene']], 'Frozen source/reference guard after run')
        write(output / 'metrics.json', state); update(output / 'progress.json', state)
        return {'status': state['status'], 'frames_processed': 100, 'failure_frame_count': failures,
                'metrics_sha256': sha(output / 'metrics.json')}
    except BaseException as error:
        state.update(status='failed_incomplete_original_frames_retained', exception_type=type(error).__name__,
                     exception_message=str(error), traceback=traceback.format_exc(), frames_processed=len(records), GT_final_eval_completed=False)
        write(output / 'failure.json', state); update(output / 'progress.json', state)
        raise


def main():
    args = arguments(); output = ROOT / 'outputs' / args.run
    require(not output.exists(), 'Unique output; preserve all original artifacts')
    if args.prepare:
        output.mkdir()
        saved_register, saved_system = odom.register_pair, odom.weighted_system
        def forbidden(*a, **k): raise AssertionError('Real registration/linear solve forbidden during preparation')
        odom.register_pair = odom.weighted_system = forbidden
        try: result = prepare(output)
        except BaseException as error:
            write(output / 'failure.json', {'status': 'preparation_failed_no_real_registration', 'exception_type': type(error).__name__,
                                          'exception_message': str(error), 'traceback': traceback.format_exc()})
            raise
        finally: odom.register_pair, odom.weighted_system = saved_register, saved_system
    else: result = execute(args, output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
