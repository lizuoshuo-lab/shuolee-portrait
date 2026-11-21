"""Explicit CPU preparation/validation for pair-local affine RGB-D odometry.

Preparation never registers a real pair or solves its normal equations. The
separate execute CLI must bind a reviewed protocol. Frozen sensor/depth/support
and last-accepted-reference policy are reused; the new model is joint8D, not a
brightness-only change with identical rank, stopping or acceptance behavior.
The association loader reads GT fields; GT matrices enter final evaluation only
after the complete estimated trajectory has been exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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

import track_rgbd_photogeometric as original
import track_rgbd_photogeometric_last_accepted as recovery
import rgbd_affine_photometric_odometry as affine

ROOT = original.ROOT
RECOVERY_PROTOCOL = ROOT / 'outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/protocol.json'
RECOVERY_PROTOCOL_SHA = 'd2adb632c2261e50bc963e3b84c31d44e92233b3f399bd45a23fafd457ce13c1'
MATH_REVIEW = ROOT / 'outputs/rgbd-affine-photometric-cpu-controls-001/root_independent_math_audit.json'
MATH_REVIEW_SHA = '96f8a6ccbc7cf0b215dbd088e51376f5e288776942032be6dfd44459ac2cfb9d'
NEW_FROZEN = {
    'track_rgbd_photogeometric_last_accepted.py': '313d7e92efb20fd212b4b65b0bfb35f94e81b89d06e1f134838742ed82471b97',
    'run_rgbd_photogeometric_last_accepted.py': 'c166fe622d530f642ebb2daa6fef658f681543384d71c5272123a7b03ee4bcd8',
    'rgbd_affine_photometric_odometry.py': '15c7f160aa0f2134ea2a40d0fc0817c5642ca9e71f053e4cb0896235ac402349',
    'check_rgbd_affine_photometric_odometry.py': 'dee2c5a2469a7889c0108f734b8b9f7a2b03990bdc6cbcefc739147b43c49a43',
    'audit_rgbd_affine_math.py': '892e44a825428577d936a802639c91b210c02e5cbada558d2f21b43e36721935',
}
REFERENCE_METRICS = {'desk': 'de38c7ced8e49453f28cc7ce3a368045a2505d78f7d14e5dcecfcc787a29b271',
                     'xyz': '14300dcbf2486357f04361bec26ccc84ff1bad7047c38d99ed0198253568b638'}
ROUNDS = [{'order': i + 1, 'round_id': scene + '-affine-lastaccepted', 'scene': scene,
           'run': 'tum-' + scene + '-affine-photogeometric-lastaccepted-transfer100-001'}
          for i, scene in enumerate(('desk', 'xyz'))]
require, sha, read, write, update = original.require, original.sha, original.read, original.write, original.update


def source_checks():
    previous = recovery.source_checks()
    require(sha(RECOVERY_PROTOCOL) == RECOVERY_PROTOCOL_SHA and sha(MATH_REVIEW) == MATH_REVIEW_SHA, 'Frozen recovery protocol/independent affine math review')
    require(read(RECOVERY_PROTOCOL)['source_sha256'] == previous, 'Exact old recovery sources')
    for name, digest in NEW_FROZEN.items(): require(sha(ROOT / 'scripts' / name) == digest, 'Frozen source ' + name)
    affine.frozen_source_check()
    return {**previous, **NEW_FROZEN,
            'track_rgbd_affine_photogeometric.py': sha(Path(__file__)),
            'run_rgbd_affine_photogeometric_validation.py': sha(ROOT / 'scripts/run_rgbd_affine_photogeometric_validation.py')}


def config_for(): return affine.AffineConfig(base=original.config_for(1.))


def config_record(): return json.loads(json.dumps(asdict(config_for())))


def reference(scene):
    folder = ROOT / ('outputs/tum-' + scene + '-photogeometric-lastaccepted-photo1-transfer100-001')
    require(sha(folder / 'metrics.json') == REFERENCE_METRICS[scene], 'Original last-accepted photo1 metrics unchanged')
    data = read(folder / 'metrics.json')
    require(data['frames_processed'] == 100 and data['status'] in ('completed', 'completed_with_pair_rejections') and
            data['configuration'] == recovery.configuration_for(scene) and data['GT_pose_used_in_optimizer'] is False,
            'Complete exact frozen non-affine last-accepted reference')
    return {'run': folder.name, 'metrics_sha256': REFERENCE_METRICS[scene], 'trajectory_sha256': sha(folder / 'trajectory_estimated.txt'),
            'configuration': data['configuration'], 'failure_frame_count': data['failure_frame_count'],
            'scope': 'Read-only previous last-accepted photo1; no retuning, reexecution or real illumination attribution.'}


def configuration_for(scene, witness=None):
    return {**(witness or reference(scene))['configuration'], 'joint_affine': config_record(), 'joint_state_dimension': 8,
            'pair_brightness_initialization': {'log_relative_gain': 0., 'relative_gain': 1., 'bias_intensity': 0.},
            'brightness_reset_for_every_pair': True, 'brightness_accumulated_across_pairs': False,
            'photometric_residual': 'exp(a)*I_reference(project(T_reference_from_current*p_current))+b-I_current',
            'photometric_scope': 'Relative correction per sensor pair; no absolute exposure gauge, sRGB linearization, radiometric calibration, prior or clamp.',
            'joint_update': 'left SE3 exp(delta_xi)@T; additive a and b; shared physical pose step-limit factor scales all8 columns'}


def integrate_affine(index, reference_index, poses, result):
    """Reuse reviewed causal pose ledger; pair brightness never enters prefix."""
    adapted = {**result, 'candidate_previous_from_current': result['candidate_reference_from_current']}
    return recovery.integrate_reference(index, reference_index, poses, adapted)


def synthetic_policy_controls():
    ledger = recovery.synthetic_controls()
    T = recovery.matrix_exp([.02, -.01, .015, .08, -.03, .04])
    base = {'accepted': True, 'failure_flags': [], 'candidate_reference_from_current': T.tolist()}
    a, ra = integrate_affine(1, 0, [np.eye(4)], {**base, 'candidate_log_relative_gain': .3, 'candidate_bias_intensity': -.2})
    b, rb = integrate_affine(1, 0, [np.eye(4)], {**base, 'candidate_log_relative_gain': -.7, 'candidate_bias_intensity': .4})
    require(np.array_equal(a, b) and ra == rb == 1, 'Pair nuisance is not an accumulated pose gauge')
    rejected, ref = integrate_affine(1, 0, [np.eye(4)], {'accepted': False, 'failure_flags': ['toy_rank_deficient8'],
                                  'candidate_reference_from_current': None, 'candidate_log_relative_gain': None, 'candidate_bias_intensity': None})
    require(np.array_equal(rejected, np.eye(4)) and ref == 0, 'Unknown rejected joint candidate retains causal reference')
    return {'status': 'passed_synthetic_ledger_only', 'reviewed_noncommuting_and_rejection_ledgers': ledger,
            'different_pair_nuisance_same_pose_prefix_exact': True, 'real_registration_or_linear_solves': 0}


def parser():
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--execute', action='store_true')
    p.add_argument('--protocol', type=Path); p.add_argument('--protocol-sha')
    p.add_argument('--round-id', choices=[r['round_id'] for r in ROUNDS]); p.add_argument('--run', required=True)
    return p


def arguments(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run): p.error('Simple unique run required')
    if args.execute != (args.protocol is not None) or args.execute != (args.protocol_sha is not None) or args.execute != (args.round_id is not None):
        p.error('Execute requires protocol/hash/round-id, prepare accepts none')
    if args.protocol_sha and not re.fullmatch(r'[a-f0-9]{64}', args.protocol_sha): p.error('Exact lowercase SHA256 required')
    return args


def prepare(output):
    sources = source_checks(); old = read(recovery.PREVIOUS_PROTOCOL); previous = read(RECOVERY_PROTOCOL)
    datasets = {}; references = {}; support = {}
    for scene in ('desk', 'xyz'):
        record, pyramid, _ = original.dataset_preflight(scene)
        require(record == old['datasets'][scene] == previous['datasets'][scene], 'All actual100 sensor metadata/gray/depth/K/index/PNG witnesses match frozen protocols')
        bundle = old['sensor_bundles'][scene]
        require(sha(ROOT / bundle['path']) == bundle['sha256'] and bundle['contains_GT_poses'] is False, 'Frozen GT-free sensor bundle')
        datasets[scene] = record; references[scene] = reference(scene); support[scene] = {}
        for level, d in enumerate(pyramid):
            config = config_for(); source = affine.frozen.make_frame(d['depth'][1], d['valid'][1], d['K'], d['rgb'][1], config.base)
            target = affine.frozen.make_frame(d['depth'][0], d['valid'][0], d['K'], d['rgb'][0], config.base)
            pair = affine.frozen.pair_correspondences(np.eye(4), source, target, config.base.geometry.distance_gates_m[level], config.base)
            measured = {'counts': pair['counts'], 'ratio': pair['ratio'],
                        'arrays': {key: original.array_record(pair[key]) for key in ('source', 'transformed', 'target', 'normals', 'source_intensity')}}
            expected = record['first_pair_identity_start'][str(level)]
            require(measured == {k: expected[k] for k in measured}, 'Same actual identity sensor/support origin, no solve')
            support[scene][str(level)] = measured
    controls = synthetic_policy_controls(); write(output / 'synthetic_policy_checks.json', controls)
    rounds = []
    for spec in ROUNDS:
        require(not (ROOT / 'outputs' / spec['run']).exists(), 'Unique predeclared actual output absent')
        cfg = configuration_for(spec['scene'], references[spec['scene']])
        require(cfg['joint_affine']['base'] == cfg['odometry'] == original.config_record(1.) and cfg['reference_policy'] == recovery.POLICY,
                'All base6D geometry/depth/support/weights/scales/gates unchanged; exact last-accepted policy')
        rounds.append({**spec, 'configuration': cfg})
    original.before_deadline(original.DEADLINE - dt.timedelta(microseconds=1)); rejected = []
    for stamp in (original.DEADLINE, original.DEADLINE + dt.timedelta(microseconds=1)):
        try: original.before_deadline(stamp)
        except ValueError: rejected.append(stamp.isoformat())
        else: raise AssertionError('Deadline equality/after accepted')
    protocol = {'status': 'ready_for_explicit_root_review_then_CPU_execution', 'source_sha256': sources,
                'original_sensor_protocol': {'path': recovery.PREVIOUS_PROTOCOL.relative_to(ROOT).as_posix(), 'sha256': recovery.PREVIOUS_PROTOCOL_SHA},
                'original_lastaccepted_protocol': {'path': RECOVERY_PROTOCOL.relative_to(ROOT).as_posix(), 'sha256': RECOVERY_PROTOCOL_SHA},
                'independent_synthetic_math_review': {'path': MATH_REVIEW.relative_to(ROOT).as_posix(), 'sha256': MATH_REVIEW_SHA},
                'rounds': rounds, 'round_order': [r['round_id'] for r in ROUNDS], 'datasets': datasets,
                'sensor_bundles': old['sensor_bundles'], 'actual_affine_identity_support': support,
                'original_lastaccepted_photo1_references': references, 'deadline_beijing': original.DEADLINE.isoformat(),
                'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'real_data_registration_runs': 0, 'GPU_runs': 0,
                'synthetic_policy_sha256': sha(output / 'synthetic_policy_checks.json'),
                'deadline_controls': {'one_us_before_passed': True, 'at_and_after_rejected': rejected},
                'limits': ['Same frozen last-accepted sensor/prefix policy, sensor200..299, depths, support, identity SE3 start, engineering weights/scales/gates; no GT tuning.',
                           'Joint8D changes nuisance observability, rank, stopping and trial/acceptance paths; not a pure brightness-only or old-ICP-byte-equivalence claim.',
                           'Each pair starts Tidentity/loggain0/bias0; gain and bias are reset, never accumulated as exposure gauge across pairs.',
                           'Unbounded exp(a) has no clamp/prior. Nonfinite overflow invalidates the whole trial; rejected/unknown candidates and all frames remain.',
                           'Relative image correction is not radiometric calibration or linear-light RGB; sRGB grayscale remains .299R+.587G+.114B.',
                           'Synthetic brightness recovery does not prove real previous rejections were illumination-caused or that real TUM will improve.',
                           'Frozen loader reads GT matrices/fields/times for association; final matrices are used only after all100 estimates are exported.',
                           'Same reviewed audit_pose_arrays for final metrics is explicit reuse, not a second independent evaluation formula.',
                           'Preparation forbids real registration and normal-equation solves. Any later execution requires explicit protocol review.',
                           'CPU host times can overlap GS GPU work; no isolated method speed or cross-sequence aggregate claim.']}
    require(source_checks() == sources and not any(k in sys.modules for k in ('torch', 'gsplat')), 'CPU/frozen source preparation')
    write(output / 'protocol.json', protocol); digest = sha(output / 'protocol.json'); parsed = []
    for row in read(output / 'protocol.json')['rounds']:
        argv = ['--execute', '--protocol', str(output / 'protocol.json'), '--protocol-sha', digest,
                '--round-id', row['round_id'], '--run', row['run']]
        args = arguments(argv)
        require(args.round_id == row['round_id'] and args.run == row['run'] and row['configuration'] == configuration_for(row['scene']), 'Actual parsed2CLI/runtime exact config')
        parsed.append({'round_id': row['round_id'], 'argv': argv, 'parsed_execute': args.execute,
                       'parsed_protocol_sha': args.protocol_sha, 'fixed_configuration': row['configuration']})
    write(output / 'cli_checks.json', {'status': 'passed', 'rounds': parsed, 'protocol_sha256': digest,
                                      'register_pair_and_6D_8D_normal_systems_forbidden': True, 'real_registration_runs': 0, 'GPU_runs': 0})
    write(output / 'source_manifest.json', {'source_sha256': sources, 'protocol_sha256': digest})
    for name in sources: shutil.copyfile(ROOT / 'scripts' / name, output / name)
    return {'status': protocol['status'], 'protocol_sha256': digest, 'source_sha256': sources,
            'real_data_registration_runs': 0, 'GPU_runs': 0}


def execute(args, output):
    raw = args.protocol.read_bytes(); require(hashlib.sha256(raw).hexdigest() == args.protocol_sha, 'Exact reviewed protocol bytes')
    protocol = json.loads(raw)
    require(protocol['status'] == 'ready_for_explicit_root_review_then_CPU_execution' and protocol['source_sha256'] == source_checks() and
            protocol['round_order'] == [r['round_id'] for r in ROUNDS] and len(protocol['rounds']) == 2, 'Reviewed exact source/order')
    row = next(r for r in protocol['rounds'] if r['round_id'] == args.round_id); spec = next(s for s in ROUNDS if s['round_id'] == args.round_id)
    require({k: row[k] for k in spec} == spec and args.run == row['run'] and row['configuration'] == configuration_for(row['scene']), 'Complete declared round/config')
    record, data, frames = original.dataset_preflight(row['scene'])
    require(record == protocol['datasets'][row['scene']] and reference(row['scene']) == protocol['original_lastaccepted_photo1_references'][row['scene']], 'Actual sensor/reference witnesses unchanged')
    bundle = protocol['sensor_bundles'][row['scene']]; require(sha(ROOT / bundle['path']) == bundle['sha256'], 'Actual GT-free bundle bytes unchanged')
    started_utc = original.before_deadline(); output.mkdir(exist_ok=False); (output / 'protocol_used.json').write_bytes(raw)
    write(output / 'configuration.json', {**row['configuration'], 'round_id': row['round_id'], 'protocol_sha256': args.protocol_sha,
                                        'source_sha256': protocol['source_sha256'], 'CPU_only': True})
    state = {'status': 'running', 'round_id': row['round_id'], 'configuration': row['configuration'], 'source_sha256': protocol['source_sha256'],
             'protocol_sha256': args.protocol_sha, 'started_utc': started_utc, 'frame_metrics': [], 'frames_processed': 0,
             'GT_pose_used_in_optimizer': False, 'ground_truth_scope': record['ground_truth_scope'], 'GPU_runs': 0}
    config = config_for(); poses = [np.eye(4)]; reference_index = 0; started = time.perf_counter(); rows = state['frame_metrics']
    def geometry(index): return [affine.frozen.make_frame(d['depth'][index], d['valid'][index], d['K'], d['rgb'][index], config.base) for d in data]
    try:
        targets = geometry(0)
        rows.append({'local_index': 0, 'frame_id': 200, 'accepted': True, 'failure_flags': [], 'status': 'fixed_first_camera_identity',
                     'estimated_c2w': poses[0].tolist(), 'reference_frame_id': 200, 'reference_gap_frames': 0,
                     'last_accepted_frame_id_before': 200, 'last_accepted_frame_id_after': 200, 'pair_initialized_from_GT': False})
        state['frames_processed'] = 1
        with (output / 'iterations.jsonl').open('x', encoding='utf-8') as stream:
            for index in range(1, 100):
                before = reference_index; sources = geometry(index)
                if index == 1: state['first_pair_registration_start_utc'] = original.before_deadline()
                try: result = affine.register_pair_affine(sources, targets, config)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as error:
                    result = {'accepted': False, 'failure_flags': ['pair_numeric_exception'], 'candidate_reference_from_current': None,
                              'candidate_log_relative_gain': None, 'candidate_relative_gain': None, 'candidate_bias_intensity': None,
                              'final_counts': {}, 'final_statistics': {}, 'iteration_history': [], 'levels': [],
                              'exception_type': type(error).__name__, 'exception_message': str(error), 'traceback': traceback.format_exc(),
                              'candidate_unknown_due_to_exception': True}
                pose, reference_index = integrate_affine(index, before, poses, result); poses.append(pose)
                pair_record = {**result, 'local_index': index, 'frame_id': 200 + index, 'current_frame_id': 200 + index,
                               'reference_frame_id': 200 + before, 'reference_gap_frames': index - before,
                               'last_accepted_frame_id_before': 200 + before, 'last_accepted_frame_id_after': 200 + reference_index,
                               'estimated_c2w': pose.tolist(), 'pair_initialized_from_GT': False,
                               'pair_initial_transform': np.eye(4).tolist(), 'pair_initial_log_relative_gain': 0., 'pair_initial_bias_intensity': 0.,
                               'brightness_accumulated_across_pairs': False, 'source_sensor_metadata': record['selected_sensor_metadata'][index],
                               'reference_sensor_metadata': record['selected_sensor_metadata'][before], 'maximum_sensor_frame_used_for_this_pair': 200 + index}
                rows.append(pair_record)
                for item in result['iteration_history']: stream.write(json.dumps({'local_index': index, 'frame_id': 200 + index,
                         'reference_frame_id': 200 + before, **item}, allow_nan=False) + '\n')
                stream.flush()
                if result['accepted']: targets = sources
                state['frames_processed'] = len(rows); update(output / 'progress.json', state)
                original.pose_text(output / 'trajectory_estimated_partial.txt', frames[:len(poses)], poses)
                print(json.dumps({'round': row['round_id'], 'current': 200 + index, 'reference': 200 + before, 'gap': index - before,
                                  'accepted': result['accepted'], 'flags': result['failure_flags'], 'loggain': result['candidate_log_relative_gain'],
                                  'bias': result['candidate_bias_intensity']}), flush=True)
        original.pose_text(output / 'trajectory_estimated.txt', frames, poses)
        # GT matrices are first accessed only after every accepted/rejected pose is exported.
        absolute_GT = np.stack([frame.c2w for frame in frames]); anchored_GT = np.linalg.inv(absolute_GT[0])[None] @ absolute_GT
        from audit_tum_trajectory import audit_pose_arrays
        trajectory = audit_pose_arrays(np.asarray(poses), anchored_GT)
        original.pose_text(output / 'trajectory_groundtruth_anchored.txt', frames, anchored_GT)
        np.savez_compressed(output / 'trajectory_arrays.npz', estimated_c2w=np.asarray(poses), evaluation_only_anchored_GT_c2w=anchored_GT,
                            rgb_timestamps=np.asarray([f.rgb_timestamp for f in frames]))
        failures = sum(bool(r['failure_flags']) for r in rows)
        state.update(status='completed' if not failures else 'completed_with_pair_rejections', frames_processed=100,
                     failure_frame_count=failures, tracked_nonanchor_frame_denominator=99, all_frame_denominator=100,
                     failure_flag_counts=dict(Counter(flag for r in rows for flag in r['failure_flags'])),
                     reference_gap_counts=dict(Counter(r['reference_gap_frames'] for r in rows[1:])), trajectory=trajectory,
                     selected_sensor_metadata=record['selected_sensor_metadata'], input_array_hashes=record['input_array_hashes'],
                     observed_wall_seconds=time.perf_counter() - started, timing_scope='CPU host includes geometry/causal I/O/export, not isolated method speed.',
                     GT_pose_evaluation_after_all_100_estimates_saved=True, limits=protocol['limits'])
        require(source_checks() == protocol['source_sha256'] and reference(row['scene']) == protocol['original_lastaccepted_photo1_references'][row['scene']], 'Frozen source/reference postguard')
        write(output / 'metrics.json', state); update(output / 'progress.json', state)
        return {'status': state['status'], 'frames_processed': 100, 'failure_frame_count': failures, 'metrics_sha256': sha(output / 'metrics.json')}
    except BaseException as error:
        state.update(status='failed_incomplete_original_frames_retained', exception_type=type(error).__name__, exception_message=str(error),
                     traceback=traceback.format_exc(), frames_processed=len(rows), GT_final_eval_completed=False)
        write(output / 'failure.json', state); update(output / 'progress.json', state); raise


def main():
    args = arguments(); output = ROOT / 'outputs' / args.run; require(not output.exists(), 'Unique output, preserve historical artifacts')
    if args.prepare:
        output.mkdir(); functions = [(affine, 'register_pair_affine'), (affine, 'normalized_system'),
                                    (affine.frozen, 'register_pair'), (affine.frozen, 'weighted_system')]
        originals = [getattr(module, name) for module, name in functions]
        def forbidden(*a, **k): raise AssertionError('Real registration/6D/8D normal-equation solve forbidden during preparation')
        for module, name in functions: setattr(module, name, forbidden)
        try: result = prepare(output)
        except BaseException as error:
            write(output / 'failure.json', {'status': 'preparation_failed_no_real_registration', 'exception_type': type(error).__name__,
                                          'exception_message': str(error), 'traceback': traceback.format_exc()}); raise
        finally:
            for (module, name), function in zip(functions, originals): setattr(module, name, function)
    else: result = execute(args, output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
