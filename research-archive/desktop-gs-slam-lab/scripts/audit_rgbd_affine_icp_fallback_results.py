"""Read-only terminal fusion audit; explicit frozen b549/e807 numerical reuse.

Never imports the fusion integration, sensor loader or numerical registration.
Preparation uses only synthetic rows and terminal older evidence. Actual audit
requires the complete two-child terminal ledger, a separately reviewed package
SHA, and immutable files; it never opens active progress to decide readiness.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime
from decimal import Decimal
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
import audit_rgbd_photogeometric_results as metric_helper
import audit_rgbd_affine_real_results as step_helper

ROOT = Path(__file__).resolve().parents[1]
FUSION_PREP = ROOT / 'outputs/rgbd-affine-icp-fallback-cpu-preparation-001'
FUSION_PROTOCOL_SHA = '358a418359ef4f2a78c9ad73ac78a7fa69582457843d23cb28cd142e018bf077'
PREP = ROOT / 'outputs/rgbd-affine-icp-fallback-result-audit-cpu-preparation-001'
BATCH = ROOT / 'outputs/rgbd-affine-icp-fallback-validation-001'
HELPERS = {'audit_rgbd_photogeometric_results.py': 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53',
           'audit_rgbd_affine_real_results.py': 'b549a0af2642b2e203e8f3b5f0deae46dcaf2458f608d7e27435c87b33f8c6d5'}
SENSOR_KEYS = ('frame_id', 'rgb_index', 'depth_index', 'rgb_timestamp', 'depth_timestamp', 'rgb_depth_abs_dt_s', 'rgb_path', 'depth_path')
FALLBACK_KEYS = {'current_frame_id', 'previous_frame_id', 'original_accepted', 'original_failure_flags', 'candidate_previous_from_current', 'candidate_direction', 'current_sensor', 'previous_sensor', 'original_current_rgb_timestamp', 'source_core_sha256', 'original_full_metric_row_sha256', 'original_metric_row_index', 'original_final_counts', 'original_final_statistics', 'eligible', 'eligibility_checks', 'ineligible_reasons'}
CANDIDATE_KEYS = {'frame_id', 'primary_reference_frame_id', 'primary', 'fallback', 'primary_current_sensor', 'primary_reference_sensor', 'pair_initial_transform', 'pair_initial_log_relative_gain', 'pair_initial_bias_intensity', 'brightness_accumulated_across_pairs'}
DECISION_KEYS = {'selection', 'accepted', 'failure_flags', 'primary_rejected', 'fallback_selected', 'fallback_was_eligible', 'primary_failure_flags', 'fallback_ineligible_reasons'}
PAIR_EXTRA_KEYS = {'local_index', 'current_frame_id', 'reference_frame_id', 'reference_gap_frames', 'fallback_reference_frame_id', 'last_accepted_frame_id_before', 'last_accepted_frame_id_after', 'estimated_c2w', 'maximum_sensor_frame_used', 'pair_initialized_from_GT'}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    require(len(raw) < 1024 * 1024, 'Audit/preparation artifact below1MiB')
    with Path(path).open('xb') as stream: stream.write(raw)


def sensor(metadata): return {k: metadata[k] for k in SENSOR_KEYS}


def proper_pose(value, tol=1e-8):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == (4, 4) and np.isfinite(a).all(), 'Finite4x4 camera/candidate')
    require(np.max(np.abs(a[3] - [0, 0, 0, 1])) < 1e-12 and np.max(np.abs(a[:3, :3].T @ a[:3, :3] - np.eye(3))) < tol and abs(np.linalg.det(a[:3, :3]) - 1) < tol, 'Proper rigid homogeneous pose')
    return a


def primary_steps(primary):
    require(primary['accepted'] == (not primary['failure_flags']), 'Primary accepted and all primary flags agree')
    if primary.get('candidate_unknown_due_to_exception'):
        require(not primary['accepted'] and primary['candidate_reference_from_current'] is None and primary['levels'] == [] and primary['iteration_history'] == [] and 'pair_numeric_exception' in primary['failure_flags'], 'Unknown numeric candidate honestly retains empty unavailable levels/iterations')
    else:
        proper_pose(primary['candidate_reference_from_current'])
        require([level['level'] for level in primary['levels']] == [0, 1, 2], 'All ordered primary pyramid levels retained')
        for row in primary['iteration_history']:
            require(row['level'] in (0, 1, 2), 'Primary step level belongs to exact pyramid')
            if 'proposed_delta_8d' in row:
                proposed = np.asarray(row['proposed_delta_8d'], dtype=float)
                require(proposed.shape == (8,) and np.isfinite(proposed).all(), 'This row has exactly finite8D proposal')
            if row['accepted']:
                require('proposed_delta_8d' in row, 'Accepted primary step has own current-row proposal')
                accepted = np.asarray(row['accepted_delta_8d'], dtype=float)
                require(accepted.shape == (8,) and np.isfinite(accepted).all(), 'Exactly finite8D accepted update')
    return step_helper.step_ledger(primary)


def audit_pair(index, reference, poses, row, fallback, metadata):
    require(set(row) == CANDIDATE_KEYS | DECISION_KEYS | PAIR_EXTRA_KEYS, 'Exact immutable fusion pair schema')
    require(row['frame_id'] == row['local_index'] == row['current_frame_id'] == index and row['primary_reference_frame_id'] == row['reference_frame_id'] == row['last_accepted_frame_id_before'] == reference, 'Primary/current reference identities exact')
    require(row['reference_gap_frames'] == index - reference and row['fallback_reference_frame_id'] == index - 1 and row['maximum_sensor_frame_used'] == index and row['pair_initialized_from_GT'] is False, 'Reference gap/adjacency/causal sensor boundary')
    require(row['primary_current_sensor'] == sensor(metadata[index]) and row['primary_reference_sensor'] == sensor(metadata[reference]), 'Primary physically matches current and last FINAL accepted sensor')
    require(np.array_equal(row['pair_initial_transform'], np.eye(4)) and row['pair_initial_log_relative_gain'] == row['pair_initial_bias_intensity'] == 0 and row['brightness_accumulated_across_pairs'] is False, 'Original identity/a0/b0 pair initialization')
    require(set(fallback) == FALLBACK_KEYS and set(fallback['current_sensor']) == set(fallback['previous_sensor']) == set(SENSOR_KEYS) and row['fallback'] == fallback, 'Every fallback field equals frozen GT-free witness')
    require(fallback['current_frame_id'] == index and fallback['previous_frame_id'] == index - 1 and fallback['current_sensor'] == sensor(metadata[index]) and fallback['previous_sensor'] == sensor(metadata[index - 1]) and fallback['original_current_rgb_timestamp'] == metadata[index]['rgb_timestamp'] and fallback['candidate_direction'] == 'current -> immediately previous SENSOR frame', 'Fallback physical adjacent sensor identity/timestamp/direction')
    checks = fallback['eligibility_checks']
    require(fallback['eligible'] == all(checks.values()) and fallback['ineligible_reasons'] == [k for k, v in checks.items() if not v], 'Original fallback eligibility/reasons preserved')
    if fallback['eligible']: require(fallback['original_accepted'] is True and fallback['original_failure_flags'] == [], 'Eligible cached ICP original acceptance/flags')
    require(np.array_equal(poses[reference], poses[index - 1]), 'Held ledger last accepted prefix equals preceding final pose; physical SENSOR IDs may differ')
    primary = row['primary']; stats = primary_steps(primary)
    if primary['accepted']:
        expected = poses[reference] @ proper_pose(primary['candidate_reference_from_current']); selection = 'primary_affine'; accepted = True; final_flags = []
    elif fallback['eligible']:
        expected = poses[index - 1] @ proper_pose(fallback['candidate_previous_from_current']); selection = 'adjacent_depth_ICP_fallback'; accepted = True; final_flags = []
    else:
        expected = poses[index - 1].copy(); selection = 'hold_both_rejected'; accepted = False
        final_flags = ['primary/' + f for f in primary['failure_flags']] + ['fallback/' + f for f in fallback['ineligible_reasons']]
    next_reference = index if accepted else reference
    decision = {'selection': selection, 'accepted': accepted, 'failure_flags': final_flags, 'primary_rejected': not primary['accepted'],
                'fallback_selected': selection == 'adjacent_depth_ICP_fallback', 'fallback_was_eligible': fallback['eligible'],
                'primary_failure_flags': primary['failure_flags'], 'fallback_ineligible_reasons': fallback['ineligible_reasons']}
    require({k: row[k] for k in DECISION_KEYS} == decision, 'Independent primary priority/fallback/dualhold selection and separate flags')
    require(np.array_equal(expected, poses[index]) and np.array_equal(row['estimated_c2w'], poses[index]) and row['last_accepted_frame_id_after'] == next_reference, 'Exact independent c2w composition/held pose and accepted SENSOR promotion')
    return next_reference, stats


def terminal_gate(batch, protocol):
    require(batch['status'] == 'all_two_predeclared_CPU_fusion_rounds_completed' and batch['runs_started'] == 2 and batch['current_run'] is None and len(batch['run_records']) == 2 and batch['GPU_runs'] == 0, 'Only both genuine terminal children allow actual audit')
    require(batch['protocol_sha256'] == FUSION_PROTOCOL_SHA and batch['source_sha256'] == protocol['source_sha256'], 'Actual batch source/protocol exact')
    previous = None
    for declared, record in zip(protocol['rounds'], batch['run_records']):
        require(record['returncode'] == 0 and all(record[k] == declared[k] for k in ('order', 'round_id', 'scene', 'run', 'configuration')), 'Exact two declared commands and actual exit0')
        start, end = (datetime.fromisoformat(record[k]) for k in ('started_utc', 'ended_utc'))
        require(start < end and start < datetime.fromisoformat(protocol['deadline_beijing']) and (previous is None or previous <= start), 'Serial actual interval and late launch cutoff')
        previous = end


def text_check(path, metadata, poses):
    rows = metric_helper.rows(path); require(len(rows) == len(poses), 'All trajectory text rows including held/rejected frames')
    maximum = angle = 0.; previous = None
    for fields, meta, pose in zip(rows, metadata, poses):
        require(len(fields) == 8, 'TUM timestamp tx ty tz qx qy qz qw')
        stamp = Decimal(fields[0]); require(stamp == Decimal(f"{meta['rgb_timestamp']:.9f}") and (previous is None or previous < stamp), 'Exact monotonically increasing actual RGB times')
        previous = stamp; a = np.asarray(fields[1:], dtype=float); require(np.isfinite(a).all() and abs(np.linalg.norm(a[3:]) - 1) < 1e-10, 'Finite position/unit xyzw quaternion')
        maximum = max(maximum, step_helper.close(a[:3], pose[:3, 3], 'Saved TUM position', 1e-11))
        angle = max(angle, float(Rotation.from_matrix(Rotation.from_quat(a[3:]).as_matrix().T @ pose[:3, :3]).magnitude()))
    require(angle < 1e-10, 'Saved TUM quaternion matches c2w')
    return {'rows': len(rows), 'translation_max_abs_difference_m': maximum, 'rotation_max_difference_rad': angle}


def synthetic_controls(protocol):
    delta = np.array([.01, -.02, .015, .02, -.03, .01, .01, .002]); T = step_helper.se3(delta)
    trial = {'scale': 1., 'candidate_transform': T.tolist(), 'candidate_log_gain': float(delta[6]), 'candidate_bias': float(delta[7]), 'same_correspondence_denominator': 32, 'objective': 9., 'invalid_reason': None}
    step = {'level': 0, 'transform_before': np.eye(4).tolist(), 'log_gain_before': 0., 'bias_before': 0., 'proposed_delta_8d': delta.tolist(), 'shared_pose_limit_factor': 1., 'line_search_trials': [trial], 'counts': {'joint_correspondences': 32}, 'normalized_robust_objective': 10., 'accepted': True, 'accepted_scale': 1., 'fixed_correspondence_objective_after': 9., 'accepted_delta_8d': delta.tolist(), 'transform_after': T.tolist(), 'log_gain_after': float(delta[6]), 'bias_after': float(delta[7])}
    primary = {'accepted': True, 'failure_flags': [], 'candidate_reference_from_current': T.tolist(), 'candidate_log_relative_gain': float(delta[6]), 'candidate_relative_gain': math.exp(float(delta[6])), 'candidate_bias_intensity': float(delta[7]), 'iteration_history': [step], 'levels': [{'level': i, 'iterations_recorded': int(i == 0), 'accepted_updates': int(i == 0)} for i in range(3)], 'final_statistics': {'joint_rank': 8, 'column_normalized_hessian_condition': 1.}}
    metadata = [{'frame_id': i, 'rgb_index': i, 'depth_index': i, 'rgb_timestamp': 100. + i * .03, 'depth_timestamp': 100. + i * .03 + .01, 'rgb_depth_abs_dt_s': .01, 'rgb_path': f'rgb/{i}.png', 'depth_path': f'depth/{i}.png'} for i in range(4)]
    F = np.eye(4); F[:3, :3] = Rotation.from_rotvec([-.08, .03, .05]).as_matrix(); F[:3, 3] = [-.02, .04, .01]
    checks = {'original_ICP_accepted': True, 'original_failure_flags_empty': True, 'current_and_previous_IDs_are_adjacent': True, 'current_and_previous_sensor_metadata_exact': True, 'original_current_timestamp_exact': True, 'frozen_ICP_core_source_exact': True, 'current_to_previous_direction': True, 'candidate_proper_rigid': True}
    fallback = {'current_frame_id': 1, 'previous_frame_id': 0, 'original_accepted': True, 'original_failure_flags': [], 'candidate_previous_from_current': F.tolist(), 'candidate_direction': 'current -> immediately previous SENSOR frame', 'current_sensor': sensor(metadata[1]), 'previous_sensor': sensor(metadata[0]), 'original_current_rgb_timestamp': metadata[1]['rgb_timestamp'], 'source_core_sha256': {}, 'original_full_metric_row_sha256': '0' * 64, 'original_metric_row_index': 1, 'original_final_counts': {}, 'original_final_statistics': {}, 'eligible': True, 'eligibility_checks': checks, 'ineligible_reasons': []}
    C = np.eye(4); C[:3, :3] = Rotation.from_rotvec([.2, -.1, .3]).as_matrix(); C[:3, 3] = [1, .5, -.2]
    def make_row(p, f, index=1, reference=0, initial=C, result=None):
        if p['accepted']: expected = initial @ np.asarray(p['candidate_reference_from_current']); chosen = 'primary_affine'; flags = []; accept = True
        elif f['eligible']: expected = initial @ np.asarray(f['candidate_previous_from_current']); chosen = 'adjacent_depth_ICP_fallback'; flags = []; accept = True
        else: expected = initial.copy(); chosen = 'hold_both_rejected'; flags = ['primary/' + x for x in p['failure_flags']] + ['fallback/' + x for x in f['ineligible_reasons']]; accept = False
        return {'frame_id': index, 'primary_reference_frame_id': reference, 'primary': copy.deepcopy(p), 'fallback': copy.deepcopy(f), 'primary_current_sensor': sensor(metadata[index]), 'primary_reference_sensor': sensor(metadata[reference]), 'pair_initial_transform': np.eye(4).tolist(), 'pair_initial_log_relative_gain': 0., 'pair_initial_bias_intensity': 0., 'brightness_accumulated_across_pairs': False,
            'selection': chosen, 'accepted': accept, 'failure_flags': flags, 'primary_rejected': not p['accepted'], 'fallback_selected': chosen == 'adjacent_depth_ICP_fallback', 'fallback_was_eligible': f['eligible'], 'primary_failure_flags': p['failure_flags'], 'fallback_ineligible_reasons': f['ineligible_reasons'],
            'local_index': index, 'current_frame_id': index, 'reference_frame_id': reference, 'reference_gap_frames': index - reference, 'fallback_reference_frame_id': index - 1, 'last_accepted_frame_id_before': reference, 'last_accepted_frame_id_after': index if accept else reference, 'estimated_c2w': (expected if result is None else result).tolist(), 'maximum_sensor_frame_used': index, 'pair_initialized_from_GT': False}
    positive = make_row(primary, fallback); audit_pair(1, 0, np.stack([C, np.asarray(positive['estimated_c2w'])]), positive, fallback, metadata)
    rejected = copy.deepcopy(primary); rejected.update(accepted=False, failure_flags=['ill_conditioned_joint_system'])
    chosen = make_row(rejected, fallback); audit_pair(1, 0, np.stack([C, np.asarray(chosen['estimated_c2w'])]), chosen, fallback, metadata)
    bad_fallback = copy.deepcopy(fallback); bad_fallback.update(eligible=False, original_accepted=False, ineligible_reasons=['original_ICP_accepted']); bad_fallback['eligibility_checks']['original_ICP_accepted'] = False
    held = make_row(rejected, bad_fallback); audit_pair(1, 0, np.stack([C, C]), held, bad_fallback, metadata)
    after_holds = copy.deepcopy(fallback); after_holds.update(current_frame_id=3, previous_frame_id=2, current_sensor=sensor(metadata[3]), previous_sensor=sensor(metadata[2]), original_current_rgb_timestamp=metadata[3]['rgb_timestamp'])
    followed = make_row(rejected, after_holds, index=3); audit_pair(3, 0, np.stack([C, C, C, C @ F]), followed, after_holds, metadata)
    negative = {}
    def rejects(name, operation):
        try: operation()
        except (ValueError, KeyError, TypeError, IndexError): negative[name] = 'rejected_as_required'
        else: raise AssertionError('Negative control was accepted: ' + name)
    def mutate_row(key, value):
        edited = copy.deepcopy(positive); edited[key] = value
        return lambda: audit_pair(1, 0, np.stack([C, C @ T]), edited, fallback, metadata)
    for name, key, value in [('wrong_primary_reference_ID', 'primary_reference_frame_id', 1), ('wrong_adjacent_reference_ID', 'fallback_reference_frame_id', 9), ('brightness_not_reset', 'pair_initial_bias_intensity', .2), ('GT_initialization_flag', 'pair_initialized_from_GT', True), ('wrong_promotion', 'last_accepted_frame_id_after', 0), ('lost_primary_flags', 'primary_failure_flags', ['invented']), ('extra_absolute_pose', 'estimated_absolute_ICP_c2w', np.eye(4).tolist())]: rejects(name, mutate_row(key, value))
    for name, result in [('inverse_composition', C @ np.linalg.inv(T)), ('left_composition', T @ C)]:
        edited = copy.deepcopy(positive); edited['estimated_c2w'] = result.tolist()
        rejects(name, lambda e=edited, p=result: audit_pair(1, 0, np.stack([C, p]), e, fallback, metadata))
    for name, mutation in [('8D_extra_component', lambda p: p['iteration_history'][0]['proposed_delta_8d'].append(0.)), ('8D_NaN_component', lambda p: p['iteration_history'][0]['proposed_delta_8d'].__setitem__(0, float('nan'))), ('accepted_without_current_proposal', lambda p: p['iteration_history'][0].pop('proposed_delta_8d')), ('missing_pyramid_level', lambda p: p['levels'].pop()), ('wrong_trial_denominator', lambda p: p['iteration_history'][0]['line_search_trials'][0].__setitem__('same_correspondence_denominator', 31)), ('non_descending_trial', lambda p: p['iteration_history'][0]['line_search_trials'][0].__setitem__('objective', 11.))]:
        edited = copy.deepcopy(primary); mutation(edited); rejects(name, lambda p=edited: primary_steps(p))
    edited = copy.deepcopy(positive); edited['fallback']['current_sensor']['GT_c2w'] = np.eye(4).tolist(); rejects('GT_witness_metadata_injection', lambda: audit_pair(1, 0, np.stack([C, C @ T]), edited, fallback, metadata))
    fake = {'status': 'all_two_predeclared_CPU_fusion_rounds_completed', 'runs_started': 2, 'current_run': None, 'run_records': [], 'GPU_runs': 0, 'protocol_sha256': FUSION_PROTOCOL_SHA, 'source_sha256': protocol['source_sha256']}
    deadline = datetime.fromisoformat(protocol['deadline_beijing'])
    from datetime import timedelta
    for i, declared in enumerate(protocol['rounds']):
        fake['run_records'].append({**declared, 'returncode': 0, 'started_utc': (deadline - timedelta(minutes=10 - i * 2)).isoformat(), 'ended_utc': (deadline - timedelta(minutes=9 - i * 2)).isoformat()})
    terminal_gate(fake, protocol)
    for name, mutation in [('child_exit1', lambda b: b['run_records'][0].__setitem__('returncode', 1)), ('running_batch', lambda b: b.__setitem__('status', 'running')), ('missing_second_child', lambda b: b['run_records'].pop()), ('deadline_equal_launch', lambda b: b['run_records'][0].__setitem__('started_utc', protocol['deadline_beijing'])), ('overlapping_children', lambda b: b['run_records'][1].__setitem__('started_utc', b['run_records'][0]['started_utc'])), ('source_mismatch', lambda b: b.__setitem__('source_sha256', {}))]:
        edited = copy.deepcopy(fake); mutation(edited); rejects(name, lambda b=edited: terminal_gate(b, protocol))
    return {'status': 'passed_independent_synthetic_primary_fallback_hold_and_terminal_controls', 'positive_cases': 5, 'negative_controls': negative, 'actual_audits': 0, 'real_registration_runs': 0, 'real_children_started': 0, 'GPU_calls': 0,
            'limits': ['Synthetic ledger/trials are labelled controls, never actual result evidence.', 'Valid held Cref and Cprev poses are equal; reference-ID/candidate direction negatives are used instead of claiming those equal matrices should differ.']}


def prepare():
    PREP.mkdir(exist_ok=False); (PREP / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        require(sha(FUSION_PREP / 'protocol.json') == FUSION_PROTOCOL_SHA, 'Frozen fusion protocol')
        protocol = read(FUSION_PREP / 'protocol.json'); locks = {}
        def lock(path, expected=None):
            value = sha(path); require(expected is None or value == expected, 'Reviewed preparation/source lock'); locks[path.relative_to(ROOT).as_posix()] = value
        lock(FUSION_PREP / 'protocol.json', FUSION_PROTOCOL_SHA); lock(FUSION_PREP / 'root_protocol_review.json')
        review = read(FUSION_PREP / 'root_protocol_review.json'); require(review['status'] == 'passed_root_source_and_independent_CPU_review_authorized_exact_two_rounds' and review['protocol_sha256'] == FUSION_PROTOCOL_SHA, 'Actual preparation root review lineage')
        for name, value in HELPERS.items(): lock(ROOT / 'scripts' / name, value)
        for name, value in protocol['source_sha256'].items(): lock(ROOT / 'scripts' / name, value); lock(FUSION_PREP / 'source_snapshot' / name, value)
        for section in ['datasets', 'fallback_candidate_witnesses', 'old_full8D_references']:
            for value in protocol[section].values(): lock(ROOT / value['path'], value['sha256'])
        reuse = protocol['full_input_preflight_reuse']; lock(ROOT / reuse['protocol_path'], reuse['protocol_sha256']); lock(ROOT / reuse['independent_review_path'], reuse['independent_review_sha256'])
        old = protocol['independent_original_ICP_audit']; lock(ROOT / old['path'], old['sha256'])
        for name, value in review['additional_readonly_old_full_result_locks'].items(): lock(ROOT / name, value)
        lock(ROOT / 'outputs/rgbd-affine-icp-fallback-independent-cpu-review-002/metrics.json', review['independent_review_sha256'])
        compile(Path(__file__).read_text(encoding='utf-8'), str(Path(__file__)), 'exec')
        own_tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        imports = [ast.unparse(n) for n in ast.walk(own_tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        require(not any(any(bad in value for bad in ['track_rgbd', 'train_tum', 'tum_rgbd', 'torch', 'gsplat']) for value in imports), 'No hybrid/optimizer/loader/renderer import')
        checks = synthetic_controls(protocol); write(PREP / 'CPU_controls.json', checks)
        package = {'status': 'prepared_read_only_terminal_fusion_result_audit_no_actual_execution', 'audit_source_sha256': sha(Path(__file__)), 'fusion_protocol_sha256': FUSION_PROTOCOL_SHA,
            'batch_relative_path': BATCH.relative_to(ROOT).as_posix(), 'actual_output_relative_path': (BATCH / 'independent_affine_icp_fallback_result_audit.json').relative_to(ROOT).as_posix(),
            'root_source_review_relative_path': (PREP / 'root_source_review.json').relative_to(ROOT).as_posix(), 'root_source_review_expected_status': 'passed_root_fallback_result_audit_source_review',
            'source_and_lineage_file_locks': locks, 'CPU_controls_sha256': sha(PREP / 'CPU_controls.json'), 'reused_numerical_helpers': HELPERS,
            'expected_frames': 1297, 'expected_nonanchor_pairs': 1295, 'expected_scenes': ['desk', 'xyz'], 'actual_audits': 0, 'registration_runs': 0, 'GPU_calls': 0,
            'limits': ['Frozen e807 rawGT/trajectory formula and b549 saved8D-trial ledger explicitly reused; no new optimizer or third numerical formulation.', 'Persisted rank/objective consistency is checked; unsaved correspondences/Hessians are not recomputed.', 'No active progress or new result metrics are consumed in SOURCE/CPU preparation.']}
        write(PREP / 'protocol.json', package)
        require(all(sha(ROOT / p) == d for p, d in locks.items()) and not any(n in sys.modules for n in ('torch', 'gsplat')), 'Preparation unchanged-file and CPU boundary')
        print(json.dumps({'status': package['status'], 'audit_source_sha256': package['audit_source_sha256'], 'protocol_sha256': sha(PREP / 'protocol.json'), 'CPU_controls_sha256': package['CPU_controls_sha256'], 'negative_controls': len(checks['negative_controls']), 'actual_audits': 0, 'GPU_calls': 0}))
    except BaseException as error:
        write(PREP / 'failure.json', {'status': 'audit_preparation_failed_retained', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'actual_audits': 0, 'GPU_calls': 0}); raise


def audit(package_sha):
    require(sha(PREP / 'protocol.json') == package_sha, 'Exact separately reviewed audit package SHA')
    package = read(PREP / 'protocol.json'); require(package['audit_source_sha256'] == sha(Path(__file__)), 'Audit SOURCE frozen')
    reviewed = read(ROOT / package['root_source_review_relative_path'])
    require(reviewed['status'] == package['root_source_review_expected_status'] and reviewed['protocol_sha256'] == package_sha and reviewed['audit_source_sha256'] == package['audit_source_sha256'], 'Explicit actual audit source review')
    target = ROOT / package['actual_output_relative_path']; require(not target.exists(), 'Unique actual audit artifact')
    watched = {}
    def watch(path, expected=None):
        value = sha(path); require(expected is None or value == expected, 'Frozen consumed file ' + str(path)); watched[Path(path)] = value
    try:
        for path, expected in package['source_and_lineage_file_locks'].items(): watch(ROOT / path, expected)
        watch(Path(__file__), package['audit_source_sha256']); watch(PREP / 'protocol.json', package_sha); watch(PREP / 'source_script.py', package['audit_source_sha256']); watch(PREP / 'CPU_controls.json', package['CPU_controls_sha256']); watch(ROOT / package['root_source_review_relative_path'])
        protocol = read(FUSION_PREP / 'protocol.json')
        # metrics.json exists only after the driver has collected both children.
        # It is the first new-run artifact consumed, before any child/progress read.
        watch(BATCH / 'metrics.json'); batch = read(BATCH / 'metrics.json'); terminal_gate(batch, protocol)
        watch(BATCH / 'protocol_used.json'); require((BATCH / 'protocol_used.json').read_bytes() == (FUSION_PREP / 'protocol.json').read_bytes(), 'Actual batch protocol bytes')
        inputs = {}; truth = {}; cached = {}; sensor_files = 0
        for scene in ['desk', 'xyz']:
            entry = protocol['datasets'][scene]; w = read(ROOT / entry['path']); inputs[scene] = w
            for name, expected in w['index_sha256'].items(): watch(Path(w['dataset']) / name, expected)
            for row in w['raw_sensor_files']:
                for key in ['rgb_path', 'depth_path']: watch(Path(w['dataset']) / row[key]['path'], row[key]['sha256']); sensor_files += 1
            truth[scene] = metric_helper.truth_from_raw(w)
            witness = read(ROOT / protocol['fallback_candidate_witnesses'][scene]['path']); cached[scene] = witness
            require(witness['contains_GT_pose_or_absolute_ICP_pose'] is False and len(witness['rows']) == entry['frames'] - 1, 'Full frozen relative-only fallback payload')
        results = []; all_rows = all_pairs = all_steps = all_trials = 0
        for declared, record in zip(protocol['rounds'], batch['run_records']):
            folder = ROOT / 'outputs' / declared['run']; watch(folder / 'metrics.json', record['metrics_sha256']); watch(BATCH / (record['round_id'] + '.log'), record['log_sha256'])
            for path in folder.iterdir():
                if path.is_file(): watch(path)
            require((folder / 'protocol_used.json').read_bytes() == (FUSION_PREP / 'protocol.json').read_bytes(), 'Exact actual child protocol bytes')
            m = read(folder / 'metrics.json'); count = declared['configuration']['frames']; w = inputs[declared['scene']]; metadata = w['selected_sensor_metadata']
            require(m['status'] in ['completed', 'completed_with_final_pair_rejections'] and m['frames_processed'] == count and m['configuration'] == declared['configuration'] and m['source_sha256'] == protocol['source_sha256'] and m['protocol_sha256'] == FUSION_PROTOCOL_SHA and m['GPU_runs'] == 0, 'Complete actual frozen child result')
            require(m['GT_pose_used_in_optimizer'] is False and m['GT_pose_used_in_selection'] is False and m['fallback_uses_absolute_ICP_pose'] is False and m['GT_pose_evaluation_after_all_estimates_saved'] is True and m['ground_truth_scope'] == w['ground_truth_scope'], 'Honest association/final GT boundary')
            require(m['selected_sensor_metadata'] == metadata and m['all_frame_denominator'] == count and m['tracked_nonanchor_frame_denominator'] == count - 1, 'Complete metadata/all-frame and adjacent denominators')
            require(datetime.fromisoformat(record['started_utc']) <= datetime.fromisoformat(m['started_utc']) <= datetime.fromisoformat(m['first_pair_registration_start_utc']) < datetime.fromisoformat(protocol['deadline_beijing']) and datetime.fromisoformat(m['first_pair_registration_start_utc']) < datetime.fromisoformat(record['ended_utc']), 'Actual first registration guarded after child start and before cutoff')
            config = read(folder / 'configuration.json'); require(config == {**declared['configuration'], 'source_sha256': protocol['source_sha256'], 'protocol_sha256': FUSION_PROTOCOL_SHA, 'CPU_only': True}, 'Full configuration duplicate unchanged')
            rows = m['frame_metrics']; require(len(rows) == count and [r['frame_id'] for r in rows] == list(range(count)), 'Every actual requested row retained')
            poses = np.stack([proper_pose(r['estimated_c2w']) for r in rows]); require(np.array_equal(poses[0], np.eye(4)), 'Identity first camera')
            expected_anchor = {'frame_id': 0, 'local_index': 0, 'accepted': True, 'failure_flags': [], 'selection': 'fixed_first_camera_identity', 'primary_rejected': False, 'fallback_selected': False, 'reference_frame_id': 0, 'reference_gap_frames': 0, 'last_accepted_frame_id_after': 0, 'estimated_c2w': np.eye(4).tolist()}
            require(rows[0] == expected_anchor, 'Exact first-camera identity/flags anchor')
            manifest = read(folder / 'pair_manifest.json'); expected_files = {f'pairs/{i:04d}.json' for i in range(count)} | {f'candidates/{i:04d}.json' for i in range(1, count)}
            require(set(manifest) == expected_files and {str(path.relative_to(folder)).replace('\\', '/') for sub in ['pairs', 'candidates'] for path in (folder / sub).glob('*.json')} == expected_files, 'All immutable actual pair/candidate files, no omissions or extras')
            for rel, entry in manifest.items():
                watch(folder / rel, entry['sha256']); require((folder / rel).stat().st_size == entry['bytes'], 'Immutable manifest byte count')
            require(read(folder / 'pairs/0000.json') == rows[0], 'Anchor immutable equality')
            reference = 0; aggregate = Counter(); final_flags = Counter(); primary_flags = Counter(); selections = Counter([rows[0]['selection']]); primary_rejected = []; final_rejected = []; fallback_selected = []; max_step_error = 0.
            with (folder / 'iterations.jsonl').open(encoding='utf-8') as stream:
                for index, row in enumerate(rows[1:], 1):
                    require(read(folder / f'pairs/{index:04d}.json') == row and read(folder / f'candidates/{index:04d}.json') == {k: row[k] for k in CANDIDATE_KEYS}, 'Immutable pair/candidate equals complete metric row')
                    before = reference; reference, stats = audit_pair(index, reference, poses, row, cached[declared['scene']]['rows'][index - 1], metadata)
                    for key in ['steps', 'trials', 'accepted_updates']: aggregate[key] += stats.get(key, 0)
                    max_step_error = max(max_step_error, stats.get('independent_SciPy_SE3_max_abs_difference', 0.))
                    for item in row['primary']['iteration_history']:
                        line = stream.readline(); require(bool(line) and json.loads(line) == {'frame_id': index, 'primary_reference_frame_id': before, **item}, 'Every primary trial/history stream row equals immutable primary evidence')
                    final_flags.update(row['failure_flags']); primary_flags.update(row['primary']['failure_flags']); selections[row['selection']] += 1
                    if row['primary_rejected']: primary_rejected.append(index)
                    if row['failure_flags']: final_rejected.append(index)
                    if row['fallback_selected']: fallback_selected.append(index)
                require(stream.read().strip() == '', 'No extra primary iteration rows')
            require(len(final_rejected) == m['final_failure_frame_count'] == m['failure_frame_count'] == record['final_failure_frame_count'] and len(primary_rejected) == m['primary_failure_frame_count'] == record['primary_failure_frame_count'] and len(fallback_selected) == m['fallback_selected_frame_count'] == record['fallback_selected_frame_count'], 'Separate primary/fallback/final counts match actual batch')
            require(dict(final_flags) == m['final_failure_flag_counts'] and dict(primary_flags) == m['primary_failure_flag_counts'] and dict(selections) == m['selection_counts'] and m['status'] == ('completed' if not final_rejected else 'completed_with_final_pair_rejections'), 'All separate final/primary flag histograms and completed state exact')
            progress = read(folder / 'progress.json'); require(progress == {k: v for k, v in m.items() if k != 'frame_metrics'}, 'Now-terminal compact progress equals completed metrics')
            require(not (folder / 'failure.json').exists(), 'Successful terminal cannot hide failure artifact')
            with np.load(folder / 'trajectory_arrays.npz', allow_pickle=False) as saved:
                require(set(saved.files) == {'estimated_c2w', 'evaluation_only_anchored_GT_c2w', 'rgb_timestamps'} and np.array_equal(saved['estimated_c2w'], poses) and np.array_equal(saved['rgb_timestamps'], [r['rgb_timestamp'] for r in metadata]), 'Every actual pose/timestamp NPZ equals metric rows')
                gt_error = step_helper.close(saved['evaluation_only_anchored_GT_c2w'], truth[declared['scene']], 'Raw GT xyzw/first camera anchor', 3e-14)
            metrics = metric_helper.independent_metrics(poses, truth[declared['scene']]); differences = {}
            for key, value in metrics.items():
                a, b = np.asarray(value), np.asarray(m['trajectory'][key]); require(a.shape == b.shape and np.isfinite(b).all(), 'Exact complete trajectory metric shape/finiteness')
                differences[key] = float(np.max(np.abs(a - b))); require(differences[key] < (1e-7 if 'rotation' in key else 1e-11), 'Independent e807 raw/aligned/adjacent trajectory values')
            estimate_text = text_check(folder / 'trajectory_estimated.txt', metadata, poses)
            gt_text = text_check(folder / 'trajectory_groundtruth_anchored.txt', metadata, truth[declared['scene']])
            partial_text = text_check(folder / 'trajectory_estimated_partial.txt', metadata, poses)
            all_rows += count; all_pairs += count - 1; all_steps += aggregate['steps']; all_trials += aggregate['trials']
            results.append({'run': declared['run'], 'scene': declared['scene'], 'frames': count, 'pairs': count - 1, 'metrics_sha256': record['metrics_sha256'], 'primary_rejected_frame_ids': primary_rejected, 'fallback_selected_frame_ids': fallback_selected, 'final_rejected_frame_ids': final_rejected, 'primary_failure_flag_counts': dict(primary_flags), 'final_failure_flag_counts': dict(final_flags), 'selection_counts': dict(selections), 'last_final_accepted_SENSOR_reference_id': reference, 'step_ledger_counts': dict(aggregate), 'independent_SciPy_SE3_max_abs_difference': max_step_error, 'raw_GT_anchor_max_abs_difference': gt_error, 'metric_max_abs_differences': differences, 'independently_recomputed_scalar_metrics': {k: v for k, v in metrics.items() if not isinstance(v, list)}, 'estimated_trajectory_text': estimate_text, 'GT_trajectory_text': gt_text, 'terminal_partial_text': partial_text})
        require(all_rows == 1297 and all_pairs == 1295 and sensor_files == 2594, 'Both entire trajectories and every actual sensor PNG')
        require(all(sha(path) == value for path, value in watched.items()) and not any(n in sys.modules for n in ['torch', 'gsplat', 'track_rgbd_affine_icp_fallback', 'track_rgbd_icp', 'tum_rgbd']), 'Every consumed file unchanged; no solver/loader/GPU import')
        result = {'status': 'passed_independent_terminal_affine_ICP_fallback_result_audit', 'audit_source_sha256': sha(Path(__file__)), 'audit_package_protocol_sha256': package_sha, 'fusion_protocol_sha256': FUSION_PROTOCOL_SHA, 'metric_helper_sha256': HELPERS['audit_rgbd_photogeometric_results.py'], 'saved8D_step_helper_sha256': HELPERS['audit_rgbd_affine_real_results.py'], 'batch_metrics_sha256': sha(BATCH / 'metrics.json'), 'actual_rounds': 2, 'complete_frame_rows': all_rows, 'tracked_nonanchor_pairs': all_pairs, 'all_primary_saved_steps': all_steps, 'all_primary_saved_trials': all_trials, 'raw_sensor_PNGs_rehashed': sensor_files, 'consumed_files_rehashed_unchanged': len(watched), 'runs': results, 'GPU_calls': 0,
            'limits': ['Explicit frozen b549 saved8D-trial and e807 rawGT/trajectory math reuse; no optimizer rerun or third formula.', 'All ranks/robust objectives are checked as persisted consistency, not recreated from unexported correspondences.', 'Independent cached ICP and original8D evidence is reused exactly as locked SOURCE/input lineage, not new supervision.', 'Loader association reads GT timestamps/fields; reviewed matrix-value usage is final evaluation only. Flags alone do not prove isolation.', 'A final accepted fallback may coexist with primary rejection; both records and every held pose remain in all-frame results.', 'This full causal path comparison is not proof of brightness causality, restored earlier missing motion or complete SLAM.']}
        write(target, result)
        print(json.dumps({'status': result['status'], 'frames': all_rows, 'pairs': all_pairs, 'steps': all_steps, 'trials': all_trials, 'metrics_sha256': sha(target), 'GPU_calls': 0}))
    except BaseException as error:
        failure = BATCH / 'independent_affine_icp_fallback_result_audit_failure.json'
        if not failure.exists(): write(failure, {'status': 'actual_readonly_audit_failure_retained', 'audit_source_sha256': sha(Path(__file__)), 'audit_package_protocol_sha256': package_sha, 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'GPU_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--audit', action='store_true'); parser.add_argument('--protocol-sha')
    args = parser.parse_args(); require(args.audit == (args.protocol_sha is not None), 'Actual audit requires reviewed package SHA; preparation does not accept it')
    if args.prepare: prepare()
    else: audit(args.protocol_sha)


if __name__ == '__main__': main()
