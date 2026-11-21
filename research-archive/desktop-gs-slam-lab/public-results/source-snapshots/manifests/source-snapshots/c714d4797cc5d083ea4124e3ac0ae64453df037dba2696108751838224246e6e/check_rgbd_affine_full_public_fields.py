"""Independent JSON field-preservation check; no collector or solver imports.

This source is prepared for root review before execution. A later explicit
--execute writes one new audit JSON, never any existing experiment artifact.
It checks stored values, not GT geometry, registration, ranks or trajectories.
The 24-column mapping below is the public format specification, not a call to
the collector's compact_frames/compact_field_check implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import traceback


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = 'outputs/rgbd-affine-full-comparison-001'
PREP = 'outputs/rgbd-affine-full-cpu-preparation-001'
BATCH = 'outputs/rgbd-affine-full-validation-001'
ADDENDUM_PREP = 'outputs/rgbd-affine-full-6d-proposal-addendum-cpu-preparation-001'
SUMMARY_SHA = 'd6a5f9e304579e98b0b97912762eb6c7618273c89398bf3c95e09809e76635d3'
AUDIT_SHA = 'a9480a1a6a7134b8811dc8b500045419f1fbb1dbfaf4dbb41caae6f050e4afd5'
ADDENDUM_SHA = '1d75beecb78ed33a93c1072ddb93e6e494ccdc5a1f3b2ebb3f95887808642a00'
COMPLETION_SHA = '2afb16ccd758bb3de63709d2b5024200f15ae6cda441abe6417cb4211e4bccca'
COLLECTOR_SHA = '13e05091ee1c4f8be87c4f11fa88f7efda96f73e0780039c62d54b0965f72309'
PROTOCOL_SHA = 'ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a'
AUDITOR_SHA = '155bfe57fbf873d7717eac40d8ac9a18134ee7f7824eed92efab800d5577c690'
ADDENDUM_SOURCE_SHA = 'c5087f447fb833684096ea48dacdf93f25d9199b9a41355b8395c36a0058c8e3'
ADDENDUM_PROTOCOL_SHA = '76f4c95fca28268c9ff0948c3c546461a8f3462d242f876d94080447b1eb6ba1'
ADDENDUM_REVIEW_SHA = 'cba08e8344042bec0ea6a373fde3f623ec0374714bdc633bdfa0ff499b1e34ff'

ARMS = (
    ('desk', 6, 549, 'tum-desk-lastaccepted-6d-full549-001',
     '317cc482a07d462c89283cd50531552ac34edc326443d6e1074786512fa0cd71'),
    ('desk', 8, 549, 'tum-desk-lastaccepted-8d-full549-001',
     '52bf33985013dda1f654348aae2814aa1d5c797c2e0c2abf3ba782dac31a4262'),
    ('xyz', 6, 748, 'tum-xyz-lastaccepted-6d-full748-001',
     'd34b698dbac30400856f88c9f49f6d7c81ec4e632e463f1758fa1b43ca301fb6'),
    ('xyz', 8, 748, 'tum-xyz-lastaccepted-8d-full748-001',
     '312259b7fd1545b2ff30ba05146a8a92456cf0b8470616a70bea1924f9539f9f'),
)

# Sources: frame=original frame_metrics[i]; sensor=selected_sensor_metadata[i];
# stats=frame.final_statistics when present. Missing optional fields map to null.
# For aliases, key PRESENCE wins, including an explicit null: never truthiness.
FIELD_MAPPING = (
    ('frame_id', 'frame', 'frame_id'),
    ('rgb_index', 'sensor', 'rgb_index'),
    ('depth_index', 'sensor', 'depth_index'),
    ('gt_index', 'sensor', 'gt_index'),
    ('rgb_timestamp', 'sensor', 'rgb_timestamp'),
    ('accepted', 'frame', 'accepted'),
    ('failure_flags', 'frame', 'failure_flags'),
    ('reference_frame_id', 'frame', 'reference_frame_id'),
    ('reference_gap_frames', 'frame', 'reference_gap_frames'),
    ('last_accepted_frame_id_before', 'frame', 'last_accepted_frame_id_before'),
    ('last_accepted_frame_id_after', 'frame', 'last_accepted_frame_id_after'),
    ('estimated_c2w', 'frame', 'estimated_c2w'),
    ('candidate_reference_from_current', 'optional_frame', 'candidate_reference_from_current'),
    ('candidate_log_relative_gain', 'optional_frame', 'candidate_log_relative_gain'),
    ('candidate_relative_gain', 'optional_frame', 'candidate_relative_gain'),
    ('candidate_bias_intensity', 'optional_frame', 'candidate_bias_intensity'),
    ('final_rank', 'stats_alias', ('joint_rank', 'rank')),
    ('column_normalized_hessian_condition', 'optional_stats', 'column_normalized_hessian_condition'),
    ('plane_rmse_m', 'optional_stats', 'plane_rmse_m'),
    ('intensity_rmse', 'stats_alias', ('affine_corrected_intensity_rmse', 'intensity_rmse')),
    ('raw_translation_error_m', 'per_frame', 'per_frame_raw_translation_error_m'),
    ('SE3_ATE_translation_error_m', 'per_frame', 'per_frame_se3_aligned_ate_translation_error_m'),
    ('adjacent_RPE_translation_m', 'per_interval', 'per_interval_rpe_translation_error_m'),
    ('adjacent_RPE_rotation_deg', 'per_interval', 'per_interval_rpe_rotation_error_deg'),
)
SCALARS = (
    'raw_anchored_translation_rmse_m', 'se3_aligned_ate_translation_rmse_m',
    'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_deg',
    'raw_rotation_rmse_deg',
)


def require(condition, label):
    if not condition:
        raise ValueError(label)


def exact(actual, expected, label):
    """Typed JSON equality: preserve null/bool and float sign/round-trip bits."""
    require(type(actual) is type(expected), label + ': JSON type differs')
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), label + ': object keys differ')
        for key in expected:
            exact(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), label + ': list length differs')
        for i, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, f'{label}[{i}]')
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isfinite(expected), label + ': nonfinite float')
        require(actual.hex() == expected.hex(), label + ': stored float differs')
    else:
        require(actual == expected, label + ': stored value differs')


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'Duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def invalid_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


class Inputs:
    def __init__(self):
        self.watched = {}

    def path(self, relative):
        require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Relative lab path required')
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Existing file inside lab: ' + relative)
        return path

    def raw(self, relative, digest=None, size=None):
        path = self.path(relative)
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        require(digest is None or observed == digest, 'Bound SHA256: ' + relative)
        require(size is None or len(raw) == size, 'Bound byte count: ' + relative)
        key = path.relative_to(ROOT).as_posix()
        if key in self.watched:
            exact(self.watched[key], {'sha256': observed, 'bytes': len(raw)}, 'Repeated read unchanged: ' + key)
        self.watched[key] = {'sha256': observed, 'bytes': len(raw)}
        return raw

    def json(self, relative, digest=None, size=None):
        return json.loads(self.raw(relative, digest, size), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)

    def recheck(self):
        for relative, binding in list(self.watched.items()):
            self.raw(relative, binding['sha256'], binding['bytes'])


def expected_cell(kind, key, frame, sensor, trajectory, index):
    if kind == 'frame':
        return frame[key], 'present'
    if kind == 'sensor':
        return sensor[key], 'present'
    if kind == 'per_frame':
        return trajectory[key][index], 'present'
    if kind == 'per_interval':
        return (trajectory[key][index - 1], 'present') if index else (None, 'anchor_no_interval')
    values = frame if kind == 'optional_frame' else frame.get('final_statistics', {})
    require(isinstance(values, dict), 'Optional field container is an object')
    choices = key if kind == 'stats_alias' else (key,)
    for candidate in choices:
        if candidate in values:
            value = values[candidate]
            return value, candidate + (':explicit_null' if value is None else ':present')
    return None, 'absent_exported_as_null'


def inspect_frames(compact, metrics, dimension, n):
    columns = [entry[0] for entry in FIELD_MAPPING]
    exact(compact['columns'], columns, 'Public column order and every declared column')
    require(len(columns) == len(set(columns)) == 24, 'Exactly 24 distinct columns')
    require(compact['status'] == 'all_original_frames_retained' and compact['model_dimension'] == dimension,
            'Public frames status/dimension')
    require(compact['full_iteration_history_retained_in_original_metrics_and_immutable_pairs'] is True,
            'History retention declaration')
    exact(compact['RPE_row_convention'], 'row i stores interval i-1 -> i; anchor null', 'RPE convention')
    exact(compact['null_semantics'],
          'First anchor/no exported value/inapplicable6D nuisance; never replaced with zero or a selected best frame.',
          'Null convention')
    rows = metrics['frame_metrics']; sensors = metrics['selected_sensor_metadata']; trajectory = metrics['trajectory']
    require(len(rows) == len(sensors) == len(compact['rows']) == n, 'All original and compact rows retained')
    for key in ('per_frame_raw_translation_error_m', 'per_frame_se3_aligned_ate_translation_error_m'):
        require(len(trajectory[key]) == n, 'Complete per-frame metric array: ' + key)
    for key in ('per_interval_rpe_translation_error_m', 'per_interval_rpe_rotation_error_deg'):
        require(len(trajectory[key]) == n - 1, 'Complete interval metric array: ' + key)
    states = {column: Counter() for column in columns}; nulls = Counter(); flags = Counter(); gaps = Counter()
    rejected = []; segments = []; last_accepted = 0; max_gap = 0
    for index, (frame, sensor, exported) in enumerate(zip(rows, sensors, compact['rows'])):
        require(type(exported) is list and len(exported) == 24, 'No missing compact cell')
        exact(frame['frame_id'], index, 'Original frame order')
        exact(sensor['frame_id'], index, 'Original sensor order')
        for column_index, (column, kind, key) in enumerate(FIELD_MAPPING):
            expected, state = expected_cell(kind, key, frame, sensor, trajectory, index)
            exact(exported[column_index], expected, f'frame {index} column {column}')
            states[column][state] += 1
            nulls[column] += exported[column_index] is None
        require(type(frame['accepted']) is bool and type(frame['failure_flags']) is list,
                'Accepted boolean and failure list preserved')
        require(frame['accepted'] == (len(frame['failure_flags']) == 0), 'Original single-route decision semantics')
        flags.update(frame['failure_flags'])
        if index == 0:
            require(frame['accepted'] and frame['reference_frame_id'] == frame['reference_gap_frames'] == 0,
                    'Explicit first-camera anchor')
            continue
        require('candidate_reference_from_current' in frame, 'Nonanchor candidate field must exist, including null')
        exact(frame['reference_frame_id'], last_accepted, 'Held reference before current row')
        exact(frame['last_accepted_frame_id_before'], last_accepted, 'Before reference identity')
        exact(frame['reference_gap_frames'], index - last_accepted, 'Stored reference gap')
        exact(frame['source_sensor_metadata'], sensor, 'Full current metadata, beyond exported four fields')
        exact(frame['reference_sensor_metadata'], sensors[last_accepted], 'Full held-reference metadata')
        gaps[str(frame['reference_gap_frames'])] += 1
        max_gap = max(max_gap, frame['reference_gap_frames'])
        if frame['accepted']:
            last_accepted = index
        else:
            rejected.append(index)
            exact(frame['estimated_c2w'], rows[index - 1]['estimated_c2w'], 'Rejected prefix copied exactly')
            if not segments or segments[-1]['last_frame_id'] + 1 != index:
                segments.append({'first_frame_id': index, 'last_frame_id': index, 'frames': 1,
                                 'flags': sorted(set(frame['failure_flags'])), 'reference_ids': [frame['reference_frame_id']]})
            else:
                segment = segments[-1]; segment['last_frame_id'] = index; segment['frames'] += 1
                segment['flags'] = sorted(set(segment['flags']).union(frame['failure_flags']))
                segment['reference_ids'] = sorted(set(segment['reference_ids']).union([frame['reference_frame_id']]))
        exact(frame['last_accepted_frame_id_after'], last_accepted, 'After reference identity')
    exact(dict(flags), metrics['failure_flag_counts'], 'All original flag occurrences')
    exact(dict(gaps), metrics['reference_gap_counts'], 'All reference gaps')
    exact(len(rejected), metrics['failure_frame_count'], 'Original failure count')
    exact(last_accepted, metrics['last_accepted_reference_id'], 'Final held reference')
    return {'rows': n, 'cells_checked': n * 24, 'column_source_states': {k: dict(v) for k, v in states.items()},
            'null_cells_by_column': dict(nulls), 'failure_flags': dict(flags), 'rejected_ids': rejected,
            'gap_counts': dict(gaps), 'maximum_gap': max_gap, 'last_reference': last_accepted,
            'segments': segments, 'longest_segment': max((x['frames'] for x in segments), default=0)}


def run_check(inputs):
    source_bytes = inputs.raw('scripts/check_rgbd_affine_full_public_fields.py')
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    summary = inputs.json(PUBLIC + '/summary.json', SUMMARY_SHA)
    completion = inputs.json(PUBLIC + '/completion.json', COMPLETION_SHA)
    audit = inputs.json(BATCH + '/independent_full_affine_result_audit.json', AUDIT_SHA)
    addendum = inputs.json('outputs/rgbd-affine-full-6d-proposal-addendum-001/metrics.json', ADDENDUM_SHA)
    protocol = inputs.json(PREP + '/protocol.json', PROTOCOL_SHA)
    batch = inputs.json(BATCH + '/metrics.json', summary['batch_metrics_sha256'])
    require(summary['status'] == 'completed_after_independent_full_result_audit' and
            audit['status'] == 'passed_independent_full_input_reference_step_and_trajectory_audit' and
            addendum['status'] == 'passed_current_row_6d_proposal_field_addendum', 'Three actual terminal bindings')
    exact(summary['audit_sha256'], AUDIT_SHA, 'Summary old audit binding')
    exact(summary['audit_source_sha256'], AUDITOR_SHA, 'Summary auditor source')
    exact(audit['audit_source_sha256'], AUDITOR_SHA, 'Original auditor source')
    exact(audit['protocol_sha256'], PROTOCOL_SHA, 'Original audit protocol')
    exact(summary['protocol_sha256'], PROTOCOL_SHA, 'Summary protocol')
    exact(summary['source_sha256'], COLLECTOR_SHA, 'Frozen collector identity')
    exact(summary['complete_new_frame_rows'], 2594, 'Top-level all-frame denominator')
    exact(summary['tracked_new_pairs'], 2590, 'Top-level tracked denominator')
    exact(summary['CPU_round_returncodes'], [0, 0, 0, 0], 'Actual four returncodes')
    require(summary['GPU_calls'] == audit['GPU_calls'] == addendum['GPU_calls'] == 0, 'CPU-only artifacts')
    exact(addendum['original_actual_audit_sha256'], AUDIT_SHA, 'Addendum original audit')
    exact(addendum['original_summary_sha256'], SUMMARY_SHA, 'Addendum frozen summary')
    exact(addendum['original_frozen_auditor_source_sha256'], AUDITOR_SHA, 'Addendum old auditor')
    exact(addendum['source_sha256'], ADDENDUM_SOURCE_SHA, 'Addendum new source')
    exact(addendum['protocol_sha256'], ADDENDUM_PROTOCOL_SHA, 'Addendum protocol binding')
    exact(addendum['root_review_sha256'], ADDENDUM_REVIEW_SHA, 'Addendum root review')
    require(addendum['all_input_bytes_before_after_unchanged'] is True and
            addendum['original_audit_and_summary_unchanged'] is True and addendum['new_trajectory_metrics'] == 0,
            'Addendum leaves old data and metrics intact')
    inputs.raw('scripts/audit_rgbd_full_6d_proposal_addendum.py', ADDENDUM_SOURCE_SHA)
    inputs.json(ADDENDUM_PREP + '/protocol.json', ADDENDUM_PROTOCOL_SHA)
    add_review = inputs.json(ADDENDUM_PREP + '/root_source_review.json', ADDENDUM_REVIEW_SHA)
    require('passed' in add_review['status'], 'Actual addendum source review passed')
    inputs.raw('scripts/audit_rgbd_affine_full_results.py', AUDITOR_SHA)
    inputs.raw('scripts/summarize_rgbd_affine_full_results.py', COLLECTOR_SHA)
    inputs.raw('scripts/audit_rgbd_photogeometric_results.py', audit['metric_helper_sha256'])
    inputs.raw('scripts/audit_rgbd_affine_real_results.py', audit['saved8D_step_helper_sha256'])
    exact(summary['metric_helper_sha256'], audit['metric_helper_sha256'], 'Stored metric helper identity')
    exact(summary['saved8D_step_helper_sha256'], audit['saved8D_step_helper_sha256'], 'Stored step helper identity')
    inputs.raw(PREP + '/root_protocol_review.json', audit['root_review_sha256'])
    inputs.raw(PREP + '/root-independent-full-preflight-review.json', audit['independent_preflight_sha256'])
    exact(summary['frozen_pipeline_source_sha256'], protocol['source_sha256'], 'All current original source bindings')
    for name, digest in protocol['source_sha256'].items():
        inputs.raw('scripts/' + name, digest)
        inputs.raw(PREP + '/source_snapshot/' + name, digest)

    expected_files = {'summary.json', 'source_script.py', 'input_hashes.json'}
    expected_files.update(f'input-hashes-{i:02d}.json' for i in range(6))
    expected_files.update(f'{scene}-{dim}d-{suffix}.json' for scene, dim, *_ in ARMS for suffix in ('frames', 'summary'))
    require(len(expected_files) == 17 and set(completion['files']) == expected_files, 'All 17 completion files exactly')
    require(completion['status'] == 'completed' and completion['source_sha256'] == COLLECTOR_SHA and
            completion['GPU_calls'] == 0, 'Original completion binding')
    for name, binding in completion['files'].items():
        inputs.raw(PUBLIC + '/' + name, binding['sha256'], binding['bytes'])
    input_manifest = inputs.json(PUBLIC + '/input_hashes.json', summary['input_hash_manifest']['sha256'])
    exact(summary['input_hash_manifest']['path'], 'input_hashes.json', 'Original input manifest path')
    require(input_manifest['all_entries_retained'] is True and len(input_manifest['pages']) == 6, 'All six original input pages')
    original_input_hashes = {}
    for i, page in enumerate(input_manifest['pages']):
        exact(page['path'], f'input-hashes-{i:02d}.json', 'Original input page order')
        data = inputs.json(PUBLIC + '/' + page['path'], page['sha256'], page['bytes'])
        require(len(data) == page['entries'] and not set(data).intersection(original_input_hashes), 'Unique original input hash entries')
        original_input_hashes.update(data)
    exact(len(original_input_hashes), input_manifest['total_entries'], 'Complete original input manifest count')
    exact(len(original_input_hashes), summary['consumed_files_rehashed_unchanged'], 'Original collector count declaration')

    require(batch['status'] == 'all_four_predeclared_CPU_full_sequence_rounds_completed' and batch['runs_started'] == 4,
            'Actual original batch completed')
    exact(batch['protocol_sha256'], PROTOCOL_SHA, 'Batch protocol')
    exact(audit['batch_metrics_sha256'], summary['batch_metrics_sha256'], 'Audit and summary bind same batch')
    require(len(batch['run_records']) == len(audit['runs']) == len(protocol['rounds']) == 4, 'Four records in every original ledger')
    exact(set(summary['scene_results']), {'desk', 'xyz'}, 'Both scenes retained')
    reports = []; add_runs = {record['run']: record for record in addendum['runs']}
    for arm_index, (scene, dimension, n, run, metrics_sha) in enumerate(ARMS):
        key = f'{scene}-{dimension}d'
        declared = protocol['rounds'][arm_index]; terminal = batch['run_records'][arm_index]; checked = audit['runs'][arm_index]
        require(declared['run'] == terminal['run'] == checked['run'] == run, 'Fixed arm order')
        exact(terminal['metrics_sha256'], metrics_sha, 'Original actual metrics bound')
        exact(checked['metrics_sha256'], metrics_sha, 'Independent audit original metrics bound')
        require(terminal['returncode'] == 0 and declared['model_dimension'] == dimension and declared['scene'] == scene,
                'Declared model and actual completion')
        exact(terminal['configuration'], declared['configuration'], 'Actual terminal declared configuration')
        m = inputs.json('outputs/' + run + '/metrics.json', metrics_sha)
        require(m['frames_processed'] == n and m['all_frame_denominator'] == n and
                m['tracked_nonanchor_frame_denominator'] == n - 1 and
                m['GT_pose_used_in_optimizer'] is False and m['GT_pose_evaluation_after_all_estimates_saved'] is True,
                'Original full denominators and declared GT scope')
        exact(m['source_sha256'], protocol['source_sha256'], 'Original run source bindings')
        exact(m['configuration'], declared['configuration'], 'Original declared configuration')
        witness_binding = protocol['datasets'][scene]
        witness = inputs.json(witness_binding['path'], witness_binding['sha256'])
        exact(m['selected_sensor_metadata'], witness['selected_sensor_metadata'], 'All sensor metadata bound to preparation')
        exact(m['input_array_hashes'], witness['input_array_hashes'], 'Original sensor array hash identity')
        exact(summary['scene_results'][scene]['association'], witness['association'], 'Public original association declaration')
        compact = inputs.json(PUBLIC + '/' + key + '-frames.json')
        counts = inspect_frames(compact, m, dimension, n)
        arm = inputs.json(PUBLIC + '/' + key + '-summary.json')
        top_arms = summary['scene_results'][scene]['arms']
        require(len(top_arms) == 2, 'Both original model arms retained')
        exact(top_arms[0 if dimension == 6 else 1], arm, 'Top summary arm equals standalone existing summary')
        expected_arm = {
            'run': run, 'model_dimension': dimension, 'metrics_sha256': metrics_sha,
            'configuration': declared['configuration'], 'status': m['status'], 'frames': n,
            'tracked_pairs': n - 1, 'failure_frame_count': len(counts['rejected_ids']),
            'all_rejected_frame_ids': counts['rejected_ids'], 'failure_flag_counts': counts['failure_flags'],
            'rejection_segments': counts['segments'], 'longest_rejection_segment': counts['longest_segment'],
            'reference_gap_counts': counts['gap_counts'], 'maximum_reference_gap_frames': counts['maximum_gap'],
            'last_accepted_reference_id': counts['last_reference'],
            'trajectory_scalars': {name: m['trajectory'][name] for name in SCALARS},
            'endpoint_raw_translation_m': checked['endpoint_raw_translation_m'],
            'alignment_scale': m['trajectory']['alignment_scale'], 'observed_wall_seconds_including_IO': m['observed_wall_seconds'],
            'started_utc': terminal['started_utc'], 'ended_utc': terminal['ended_utc'], 'returncode': terminal['returncode'],
            'all_frame_file': key + '-frames.json',
            'all_frame_file_sha256': completion['files'][key + '-frames.json']['sha256'],
            'audited_saved_step_counts': checked['all_pair_step_checks'],
        }
        final_frame = m['frame_metrics'][-1]; final_sensor = m['selected_sensor_metadata'][-1]
        exact(m['last_completed_frame_id'], n - 1, 'Original terminal last processed frame')
        exact(m['last_pair_accepted'], final_frame['accepted'], 'Original terminal last pair decision')
        exact(m['last_pair_flags'], final_frame['failure_flags'], 'Original terminal last pair flags')
        expected_arm['last_camera'] = {
            'frame_id': final_frame['frame_id'], 'rgb_timestamp': final_sensor['rgb_timestamp'],
            'accepted': final_frame['accepted'], 'failure_flags': final_frame['failure_flags'],
            'reference_frame_id': final_frame['reference_frame_id'],
            'last_accepted_frame_id_after': final_frame['last_accepted_frame_id_after'],
            'estimated_c2w': final_frame['estimated_c2w'],
            'candidate_reference_from_current': final_frame.get('candidate_reference_from_current'),
            'endpoint_raw_translation_m': checked['endpoint_raw_translation_m'],
            'rejected_endpoint_policy': 'If rejected, this is the retained preceding estimate at the final sensor timestamp, not the last accepted sensor timestamp or a removed frame.',
        }
        exact(arm, expected_arm, 'Every public arm summary field and endpoint')
        for public_key, check_key in (('all_rejected_frame_ids', 'all_rejected_frame_ids'),
                                      ('failure_frame_count', 'failure_frame_count'),
                                      ('failure_flag_counts', 'failure_flag_counts'),
                                      ('maximum_reference_gap_frames', 'maximum_reference_gap_frames'),
                                      ('last_accepted_reference_id', 'last_accepted_reference_id')):
            exact(arm[public_key], checked[check_key], 'Existing audit retained field ' + public_key)
        endpoint_difference_m = abs(checked['endpoint_raw_translation_m'] - m['trajectory']['per_frame_raw_translation_error_m'][-1])
        require(endpoint_difference_m <= 1e-11, 'Audited endpoint versus stored final raw error; metres only')
        if dimension == 6:
            require(run in add_runs and add_runs[run]['metrics_sha256'] == metrics_sha, 'Both original 6D metrics bind passed addendum')
            require(add_runs[run]['all_current_proposed_and_accepted_fields_present_finite6'] is True,
                    'Actual current-row 6D proposal guard passed')
        reports.append({'run': run, 'scene': scene, 'model_dimension': dimension, 'metrics_sha256': metrics_sha,
                        **counts, 'final_sensor_metadata': final_sensor, 'last_camera': arm['last_camera'],
                        'stored_endpoint_vs_audited_endpoint_difference_m': endpoint_difference_m,
                        'all_public_scalar_values_exact_original': True, 'standalone_and_top_summary_exact': True})

    # Only cross-check entries actually consumed here. Other original raw PNG /
    # pair hashes are preserved in the six pages, not falsely called rehashed.
    original_entries_rechecked = 0
    for relative, binding in inputs.watched.items():
        if relative in original_input_hashes:
            exact(binding['sha256'], original_input_hashes[relative], 'Original ledger consumed input: ' + relative)
            original_entries_rechecked += 1
    require(sum(report['rows'] for report in reports) == 2594 and
            sum(report['cells_checked'] for report in reports) == 62256, 'All 2594 rows and 62256 typed cells')
    inputs.recheck()
    return {'status': 'passed_independent_full_public_field_preservation_check',
            'source_sha256': source_digest,
            'summary_sha256': SUMMARY_SHA, 'original_audit_sha256': AUDIT_SHA, 'proposal_addendum_sha256': ADDENDUM_SHA,
            'completion_sha256': COMPLETION_SHA, 'completion_files_checked': 17,
            'frames_checked': 2594, 'tracked_pairs_retained': 2590, 'columns_checked_per_frame': 24,
            'typed_cells_checked': 62256, 'field_mapping': FIELD_MAPPING, 'runs': reports,
            'original_manifest_entries_preserved': len(original_input_hashes),
            'original_manifest_consumed_entries_rehashed_here': original_entries_rechecked,
            'files_read_and_rehashed_unchanged': inputs.watched, 'GPU_calls': 0, 'registration_calls': 0,
            'new_GT_metric_calculations': 0,
            'limits': ['Independent serialization/field audit, not another GT trajectory or solver implementation.',
                       'All stored scalars/per-frame errors are compared to original arrays; none are refitted.',
                       'Original audit and supplementary guard are explicitly bound, not rerun or replaced.',
                       'Missing optional fields and explicit nulls have separate source-state counts; exported null is never zero.',
                       'Only consumed input-ledger entries are rehashed here; other historical entries remain preserved.']}


def write_new(path, value):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    require(len(data) <= 1024 * 1024, 'New field-audit JSON must remain at most 1 MiB; no truncation')
    with path.open('xb') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run only after independent source review')
    parser.add_argument('--run', default='rgbd-affine-full-public-field-audit-001')
    args = parser.parse_args()
    require(args.execute, 'Source prepared only; explicit --execute is required')
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run) is not None, 'Safe new output name')
    output = ROOT / 'outputs' / args.run
    output.mkdir(exist_ok=False)
    (output / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    inputs = Inputs()
    try:
        result = run_check(inputs)
        write_new(output / 'metrics.json', result)
        print(json.dumps({'status': result['status'], 'run': args.run,
                          'metrics_sha256': hashlib.sha256((output / 'metrics.json').read_bytes()).hexdigest(),
                          'frames': result['frames_checked'], 'typed_cells': result['typed_cells_checked']}))
    except BaseException as error:
        write_new(output / 'failure.json', {'status': 'independent_public_field_audit_failed',
                  'exception_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                  'consumed_inputs_before_failure': inputs.watched, 'old_artifacts_modified': False, 'GPU_calls': 0})
        raise


if __name__ == '__main__':
    main()
