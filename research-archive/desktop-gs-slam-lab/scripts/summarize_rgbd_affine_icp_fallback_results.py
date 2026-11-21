"""Stdlib-only audited hybrid export; copies stored values, never fits metrics.

Prepared for root source review. Only explicit --execute may collect.
Public pages preserve every final primary/fallback candidate and frame field.
Primary iteration_history stays in immutable local evidence with count/hash;
no trial stream is claimed to be in the public final-candidate subset.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREP = 'outputs/rgbd-affine-icp-fallback-cpu-preparation-001'
BATCH = 'outputs/rgbd-affine-icp-fallback-validation-001'
AUDIT_PREP = 'outputs/rgbd-affine-icp-fallback-result-audit-cpu-preparation-001'
OLD_FULL = 'outputs/rgbd-affine-full-comparison-001'
OLD_ICP = 'outputs/tum-online-v4-full-sequence-recovery-summary-001'
PROTOCOL_SHA = '358a418359ef4f2a78c9ad73ac78a7fa69582457843d23cb28cd142e018bf077'
AUDIT_SHA = '69b3856ca466d079c3dc485a6327d668aa88067d161b534b8f4d9c899da87715'
AUDIT_SOURCE_SHA = 'f9536e952922fc13ce9702725c064ae2c0e31b04574b4973e5d13f9d4e855096'
AUDIT_PACKAGE_SHA = '08650f8fc7f93a080b0ebb335a7b2c671f1c3d77639d64c269699b9514ae4a92'
AUDIT_REVIEW_SHA = '97271860b46f186b1fe9082cf23b831c228cf462860b597dd0487c72fdb7946f'
BATCH_SHA = '51f51ab169853f26384e8678e3deae436924c10f142e5d0129c2904a0d39fa07'
OLD_FULL_SHA = 'd6a5f9e304579e98b0b97912762eb6c7618273c89398bf3c95e09809e76635d3'
OLD_FULL_AUDIT_SHA = 'a9480a1a6a7134b8811dc8b500045419f1fbb1dbfaf4dbb41caae6f050e4afd5'
ADDENDUM_SHA = '1d75beecb78ed33a93c1072ddb93e6e494ccdc5a1f3b2ebb3f95887808642a00'
OLD_ICP_SHA = '97b65b413abd5b18eb8677a20ff7d1eef422205289d929d373f4ea5edd10c81d'
OLD_ICP_AUDIT_SHA = '70f65dfced84eabf1ec313e93a908062d565a19401b1ae4c49b0d4d71801417e'
ARMS = (
    ('desk', 549, 'tum-desk-affine-icp-fallback-full549-001', 'f4b6cef8240f96054163dc3ac4f354cc00bdf98a5f484569254b3b6b5bb35b3d'),
    ('xyz', 748, 'tum-xyz-affine-icp-fallback-full748-001', 'c60f3b5827bccbee82996cc54fcafa09be6c8c91107c8af133ebe44ec933569c'),
)
MAX_BYTES = 1024 * 1024
MAX_ROWS = 200
SCALARS = ('raw_anchored_translation_rmse_m', 'se3_aligned_ate_translation_rmse_m',
           'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_deg', 'raw_rotation_rmse_deg')
ERROR_ARRAYS = {
    'raw_translation_error_m': ('per_frame_raw_translation_error_m', False),
    'SE3_ATE_translation_error_m': ('per_frame_se3_aligned_ate_translation_error_m', False),
    'adjacent_RPE_translation_m': ('per_interval_rpe_translation_error_m', True),
    'adjacent_RPE_rotation_deg': ('per_interval_rpe_rotation_error_deg', True),
}
CANDIDATE_FIELDS = ('frame_id', 'primary_reference_frame_id', 'primary', 'fallback', 'primary_current_sensor',
                    'primary_reference_sensor', 'pair_initial_transform', 'pair_initial_log_relative_gain',
                    'pair_initial_bias_intensity', 'brightness_accumulated_across_pairs')


def require(condition, label):
    if not condition:
        raise ValueError(label)


def encoded(value):
    """Named Python JSON canonical encoding, not a claim of RFC8785/JCS."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def equal(left, right, label):
    require(encoded(left) == encoded(right), label + ': typed persisted JSON differs')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


class Inputs:
    def __init__(self):
        self.files = {}

    def raw(self, relative, expected=None, size=None):
        require(not Path(relative).is_absolute(), 'Lab-relative path required')
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Existing input in lab: ' + relative)
        raw = path.read_bytes(); sha = digest(raw); key = path.relative_to(ROOT).as_posix()
        require(expected is None or sha == expected, 'Frozen input SHA: ' + relative)
        require(size is None or len(raw) == size, 'Frozen input bytes: ' + relative)
        binding = {'sha256': sha, 'bytes': len(raw)}
        if key in self.files:
            equal(binding, self.files[key], 'Input changed on repeated read')
        self.files[key] = binding
        return raw

    def json(self, relative, expected=None, size=None):
        return json.loads(self.raw(relative, expected, size), object_pairs_hook=unique_object, parse_constant=reject_constant)

    def recheck(self):
        for path, binding in list(self.files.items()):
            self.raw(path, binding['sha256'], binding['bytes'])


def gate(inputs, supplied_audit_sha):
    require(supplied_audit_sha == AUDIT_SHA, 'Only the exact completed actual audit is admissible')
    protocol = inputs.json(PREP + '/protocol.json', PROTOCOL_SHA)
    package = inputs.json(AUDIT_PREP + '/protocol.json', AUDIT_PACKAGE_SHA)
    review = inputs.json(AUDIT_PREP + '/root_source_review.json', AUDIT_REVIEW_SHA)
    audit = inputs.json(BATCH + '/independent_affine_icp_fallback_result_audit.json', AUDIT_SHA)
    batch = inputs.json(BATCH + '/metrics.json', BATCH_SHA)
    require(audit['status'] == 'passed_independent_terminal_affine_ICP_fallback_result_audit' and
            audit['audit_source_sha256'] == AUDIT_SOURCE_SHA and audit['audit_package_protocol_sha256'] == AUDIT_PACKAGE_SHA and
            audit['fusion_protocol_sha256'] == PROTOCOL_SHA and audit['batch_metrics_sha256'] == BATCH_SHA and
            audit['actual_rounds'] == 2 and audit['complete_frame_rows'] == 1297 and audit['tracked_nonanchor_pairs'] == 1295 and
            audit['all_primary_saved_steps'] == 33900 and audit['all_primary_saved_trials'] == 35746 and audit['GPU_calls'] == 0,
            'Completed actual independent 1297-row / 1295-pair audit, never source-only evidence')
    require(review['status'] == 'passed_root_fallback_result_audit_source_review' and
            review['protocol_sha256'] == AUDIT_PACKAGE_SHA and review['audit_source_sha256'] == AUDIT_SOURCE_SHA,
            'Exact root-reviewed auditor lineage')
    inputs.raw('scripts/audit_rgbd_affine_icp_fallback_results.py', AUDIT_SOURCE_SHA)
    inputs.raw(AUDIT_PREP + '/source_script.py', AUDIT_SOURCE_SHA)
    inputs.raw(AUDIT_PREP + '/CPU_controls.json', package['CPU_controls_sha256'])
    for path, sha in package['source_and_lineage_file_locks'].items():
        inputs.raw(path, sha)
    for name, sha in protocol['source_sha256'].items():
        inputs.raw('scripts/' + name, sha)
        inputs.raw(PREP + '/source_snapshot/' + name, sha)
    require(batch['status'] == 'all_two_predeclared_CPU_fusion_rounds_completed' and batch['runs_started'] == 2 and
            batch['current_run'] is None and len(batch['run_records']) == 2 and batch['GPU_runs'] == 0 and
            len(audit['runs']) == len(protocol['rounds']) == 2, 'Only both actual terminal children allow collection')
    equal(batch['source_sha256'], protocol['source_sha256'], 'Batch frozen source set')
    require(batch['protocol_sha256'] == PROTOCOL_SHA, 'Batch protocol exact')
    require(inputs.raw(BATCH + '/protocol_used.json') == inputs.raw(PREP + '/protocol.json'), 'Original protocol exact bytes')
    previous_end = None; ready = []
    for i, (scene, n, run, sha) in enumerate(ARMS):
        declared, terminal, checked = protocol['rounds'][i], batch['run_records'][i], audit['runs'][i]
        require(declared['run'] == terminal['run'] == checked['run'] == run and checked['metrics_sha256'] == terminal['metrics_sha256'] == sha,
                'Exact predeclared run and metrics bytes')
        for key in ('order', 'round_id', 'scene', 'configuration'):
            equal(terminal[key], declared[key], 'Original batch declared ' + key)
        require(terminal['returncode'] == 0 and checked['frames'] == terminal['frames_processed'] == n,
                'Genuine exit zero and every frame')
        start, end = datetime.fromisoformat(terminal['started_utc']), datetime.fromisoformat(terminal['ended_utc'])
        require(start < end and start < datetime.fromisoformat(protocol['deadline_beijing']) and
                (previous_end is None or previous_end <= start), 'Serial launches before fixed deadline')
        previous_end = end
        inputs.raw(BATCH + '/' + terminal['round_id'] + '.log', terminal['log_sha256'])
        metrics = inputs.json('outputs/' + run + '/metrics.json', sha)
        require(metrics['status'] in ('completed', 'completed_with_final_pair_rejections') and
                metrics['frames_processed'] == metrics['all_frame_denominator'] == n and
                metrics['tracked_nonanchor_frame_denominator'] == n - 1 and len(metrics['frame_metrics']) == n,
                'Original terminal all-frame denominators')
        equal(metrics['configuration'], declared['configuration'], 'Child fixed configuration')
        equal(metrics['source_sha256'], protocol['source_sha256'], 'Child frozen sources')
        require(metrics['protocol_sha256'] == PROTOCOL_SHA and metrics['GT_pose_used_in_optimizer'] is False and
                metrics['GT_pose_used_in_selection'] is False and metrics['fallback_uses_absolute_ICP_pose'] is False and
                metrics['GT_pose_evaluation_after_all_estimates_saved'] is True and metrics['GPU_runs'] == 0,
                'Recorded sensor-only optimization / selection and posthoc evaluation boundary')
        witness_spec = protocol['datasets'][scene]; witness = inputs.json(witness_spec['path'], witness_spec['sha256'])
        equal(metrics['selected_sensor_metadata'], witness['selected_sensor_metadata'], 'All declared sensor identities')
        fallback_spec = protocol['fallback_candidate_witnesses'][scene]
        fallback = inputs.json(fallback_spec['path'], fallback_spec['sha256'])
        require(fallback['contains_GT_pose_or_absolute_ICP_pose'] is False and len(fallback['rows']) == n - 1,
                'Frozen relative-only fallback witness')
        ready.append((scene, n, run, sha, terminal, checked, metrics, witness, fallback))
    return protocol, batch, audit, ready


def old_references(inputs, protocol):
    full = inputs.json(OLD_FULL + '/summary.json', OLD_FULL_SHA)
    audit = inputs.json('outputs/rgbd-affine-full-validation-001/independent_full_affine_result_audit.json', OLD_FULL_AUDIT_SHA)
    add = inputs.json('outputs/rgbd-affine-full-6d-proposal-addendum-001/metrics.json', ADDENDUM_SHA)
    require(full['status'] == 'completed_after_independent_full_result_audit' and full['audit_sha256'] == OLD_FULL_AUDIT_SHA and
            audit['status'] == 'passed_independent_full_input_reference_step_and_trajectory_audit' and
            add['status'] == 'passed_current_row_6d_proposal_field_addendum' and add['original_summary_sha256'] == OLD_FULL_SHA and
            add['original_actual_audit_sha256'] == OLD_FULL_AUDIT_SHA, 'Original full model evidence retained, addendum is separate 6D field scope')
    old_icp = inputs.json(OLD_ICP + '/summary.json', OLD_ICP_SHA)
    icp_audit = inputs.json(OLD_ICP + '/independent_summary_audit.json', OLD_ICP_AUDIT_SHA)
    require(icp_audit['status'] == 'passed_independent_lineage_full_trajectory_chunk_and_tensor_audit' and
            icp_audit['summary_sha256'] == OLD_ICP_SHA, 'Old ICP is reused from its original independent full audit')
    ledger = inputs.json(OLD_ICP + '/input_hashes.json', old_icp['input_hashes_file_sha256'])
    old_hashes = {key.replace('\\', '/'): value for key, value in ledger['input_sha256'].items()}
    result = {}
    for scene, n, *_ in ARMS:
        arm = next(x for x in full['scene_results'][scene]['arms'] if x['model_dimension'] == 8)
        binding = protocol['old_full8D_references'][scene]
        old = inputs.json(binding['path'], binding['sha256'])
        require(arm['metrics_sha256'] == binding['sha256'] and arm['frames'] == n and
                any(x['run'] == arm['run'] and x['metrics_sha256'] == binding['sha256'] for x in audit['runs']),
                'Original fixed full 8D run/audit identity')
        equal(arm['trajectory_scalars'], {k: old['trajectory'][k] for k in SCALARS}, 'Reused original 8D scalars')
        baseline = old_icp['scenes'][scene]['baseline']; original_metric_path = 'outputs/' + baseline['run'] + '/metrics.json'
        original_icp = inputs.json(original_metric_path, old_hashes[original_metric_path])
        require(original_icp['frames_processed'] == n and any(x['run'] == baseline['run'] and x['full_frames'] == n
                for x in icp_audit['six_complete_trajectory_checks']), 'Both complete original ICP trajectories retained')
        old_icp_metrics = baseline['metrics']
        mapping = {'raw_anchored_translation_rmse_m': 'raw_anchored_translation_rmse_m',
                   'se3_aligned_ate_translation_rmse_m': 'se3_aligned_ate_translation_rmse_m',
                   'adjacent_rpe_translation_rmse_m': 'adjacent_rpe_translation_rmse_m',
                   'adjacent_rpe_rotation_rmse_deg': 'adjacent_rpe_rotation_rmse_degrees',
                   'raw_rotation_rmse_deg': 'raw_anchored_rotation_rmse_degrees'}
        # The ICP comparison deliberately copies the already audited 97b values,
        # not its near-equal original metrics or a third numerical calculation.
        result[scene] = {'old_full8D': {'run': arm['run'], 'metrics_sha256': arm['metrics_sha256'],
            'frames': n, 'trajectory_scalars': arm['trajectory_scalars'], 'endpoint_raw_translation_m': arm['endpoint_raw_translation_m'],
            'failure_frame_count': arm['failure_frame_count'], 'last_camera': arm['last_camera'],
            'configuration': arm['configuration'], 'observed_wall_seconds_including_IO': arm['observed_wall_seconds_including_IO'],
            'summary_sha256': OLD_FULL_SHA, 'audit_sha256': OLD_FULL_AUDIT_SHA,
            'scope': 'Original full 8D result, never rerun; the 1d75 addendum concerns the old 6D proposal guard only.'},
            'old_fullICP': {'run': baseline['run'], 'original_metrics_sha256': old_hashes[original_metric_path],
            'frames': n, 'trajectory_scalars': {new: old_icp_metrics[old_key] for new, old_key in mapping.items()},
            'scalar_key_mapping_from_audited_summary': mapping,
            'endpoint_raw_translation_m': old_icp_metrics['raw_endpoint_translation_error_m'],
            'failure_frame_count': baseline['failure_frame_count'], 'stored_original_timing_s': baseline['timing_s'],
            'summary_sha256': OLD_ICP_SHA, 'audit_sha256': OLD_ICP_AUDIT_SHA,
            'scope': 'Old readonly full ICP control and cached relative candidate source; not an equal-cost rerun.'},
            '_old_sensor_metadata': old['selected_sensor_metadata']}
    return result


def frame_subset(frame, sensor, trajectory, index, pair_binding, candidate_binding, run):
    copied = dict(frame); history_binding = None
    if 'primary' in frame:
        primary = dict(frame['primary']); history = primary.pop('iteration_history')
        history_binding = {'rows': len(history), 'line_search_trials': sum(len(row.get('line_search_trials', [])) for row in history),
                           'sha256': digest(encoded(history)), 'encoding': 'Python json.dumps ensure_ascii=False sort_keys=True separators=(comma,colon) allow_nan=False UTF8; no trailing newline',
                           'scope': 'The full primary iteration_history, including trials, stays in the original local pair/candidate and JSONL; not copied into this page.'}
        copied['primary'] = primary
        reconstructed = {**copied, 'primary': {**primary, 'iteration_history': history}}
    else:
        require(index == 0, 'Only anchor lacks original primary/fallback attempts')
        reconstructed = copied
    equal(reconstructed, frame, 'All original frame fields except explicitly bound history retained')
    errors = {label: (None if interval and index == 0 else trajectory[key][index - 1 if interval else index])
              for label, (key, interval) in ERROR_ARRAYS.items()}
    return {'frame': copied, 'sensor_metadata': sensor, 'stored_errors': errors,
            'primary_history_local_evidence': history_binding,
            'immutable_local_evidence': {'pair_path': f'outputs/{run}/pairs/{index:04d}.json', 'pair_binding': pair_binding,
                'candidate_path': f'outputs/{run}/candidates/{index:04d}.json' if index else None,
                'candidate_binding': candidate_binding}}


def page_object(scene, page_index, rows):
    return {'status': 'all_final_candidate_frame_fields_retained_history_bound_locally', 'scene': scene,
            'page_index': page_index, 'first_frame_id': rows[0]['frame']['frame_id'],
            'last_frame_id': rows[-1]['frame']['frame_id'], 'rows': rows,
            'stored_error_mapping': ERROR_ARRAYS, 'RPE_row_convention': 'row i contains original interval i-1 to i; anchor null',
            'scope': 'All source frame fields and complete primary/fallback FINAL candidates retained; only primary.iteration_history omitted and hash/count-bound locally. Anchor absence and explicit nulls are never replaced with invented candidates.'}


def save_new(path, obj):
    raw = encoded(obj) + b'\n'
    require(len(raw) < MAX_BYTES, 'Public JSON must be below 1 MiB; never truncate: ' + path.name)
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': path.name, 'bytes': len(raw), 'sha256': digest(raw)}


def collect(args):
    inputs = Inputs(); source = inputs.raw('scripts/summarize_rgbd_affine_icp_fallback_results.py')
    protocol, batch, audit, runs = gate(inputs, args.audit_sha)
    references = old_references(inputs, protocol)
    output = ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False)
    (output / 'source_script.py').write_bytes(source)
    try:
        scene_summaries = {}; artifacts = []; total_rows = total_steps = total_trials = 0
        for scene, n, run, metrics_sha, terminal, checked, m, witness, fallback in runs:
            equal(m['selected_sensor_metadata'], references[scene].pop('_old_sensor_metadata'), 'New and original full8D exact sensor sequence')
            manifest = inputs.json('outputs/' + run + '/pair_manifest.json')
            expected_paths = {f'pairs/{i:04d}.json' for i in range(n)} | {f'candidates/{i:04d}.json' for i in range(1, n)}
            require(set(manifest) == expected_paths, 'Complete immutable pair/candidate evidence')
            iterations_path = 'outputs/' + run + '/iterations.jsonl'; inputs.raw(iterations_path)
            pages = []; buffer = []; selections = Counter(); primary_flags = Counter(); final_flags = Counter()
            primary_rejected = []; fallback_selected = []; final_rejected = []; step_count = trial_count = 0
            for key, interval in ERROR_ARRAYS.values():
                require(len(m['trajectory'][key]) == n - int(interval), 'Every stored metric array entry retained')
            for i, (frame, sensor) in enumerate(zip(m['frame_metrics'], m['selected_sensor_metadata'])):
                require(frame['frame_id'] == sensor['frame_id'] == i, 'Every original associated frame ID')
                pair_path = f'pairs/{i:04d}.json'; pb = manifest[pair_path]
                equal(inputs.json('outputs/' + run + '/' + pair_path, pb['sha256'], pb['bytes']), frame, 'Immutable original pair equality')
                cb = None
                if i:
                    candidate_path = f'candidates/{i:04d}.json'; cb = manifest[candidate_path]
                    equal(inputs.json('outputs/' + run + '/' + candidate_path, cb['sha256'], cb['bytes']),
                          {key: frame[key] for key in CANDIDATE_FIELDS}, 'Immutable original final-candidate evidence')
                    equal(frame['fallback'], fallback['rows'][i - 1], 'Every original cached relative fallback field')
                    primary_flags.update(frame['primary']['failure_flags'])
                selections[frame['selection']] += 1; final_flags.update(frame['failure_flags'])
                if frame['primary_rejected']: primary_rejected.append(i)
                if frame['fallback_selected']: fallback_selected.append(i)
                if frame['failure_flags']: final_rejected.append(i)
                row = frame_subset(frame, sensor, m['trajectory'], i, pb, cb, run)
                if row['primary_history_local_evidence'] is not None:
                    step_count += row['primary_history_local_evidence']['rows']
                    trial_count += row['primary_history_local_evidence']['line_search_trials']
                candidate_page = page_object(scene, len(pages), buffer + [row])
                if buffer and (len(buffer) == MAX_ROWS or len(encoded(candidate_page)) + 1 >= MAX_BYTES):
                    name = f'{scene}-frames-{len(pages):02d}.json'
                    entry = save_new(output / name, page_object(scene, len(pages), buffer))
                    pages.append({**entry, 'rows': len(buffer), 'first_frame_id': buffer[0]['frame']['frame_id'], 'last_frame_id': buffer[-1]['frame']['frame_id']})
                    buffer = []
                require(len(encoded(page_object(scene, len(pages), [row]))) + 1 < MAX_BYTES, 'Single complete frame exceeds limit; fail instead of dropping fields')
                buffer.append(row)
            if buffer:
                entry = save_new(output / f'{scene}-frames-{len(pages):02d}.json', page_object(scene, len(pages), buffer))
                pages.append({**entry, 'rows': len(buffer), 'first_frame_id': buffer[0]['frame']['frame_id'], 'last_frame_id': buffer[-1]['frame']['frame_id']})
            require(sum(page['rows'] for page in pages) == n and all(page['rows'] <= MAX_ROWS for page in pages), 'Every frame appears in <=200-row pages')
            for values, key in [(primary_rejected, 'primary_rejected_frame_ids'), (fallback_selected, 'fallback_selected_frame_ids'), (final_rejected, 'final_rejected_frame_ids')]:
                equal(values, checked[key], 'New independent audit exact decision IDs: ' + key)
            for value, key in [(dict(selections), 'selection_counts'), (dict(primary_flags), 'primary_failure_flag_counts'), (dict(final_flags), 'final_failure_flag_counts')]:
                equal(value, m[key], 'Stored all-frame histogram: ' + key); equal(value, checked[key], 'Audit histogram: ' + key)
            require(step_count == checked['step_ledger_counts']['steps'] and trial_count == checked['step_ledger_counts']['trials'], 'All omitted local history rows/trials counted and hash-bound')
            require(len(primary_rejected) == m['primary_failure_frame_count'] == terminal['primary_failure_frame_count'] and
                    len(fallback_selected) == m['fallback_selected_frame_count'] == terminal['fallback_selected_frame_count'] and
                    len(final_rejected) == m['final_failure_frame_count'] == terminal['final_failure_frame_count'], 'Separate attempt and final counts')
            current = {'run': run, 'metrics_sha256': metrics_sha, 'frames': n, 'tracked_pairs': n - 1,
                'configuration': m['configuration'], 'source_sha256': m['source_sha256'], 'status': m['status'],
                'selection_counts': dict(selections), 'primary_rejected_frame_ids': primary_rejected,
                'fallback_selected_frame_ids': fallback_selected, 'final_rejected_frame_ids': final_rejected,
                'primary_failure_frame_count': len(primary_rejected), 'fallback_selected_frame_count': len(fallback_selected),
                'final_failure_frame_count': len(final_rejected), 'primary_failure_flag_counts': dict(primary_flags),
                'final_failure_flag_counts': dict(final_flags), 'trajectory_scalars': {key: m['trajectory'][key] for key in SCALARS},
                'endpoint_raw_translation_m': m['trajectory']['per_frame_raw_translation_error_m'][-1],
                'alignment_scale': m['trajectory']['alignment_scale'], 'last_final_reference_id': m['last_final_reference_id'],
                'last_frame': {key: value for key, value in m['frame_metrics'][-1].items() if key not in ('primary', 'fallback')},
                'last_sensor_metadata': m['selected_sensor_metadata'][-1], 'observed_wall_seconds_including_cached_ICP_input_IO': m['observed_wall_seconds'],
                'timing_scope': m['timing_scope'], 'started_utc': terminal['started_utc'], 'ended_utc': terminal['ended_utc'],
                'returncode': terminal['returncode'], 'local_primary_history_rows': step_count, 'local_primary_trial_count': trial_count,
                'local_iterations_jsonl': {'path': iterations_path, **inputs.files[iterations_path]}, 'public_frame_pages': pages}
            scene_summary = {'scene': scene, 'association': witness['association'], 'new_hybrid': current, **references[scene],
                'comparison_scope': 'Fixed complete causal pipelines; reused full8D and fullICP; no metric recalculation, no GT-based candidate choice, no isolated brightness causality or cached-ICP speed/FPS comparison.'}
            artifacts.extend(pages); artifacts.append(save_new(output / (scene + '-summary.json'), scene_summary))
            scene_summaries[scene] = scene_summary; total_rows += n; total_steps += step_count; total_trials += trial_count
        require((total_rows, total_steps, total_trials) == (1297, 33900, 35746), 'All actual frames and full primary history binding counts')
        inputs.recheck()
        entries = list(sorted(inputs.files.items())); hash_pages = []
        for index, offset in enumerate(range(0, len(entries), 500)):
            hash_pages.append(save_new(output / f'input-hashes-{index:02d}.json', dict(entries[offset:offset + 500])))
        summary = {'status': 'completed_after_actual_independent_hybrid_result_audit', 'source_sha256': digest(source),
            'fusion_protocol_sha256': PROTOCOL_SHA, 'actual_independent_audit_sha256': AUDIT_SHA,
            'actual_auditor_source_sha256': AUDIT_SOURCE_SHA, 'audit_package_protocol_sha256': AUDIT_PACKAGE_SHA,
            'audit_root_review_sha256': AUDIT_REVIEW_SHA, 'batch_metrics_sha256': BATCH_SHA,
            'old_full8D_summary_sha256': OLD_FULL_SHA, 'old_full_audit_sha256': OLD_FULL_AUDIT_SHA,
            'old_6D_proposal_addendum_sha256': ADDENDUM_SHA, 'old_fullICP_summary_sha256': OLD_ICP_SHA,
            'old_fullICP_audit_sha256': OLD_ICP_AUDIT_SHA, 'source_sha256_original_hybrid': protocol['source_sha256'],
            'CPU_child_returncodes': [row['returncode'] for row in batch['run_records']], 'all_new_frame_rows': total_rows,
            'all_new_nonanchor_pairs': 1295, 'all_local_primary_history_rows': total_steps, 'all_local_primary_trial_count': total_trials,
            'scenes': scene_summaries, 'public_json_max_bytes_exclusive': MAX_BYTES, 'public_page_max_rows': MAX_ROWS,
            'consumed_files_rehashed_unchanged': len(inputs.files), 'input_hash_pages': hash_pages,
            'GPU_calls': 0, 'new_metric_calculations': 0, 'registration_calls': 0,
            'limits': ['Public subset retains all per-frame FINAL candidates, brightness, primary/fallback/final flags, references, sensors, poses and four stored error arrays.',
                       'All original primary trials/iteration histories stay in local immutable pair/candidate and JSONL; canonical hash/count only is public.',
                       'Actual new f953/e807/b549 audit is separately bound. Old 1d75 is only an original 6D proposal-field guard, not a hybrid metric audit.',
                       'Zero final flags means selected branches passed engineering gates, not true pose accuracy/convergence or complete SLAM.',
                       'Cached ICP inputs exclude cost of newly computing these candidates; no isolated runtime/FPS claim or cross-scene mean.']}
        artifacts.extend(hash_pages); artifacts.append(save_new(output / 'summary.json', summary))
        save_new(output / 'completion.json', {'status': 'completed', 'source_sha256': digest(source), 'files': artifacts,
            'source_snapshot': {'path': 'source_script.py', 'sha256': digest(source), 'bytes': len(source)}, 'old_results_modified': False, 'GPU_calls': 0})
        return {'status': summary['status'], 'run': args.run, 'frames': total_rows, 'summary_sha256': digest((output / 'summary.json').read_bytes())}
    except BaseException as error:
        save_new(output / 'failure.json', {'status': 'collector_failed_partial_output_preserved', 'error_type': type(error).__name__,
                 'message': str(error), 'traceback': traceback.format_exc(), 'old_results_modified': False, 'GPU_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--execute', action='store_true'); modes.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--audit-sha', required=True)
    parser.add_argument('--run', default='rgbd-affine-icp-fallback-comparison-001')
    args = parser.parse_args()
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run) is not None, 'Exclusive safe output name')
    if args.check_inputs:
        inputs = Inputs(); protocol, batch, audit, runs = gate(inputs, args.audit_sha)
        old_references(inputs, protocol); inputs.recheck()
        print(json.dumps({'status': 'actual_inputs_ready_no_summary_created', 'frames': sum(row[1] for row in runs),
                          'actual_audit_sha256': AUDIT_SHA, 'consumed_files': len(inputs.files), 'GPU_calls': 0}))
    else:
        print(json.dumps(collect(args)))


if __name__ == '__main__':
    main()
