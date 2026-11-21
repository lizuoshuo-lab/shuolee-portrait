"""Independent CPU audit of transparent full-V4 completion recovery.

--self-check exercises synthetic lineage and byte controls only. --audit is
gated on the actual completed collector before raw GT/tensors are consumed.
Never invokes either collector, optimizer, loader, registration or GPU.
Raw-GT/SciPy metric formulas explicitly reuse the separately reviewed e807
implementation through the locked 0da helper, not a third independent formula.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import audit_full_v4_evidence_helpers as independent

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROTOCOL = ROOT / 'manifests/tum-online-v4-full-sequence-protocol-001.json'
RECOVERY_PROTOCOL = ROOT / 'manifests/tum-online-v4-full-sequence-recovery-protocol-001.json'
OLD_BATCH = ROOT / 'outputs/tum-online-v4-full-sequence-001'
NEW_BATCH = ROOT / 'outputs/tum-online-v4-full-sequence-recovery-001'
OUTPUT = ROOT / 'outputs/tum-online-v4-full-sequence-recovery-summary-001'
FAILED = 'tum-xyz-online-v4-prior001-full748-001'
REPLACEMENT = 'tum-xyz-online-v4-prior001-full748-002'
LOCKS = {
    'manifests/tum-online-v4-full-sequence-protocol-001.json': 'ee635a6d65cc02dd26ff16e3f83721afa745a56b9f7a7464453833651a4963af',
    'manifests/tum-online-v4-full-sequence-recovery-protocol-001.json': '3e1a127dfbffd1885862d2f1a5eb633a36228788041f5aa3d5c8de3b5c90ea3c',
    'scripts/summarize_v4_full_sequence_controls.py': '5ce0f9dcfe8a30ec3c63560725c1783ed49f1f957df7d0483e19050b81cef5bf',
    'scripts/summarize_v4_full_sequence_recovery_controls.py': '911d8198c67c29bbf9cf8d06786560305c7478602209cff2fca2361cfc9cc76a',
    'scripts/audit_full_v4_evidence_helpers.py': '0da3578a5b5c01b3d23489e923293aa28628de4f14130e7d45a99875e74b76fb',
    'scripts/audit_rgbd_photogeometric_results.py': 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53',
    'outputs/tum-online-v4-full-sequence-recovery-001/root_protocol_review.json': '8180e1b97876b7f1f41d2ef5e19df5e4655b83f0e911059f3b05e1c875cbf2d1',
    'outputs/v4-full-sequence-recovery-collector-cpu-check-001/root_source_review.json': '2020c741620a66ea2050a2f8851e540741a18160563214ee0554dd7d8b51b77b',
}
require, sha = independent.require, independent.sha


def read(path): return json.loads(Path(path).read_bytes())


def save(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(raw) < 1024 * 1024, 'Audit JSON below1MiB without truncation')
    with Path(path).open('xb') as stream: stream.write(raw)


def stamp(value):
    a = datetime.fromisoformat(value['utc']); b = datetime.fromisoformat(value['beijing'])
    require(a.tzinfo is not None and a == b, 'UTC/Beijing timestamps agree')
    return a


def lineage_gate(original, recovery, old, new, summary):
    require(old['status'] == 'batch_stopped_failure_no_retry' and old['GPU_runs_started'] == 4 and old['current_run'] == FAILED,
            'Old failed four-attempt ledger retained')
    require(old['protocol_sha256'] == LOCKS[str(ORIGINAL_PROTOCOL.relative_to(ROOT)).replace('\\', '/')] and
            old['driver_sha256'] == original['driver_sha256'] and old['source_sha256'] == original['source_sha256'], 'Original protocol/source/driver binding')
    require(len(old['run_records']) == len(original['new_runs']) == 4, 'Original failure record retained')
    previous_end = stamp(old['started_at']); deadline = datetime.fromisoformat(original['deadline_beijing'])
    for index, (record, row) in enumerate(zip(old['run_records'], original['new_runs'])):
        require(record['run'] == row['run'] and record['configuration'] == row['configuration'] and record['order'] == index + 1, 'Original order/full configuration')
        ready, guard, popped, ended = [stamp(record[k]) for k in ('ready_at', 'spawn_guard_at', 'Popen_returned_at', 'ended_at')]
        require(previous_end <= ready <= guard <= popped <= ended and guard < deadline, 'Serial original before deadline')
        previous_end = ended
        if index < 3:
            require(record['returncode'] == 0 and record['status'] == 'completed' and record['frames'] == row['configuration']['frames'], 'First three full exit0')
        else:
            require(record['returncode'] == 1 and record['status'] == 'failed_process_exit_all_original_artifacts_retained' and
                    'metrics_sha256' not in record and 'trajectory_summary' not in record, 'Failed fourth is not fabricated terminal success')
    require(stamp(old['failed_at']) >= previous_end and old['all_original_artifacts_and_logs_retained'], 'Old failure timestamp/retention')
    expected = [r['run'] for r in original['new_runs'][:3]] + [REPLACEMENT]
    require(recovery['original_failed_run'] == FAILED and recovery['selected_final_runs'] == expected and
            recovery['new_run_count'] == 1 and len(recovery['new_runs']) == 1, 'Only declared original3+replacement selection')
    require(recovery['original_protocol_sha256'] == old['protocol_sha256'] and recovery['source_sha256'] == original['source_sha256'] and
            recovery['dataset_archives'] == original['dataset_archives'] and recovery['full_CPU_baseline_references'] == original['full_CPU_baseline_references'],
            'Recovery sources/data/full-baseline binding')
    row = recovery['new_runs'][0]; configuration = copy.deepcopy(original['new_runs'][3]['configuration']); configuration['run'] = REPLACEMENT
    require(row['run'] == REPLACEMENT and row['configuration'] == configuration and configuration['frames'] == 748 and configuration['start_frame'] == 0,
            'Frame0 restart only run-name differs from failed full748 configuration')
    binding = recovery['replacement_binding']
    require(binding['not_checkpoint_resume'] and binding['restart_from_frame'] == 0 and binding['original_failed_run'] == FAILED and
            binding['replacement_run'] == REPLACEMENT and binding['all_36_configuration_fields_only_run_differs'], 'Explicit replacement lineage')
    command = copy.deepcopy(original['new_runs'][3]['command']); command[command.index('--run') + 1] = REPLACEMENT
    require(row['command'] == command and binding['original_failed_command'] == original['new_runs'][3]['command'], 'Replacement exact command')
    require(new['status'] == 'recovery_run_completed' and new['GPU_runs_started'] == 1 and new['current_run'] is None and len(new['run_records']) == 1,
            'One terminal collected replacement, no additional attempt')
    require(new['protocol_sha256'] == LOCKS[str(RECOVERY_PROTOCOL.relative_to(ROOT)).replace('\\', '/')] and new['source_sha256'] == original['source_sha256'] and
            new['driver_sha256'] == recovery['driver_sha256'], 'Replacement ledger protocol/source/driver')
    record = new['run_records'][0]
    require(record['run'] == REPLACEMENT and record['configuration'] == configuration and record['status'] == 'completed' and
            record['returncode'] == 0 and record['frames'] == 748, 'Replacement complete748 exit0')
    started, ready, guard, popped, ended = [stamp(new['started_at'])] + [stamp(record[k]) for k in ('ready_at', 'spawn_guard_at', 'Popen_returned_at', 'ended_at')]
    require(stamp(old['failed_at']) <= started <= ready <= guard <= popped <= ended <= stamp(new['ended_at']) and guard < deadline, 'Replacement chronology/cutoff')
    require(summary['status'] == 'completed_read_only_original_three_plus_explicit_recovery' and summary['attempt_count'] == 5 and
            summary['completed_arm_count'] == 4 and summary['failed_attempt_count'] == 1 and summary['partial_failed_original_frames_not_used_as_final_trajectory'],
            'Summary discloses five attempts / one failed partial rather than original4success')
    require(summary['original_batch_status'] == old['status'] and summary['recovery_batch_status'] == new['status'] and
            summary['selection_rule'] == 'original_first_three_completed_plus_explicit_fourth_replacement_no_metric_selection', 'Summary exact lineage labels')
    require(recovery['deadline_beijing'] == original['deadline_beijing'], 'Restart retains original cutoff')
    require(summary['original_protocol_sha256'] == old['protocol_sha256'] and summary['recovery_protocol_sha256'] == new['protocol_sha256'], 'Summary protocol labels exact')
    selected = original['new_runs'][:3] + [row]
    require([summary['scenes'][r['scene']]['arms']['prior0' if r['configuration']['pose_prior_weight'] == 0 else 'prior001']['run'] for r in selected] == expected,
            'Actual summary arm identities match fixed selection')
    return selected, old['run_records'][:3] + [record]


def unpack(table):
    require(len(set(table['columns'])) == len(table['columns']), 'Unique public table columns')
    require(all(len(row) == len(table['columns']) for row in table['rows']), 'Full public table row widths')
    return [dict(zip(table['columns'], row)) for row in table['rows']]


def metric_check(actual, published, public_rows):
    mapping = {'raw_anchored_translation_rmse_m': 'raw_anchored_translation_rmse_m',
               'se3_aligned_ate_translation_rmse_m': 'se3_aligned_ate_translation_rmse_m',
               'raw_rotation_rmse_deg': 'raw_anchored_rotation_rmse_degrees',
               'adjacent_rpe_translation_rmse_m': 'adjacent_rpe_translation_rmse_m',
               'adjacent_rpe_rotation_rmse_deg': 'adjacent_rpe_rotation_rmse_degrees',
               'raw_endpoint_translation_error_m': 'raw_endpoint_translation_error_m',
               'raw_endpoint_rotation_error_degrees': 'raw_endpoint_rotation_error_degrees',
               'raw_anchored_translation_max_m': 'raw_translation_max_m', 'raw_anchored_translation_mean_m': 'raw_translation_mean_m'}
    differences = {key: abs(actual[key] - published[value]) for key, value in mapping.items()}
    require(all(error < (1e-4 if 'rotation' in key else 2e-6 if 'rpe' in key else 1e-12) for key, error in differences.items()), 'Independent scalar metric agreement / float32 rigid inverse tolerance')
    arrays = {'per_frame_raw_translation_error_m': 'raw_error_m', 'per_frame_se3_aligned_ate_translation_error_m': 'ate_error_m',
              'per_interval_rpe_translation_error_m': 'rpe_translation_m', 'per_interval_rpe_rotation_error_deg': 'rpe_rotation_deg'}
    array_diffs = {}
    for key, public in arrays.items():
        values = [r[public] for r in public_rows]
        if 'interval' in key: require(values[0] is None, 'No fictitious first RPE interval'); values = values[1:]
        error = float(np.max(np.abs(np.asarray(values) - np.asarray(actual[key])))); array_diffs[key] = error
        require(error < (1e-4 if 'rotation' in key else 2e-6 if 'rpe' in key else 1e-12), 'All public per-frame/interval metric arrays')
    return {'scalar_absolute_differences': differences, 'all_array_max_absolute_differences': array_diffs,
            'recalculated_scalar_metrics': {key: value for key, value in actual.items() if not isinstance(value, list)},
            'units': {'translation': 'meters', 'rotation': 'degrees'},
            'RPE_boundary': 'Independent generic inverse differs from collector analytic R.T inverse by persisted float32 rigidity roundoff; actual differences retained.'}


def audit():
    # Completion gate runs before GT, tensors or any run002 progress access.
    require((OUTPUT / 'completion.json').exists() and (OUTPUT / 'summary.json').exists(), 'Actual completed collector absent; no audit')
    require(not (OUTPUT / 'independent_summary_audit.json').exists(), 'Do not overwrite independent audit')
    observed = {}
    def watch(path, expected=None):
        path = Path(path).resolve(); require(path.is_relative_to(ROOT), 'Evidence remains within lab')
        digest = sha(path); require(expected is None or digest == expected, 'Actual evidence SHA ' + str(path))
        observed[path] = digest; return digest
    for name, digest in LOCKS.items(): watch(ROOT / name, digest)
    summary, completion = read(OUTPUT / 'summary.json'), read(OUTPUT / 'completion.json')
    require(completion['status'] == 'completed' and completion['source_sha256'] == LOCKS['scripts/summarize_v4_full_sequence_recovery_controls.py'] and
            summary['collector_sha256'] == completion['source_sha256'], 'Actual reviewed collector source/completion')
    for name, item in completion['files'].items():
        path = (OUTPUT / name).resolve(); require(path.is_relative_to(OUTPUT), 'Safe collector artifact')
        watch(path, item['sha256']); require(path.stat().st_size == item['bytes'], 'Collector artifact bytes')
        if path.suffix == '.json': require(path.stat().st_size <= 1024 * 1024, 'Public collector JSON under1MiB')
    inputs = read(OUTPUT / summary['input_hashes_file'])
    watch(OUTPUT / summary['input_hashes_file'], summary['input_hashes_file_sha256'])
    require(len(inputs['input_sha256']) == summary['input_hash_count'] and inputs['status'] == 'all_consumed_inputs_rehashed_unchanged', 'Collector complete consumed hash table')
    for name, digest in inputs['input_sha256'].items(): watch(ROOT / name, digest)
    original, recovery, old, new = read(ORIGINAL_PROTOCOL), read(RECOVERY_PROTOCOL), read(OLD_BATCH / 'failure.json'), read(NEW_BATCH / 'metrics.json')
    selected, records = lineage_gate(original, recovery, old, new, summary)
    require(not (OLD_BATCH / 'metrics.json').exists(), 'Original failed batch has no fabricated completed metrics')
    failure_folder = ROOT / 'outputs' / FAILED
    require(not (failure_folder / 'metrics.json').exists() and not (failure_folder / 'online_map.pt').exists(), 'Failed fourth final outputs remain absent')
    for name, digest in recovery['original_failure_artifact_sha256'].items(): watch(ROOT / name, digest)
    snapshot = recovery['original_failure_snapshot']; watch(ROOT / snapshot['manifest'], snapshot['manifest_sha256'])
    require(read(ROOT / snapshot['manifest'])['files'] == snapshot['files'], 'Full failure snapshot manifest')
    for item in snapshot['files']: watch(ROOT / item['snapshot'], item['sha256']); watch(ROOT / item['original'], item['sha256'])
    failure = read(failure_folder / 'failure.json')
    require(failure['status'] == 'failed' and failure['exception_type'] == 'PermissionError' and failure['completed_frame_count'] == 521 and
            failure['current_local_index'] == 520, 'Actual failed partial521 boundary retained')
    for name, digest in original['source_sha256'].items(): watch(ROOT / name, digest)
    for record in old['run_records'] + new['run_records']: watch(ROOT / record['log'], record['log_sha256'])
    import torch
    require(not torch.cuda.is_initialized(), 'CPU auditor starts without CUDA')
    results = []; paired_contexts = {}; scene_checks = {}
    for scene in ('desk', 'xyz'):
        witness = original['full_CPU_baseline_references'][scene]; folder = ROOT / 'outputs' / witness['run']
        for name, digest in witness['artifact_sha256'].items(): watch(ROOT / name, digest)
        baseline = read(folder / 'metrics.json'); sensors = read(folder / 'selected_sensor_metadata.json')['selected_frames']
        truth, raw_checks = independent.verify_full_raw_association(next(r['configuration']['dataset'] for r in selected if r['scene'] == scene), sensors)
        require(len(truth) == witness['frames'] == summary['scenes'][scene]['frame_count'], 'Full549/748 scene counts')
        details = read(OUTPUT / (scene + '_frames.json')); require(details['rows_not_dropped'], 'All public scene rows retained')
        require(details['sensors'] == [[s[k] for k in details['sensor_columns']] for s in sensors], 'Public actual sensor pairing')
        require(np.max(np.abs(np.asarray(details['truth_camera_centers_m']) - truth[:, :3, 3])) < 1e-12, 'Public raw GT anchor centers')
        icp = np.asarray([r['estimated_c2w'] for r in baseline['frame_metrics']]); icp_public = unpack(details['ICP'])
        require(len(icp_public) == len(icp) and [r['frame_id'] for r in icp_public] == list(range(len(icp))), 'Full ICP public rows')
        for p, b in zip(icp_public, baseline['frame_metrics']): require(p['accepted'] == b['accepted'] and p['failure_flags'] == b['failure_flags'], 'All ICP decisions/failures retained')
        metrics, structure = independent.independent_trajectory(icp, truth)
        checks = metric_check(metrics, summary['scenes'][scene]['baseline']['metrics'], icp_public)
        checks.update(scene=scene, run=witness['run'], full_frames=len(icp), camera_structure=structure,
                      failure_frame_ids=[r['frame_id'] for r in baseline['frame_metrics'] if r['failure_flags']])
        results.append(checks)
        contexts = []
        for row, record in [(r, records[selected.index(r)]) for r in selected if r['scene'] == scene]:
            folder = ROOT / row['output_relative_path']; watch(folder / 'metrics.json', record['metrics_sha256']); data = read(folder / 'metrics.json')
            require(data['status'] == 'completed' and data['configuration'] == row['configuration'] and data['selected_sensor_metadata'] == sensors and
                    data['frames_processed'] == len(truth) and data['selected_frame_ids'] == list(range(len(truth))), 'Actual complete selected V4 arm')
            compact = read(folder / 'summary.json'); chunk_checks = independent.check_chunks(folder, data, compact)
            for name, digest in chunk_checks['chunk_sha256'].items(): watch(folder / name, digest)
            rows = data['frame_metrics']; estimate = np.asarray([r['estimated_c2w'] for r in rows]); label = 'prior0' if row['configuration']['pose_prior_weight'] == 0 else 'prior001'
            public = unpack(details['arms'][label]); published = summary['scenes'][scene]['arms'][label]
            require(len(public) == len(rows) and [r['frame_id'] for r in public] == list(range(len(rows))), 'All V4 public rows')
            for index, (p, r, base) in enumerate(zip(public, rows, baseline['frame_metrics'])):
                require(p['failure_flags'] == r['failure_flags'] and p['pose_rejected'] == r.get('pose_rejected', False) and
                        p['tracking_steps_completed'] == r.get('gs_refinement_steps_completed', 0) and p['new_points_added'] == r['new_points_added'] and p['point_count'] == r['point_count'], 'All source flags/step/mapping fields')
                require(np.array_equal(r['icp']['icp_prefix_c2w_float64'], icp[index]) and r['icp']['accepted'] == base['accepted'] and r['icp']['failure_flags'] == base['failure_flags'], 'Independent ICP prefix/decisions exact')
                if index: require(r['icp']['registration']['candidate_previous_from_current'] == base['candidate_previous_from_current'], 'Independent ICP relative candidate exact')
                if r.get('pose_rejected', False): require(np.array_equal(estimate[index], estimate[index - 1]) and r['new_points_added'] == 0, 'Rejected pose held / no future map update')
                for state, key in (('initial', 'initial_loss'), ('selected', 'optimized_loss_before_mapping')):
                    expected = [r.get(key, {}).get(k) for k in ('sensor_objective', 'prior_objective', 'objective')]
                    require(p[state + '_sensor_prior_combined'] == expected, 'Public persisted selected objective values exact')
            require(published['pose_gradient_checks'] == data['pose_gradient_checks'] and published['all_mapping_events'] == data['mapping_events'] and
                    published['failure_frame_count'] == data['failure_frame_count'] == sum(bool(r['failure_flags']) for r in rows), 'All probe/map/failure records preserved')
            metrics, structure = independent.independent_trajectory(estimate, truth); checks = metric_check(metrics, published['metrics'], public)
            checks.update(scene=scene, run=row['run'], full_frames=len(estimate), camera_structure=structure, chunks=chunk_checks,
                          failure_frame_ids=[r['frame_id'] for r in rows if r['failure_flags']],
                          trajectory_text=independent.check_tum_text(folder / 'trajectory_estimated.txt', sensors, estimate))
            for filename in ('online_map.pt', 'last_checkpoint.pt', 'first_tracking_context.pt', 'first_tracking_context.json'): watch(folder / filename)
            final = torch.load(folder / 'online_map.pt', map_location='cpu', weights_only=True)
            last = torch.load(folder / 'last_checkpoint.pt', map_location='cpu', weights_only=True)
            require(np.array_equal(final['estimated_c2w'].numpy(), estimate) and final['frame_ids'] == list(range(len(estimate))), 'Final checkpoint actual trajectory')
            require(last['ground_truth_included'] is False and last['current_local_index'] == len(estimate) - 1, 'Terminal causal last checkpoint')
            require(final['configuration'] == last['configuration'] == row['configuration'] and last['frame_ids_with_stored_pose'] == final['frame_ids'], 'Actual checkpoint full configuration/frame IDs')
            forward, inverse = final['estimated_c2w'].numpy(), final['estimated_w2c'].numpy()
            require(np.array_equal(last['estimated_c2w'].numpy(), forward) and np.array_equal(last['estimated_w2c'].numpy(), inverse), 'Causal/final cameras numerical arrays exact')
            inverse_error = float(np.max(np.abs(forward @ inverse - np.eye(4))))
            require(inverse_error < 2e-6 and np.array_equal(forward[0], np.eye(4)) and np.array_equal(inverse[0], np.eye(4)), 'Frame0 identity/float32 canonical rigid inverse')
            evaluation_gt = final['evaluation_only_gt_c2w'].numpy()
            anchored_checkpoint_gt = np.stack([np.linalg.solve(evaluation_gt[0], pose) for pose in evaluation_gt])
            gt_error = float(np.max(np.abs(anchored_checkpoint_gt - truth)))
            require(gt_error < 1e-12, 'Final evaluation-only absolute GT anchors to independent raw GT')
            checks['checkpoint_camera_inverse_max_absolute_error'] = inverse_error
            checks['final_evaluation_GT_anchor_max_absolute_difference'] = gt_error
            checks['source_frame_provenance'] = independent.camera_source_provenance(final, last, data)
            require(checks['source_frame_provenance']['per_gaussian_source_frame_counts'] == published['per_gaussian_source_frame_counts'], 'Public actual Gaussian source histogram')
            context = torch.load(folder / 'first_tracking_context.pt', map_location='cpu', weights_only=True); meta = read(folder / 'first_tracking_context.json')
            require(sha(folder / 'first_tracking_context.pt') == meta['checkpoint_sha256'] and meta == data['first_tracking_context'] and meta == published['actual_first_tracking_context'], 'Actual immutable first context hash/metadata')
            require(meta['saved_before_first_pose_gradient_probe'] and meta['tracking_optimizer_steps_executed_before_save'] == 0 and meta['sensor_metadata']['frame_id'] == 1 and meta['map_source_frame_ids'] == [0], 'Actual frame0 warmup/current1 context timing')
            require(all(context[k] is False for k in ('ground_truth_included', 'future_sensor_observations_included', 'future_maps_or_trajectories_included')), 'Actual context no GT/future')
            require(context['configuration'] == row['configuration'] and context['status'] == 'first_tracking_context_saved' and context['sensor_metadata'] == meta['sensor_metadata'], 'Actual causal context configuration/status/sensor')
            contexts.append(context); results.append(checks)
        pair = independent.compare_contexts(*contexts); published_pair = summary['scenes'][scene]['actual_first_context_pair_comparison']
        for key, value in pair.items(): require(published_pair[key] == value, 'Actual seven/five byte observation exact')
        paired_contexts[scene] = pair; scene_checks[scene] = raw_checks
    require(len(results) == 6 and not torch.cuda.is_initialized(), 'Exactly six full trajectories / CPU-only')
    for path, digest in observed.items(): require(sha(path) == digest, 'Consumed original evidence unchanged after independent audit')
    save(OUTPUT / 'independent_summary_audit.json', {'status': 'passed_independent_lineage_full_trajectory_chunk_and_tensor_audit',
          'auditor_sha256': sha(Path(__file__)), 'helper_sha256': LOCKS['scripts/audit_full_v4_evidence_helpers.py'],
          'independent_metric_helper_sha256': independent.METRIC_HELPER_SHA, 'collector_sha256': summary['collector_sha256'],
          'summary_sha256': sha(OUTPUT / 'summary.json'), 'attempt_count': 5, 'original_returncodes': [r['returncode'] for r in old['run_records']],
          'replacement_returncode': new['run_records'][0]['returncode'], 'original_failed_partial_frame_count': 521,
          'original_failed_final_outputs_still_absent': True, 'selected_runs': [r['run'] for r in selected],
          'six_complete_trajectory_checks': results, 'raw_scene_checks': scene_checks, 'actual_context_byte_pairs': paired_contexts,
          'consumed_files_rehashed_unchanged': len(observed), 'GPU_calls': 0,
          'limits': ['Five attempts include the retained original failed fourth; replacement is an explicit frame0 restart, not original4success or checkpoint resume.',
                     'Reuses reviewed e807 independent raw-GT/SciPy/generic-solve/align_vectors through0da, not a third independent metric evaluator.',
                     'Translation values are meters and rotation values degrees; persisted float32 camera rigidity yields small analytical-vs-generic inverse differences, all measured.',
                     'For exact rigid E, E and inverse(E) have equal translation norm and rotation magnitude, so whole-residual inversion cannot validate RPE direction.',
                     'Raw tensor byte observations describe whole-pipeline pairs; differing warmup maps prohibit identical-frozen-map causal isolation.',
                     'Source review and execution ordering support GT optimization isolation; posthoc metadata/hashes alone cannot prove runtime causality.',
                     'No rerendering, registration, optimization, parameter tuning, speed comparison, cross-sequence mean or single-seed SD.']})
    print(json.dumps({'status': 'passed', 'trajectories': 6, 'original_attempts': 4, 'replacement_attempts': 1,
                      'output': str(OUTPUT / 'independent_summary_audit.json')}))


def self_checks(output):
    original, recovery = read(ORIGINAL_PROTOCOL), read(RECOVERY_PROTOCOL)
    t = datetime(2026, 10, 8, 18, tzinfo=timezone.utc)
    def time_record(v): return {'utc': v.isoformat(), 'beijing': v.astimezone(timezone(timedelta(hours=8))).isoformat()}
    old = {'status': 'batch_stopped_failure_no_retry', 'GPU_runs_started': 4, 'current_run': FAILED,
           'protocol_sha256': LOCKS[str(ORIGINAL_PROTOCOL.relative_to(ROOT)).replace('\\', '/')], 'driver_sha256': original['driver_sha256'],
           'source_sha256': original['source_sha256'], 'started_at': time_record(t), 'failed_at': time_record(t + timedelta(minutes=9)),
           'all_original_artifacts_and_logs_retained': True, 'run_records': []}
    for i, row in enumerate(original['new_runs']):
        start = t + timedelta(minutes=2 * i, seconds=1)
        r = {'run': row['run'], 'order': i + 1, 'configuration': row['configuration'], 'frames': row['configuration']['frames'],
             'returncode': 0 if i < 3 else 1, 'status': 'completed' if i < 3 else 'failed_process_exit_all_original_artifacts_retained'}
        r.update({k: time_record(start + timedelta(seconds=s)) for k, s in (('ready_at', 0), ('spawn_guard_at', 1), ('Popen_returned_at', 2), ('ended_at', 60))})
        if i < 3: r['metrics_sha256'] = 'a' * 64
        old['run_records'].append(r)
    row = recovery['new_runs'][0]; record = copy.deepcopy(old['run_records'][0]); start = t + timedelta(minutes=10)
    record.update(run=REPLACEMENT, configuration=row['configuration'], frames=748)
    record.update({k: time_record(start + timedelta(seconds=s)) for k, s in (('ready_at', 0), ('spawn_guard_at', 1), ('Popen_returned_at', 2), ('ended_at', 60))})
    new = {'status': 'recovery_run_completed', 'GPU_runs_started': 1, 'current_run': None, 'run_records': [record],
           'protocol_sha256': LOCKS[str(RECOVERY_PROTOCOL.relative_to(ROOT)).replace('\\', '/')], 'source_sha256': original['source_sha256'],
           'driver_sha256': recovery['driver_sha256'], 'started_at': time_record(start), 'ended_at': time_record(start + timedelta(seconds=61))}
    summary = {'status': 'completed_read_only_original_three_plus_explicit_recovery', 'attempt_count': 5, 'completed_arm_count': 4,
               'failed_attempt_count': 1, 'partial_failed_original_frames_not_used_as_final_trajectory': True,
               'original_batch_status': old['status'], 'recovery_batch_status': new['status'],
               'original_protocol_sha256': old['protocol_sha256'], 'recovery_protocol_sha256': new['protocol_sha256'],
               'selection_rule': 'original_first_three_completed_plus_explicit_fourth_replacement_no_metric_selection', 'scenes': {'desk': {'arms': {}}, 'xyz': {'arms': {}}}}
    for r in original['new_runs'][:3] + [row]: summary['scenes'][r['scene']]['arms']['prior0' if r['configuration']['pose_prior_weight'] == 0 else 'prior001'] = {'run': r['run']}
    lineage_gate(original, recovery, old, new, summary)
    cases = {}
    for name, which, mutate in [('old_relabeled_success', 2, lambda x: x.update(status='all_four_predeclared_full_sequence_runs_completed')),
                               ('old_failed_exit_overwritten', 2, lambda x: x['run_records'][-1].update(returncode=0)),
                               ('failed_final_metrics_fabricated', 2, lambda x: x['run_records'][-1].update(metrics_sha256='a' * 64)),
                               ('pending_replacement', 3, lambda x: x.update(status='running')),
                               ('extra_replacement_attempt', 3, lambda x: x.update(GPU_runs_started=2)),
                               ('incomplete_replacement', 3, lambda x: x['run_records'][0].update(frames=747)),
                               ('four_attempt_summary', 4, lambda x: x.update(attempt_count=4)),
                               ('summary_recovery_protocol_changed', 4, lambda x: x.update(recovery_protocol_sha256='b' * 64)),
                               ('recovery_deadline_extended', 1, lambda x: x.update(deadline_beijing='2026-10-09T09:00:00+08:00')),
                               ('checkpoint_resume', 1, lambda x: x['new_runs'][0]['configuration'].update(start_frame=521)),
                               ('GT_metric_selection', 1, lambda x: x.update(selected_final_runs=['best_ATE']))]:
        arguments = copy.deepcopy([original, recovery, old, new, summary]); mutate(arguments[which])
        try: lineage_gate(*arguments)
        except (ValueError, KeyError): cases[name] = 'rejected'
        else: raise AssertionError('Invalid lineage fixture accepted: ' + name)
    # Signed zero proves numeric equality is insufficient evidence of raw bytes.
    def fake(v): return SimpleNamespace(device=SimpleNamespace(type='cpu'), requires_grad=False, numpy=lambda: v)
    arrays = {k: fake(np.zeros((2, 2), np.float32)) for k in independent.INPUT_KEYS}
    params = {k: fake(np.zeros((2, 2), np.float32)) for k in independent.PARAMETER_KEYS}
    left = {**arrays, 'parameters': params, 'source_frame_ids_per_gaussian': [0, 0]}; right = copy.deepcopy(left)
    signed = np.zeros((2, 2), np.float32); signed[0, 0] = -0.; right['parameters']['means'] = fake(signed)
    pair = independent.compare_contexts(left, right)
    require(np.array_equal(left['parameters']['means'].numpy(), signed) and not pair['warmup_map_parameters']['means']['raw_bytes_equal'], 'Signed-zero byte negative control')
    output.mkdir(exist_ok=False)
    save(output / 'metrics.json', {'status': 'passed_CPU_lineage_fixtures_only', 'auditor_sha256': sha(Path(__file__)),
          'helper_sha256': sha(Path(independent.__file__)), 'negative_lineage_controls': cases,
          'valid_original3_failed4_plus_one_restart': True, 'signed_zero_numerically_equal_but_raw_bytes_differ': True,
          'reused_independent_math_controls': independent.synthetic_controls(), 'actual_summary_audits': 0,
          'live_progress_read': False, 'GPU_calls': 0, 'torch_or_gsplat_imported': any(k in sys.modules for k in ('torch', 'gsplat'))})
    (output / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': 'passed_CPU_lineage_fixtures_only', 'output': str(output), 'source_sha256': sha(Path(__file__))}))


def main():
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true'); mode.add_argument('--audit', action='store_true')
    p.add_argument('--run', default='v4-full-recovery-independent-auditor-cpu-001'); args = p.parse_args()
    if args.self_check:
        require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique simple fixture output')
        self_checks(ROOT / 'outputs' / args.run)
    else: audit()


if __name__ == '__main__': main()
