"""JSON-only collector gated by an ACTUAL independent full-affine result audit.

No registration, renderer, loader, metric fitter, torch or GPU is imported.
Trajectory numbers are explicitly reused from completed results after binding
the independent e807/b549 audit. All rejected frame IDs and per-frame records
remain; each compact artifact is limited to1MiB or the collector fails visibly.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'outputs/rgbd-affine-full-cpu-preparation-001'
BATCH = ROOT/'outputs/rgbd-affine-full-validation-001'
PROTOCOL_SHA = 'ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a'
REVIEW_SHA = '977fae21421e9c71d51c29756b8999d4b075b5b8ed53a9c16d46176c88ecd44f'
HELPER_SHA = 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53'
STEP_HELPER_SHA = 'b549a0af2642b2e203e8f3b5f0deae46dcaf2458f608d7e27435c87b33f8c6d5'
AUDIT_STATUS = 'passed_independent_full_input_reference_step_and_trajectory_audit'
SCALARS = ['raw_anchored_translation_rmse_m', 'se3_aligned_ate_translation_rmse_m',
           'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_deg', 'raw_rotation_rmse_deg']


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())


def save(path, obj):
    raw = (json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')
    require(len(raw) <= 1024*1024, 'Artifact would exceed1MiB; no row/flag truncation: '+path.name)
    with path.open('xb') as stream: stream.write(raw)


def local(relative):
    target = (ROOT/relative).resolve()
    require(target.is_relative_to(ROOT) and target.is_file(), 'Safe existing local artifact')
    return target


def gates(audit_path, audit_sha):
    watched = {}
    def watch(path, expected=None):
        observed = sha(path); require(expected is None or observed == expected, 'Bound file hash '+str(path))
        watched[path] = observed
    watch(PREP/'protocol.json', PROTOCOL_SHA); protocol = read(PREP/'protocol.json')
    watch(PREP/'root_protocol_review.json', REVIEW_SHA)
    require(protocol['status'] == 'ready_for_root_review_no_real_registration', 'Original ready protocol')
    for name, digest in protocol['source_sha256'].items():
        watch(ROOT/'scripts'/name, digest); watch(PREP/'source_snapshot'/name, digest)
    watch(audit_path, audit_sha); audit = read(audit_path)
    require(audit['status'] == AUDIT_STATUS and audit['protocol_sha256'] == PROTOCOL_SHA and
            audit['root_review_sha256'] == REVIEW_SHA and audit['actual_rounds'] == 4 and
            audit['complete_frame_rows'] == 2594 and audit['tracked_nonanchor_pairs'] == 2590 and
            audit['GPU_calls'] == 0, 'Actual complete independent result audit, not a source-only or mock review')
    watch(ROOT/'scripts/audit_rgbd_affine_full_results.py', audit['audit_source_sha256'])
    require(audit['metric_helper_sha256'] == HELPER_SHA and audit['saved8D_step_helper_sha256'] == STEP_HELPER_SHA, 'Named independent helper reuse')
    watch(ROOT/'scripts/audit_rgbd_photogeometric_results.py', HELPER_SHA)
    watch(ROOT/'scripts/audit_rgbd_affine_real_results.py', STEP_HELPER_SHA)
    watch(PREP/'root-independent-full-preflight-review.json', audit['independent_preflight_sha256'])
    watch(BATCH/'metrics.json', audit['batch_metrics_sha256']); batch = read(BATCH/'metrics.json')
    require(batch['status'] == 'all_four_predeclared_CPU_full_sequence_rounds_completed' and
            batch['runs_started'] == 4 and len(batch['run_records']) == 4 and batch['current_run'] is None and
            batch['protocol_sha256'] == PROTOCOL_SHA and batch['source_sha256'] == protocol['source_sha256'] and batch['GPU_runs'] == 0,
            'Four actual exit0 terminal rounds required')
    watched_witnesses = {}
    for scene, binding in protocol['datasets'].items():
        path = local(binding['path']); watch(path, binding['sha256']); witness = read(path)
        watched_witnesses[scene] = witness
        for filename, digest in witness['index_sha256'].items(): watch(Path(witness['dataset'])/filename, digest)
        for row in witness['raw_sensor_files']:
            for key in ('rgb_path', 'depth_path'):
                p = (Path(witness['dataset'])/row[key]['path']).resolve()
                require(p.is_relative_to(Path(witness['dataset']).resolve()), 'PNG stays inside dataset')
                watch(p, row[key]['sha256'])
    require(len(audit['runs']) == 4 and [r['run'] for r in audit['runs']] == [r['run'] for r in protocol['rounds']], 'Audit exact predeclared4 order')
    runs = []; previous_end = None
    for declared, terminal, checked in zip(protocol['rounds'], batch['run_records'], audit['runs']):
        require(all(terminal[k] == declared[k] for k in ('run', 'scene', 'order', 'round_id', 'model_dimension', 'configuration')) and terminal['returncode'] == 0,
                'Original declared run/config and genuine exit0')
        start = dt.datetime.fromisoformat(terminal['started_utc']); end = dt.datetime.fromisoformat(terminal['ended_utc'])
        require(start < end and start < dt.datetime.fromisoformat(protocol['deadline_beijing']) and
                (previous_end is None or previous_end <= start), 'Actual serial intervals before deadline'); previous_end = end
        folder = ROOT/'outputs'/declared['run']; watch(folder/'metrics.json', terminal['metrics_sha256'])
        require(checked['metrics_sha256'] == terminal['metrics_sha256'] and checked['scene'] == declared['scene'] and
                checked['model_dimension'] == declared['model_dimension'], 'Audit binds actual result bytes and arm')
        watch(BATCH/(declared['round_id']+'.log'), terminal['log_sha256'])
        watch(folder/'pair_manifest.json'); pair_manifest = read(folder/'pair_manifest.json')
        m = read(folder/'metrics.json'); n = declared['configuration']['frames']; rows = m['frame_metrics']
        require(m['status'] in ('completed', 'completed_with_pair_rejections') and m['frames_processed'] == n == checked['frames'] and
                len(rows) == n and [r['frame_id'] for r in rows] == list(range(n)) and
                m['configuration'] == declared['configuration'] and m['source_sha256'] == protocol['source_sha256'] and
                m['protocol_sha256'] == PROTOCOL_SHA and m['GT_pose_used_in_optimizer'] is False and
                m['GT_pose_evaluation_after_all_estimates_saved'] is True and m['GPU_runs'] == 0, 'Exact complete sensor-only result')
        require(m['selected_sensor_metadata'] == watched_witnesses[declared['scene']]['selected_sensor_metadata'] and
                m['input_array_hashes'] == watched_witnesses[declared['scene']]['input_array_hashes'], 'Input arrays and frame identities exact')
        require(sorted(pair_manifest) == [f'pairs/{i:04d}.json' for i in range(n)], 'All immutable pair files retained')
        for i, row in enumerate(rows):
            path = folder/f'pairs/{i:04d}.json'; v = pair_manifest[f'pairs/{i:04d}.json']; watch(path, v['sha256'])
            require(path.stat().st_size == v['bytes'] and read(path) == row, 'Exact immutable full row')
        rejected = [r['frame_id'] for r in rows if r['failure_flags']]
        require(rejected == checked['all_rejected_frame_ids'] and len(rejected) == checked['failure_frame_count'] == m['failure_frame_count'] == terminal['failure_frame_count'], 'Every rejection stays in denominator')
        require(m['failure_flag_counts'] == checked['failure_flag_counts'] and
                {str(k):v for k,v in m['reference_gap_counts'].items()} == {str(k):v for k,v in checked['reference_gap_counts'].items()} and
                m['last_accepted_reference_id'] == checked['last_accepted_reference_id'], 'Flags/gaps/reference bind audit')
        for key, value in checked['recalculated_scalar_metrics'].items():
            require(math.isclose(float(value), float(m['trajectory'][key]), rel_tol=1e-9, abs_tol=1e-7 if 'rotation' in key else 1e-11), 'Audited scalar '+key)
        runs.append((declared, terminal, checked, m))
    return protocol, batch, audit, watched_witnesses, runs, watched


def rejection_segments(rows):
    segments = []
    for row in rows[1:]:
        if not row['failure_flags']: continue
        i = row['frame_id']
        if not segments or segments[-1]['last_frame_id'] != i-1:
            segments.append({'first_frame_id': i, 'last_frame_id': i, 'frames': 0, 'flags': [], 'reference_ids': []})
        s = segments[-1]; s['last_frame_id'] = i; s['frames'] += 1
        s['flags'] = sorted(set(s['flags']) | set(row['failure_flags']))
        s['reference_ids'] = sorted(set(s['reference_ids']) | {row['reference_frame_id']})
    return segments


def compact_frames(m, dimension):
    columns = ['frame_id', 'rgb_index', 'depth_index', 'gt_index', 'rgb_timestamp', 'accepted', 'failure_flags',
        'reference_frame_id', 'reference_gap_frames', 'last_accepted_frame_id_before', 'last_accepted_frame_id_after',
        'estimated_c2w', 'candidate_reference_from_current', 'candidate_log_relative_gain', 'candidate_relative_gain',
        'candidate_bias_intensity', 'final_rank', 'column_normalized_hessian_condition', 'plane_rmse_m',
        'intensity_rmse', 'raw_translation_error_m', 'SE3_ATE_translation_error_m', 'adjacent_RPE_translation_m', 'adjacent_RPE_rotation_deg']
    rows = []
    t = m['trajectory']
    for i, (r, sensor) in enumerate(zip(m['frame_metrics'], m['selected_sensor_metadata'])):
        stats = r.get('final_statistics', {})
        rows.append([i, sensor['rgb_index'], sensor['depth_index'], sensor['gt_index'], sensor['rgb_timestamp'],
            r['accepted'], r['failure_flags'], r['reference_frame_id'], r['reference_gap_frames'],
            r['last_accepted_frame_id_before'], r['last_accepted_frame_id_after'], r['estimated_c2w'],
            r.get('candidate_reference_from_current'), r.get('candidate_log_relative_gain'), r.get('candidate_relative_gain'),
            r.get('candidate_bias_intensity'), stats.get('joint_rank', stats.get('rank')),
            stats.get('column_normalized_hessian_condition'), stats.get('plane_rmse_m'),
            stats.get('affine_corrected_intensity_rmse', stats.get('intensity_rmse')),
            t['per_frame_raw_translation_error_m'][i], t['per_frame_se3_aligned_ate_translation_error_m'][i],
            t['per_interval_rpe_translation_error_m'][i-1] if i else None,
            t['per_interval_rpe_rotation_error_deg'][i-1] if i else None])
    result = {'status': 'all_original_frames_retained', 'model_dimension': dimension, 'columns': columns, 'rows': rows,
        'RPE_row_convention': 'row i stores interval i-1 -> i; anchor null',
        'null_semantics': 'First anchor/no exported value/inapplicable6D nuisance; never replaced with zero or a selected best frame.',
        'full_iteration_history_retained_in_original_metrics_and_immutable_pairs': True}
    result['persisted_field_preservation_check'] = compact_field_check(result, m, dimension)
    return result


def compact_field_check(compact, original, dimension):
    """Lossless-subset check, not an independent metric or geometry audit."""
    columns = compact['columns']; rows = compact['rows']; complete = original['frame_metrics']
    require(len(set(columns)) == len(columns) and len(rows) == len(complete), 'Unique columns/all frames')
    direct_fields = ['frame_id', 'accepted', 'failure_flags', 'reference_frame_id', 'reference_gap_frames',
                     'last_accepted_frame_id_before', 'last_accepted_frame_id_after', 'estimated_c2w']
    optional_fields = ['candidate_reference_from_current', 'candidate_log_relative_gain',
                       'candidate_relative_gain', 'candidate_bias_intensity']
    metadata_fields = ['rgb_index', 'depth_index', 'gt_index', 'rgb_timestamp']
    retained_flags = Counter(); null_candidates = 0
    for index, (values, frame, sensor) in enumerate(zip(rows, complete, original['selected_sensor_metadata'])):
        require(len(values) == len(columns), 'Every row has every declared column')
        saved = dict(zip(columns, values))
        for key in direct_fields: require(saved[key] == frame[key], 'Persisted compact exact '+key)
        for key in optional_fields: require(saved[key] == frame.get(key), 'Persisted optional compact exact '+key)
        for key in metadata_fields: require(saved[key] == sensor[key], 'Persisted sensor compact exact '+key)
        require(saved['frame_id'] == index and saved['accepted'] == (not saved['failure_flags']), 'Frame order/decision exact')
        retained_flags.update(saved['failure_flags']); null_candidates += saved['candidate_reference_from_current'] is None
        require(saved['raw_translation_error_m'] == original['trajectory']['per_frame_raw_translation_error_m'][index] and
                saved['SE3_ATE_translation_error_m'] == original['trajectory']['per_frame_se3_aligned_ate_translation_error_m'][index],
                'Stored per-frame metric arrays retained exactly, no refit')
        if index:
            require(saved['adjacent_RPE_translation_m'] == original['trajectory']['per_interval_rpe_translation_error_m'][index-1] and
                    saved['adjacent_RPE_rotation_deg'] == original['trajectory']['per_interval_rpe_rotation_error_deg'][index-1], 'All intervals at correct destination row')
        else:
            require(saved['adjacent_RPE_translation_m'] is None and saved['adjacent_RPE_rotation_deg'] is None, 'No fake first interval')
    require(dict(retained_flags) == original['failure_flag_counts'], 'Every original multi-flag occurrence retained')
    return {'status': 'passed_exact_persisted_field_subset_check', 'frames': len(rows), 'model_dimension': dimension,
            'direct_fields_checked': direct_fields, 'optional_fields_checked': optional_fields,
            'sensor_fields_checked': metadata_fields, 'all_metric_array_rows_and_intervals_exact': True,
            'failure_flag_counts': dict(retained_flags), 'null_candidate_count_including_anchor': null_candidates,
            'meaning': 'Checks serialization/subsetting only. Independent trajectory/update audit is separately bound, not replaced.'}


def collect(args):
    protocol, batch, audit, witnesses, runs, watched = gates(args.audit, args.audit_sha)
    output = ROOT/'outputs'/args.run; output.mkdir(exist_ok=False)
    (output/'source_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        summary = {'status': 'completed_after_independent_full_result_audit', 'source_sha256': sha(Path(__file__)),
            'protocol_sha256': PROTOCOL_SHA, 'batch_metrics_sha256': sha(BATCH/'metrics.json'), 'audit_sha256': args.audit_sha,
            'audit_source_sha256': audit['audit_source_sha256'], 'complete_new_frame_rows': 2594, 'tracked_new_pairs': 2590,
            'frozen_pipeline_source_sha256': protocol['source_sha256'], 'metric_helper_sha256': HELPER_SHA,
            'saved8D_step_helper_sha256': STEP_HELPER_SHA,
            'scene_results': {}, 'CPU_round_returncodes': [r['returncode'] for r in batch['run_records']], 'GPU_calls': 0,
            'limits': ['Stored trajectory scalars reused only after independent e807/b549 audit; collector is not another metric implementation.',
                       'All rejections propagate their saved prefix; completion is not accuracy or convergence.',
                       '6D/8D whole model and acceptance paths differ; no isolated brightness causality, GT tuning or cross-scene mean.',
                       'Raw errors share first-camera gauge. ATE is posthoc SE3 without scale; RPE is adjacent. No speed/FPS claim.',
                       'Complete raw pair/step records remain in original local outputs; compact rows preserve every frame and flag.']}
        for declared, terminal, checked, m in runs:
            scene, dimension = declared['scene'], declared['model_dimension']; key = f'{scene}-{dimension}d'
            segments = rejection_segments(m['frame_metrics'])
            frame_file = key+'-frames.json'; save(output/frame_file, compact_frames(m, dimension))
            arm = {'run': declared['run'], 'model_dimension': dimension, 'metrics_sha256': terminal['metrics_sha256'],
                'configuration': declared['configuration'], 'status': m['status'], 'frames': m['frames_processed'],
                'tracked_pairs': m['tracked_nonanchor_frame_denominator'], 'failure_frame_count': m['failure_frame_count'],
                'all_rejected_frame_ids': checked['all_rejected_frame_ids'], 'failure_flag_counts': m['failure_flag_counts'],
                'rejection_segments': segments, 'longest_rejection_segment': max((x['frames'] for x in segments), default=0),
                'reference_gap_counts': m['reference_gap_counts'], 'maximum_reference_gap_frames': checked['maximum_reference_gap_frames'],
                'last_accepted_reference_id': m['last_accepted_reference_id'],
                'trajectory_scalars': {k:m['trajectory'][k] for k in SCALARS},
                'endpoint_raw_translation_m': checked['endpoint_raw_translation_m'], 'alignment_scale': m['trajectory']['alignment_scale'],
                'observed_wall_seconds_including_IO': m['observed_wall_seconds'], 'started_utc': terminal['started_utc'],
                'ended_utc': terminal['ended_utc'], 'returncode': terminal['returncode'], 'all_frame_file': frame_file,
                'all_frame_file_sha256': sha(output/frame_file), 'audited_saved_step_counts': checked['all_pair_step_checks'],
                'last_camera': {'frame_id': m['frame_metrics'][-1]['frame_id'],
                    'rgb_timestamp': m['selected_sensor_metadata'][-1]['rgb_timestamp'],
                    'accepted': m['frame_metrics'][-1]['accepted'], 'failure_flags': m['frame_metrics'][-1]['failure_flags'],
                    'reference_frame_id': m['frame_metrics'][-1]['reference_frame_id'],
                    'last_accepted_frame_id_after': m['frame_metrics'][-1]['last_accepted_frame_id_after'],
                    'estimated_c2w': m['frame_metrics'][-1]['estimated_c2w'],
                    'candidate_reference_from_current': m['frame_metrics'][-1].get('candidate_reference_from_current'),
                    'endpoint_raw_translation_m': checked['endpoint_raw_translation_m'],
                    'rejected_endpoint_policy': 'If rejected, this is the retained preceding estimate at the final sensor timestamp, not the last accepted sensor timestamp or a removed frame.'}}
            save(output/(key+'-summary.json'), arm)
            summary['scene_results'].setdefault(scene, {'association': witnesses[scene]['association'], 'arms': []})['arms'].append(arm)
        require(all(sha(p) == h for p, h in watched.items()), 'All consumed files unchanged after collection')
        summary['consumed_files_rehashed_unchanged'] = len(watched)
        # Keep every source/input hash without requiring one oversized public JSON.
        entries = sorted((str(p.relative_to(ROOT)).replace('\\','/'), h) for p,h in watched.items())
        pages = []
        for index, offset in enumerate(range(0, len(entries), 1000)):
            filename = f'input-hashes-{index:02d}.json'
            subset = dict(entries[offset:offset+1000]); save(output/filename, subset)
            pages.append({'path': filename, 'entries': len(subset), 'bytes': (output/filename).stat().st_size, 'sha256': sha(output/filename)})
        save(output/'input_hashes.json', {'total_entries': len(entries), 'pages': pages, 'all_entries_retained': True})
        summary['input_hash_manifest'] = {'path': 'input_hashes.json', 'sha256': sha(output/'input_hashes.json')}
        save(output/'summary.json', summary)
        save(output/'completion.json', {'status': 'completed', 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(output.iterdir()) if p.is_file()},
                                       'source_sha256': sha(Path(__file__)), 'GPU_calls': 0})
        return {'status': summary['status'], 'run': args.run, 'summary_sha256': sha(output/'summary.json')}
    except BaseException as error:
        save(output/'failure.json', {'status': 'collector_failed_partial_outputs_preserved', 'exception_type': type(error).__name__,
                                   'message': str(error), 'traceback': traceback.format_exc(), 'GPU_calls': 0})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=BATCH/'independent_full_affine_result_audit.json')
    p.add_argument('--audit-sha', required=True)
    p.add_argument('--run', default='rgbd-affine-full-comparison-001')
    p.add_argument('--check-inputs', action='store_true'); args = p.parse_args()
    require(re.fullmatch(r'[a-f0-9]{64}', args.audit_sha) is not None and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run) is not None, 'Exact audit hash and exclusive run')
    if args.check_inputs:
        protocol, batch, audit, witnesses, runs, watched = gates(args.audit, args.audit_sha)
        print(json.dumps({'status': 'all_actual_independent_audit_and_input_gates_passed_no_summary_written',
                          'frames': sum(m['frames_processed'] for _,_,_,m in runs), 'watched_files': len(watched), 'GPU_calls': 0}))
    else: print(json.dumps(collect(args)))


if __name__ == '__main__': main()
