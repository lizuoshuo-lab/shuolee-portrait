"""Read-only comparison of all public affine rows with immutable raw results.

This verifies copied fields, compact history counts and complete ledgers; it
does not repeat the independent trajectory formulas or solve registrations.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/rgbd-affine-photogeometric-comparison-002'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def history_counts(rows):
    return {
        'records': len(rows),
        'accepted_updates': sum(bool(row['accepted']) for row in rows),
        'stop_reasons': dict(Counter(row.get('stop_reason', 'continued') for row in rows)),
        'joint_rank_histogram': {str(k): v for k, v in Counter(row['joint_rank'] for row in rows if 'joint_rank' in row).items()},
        'line_search_trial_count': sum(len(row['line_search_trials']) for row in rows),
        'invalid_trial_counts': dict(Counter(trial['invalid_reason'] for row in rows for trial in row['line_search_trials'] if trial.get('invalid_reason'))),
    }

def main():
    summary = read(OUTPUT / 'metrics.json')
    assert summary['status'] == 'completed_read_only_four_method_comparison_after_actual_affine_audit'
    assert summary['actual_methods'] == 4 and summary['all_frame_count'] == 400
    audit_path = ROOT / 'outputs/rgbd-affine-photogeometric-real-validation-001/independent_affine_result_audit.json'
    assert sha(audit_path) == summary['affine_independent_audit_sha256']
    assert read(audit_path)['status'] == 'passed_independent_input_reference_8D_step_and_trajectory_audit'
    watched = {OUTPUT / 'metrics.json': sha(OUTPUT / 'metrics.json'), audit_path: sha(audit_path)}
    for item in summary['public_compact_summaries'] + summary['figures']:
        path = ROOT / item['path']
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        watched[path] = item['sha256']
    all_ledgers = read(OUTPUT / 'complete-reference-candidate-ledgers.json')
    assert all_ledgers['all_frame_count'] == 400 and len(all_ledgers['methods']) == 4
    results = []
    for method in summary['methods']:
        original_path = ROOT / 'outputs' / method['run'] / 'metrics.json'
        assert sha(original_path) == method['metrics_sha256']
        watched[original_path] = method['metrics_sha256']
        original = read(original_path)
        public = read(ROOT / method['public_compact_summary']['path'])
        raw_rows = original['frame_metrics']
        assert len(raw_rows) == len(public['frame_metrics']) == 100
        assert [row['frame_id'] for row in raw_rows] == list(range(200, 300))
        for key, value in original.items():
            if key not in ('frame_metrics', 'observed_wall_seconds'):
                assert public[key] == value, (method['run'], key)
        ledger = next(item for item in all_ledgers['methods'] if item['run'] == method['run'])
        assert len(ledger['all100_frame_rows']) == 100
        for original_row, public_row, ledger_row in zip(raw_rows, public['frame_metrics'], ledger['all100_frame_rows']):
            copied = {key: value for key, value in original_row.items() if key not in ('configuration', 'iteration_history')}
            exported = {key: value for key, value in public_row.items() if key != 'iteration_history_summary'}
            assert copied == exported, (method['run'], original_row['frame_id'])
            if 'iteration_history' in original_row:
                assert public_row['iteration_history_summary'] == history_counts(original_row['iteration_history'])
            assert all(original_row[key] == value for key, value in ledger_row.items())
            required = ('local_index', 'frame_id', 'reference_frame_id', 'reference_gap_frames', 'last_accepted_frame_id_before',
                        'last_accepted_frame_id_after', 'accepted', 'failure_flags', 'estimated_c2w', 'candidate_reference_from_current',
                        'candidate_log_relative_gain', 'candidate_relative_gain', 'candidate_bias_intensity')
            assert all(key in ledger_row for key in required if key in original_row)
        assert method['failure_frame_ids'] == [row['frame_id'] for row in raw_rows if row['failure_flags']]
        assert method['failure_frame_count'] == original['failure_frame_count']
        assert method['trajectory_metrics'] == {key: value for key, value in original['trajectory'].items() if not isinstance(value, list)}
        assert method['endpoint_raw_translation_error_m'] == original['trajectory']['per_frame_raw_translation_error_m'][-1]
        if method['method'] == 'affine8':
            for key in ('candidate_log_relative_gain', 'candidate_relative_gain', 'candidate_bias_intensity'):
                values = [row[key] for row in raw_rows[1:] if row.get(key) is not None]
                expected = {'defined_pair_denominator': len(values), 'all_pair_denominator': 99,
                            'mean': math.fsum(values) / len(values) if values else None,
                            'min': min(values) if values else None, 'max': max(values) if values else None}
                assert method['candidate_brightness_statistics'][key] == expected
            assert method['candidate_brightness_statistics']['includes_rejected_candidates']
        results.append({'run': method['run'], 'all_rows': 100, 'retained_rejected_frame_ids': method['failure_frame_ids']})
    assert all(sha(path) == value for path, value in watched.items())
    result = {'status': 'passed_all400_public_fields_histories_and_candidate_ledgers', 'source_sha256': sha(Path(__file__)),
              'summary_sha256': sha(OUTPUT / 'metrics.json'), 'all_rows': 400, 'methods': results,
              'consumed_artifact_hashes_verified': len(watched), 'GPU_calls': 0,
              'scope': 'Exact field preservation and count aggregation; actual independent raw-GT/SE3 trial audit remains separately hash-bound.'}
    with (OUTPUT / 'root_public_field_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'rows': 400, 'hashes': len(watched)}))

if __name__ == '__main__':
    main()
