"""Read-only actual reference-ledger and trajectory audit of recovery controls.

Reuses the independently implemented SciPy/raw-GT metric helpers from the
previous result audit, never the odometry/loader/solver/evaluation modules.
No candidate, reference, threshold, sensor or original output is changed.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

import audit_rgbd_photogeometric_results as independent

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/protocol.json'
PROTOCOL_SHA = 'd2adb632c2261e50bc963e3b84c31d44e92233b3f399bd45a23fafd457ce13c1'
BATCH = ROOT / 'outputs/rgbd-photogeometric-last-accepted-validation-001'
METRIC_HELPER_SHA = 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'exact reviewed recovery protocol')
    require(sha(Path(independent.__file__)) == METRIC_HELPER_SHA, 'independent raw-GT/metric helper version')
    protocol, batch = read(PROTOCOL), read(BATCH / 'metrics.json')
    require(batch['status'] == 'all_two_predeclared_CPU_rounds_completed' and batch['runs_started'] == 2
            and batch['current_run'] is None and len(batch['run_records']) == 2, 'both actual rounds terminal')
    require(batch['protocol_sha256'] == PROTOCOL_SHA and batch['source_sha256'] == protocol['source_sha256'], 'actual batch protocol/source')
    watched = {PROTOCOL: sha(PROTOCOL), BATCH / 'metrics.json': sha(BATCH / 'metrics.json')}
    for name, expected in protocol['source_sha256'].items():
        path = ROOT / 'scripts' / name
        require(sha(path) == expected, 'frozen source ' + name)
        watched[path] = expected
    truth = {}
    for scene, witness in protocol['datasets'].items():
        truth[scene] = independent.truth_from_raw(witness)
        for name, expected in witness['index_sha256'].items():
            path = Path(witness['dataset']) / name
            require(sha(path) == expected, 'raw GT/sensor indices unchanged')
            watched[path] = expected
        bundle = protocol['sensor_bundles'][scene]
        path = ROOT / bundle['path']
        require(sha(path) == bundle['sha256'] and not bundle['contains_GT_poses'], 'actual GT-free sensor bundle unchanged')
        watched[path] = bundle['sha256']
    reviewed = []
    previous_end = None
    for declared, record in zip(protocol['rounds'], batch['run_records']):
        require(record['run'] == declared['run'] and record['round_id'] == declared['round_id']
                and record['order'] == declared['order'] and record['configuration'] == declared['configuration']
                and record['returncode'] == 0, 'fixed order/config/returncode')
        start, end = (datetime.fromisoformat(record[key]) for key in ('started_utc', 'ended_utc'))
        require(start < end and start < datetime.fromisoformat(protocol['deadline_beijing'])
                and (previous_end is None or previous_end <= start), 'serial actual process intervals/cutoff')
        previous_end = end
        scene = declared['scene']
        folder = ROOT / 'outputs' / declared['run']
        original = read(folder / 'metrics.json')
        require(sha(folder / 'metrics.json') == record['metrics_sha256']
                and sha(BATCH / (record['round_id'] + '.log')) == record['log_sha256'], 'original metrics/log hashes')
        require((folder / 'protocol_used.json').read_bytes() == PROTOCOL.read_bytes(), 'exact actual protocol copy')
        for path in folder.iterdir():
            if path.is_file():
                watched[path] = sha(path)
        require(original['configuration'] == declared['configuration'] and original['source_sha256'] == protocol['source_sha256']
                and original['frames_processed'] == 100, 'all100 actual rows/config/source')
        witness = protocol['datasets'][scene]
        require(original['selected_sensor_metadata'] == witness['selected_sensor_metadata']
                and original['input_array_hashes'] == witness['input_array_hashes'], 'exact original sensor metadata/arrays')
        require(not original['GT_pose_used_in_optimizer'] and original['GT_pose_evaluation_after_all_100_estimates_saved'], 'reported GT scope')
        ref = protocol['original_previous_frame_photo1_references'][scene]
        old_path = ROOT / 'outputs' / ref['run'] / 'metrics.json'
        require(sha(old_path) == ref['metrics_sha256'], 'original previous-frame reference unchanged')
        old = read(old_path)
        watched[old_path] = sha(old_path)
        require(original['configuration']['odometry'] == old['configuration']['odometry'], 'engineering objective unchanged')
        rows = original['frame_metrics']
        require([row['frame_id'] for row in rows] == list(range(200, 300)), 'every requested frame retained')
        estimated = np.array([row['estimated_c2w'] for row in rows])
        require(np.array_equal(estimated[0], np.eye(4)), 'same first-camera gauge')
        last = 0
        gap_counts, flag_counts = Counter(), Counter()
        failure_ids, recovery_pairs = [], []
        prefix_max = 0.
        for i, row in enumerate(rows[1:], 1):
            require(row['current_frame_id'] == row['frame_id'] == 200+i
                    and row['reference_frame_id'] == row['last_accepted_frame_id_before'] == 200+last
                    and row['reference_gap_frames'] == i-last, 'last-accepted reference ID and gap')
            require(row['reference_sensor_metadata'] == witness['selected_sensor_metadata'][last]
                    and row['source_sensor_metadata'] == witness['selected_sensor_metadata'][i], 'actual referenced sensor identity')
            require(row['maximum_sensor_frame_used_for_this_pair'] == 200+i and not row['pair_initialized_from_GT'], 'causal sensor range/initialization')
            require(row['accepted'] == (not bool(row['failure_flags'])), 'acceptance and full flags')
            candidate = row['candidate_reference_from_current']
            require(candidate == row['candidate_previous_from_current'], 'frozen candidate key means selected target')
            gap_counts[i-last] += 1
            flag_counts.update(row['failure_flags'])
            previous_reference = last
            if row['accepted']:
                transform = np.array(candidate)
                require(transform.shape == (4, 4) and np.isfinite(transform).all(), 'actual accepted finite candidate')
                expected = estimated[last] @ transform
                last = i
            else:
                require(np.array_equal(estimated[last], estimated[i-1]), 'held prefix equals last accepted pose')
                expected = estimated[i-1]
                failure_ids.append(200+i)
            difference = float(np.max(np.abs(expected-estimated[i])))
            prefix_max = max(prefix_max, difference)
            require(difference == 0., 'exact reference-pose composition or rejected hold')
            require(row['last_accepted_frame_id_after'] == 200+last, 'reference changes only when accepted')
            if i-previous_reference > 1:
                recovery_pairs.append({'current_frame_id': 200+i, 'reference_frame_id': 200+previous_reference,
                                       'gap': i-previous_reference, 'accepted': row['accepted'], 'failure_flags': row['failure_flags']})
        require(len(failure_ids) == original['failure_frame_count'] == record['failure_frame_count']
                and dict(flag_counts) == original['failure_flag_counts'], 'no failure omitted')
        require({str(k): v for k, v in gap_counts.items()} == original['reference_gap_counts'], 'complete gap histogram')
        with np.load(folder / 'trajectory_arrays.npz', allow_pickle=False) as saved:
            require(np.array_equal(saved['estimated_c2w'], estimated), 'actual saved estimates')
            gt_diff = float(np.max(np.abs(saved['evaluation_only_anchored_GT_c2w']-truth[scene])))
            require(gt_diff < 3e-14, 'raw-GT xyzw gauge reconstruction')
        calculated = independent.independent_metrics(estimated, truth[scene])
        differences = {key: float(np.max(np.abs(np.asarray(value)-np.asarray(original['trajectory'][key]))))
                       for key, value in calculated.items()}
        require(all(error < (1e-7 if 'rotation' in key else 1e-12) for key, error in differences.items()), 'independent metric agreement')
        first_old_rejected_local = next((i for i, row in enumerate(old['frame_metrics']) if row['failure_flags']), 100)
        common_initial_prefix = float(np.max(np.abs(estimated[:first_old_rejected_local+1]
                                      - np.array([row['estimated_c2w'] for row in old['frame_metrics'][:first_old_rejected_local+1]]))))
        reviewed.append({'scene': scene, 'new_run': declared['run'], 'previous_frame_run': ref['run'],
                         'complete_frames': 100, 'tracked_denominator': 99, 'failure_frame_ids': failure_ids,
                         'failure_flag_counts': dict(flag_counts), 'reference_gap_counts': dict(gap_counts),
                         'all_larger_gap_attempts': recovery_pairs, 'prefix_max_abs_difference': prefix_max,
                         'raw_GT_matrix_max_abs_difference': gt_diff,
                         'initial_prefix_through_original_first_rejection_max_abs_difference': common_initial_prefix,
                         'independent_metric_max_abs_differences': differences,
                         'recalculated_scalar_metrics': {key: value for key, value in calculated.items() if not isinstance(value, list)},
                         'recalculated_endpoint_raw_error_m': calculated['per_frame_raw_translation_error_m'][-1]})
    require(all(sha(path) == expected for path, expected in watched.items()), 'all consumed sources/artifacts unchanged')
    result = {'status': 'passed', 'audit_source_sha256': sha(Path(__file__)), 'independent_metric_helper_sha256': METRIC_HELPER_SHA,
              'protocol_sha256': PROTOCOL_SHA, 'batch_metrics_sha256': sha(BATCH / 'metrics.json'),
              'actual_rounds': 2, 'complete_frame_rows': 200, 'tracked_nonanchor_rows': 198,
              'read_only_consumed_files_rehashed': len(watched), 'scenes': reviewed, 'GPU_calls': 0,
              'limits': ['Frozen optimizer is not re-executed; actual reference IDs, sensor metadata and accepted/held prefix algebra are checked.',
                         'Reuses the previous independent raw-GT/SciPy metric implementation, not a third independent evaluator.',
                         'Reference gaps may change registration difficulty. Algebra validity does not prove all recovery attempts accurate.',
                         'GT-isolation evidence requires frozen source review/execution ordering; metadata and hashes alone are insufficient.',
                         'Both runs use the same previously examined Freiburg1 clips; no unseen-domain, loop-closure or full-SLAM claim.']}
    target = BATCH / 'independent_reference_trajectory_audit.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'rounds': 2, 'frames': 200,
                      'failure_ids': {r['scene']: r['failure_frame_ids'] for r in reviewed},
                      'larger_gap_attempts': {r['scene']: len(r['all_larger_gap_attempts']) for r in reviewed}}))


if __name__ == '__main__':
    main()
