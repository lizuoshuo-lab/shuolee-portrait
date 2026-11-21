"""Read-only CPU audit of the reviewed two actual affine RGB-D rounds.

No registration/odometry/loader/evaluation modules are imported. Explicitly
reuses reviewed e807 raw-GT/SciPy metrics, rather than claiming a third formula.
SciPy expm independently checks every saved8D update/trial and its left-SE3
direction; persisted ranks/costs are checked for consistency, not recomputed
from unexported correspondence matrices. A rejected pair is never discarded.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import audit_rgbd_photogeometric_results as independent

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/rgbd-affine-photogeometric-real-cpu-preparation-001'
PROTOCOL = PREP / 'protocol.json'
PROTOCOL_SHA = '9d8e674827a83f2c29e7dc901f2bcb8e1203b5513d89692b5aa92b34db0d8a63'
ROOT_REVIEW_SHA = '6aa84318ebb9b28448c156b44124bac07f6425c9bd69a3450e4b30b849085120'
HELPER_SHA = 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53'
BATCH = ROOT / 'outputs/rgbd-affine-photogeometric-real-validation-001'
require, sha, read = independent.require, independent.sha, independent.read


def se3(delta):
    x, y, z = delta[3:6]; algebra = np.zeros((4, 4))
    algebra[:3, :3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]; algebra[:3, 3] = delta[:3]
    return expm(algebra)


def close(a, b, message, tol=3e-12):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))); require(error < tol, message); return error


def step_ledger(result):
    if result.get('candidate_unknown_due_to_exception'):
        require(not result['accepted'] and result['candidate_reference_from_current'] is None and 'pair_numeric_exception' in result['failure_flags'], 'Unknown exception candidate honestly retained')
        return {'unknown_due_to_numeric_exception': True, 'steps': 0, 'trials': 0}
    transform = np.eye(4); a = 0.; b = 0.; trials = 0; updates = 0; ranks = Counter(); invalid = Counter(); stops = Counter(); maximum = 0.
    level_counts = Counter(); level_updates = Counter()
    for row in result['iteration_history']:
        level = row['level']; level_counts[level] += 1
        maximum = max(maximum, close(row['transform_before'], transform, 'Every step starts at saved preceding left-SE3 state'))
        require(row['log_gain_before'] == a and row['bias_before'] == b, 'Joint nuisance state chain starts0 and persists only within pair')
        if 'joint_rank' in row: ranks[row['joint_rank']] += 1
        if 'stop_reason' in row: stops[row['stop_reason']] += 1
        if 'proposed_delta_8d' in row:
            delta = np.asarray(row['proposed_delta_8d']) * row['shared_pose_limit_factor']; require(delta.shape == (8,) and np.isfinite(delta).all(), 'Finite full8D direction')
            chosen = None
            for trial in row['line_search_trials']:
                trials += 1; scale = trial['scale']; expected_T = se3(delta * scale) @ transform
                maximum = max(maximum, close(trial['candidate_transform'], expected_T, 'Independent SciPy left-SE3 trial'))
                require(trial['candidate_log_gain'] == float(a + delta[6] * scale) and trial['candidate_bias'] == float(b + delta[7] * scale), 'Unclamped additive nuisance trial')
                require(trial['same_correspondence_denominator'] == row['counts']['joint_correspondences'], 'Fixed trial denominator')
                if trial['objective'] is None: require(trial['invalid_reason'] is not None, 'Whole invalid trial retained'); invalid[trial['invalid_reason']] += 1
                else:
                    require(math.isfinite(trial['objective']) and trial['invalid_reason'] is None, 'Finite valid whole objective')
                    if trial['objective'] < row['normalized_robust_objective']:
                        require(chosen is None, 'Only first descending trial selected'); chosen = trial
            require(row['accepted'] == (chosen is not None), 'Selection exactly strict first finite descent')
            if chosen is not None:
                require(chosen == row['line_search_trials'][-1] and row['accepted_scale'] == chosen['scale'] and row['fixed_correspondence_objective_after'] == chosen['objective'], 'First descent terminates search')
                maximum = max(maximum, close(row['accepted_delta_8d'], delta * chosen['scale'], 'All8 accepted delta components retained'))
                transform = np.asarray(chosen['candidate_transform']); a = chosen['candidate_log_gain']; b = chosen['candidate_bias']; updates += 1; level_updates[level] += 1
                maximum = max(maximum, close(row['transform_after'], transform, 'Accepted after-state exact'))
                require(row['log_gain_after'] == a and row['bias_after'] == b, 'Accepted nuisance after-state exact')
        else: require(not row['accepted'] and not row['line_search_trials'], 'No invented step without direction')
    maximum = max(maximum, close(result['candidate_reference_from_current'], transform, 'Final full candidate equals last within-pair state'))
    require(result['candidate_log_relative_gain'] == a and result['candidate_bias_intensity'] == b and result['candidate_relative_gain'] == math.exp(a), 'Final affine correction equals unbounded exp(a)/b')
    require(len(result['levels']) == 3, 'All three levels retained')
    for row in result['levels']:
        require(row['iterations_recorded'] == level_counts[row['level']] and row['accepted_updates'] == level_updates[row['level']], 'Exact level iteration/update counts')
    if result['accepted']:
        require(result['final_statistics']['joint_rank'] == 8 and result['final_statistics']['column_normalized_hessian_condition'] is not None, 'Accepted full8D persisted rank')
    return {'steps': len(result['iteration_history']), 'trials': trials, 'accepted_updates': updates,
            'persisted_step_rank_histogram': dict(ranks), 'invalid_trial_counts': dict(invalid), 'step_stop_counts': dict(stops),
            'independent_SciPy_SE3_max_abs_difference': maximum}


def text_check(path, metadata, poses):
    rows = independent.rows(path); require(len(rows) == 100 and all(len(r) == 8 for r in rows), 'All100 TUM text rows')
    positional = 0.; angular = 0.; previous = None
    for fields, witness, expected in zip(rows, metadata, poses):
        stamp = Decimal(fields[0]); require(stamp == Decimal(f"{witness['rgb_timestamp']:.9f}") and (previous is None or stamp > previous), 'Actual paired9decimal timestamps'); previous = stamp
        values = np.asarray(fields[1:], dtype=float); positional = max(positional, close(values[:3], expected[:3, 3], 'Actual text translation', 1e-11))
        angular = max(angular, float(Rotation.from_matrix(Rotation.from_quat(values[3:]).as_matrix().T @ expected[:3, :3]).magnitude()))
    require(angular < 1e-10, 'Actual xyzw/c2w text rotation')
    return {'rows': len(rows), 'translation_max_abs_difference_m': positional, 'rotation_max_difference_rad': angular}


def main():
    require(sha(PROTOCOL) == PROTOCOL_SHA and sha(PREP / 'root_protocol_review.json') == ROOT_REVIEW_SHA and sha(Path(independent.__file__)) == HELPER_SHA, 'Exact reviewed protocol/review/independent metric source')
    protocol = read(PROTOCOL); ledger = read(BATCH / 'metrics.json')
    require(ledger['status'] == 'all_two_predeclared_CPU_rounds_completed' and ledger['runs_started'] == 2 and ledger['current_run'] is None and len(ledger['run_records']) == 2, 'Only both actual terminal rounds audited')
    require(ledger['protocol_sha256'] == PROTOCOL_SHA and ledger['source_sha256'] == protocol['source_sha256'], 'Actual ledger source/protocol')
    watched = {}; inputs = Counter(); scenes = []; truth = {}
    def watch(path, expected=None):
        digest = sha(path); require(expected is None or digest == expected, 'Actual consumed SHA ' + str(path)); watched[path] = digest
    watch(PROTOCOL, PROTOCOL_SHA); watch(BATCH / 'metrics.json'); watch(PREP / 'root_protocol_review.json', ROOT_REVIEW_SHA); watch(Path(independent.__file__), HELPER_SHA)
    for name, expected in protocol['source_sha256'].items(): watch(ROOT / 'scripts' / name, expected); watch(PREP / name, expected)
    for scene, witness in protocol['datasets'].items():
        dataset = Path(witness['dataset'])
        for name, expected in witness['index_sha256'].items(): watch(dataset / name, expected); inputs['index_files'] += 1
        for row in witness['raw_sensor_files']:
            for kind in ('rgb_path', 'depth_path'): watch(dataset / row[kind]['path'], row[kind]['sha256']); inputs['raw_sensor_PNGs'] += 1
        bundle = protocol['sensor_bundles'][scene]; require(bundle['contains_GT_poses'] is False, 'GT-free sensor bundle'); watch(ROOT / bundle['path'], bundle['sha256'])
        with np.load(ROOT / bundle['path'], allow_pickle=False) as saved:
            require(len(saved.files) == 15, 'Exact15 sensor arrays per scene')
            for level, hashes in witness['input_array_hashes'].items():
                for key, frozen in hashes.items():
                    value = saved['level' + level + '_' + key]
                    require(list(value.shape) == frozen['shape'] and str(value.dtype) == frozen['dtype'] and value.nbytes == frozen['bytes'] and hashlib.sha256(value.tobytes(order='C')).hexdigest() == frozen['sha256'], 'Actual declared sensor array'); inputs['sensor_arrays'] += 1
                require(np.array_equal(saved['level' + level + '_rgb'].astype(np.float64) @ np.array([.299, .587, .114]), saved['level' + level + '_gray']), 'Actual fixed gray weights')
        truth[scene] = independent.truth_from_raw(witness)
    previous_end = None
    for declared, record in zip(protocol['rounds'], ledger['run_records']):
        require(all(record[k] == declared[k] for k in ('round_id', 'run', 'order', 'configuration')) and record['returncode'] == 0, 'Declared exact row/exit0')
        start, end = [datetime.fromisoformat(record[k]) for k in ('started_utc', 'ended_utc')]
        require(start < end and start < datetime.fromisoformat(protocol['deadline_beijing']) and (previous_end is None or previous_end <= start), 'Actual serial intervals/start cutoff'); previous_end = end
        folder = ROOT / 'outputs' / declared['run']; watch(folder / 'metrics.json', record['metrics_sha256']); watch(BATCH / (record['round_id'] + '.log'), record['log_sha256'])
        require((folder / 'protocol_used.json').read_bytes() == PROTOCOL.read_bytes(), 'Exact actual protocol bytes')
        for path in folder.iterdir():
            if path.is_file(): watch(path)
        data = read(folder / 'metrics.json'); witness = protocol['datasets'][declared['scene']]
        require(data['configuration'] == declared['configuration'] and data['source_sha256'] == protocol['source_sha256'] and data['frames_processed'] == 100 and data['status'] in ('completed', 'completed_with_pair_rejections'), 'Actual complete frozen configuration/source')
        require(data['selected_sensor_metadata'] == witness['selected_sensor_metadata'] and data['input_array_hashes'] == witness['input_array_hashes'], 'All actual sensors match reviewed input witnesses')
        require(data['GT_pose_evaluation_after_all_100_estimates_saved'] and data['GT_pose_used_in_optimizer'] is False, 'Reported final-only GT scope')
        ref = protocol['original_lastaccepted_photo1_references'][declared['scene']]; old = ROOT / 'outputs' / ref['run']; watch(old / 'metrics.json', ref['metrics_sha256']); watch(old / 'trajectory_estimated.txt', ref['trajectory_sha256'])
        require(data['configuration']['odometry'] == read(old / 'metrics.json')['configuration']['odometry'] == data['configuration']['joint_affine']['base'], 'All base weights/scales/gates remain exact oldphoto1')
        rows = data['frame_metrics']; require([r['frame_id'] for r in rows] == list(range(200, 300)), 'All requested100 rows retained')
        poses = np.asarray([r['estimated_c2w'] for r in rows]); require(np.array_equal(poses[0], np.eye(4)), 'Actual anchor identity')
        reference = 0; flags = Counter(); gaps = Counter(); rejected = []; bigger_gaps = []; stepstats = []; maximum = 0.
        for index, row in enumerate(rows[1:], 1):
            require(row['current_frame_id'] == row['frame_id'] == 200 + index and row['reference_frame_id'] == row['last_accepted_frame_id_before'] == 200 + reference and row['reference_gap_frames'] == index - reference, 'Current/last-accepted sensor/reference/gap IDs')
            require(row['reference_sensor_metadata'] == witness['selected_sensor_metadata'][reference] and row['source_sensor_metadata'] == witness['selected_sensor_metadata'][index] and row['maximum_sensor_frame_used_for_this_pair'] == 200 + index, 'Actual causal referenced sensor metadata')
            require(not row['pair_initialized_from_GT'] and np.array_equal(row['pair_initial_transform'], np.eye(4)) and row['pair_initial_log_relative_gain'] == row['pair_initial_bias_intensity'] == 0. and row['brightness_accumulated_across_pairs'] is False, 'Every pair resets pose/loggain/bias without GT or nuisance accumulation')
            require(row['accepted'] == (not row['failure_flags']), 'Actual acceptance/retained flags'); flags.update(row['failure_flags']); gaps[index - reference] += 1
            stepstats.append({'frame_id': row['frame_id'], **step_ledger(row)})
            if row['reference_gap_frames'] > 1: bigger_gaps.append({k: row[k] for k in ('frame_id', 'reference_frame_id', 'reference_gap_frames', 'accepted', 'failure_flags')})
            if row['accepted']: expected = poses[reference] @ np.asarray(row['candidate_reference_from_current']); reference = index
            else: require(np.array_equal(poses[reference], poses[index - 1]), 'Reject prefix equals held reference'); expected = poses[index - 1]; rejected.append(row['frame_id'])
            error = float(np.max(np.abs(expected - poses[index]))); maximum = max(maximum, error); require(error == 0. and row['last_accepted_frame_id_after'] == 200 + reference, 'Exact accepted Cref@T or previous held prefix')
        require(len(rejected) == data['failure_frame_count'] == record['failure_frame_count'] and dict(flags) == data['failure_flag_counts'] and {str(k): v for k, v in gaps.items()} == data['reference_gap_counts'], 'All failures/denominators/gaps preserved')
        with np.load(folder / 'trajectory_arrays.npz', allow_pickle=False) as saved:
            require(np.array_equal(saved['estimated_c2w'], poses), 'Actual NPZ estimates'); gt_error = close(saved['evaluation_only_anchored_GT_c2w'], truth[declared['scene']], 'Independent raw xyzw GT gauge', 3e-14)
        metrics = independent.independent_metrics(poses, truth[declared['scene']]); differences = {k: float(np.max(np.abs(np.asarray(v) - np.asarray(data['trajectory'][k])))) for k, v in metrics.items()}
        require(all(v < (1e-7 if 'rotation' in k else 1e-12) for k, v in differences.items()), 'Independent actual trajectory metrics')
        scenes.append({'scene': declared['scene'], 'run': declared['run'], 'metrics_sha256': record['metrics_sha256'], 'complete_frames': 100, 'tracked_pairs': 99,
                       'failure_frame_ids': rejected, 'failure_flag_counts': dict(flags), 'reference_gap_counts': dict(gaps), 'all_larger_gap_attempts': bigger_gaps,
                       'exact_prefix_max_abs_difference': maximum, 'raw_GT_anchor_max_abs_difference': gt_error, 'step_checks_all99_pairs': stepstats,
                       'trajectory_text': text_check(folder / 'trajectory_estimated.txt', witness['selected_sensor_metadata'], poses),
                       'independent_metric_max_abs_differences': differences,
                       'recalculated_scalar_metrics': {k: v for k, v in metrics.items() if not isinstance(v, list)}, 'endpoint_raw_translation_m': metrics['per_frame_raw_translation_error_m'][-1]})
    require(all(sha(path) == digest for path, digest in watched.items()), 'All consumed source/evidence unchanged')
    result = {'status': 'passed_independent_input_reference_8D_step_and_trajectory_audit', 'audit_source_sha256': sha(Path(__file__)), 'metric_helper_sha256': HELPER_SHA,
              'protocol_sha256': PROTOCOL_SHA, 'root_review_sha256': ROOT_REVIEW_SHA, 'batch_metrics_sha256': sha(BATCH / 'metrics.json'), 'actual_rounds': 2, 'complete_frame_rows': 200,
              'tracked_nonanchor_pairs': 198, 'actual_input_checks': dict(inputs), 'consumed_files_rehashed_unchanged': len(watched), 'scenes': scenes, 'GPU_calls': 0,
              'limits': ['Reuses reviewed e807 raw-GT/SciPy/generic-solve metrics; no third independent formula or optimizer reexecution.',
                         'Every saved trial/8D update/left-SE3 chain independently checked with SciPy expm; ranks/costs are persisted-consistency checks, not rebuilt correspondence SVD/objectives.',
                         'Candidate acceptance is an engineering decision, not GT correctness or proof that old failures were illumination-caused.',
                         'GT-isolation evidence depends on reviewed frozen source and ordering; artifact flags/hashes alone cannot prove execution causality.',
                         'No retry, parameter selection, GPU, cross-scene mean or isolated speed comparison.']}
    target = BATCH / 'independent_affine_result_audit.json'; raw = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode(); require(len(raw) < 1024 * 1024, 'Small audit artifact')
    with target.open('xb') as stream: stream.write(raw)
    print(json.dumps({'status': result['status'], 'frames': 200, 'failure_ids': {r['scene']: r['failure_frame_ids'] for r in scenes}, 'artifact': str(target)}))


if __name__ == '__main__': main()
