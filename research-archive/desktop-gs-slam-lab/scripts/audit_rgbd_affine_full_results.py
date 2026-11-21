"""CPU-only audit of four terminal full RGB-D trajectories and retained rejects.

Explicit reuse: e807 raw-GT/SciPy metrics and b549 saved8D SciPy step audit.
No optimizer, loader or renderer is imported. Real correspondence ranks and
objectives are persisted-consistency checks; this does not recompute them.
"""
from collections import Counter
from datetime import datetime
from decimal import Decimal
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import audit_rgbd_photogeometric_results as independent
import audit_rgbd_affine_real_results as affine_audit

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/rgbd-affine-full-cpu-preparation-001'
BATCH = ROOT / 'outputs/rgbd-affine-full-validation-001'
PROTOCOL_SHA = 'ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a'
ROOT_REVIEW_SHA = '977fae21421e9c71d51c29756b8999d4b075b5b8ed53a9c16d46176c88ecd44f'
PREFLIGHT_SHA = 'd815c85825decc9d7330594033416fbfdf1dfc09d472c081176a4329265dfdeb'
HELPER_SHA = 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53'
AFFINE_AUDIT_SHA = 'b549a0af2642b2e203e8f3b5f0deae46dcaf2458f608d7e27435c87b33f8c6d5'
require, sha, read = independent.require, independent.sha, independent.read


def six_step_ledger(pair):
    if pair.get('candidate_unknown_due_to_exception'):
        require(not pair['accepted'] and pair['candidate_reference_from_current'] is None,
                'Unknown numeric exception retained')
        return {'steps': 0, 'trials': 0, 'accepted_updates': 0, 'unknown_exception': True}
    require([level['level'] for level in pair['levels']] == [0, 1, 2], 'All three ordered 6D pyramid level summaries retained')
    transform = np.eye(4); levels = Counter(); updates_by_level = Counter()
    maximum = 0.; trials = 0; updates = 0
    for row in pair['iteration_history']:
        levels[row['level']] += 1
        maximum = max(maximum, affine_audit.close(row['T_previous_from_current_before'], transform, '6D before chain'))
        if 'proposed_delta_twist_m_rad' in row:
            proposed = np.asarray(row['proposed_delta_twist_m_rad'], dtype=float)
            require(proposed.shape == (6,) and np.isfinite(proposed).all(), 'Exactly finite six-dimensional proposed physical twist')
        chosen = None
        for trial in row['line_search_trials']:
            trials += 1
            require(trial['same_correspondence_denominator'] == row['counts']['joint_correspondences'], '6D fixed support denominator')
            objective = trial['objective']
            if objective is not None:
                require(math.isfinite(objective), '6D valid finite persisted objective')
                if objective < row['normalized_robust_objective']:
                    require(chosen is None, '6D first descending trial'); chosen = trial
        require(row['accepted'] == (chosen is not None), '6D strict persisted descent')
        if chosen is not None:
            require(chosen == row['line_search_trials'][-1] and row['line_search_scale'] == chosen['scale']
                    and row['fixed_correspondence_objective_after'] == chosen['objective'], '6D first descent terminates search')
            accepted_delta = np.asarray(row['accepted_delta_twist_m_rad'], dtype=float)
            require(accepted_delta.shape == (6,) and np.isfinite(accepted_delta).all(), 'Exactly finite six-dimensional accepted physical twist')
            delta = proposed * row['step_limit_factor'] * chosen['scale']
            maximum = max(maximum, affine_audit.close(accepted_delta, delta, '6D limited physical update'))
            transform = affine_audit.se3(delta) @ transform
            maximum = max(maximum, affine_audit.close(row['T_previous_from_current_after'], transform, '6D independent SciPy left expm'))
            updates += 1; updates_by_level[row['level']] += 1
        else:
            require('T_previous_from_current_after' not in row, '6D rejected step has no invented after state')
    maximum = max(maximum, affine_audit.close(pair['candidate_reference_from_current'], transform, '6D final candidate chain'))
    require(pair['candidate_previous_from_current'] == pair['candidate_reference_from_current'], '6D alias direction explicit')
    for level in pair['levels']:
        require(level['iterations_recorded'] == levels[level['level']] and level['accepted_updates'] == updates_by_level[level['level']], '6D all level counts')
    return {'steps': len(pair['iteration_history']), 'trials': trials, 'accepted_updates': updates,
            'independent_SciPy_SE3_max_abs_difference': maximum}


def text_check(path, metadata, poses):
    rows = independent.rows(path); require(len(rows) == len(poses), 'All requested trajectory text rows')
    maximum = 0.; angle = 0.; previous = None
    for fields, meta, pose in zip(rows, metadata, poses):
        require(len(fields) == 8, 'Actual TUM xyzw row')
        stamp = Decimal(fields[0]); require(stamp == Decimal(f"{meta['rgb_timestamp']:.9f}") and (previous is None or stamp > previous), '9decimal paired timestamps')
        previous = stamp; values = np.asarray(fields[1:], dtype=float)
        maximum = max(maximum, affine_audit.close(values[:3], pose[:3, 3], 'Text position', 1e-11))
        angle = max(angle, float(Rotation.from_matrix(Rotation.from_quat(values[3:]).as_matrix().T @ pose[:3, :3]).magnitude()))
    require(angle < 1e-10, 'Text xyzw orientation')
    return {'rows': len(rows), 'translation_max_abs_difference_m': maximum, 'rotation_max_difference_rad': angle}


def main():
    watched = {}
    def watch(path, expected=None):
        digest = sha(path); require(expected is None or digest == expected, 'Frozen consumed file ' + str(path)); watched[path] = digest
    watch(PREP/'protocol.json', PROTOCOL_SHA); protocol = read(PREP/'protocol.json')
    watch(PREP/'root_protocol_review.json', ROOT_REVIEW_SHA)
    watch(PREP/'root-independent-full-preflight-review.json', PREFLIGHT_SHA)
    require(read(PREP/'root-independent-full-preflight-review.json')['status'] == 'passed_independent_full_preflight_review_no_actual_execution', 'Independent preflight passed')
    watch(Path(independent.__file__), HELPER_SHA); watch(Path(affine_audit.__file__), AFFINE_AUDIT_SHA)
    watch(BATCH/'metrics.json'); batch = read(BATCH/'metrics.json')
    require(batch['status'] == 'all_four_predeclared_CPU_full_sequence_rounds_completed' and batch['runs_started'] == 4 and batch['current_run'] is None and len(batch['run_records']) == 4 and batch['GPU_runs'] == 0, 'Exactly all four actual terminal CPU rounds')
    require(batch['protocol_sha256'] == PROTOCOL_SHA and batch['source_sha256'] == protocol['source_sha256'], 'Batch frozen version')
    require((BATCH/'protocol_used.json').read_bytes() == (PREP/'protocol.json').read_bytes(), 'Exact batch protocol bytes'); watch(BATCH/'protocol_used.json')
    for name, digest in protocol['source_sha256'].items(): watch(ROOT/'scripts'/name, digest); watch(PREP/'source_snapshot'/name, digest)
    truth = {}; witnesses = {}; sensor_files = 0
    for scene, binding in protocol['datasets'].items():
        watch(ROOT/binding['path'], binding['sha256']); w = read(ROOT/binding['path']); witnesses[scene] = w
        for name, digest in w['index_sha256'].items(): watch(Path(w['dataset'])/name, digest)
        for row in w['raw_sensor_files']:
            for key in ('rgb_path', 'depth_path'): watch(Path(w['dataset'])/row[key]['path'], row[key]['sha256']); sensor_files += 1
        truth[scene] = independent.truth_from_raw(w)
    runs = []; previous_end = None; total_rows = 0; total_steps = 0; total_trials = 0
    for declared, record in zip(protocol['rounds'], batch['run_records']):
        require(record['returncode'] == 0 and all(record[k] == declared[k] for k in ('order','round_id','scene','model_dimension','run','configuration')), 'Exact declared round and actual exit0')
        start, end = (datetime.fromisoformat(record[k]) for k in ('started_utc','ended_utc'))
        require(start < end and start < datetime.fromisoformat(protocol['deadline_beijing']) and (previous_end is None or previous_end <= start), 'Serial actual interval and cutoff'); previous_end = end
        folder = ROOT/'outputs'/declared['run']; watch(folder/'metrics.json', record['metrics_sha256']); watch(BATCH/(record['round_id']+'.log'), record['log_sha256'])
        require((folder/'protocol_used.json').read_bytes() == (PREP/'protocol.json').read_bytes(), 'Exact child protocol')
        for path in folder.iterdir():
            if path.is_file(): watch(path)
        data = read(folder/'metrics.json'); w = witnesses[declared['scene']]; count = declared['configuration']['frames']
        require(data['status'] in ('completed','completed_with_pair_rejections') and data['frames_processed'] == count and data['configuration'] == declared['configuration'] and data['source_sha256'] == protocol['source_sha256'] and data['protocol_sha256'] == PROTOCOL_SHA, 'Complete fixed actual result')
        require(data['GT_pose_used_in_optimizer'] is False and data['GT_pose_evaluation_after_all_estimates_saved'] is True and data['GPU_runs'] == 0, 'Final evaluation GT boundary')
        require(data['selected_sensor_metadata'] == w['selected_sensor_metadata'] and data['input_array_hashes'] == w['input_array_hashes'], 'Full preflight sensor witnesses exact')
        rows = data['frame_metrics']; require([r['frame_id'] for r in rows] == list(range(count)), 'No requested frame omitted')
        manifest = read(folder/'pair_manifest.json'); require(list(sorted(manifest)) == [f'pairs/{i:04d}.json' for i in range(count)], 'All immutable pairs manifested')
        for index, row in enumerate(rows):
            path = folder/f'pairs/{index:04d}.json'; entry = manifest[f'pairs/{index:04d}.json']; watch(path,entry['sha256']); require(path.stat().st_size == entry['bytes'] and read(path) == row, 'Immutable pair equals final row')
        poses = np.asarray([r['estimated_c2w'] for r in rows]); require(np.array_equal(poses[0],np.eye(4)), 'Identity anchor')
        require(np.max(np.abs(poses[:,3,:]-np.array([0.,0.,0.,1.]))) == 0. and np.max(np.abs(poses[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3]-np.eye(3))) < 1e-10 and np.max(np.abs(np.linalg.det(poses[:,:3,:3])-1.)) < 1e-10, 'Every pose proper rigid')
        reference = 0; flags = Counter(); gaps = Counter(); rejected = []; aggregate = Counter(); se3_error = 0.; expected_iterations = []
        for index,row in enumerate(rows[1:],1):
            require(row['local_index'] == row['current_frame_id'] == index and row['reference_frame_id'] == row['last_accepted_frame_id_before'] == reference and row['reference_gap_frames'] == index-reference, 'Exact causal reference IDs')
            require(row['source_sensor_metadata'] == w['selected_sensor_metadata'][index] and row['reference_sensor_metadata'] == w['selected_sensor_metadata'][reference] and row['maximum_sensor_frame_used_for_this_pair'] == index, 'Causal sensor witnesses')
            require(not row['pair_initialized_from_GT'] and np.array_equal(row['pair_initial_transform'],np.eye(4)), 'Sensor-only identity pair initialization')
            if not row.get('candidate_unknown_due_to_exception'):
                require([level['level'] for level in row['levels']] == [0, 1, 2], 'All ordered pyramid levels retained in both models')
            if declared['model_dimension'] == 8:
                require(row['pair_initial_log_relative_gain'] == row['pair_initial_bias_intensity'] == 0. and row['brightness_accumulated_across_pairs'] is False, 'Brightness reset each pair')
                stats = affine_audit.step_ledger(row)
            else: stats = six_step_ledger(row)
            for key in ('steps','trials','accepted_updates'): aggregate[key] += stats.get(key,0)
            se3_error = max(se3_error,stats.get('independent_SciPy_SE3_max_abs_difference',0.))
            expected_iterations.extend({'local_index':index,'frame_id':index,'reference_frame_id':reference,**item} for item in row['iteration_history'])
            require(row['accepted'] == (not row['failure_flags']), 'All rejection flags retained'); flags.update(row['failure_flags']); gaps[index-reference] += 1
            if row['accepted']: expected = poses[reference]@np.asarray(row['candidate_reference_from_current']); reference = index
            else: expected = poses[index-1]; rejected.append(index)
            require(np.array_equal(expected,poses[index]) and row['last_accepted_frame_id_after'] == reference, 'Exact integrated or held prefix')
        actual_iterations = [json.loads(line) for line in (folder/'iterations.jsonl').read_text(encoding='utf-8').splitlines()]
        require(actual_iterations == expected_iterations, 'All iteration rows duplicate exact saved pairs, none omitted')
        require(len(rejected) == data['failure_frame_count'] == record['failure_frame_count'] and dict(flags) == data['failure_flag_counts'] and {str(k):v for k,v in gaps.items()} == data['reference_gap_counts'], 'Complete reject/gap denominators')
        require(data['last_accepted_reference_id'] == reference and data['all_frame_denominator'] == count and data['tracked_nonanchor_frame_denominator'] == count-1, 'Final reference and all-frame denominator')
        with np.load(folder/'trajectory_arrays.npz',allow_pickle=False) as saved:
            require(np.array_equal(saved['estimated_c2w'],poses) and np.array_equal(saved['rgb_timestamps'],[m['rgb_timestamp'] for m in w['selected_sensor_metadata']]), 'Full NPZ estimate/timestamps')
            gt_error = affine_audit.close(saved['evaluation_only_anchored_GT_c2w'],truth[declared['scene']], 'Raw xyzw/GT gauge',3e-14)
        metrics = independent.independent_metrics(poses,truth[declared['scene']]); differences = {k:float(np.max(np.abs(np.asarray(v)-np.asarray(data['trajectory'][k])))) for k,v in metrics.items()}
        require(all(v < (1e-7 if 'rotation' in k else 1e-11) for k,v in differences.items()), 'Independent full trajectory metrics')
        text = text_check(folder/'trajectory_estimated.txt',w['selected_sensor_metadata'],poses)
        total_rows += count; total_steps += aggregate['steps']; total_trials += aggregate['trials']
        runs.append({'run':declared['run'],'scene':declared['scene'],'model_dimension':declared['model_dimension'],'metrics_sha256':record['metrics_sha256'],'frames':count,'failure_frame_count':len(rejected),'all_rejected_frame_ids':rejected,'failure_flag_counts':dict(flags),'reference_gap_counts':dict(gaps),'last_accepted_reference_id':reference,'maximum_reference_gap_frames':max(gaps),'all_pair_step_checks':dict(aggregate),'independent_SciPy_SE3_max_abs_difference':se3_error,'raw_GT_anchor_max_abs_difference':gt_error,'metric_max_abs_differences':differences,'recalculated_scalar_metrics':{k:v for k,v in metrics.items() if not isinstance(v,list)},'endpoint_raw_translation_m':metrics['per_frame_raw_translation_error_m'][-1],'trajectory_text':text})
    require(total_rows == 2594 and all(sha(p)==h for p,h in watched.items()), 'All2594rows and all consumed files unchanged')
    result = {'status':'passed_independent_full_input_reference_step_and_trajectory_audit','audit_source_sha256':sha(Path(__file__)),'metric_helper_sha256':HELPER_SHA,'saved8D_step_helper_sha256':AFFINE_AUDIT_SHA,'protocol_sha256':PROTOCOL_SHA,'root_review_sha256':ROOT_REVIEW_SHA,'independent_preflight_sha256':PREFLIGHT_SHA,'batch_metrics_sha256':sha(BATCH/'metrics.json'),'actual_rounds':4,'complete_frame_rows':total_rows,'tracked_nonanchor_pairs':total_rows-4,'raw_sensor_PNGs_rehashed':sensor_files,'consumed_files_rehashed_unchanged':len(watched),'steps':total_steps,'trials':total_trials,'runs':runs,'GPU_calls':0,'limits':['Explicit e807 rawGT/SciPy metrics and b549 saved8D-step helper reuse; no third numerical formulation or optimizer rerun.','Independent SciPy verifies saved pose update chains; real ranks/objectives are persisted consistency, not reconstructed correspondence SVD/costs.6D trial transforms were not exported; only accepted6D updates can be checked.','Preflight30full/30short-array byte audit is reused explicitly; actual source/rawPNG/input-witness hashes are checked again.','GT isolation rests on reviewed source/order, not flags alone. Accepted registration is not GT accuracy; held poses and every rejection remain in all-frame metrics.','Comparing6D/8D whole paths is not isolated brightness causality or cross-scene generalization.']}
    raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<1024*1024,'Small complete audit artifact')
    target=BATCH/'independent_full_affine_result_audit.json'
    with target.open('xb') as stream: stream.write(raw)
    print(json.dumps({'status':result['status'],'frames':total_rows,'pairs':total_rows-4,'steps':total_steps,'trials':total_trials,'file':str(target)}))


if __name__ == '__main__': main()
