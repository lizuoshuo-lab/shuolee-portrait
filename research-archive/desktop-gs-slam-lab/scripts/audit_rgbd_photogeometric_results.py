"""Independent, read-only audit of the four fixed CPU odometry rounds.

No solver/loader/trajectory-evaluation module is imported. Raw GT xyzw poses
are reconstructed with SciPy, generic solves form relative transforms, and
SciPy align_vectors supplies the scale-free positional alignment. Rejected
poses, compact frame fields and actual sensor bundle bytes are all checked.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'outputs/rgbd-photogeometric-real-cpu-preparation-003/protocol.json'
PROTOCOL_SHA = '81325bceecd336b678e8cadda1af4bf926fd3c7e5788acbb0db128524ccc265b'
BATCH = ROOT / 'outputs/rgbd-photogeometric-real-validation-001'
COMPARISON = ROOT / 'outputs/rgbd-photogeometric-comparison-001'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rmse(values):
    return math.sqrt(math.fsum(float(x) ** 2 for x in values) / len(values))


def rows(path):
    return [line.split() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def truth_from_raw(witness):
    dataset = Path(witness['dataset'])
    rgb, depth, gt = (rows(dataset / name) for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt'))
    gt_times = np.array([float(row[0]) for row in gt])
    poses = []
    for meta in witness['selected_sensor_metadata']:
        require(rgb[meta['rgb_index']][1] == meta['rgb_path'], 'actual RGB index/path')
        require(depth[meta['depth_index']][1] == meta['depth_path'], 'actual depth index/path')
        stamp = float(rgb[meta['rgb_index']][0])
        insertion = np.searchsorted(gt_times, stamp)
        candidates = [i for i in (insertion-1, insertion) if 0 <= i < len(gt_times)]
        nearest = min(candidates, key=lambda i: (abs(gt_times[i]-stamp), i))
        require(nearest == meta['gt_index'], 'nearest raw GT timestamp, no interpolation')
        row = np.array(gt[nearest], dtype=float)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses.append(pose)
    absolute = np.stack(poses)
    return np.stack([np.linalg.solve(absolute[0], pose) for pose in absolute])


def independent_metrics(estimated, truth):
    a, b = estimated[:, :3, 3], truth[:, :3, 3]
    raw = np.linalg.norm(a-b, axis=1)
    ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
    alignment, _ = Rotation.align_vectors(bc, ac)
    aligned = np.linalg.norm(alignment.apply(ac)-bc, axis=1)
    raw_rotation = [math.degrees(Rotation.from_matrix(g[:3, :3].T @ e[:3, :3]).magnitude())
                    for e, g in zip(estimated, truth)]
    interval_t, interval_r = [], []
    for i in range(1, len(estimated)):
        relative_e = np.linalg.solve(estimated[i-1], estimated[i])
        relative_g = np.linalg.solve(truth[i-1], truth[i])
        error = np.linalg.solve(relative_g, relative_e)
        interval_t.append(float(np.linalg.norm(error[:3, 3])))
        interval_r.append(math.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))
    return {
        'raw_anchored_translation_rmse_m': rmse(raw),
        'raw_anchored_translation_mean_m': float(np.mean(raw)),
        'raw_anchored_translation_max_m': float(np.max(raw)),
        'raw_rotation_rmse_deg': rmse(raw_rotation),
        'se3_aligned_ate_translation_rmse_m': rmse(aligned),
        'se3_aligned_ate_translation_mean_m': float(np.mean(aligned)),
        'se3_aligned_ate_translation_max_m': float(np.max(aligned)),
        'adjacent_rpe_translation_rmse_m': rmse(interval_t),
        'adjacent_rpe_rotation_rmse_deg': rmse(interval_r),
        'per_frame_raw_translation_error_m': raw.tolist(),
        'per_frame_se3_aligned_ate_translation_error_m': aligned.tolist(),
        'per_interval_rpe_translation_error_m': interval_t,
        'per_interval_rpe_rotation_error_deg': interval_r,
    }


def main():
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'exact root-reviewed protocol')
    protocol, ledger, comparison = read(PROTOCOL), read(BATCH / 'metrics.json'), read(COMPARISON / 'metrics.json')
    require(ledger['status'] == 'all_four_predeclared_CPU_rounds_completed' and ledger['runs_started'] == 4
            and ledger['current_run'] is None and len(ledger['run_records']) == 4, 'all four actual terminal rounds')
    require(comparison['batch_metrics_sha256'] == sha(BATCH / 'metrics.json'), 'comparison actual ledger')
    require(comparison['summarizer_sha256'] == sha(ROOT / 'scripts/summarize_rgbd_photogeometric.py'), 'summary source')
    for name, expected in protocol['source_sha256'].items():
        require(sha(ROOT / 'scripts' / name) == expected, 'frozen source ' + name)
    input_checks = {'index_files': 0, 'raw_sensor_files': 0, 'sensor_arrays': 0}
    truth = {}
    for scene, witness in protocol['datasets'].items():
        dataset = Path(witness['dataset'])
        for name, expected in witness['index_sha256'].items():
            require(sha(dataset / name) == expected, 'raw index unchanged')
            input_checks['index_files'] += 1
        for frame in witness['raw_sensor_files']:
            for kind in ('rgb_path', 'depth_path'):
                require(sha(dataset / frame[kind]['path']) == frame[kind]['sha256'], 'raw PNG unchanged')
                input_checks['raw_sensor_files'] += 1
        bundle = protocol['sensor_bundles'][scene]
        require(not bundle['contains_GT_poses'] and sha(ROOT / bundle['path']) == bundle['sha256'], 'GT-free bundle SHA')
        with np.load(ROOT / bundle['path'], allow_pickle=False) as saved:
            require(len(saved.files) == 15, 'exact RGB/depth/valid/gray/K three-level bundle')
            for level, hashes in witness['input_array_hashes'].items():
                for key, frozen in hashes.items():
                    array = saved[f'level{level}_{key}']
                    require(list(array.shape) == frozen['shape'] and str(array.dtype) == frozen['dtype']
                            and array.nbytes == frozen['bytes']
                            and hashlib.sha256(array.tobytes(order='C')).hexdigest() == frozen['sha256'], 'actual sensor array')
                    input_checks['sensor_arrays'] += 1
                rgb = saved[f'level{level}_rgb'].astype(np.float64)
                gray = rgb @ np.array([.299, .587, .114], dtype=np.float64)
                require(np.array_equal(gray, saved[f'level{level}_gray']), 'actual RGB-derived gray, no GT')
        truth[scene] = truth_from_raw(witness)
    methods = []
    last_end = None
    for frozen, record in zip(protocol['rounds'], ledger['run_records']):
        require(record['round_id'] == frozen['round_id'] and record['run'] == frozen['run']
                and record['order'] == frozen['order'] and record['configuration'] == frozen['configuration']
                and record['returncode'] == 0, 'declared order, config and exit0')
        start, end = (datetime.fromisoformat(record[key]) for key in ('started_utc', 'ended_utc'))
        require(start < end and (last_end is None or last_end <= start), 'serial process intervals')
        last_end = end
        require(start < datetime.fromisoformat(protocol['deadline_beijing']), 'spawn before cutoff')
        folder = ROOT / 'outputs' / record['run']
        require(sha(folder / 'metrics.json') == record['metrics_sha256']
                and sha(BATCH / (record['round_id'] + '.log')) == record['log_sha256'], 'actual original/log SHA')
        original = read(folder / 'metrics.json')
        require(original['frames_processed'] == 100 and original['configuration'] == frozen['configuration']
                and original['source_sha256'] == protocol['source_sha256'], 'actual full count/config/source')
        require(original['selected_sensor_metadata'] == protocol['datasets'][frozen['scene']]['selected_sensor_metadata']
                and original['input_array_hashes'] == protocol['datasets'][frozen['scene']]['input_array_hashes'], 'actual declared observations')
        require(original['GT_pose_evaluation_after_all_100_estimates_saved'] and not original['GT_pose_used_in_optimizer'], 'reported GT scope')
        rows_original = original['frame_metrics']
        require([row['frame_id'] for row in rows_original] == list(range(200, 300)), 'all100 frame IDs')
        estimated = np.array([row['estimated_c2w'] for row in rows_original])
        require(np.array_equal(estimated[0], np.eye(4)), 'fixed initial gauge')
        flags = Counter(flag for row in rows_original for flag in row['failure_flags'])
        rejected = [row['frame_id'] for row in rows_original if row['failure_flags']]
        require(flags == original['failure_flag_counts'] and len(rejected) == original['failure_frame_count']
                == record['failure_frame_count'], 'all failures retained')
        prefix_max = 0.
        for i, row in enumerate(rows_original[1:], 1):
            require(row['accepted'] == (not bool(row['failure_flags'])), 'accept/reject ledger')
            expected = estimated[i-1] @ np.array(row['candidate_previous_from_current']) if row['accepted'] else estimated[i-1]
            prefix_max = max(prefix_max, float(np.max(np.abs(expected-estimated[i]))))
        require(prefix_max == 0., 'exact accepted composition or previous-pose rejection propagation')
        with np.load(folder / 'trajectory_arrays.npz', allow_pickle=False) as saved:
            require(np.array_equal(saved['estimated_c2w'], estimated), 'NPZ actual estimates')
            gt_diff = float(np.max(np.abs(saved['evaluation_only_anchored_GT_c2w']-truth[frozen['scene']])))
            require(gt_diff < 3e-14, 'raw xyzw GT independent reconstruction')
        compact = read(COMPARISON / (record['round_id'] + '-summary.json'))
        require(len(compact['frame_metrics']) == 100, 'compact all100 frames')
        for full, short in zip(rows_original, compact['frame_metrics']):
            retained = {key: value for key, value in full.items() if key not in ('configuration', 'iteration_history')}
            require(retained == {key: value for key, value in short.items() if key != 'iteration_history_summary'}, 'every retained frame field exact')
            if 'iteration_history' in full:
                history = full['iteration_history']
                expected = {'records': len(history), 'accepted_updates': sum(bool(i['accepted']) for i in history),
                            'stop_reasons': dict(Counter(i.get('stop_reason', 'continued') for i in history))}
                require(short['iteration_history_summary'] == expected, 'all iteration counts/stops preserved')
        recalculated = independent_metrics(estimated, truth[frozen['scene']])
        differences = {key: float(np.max(np.abs(np.asarray(value)-np.asarray(original['trajectory'][key]))))
                       for key, value in recalculated.items()}
        require(all(error < (1e-7 if 'rotation' in key else 1e-12) for key, error in differences.items()), 'independent metrics agreement')
        methods.append({'name': record['round_id'], 'original_metrics_sha256': record['metrics_sha256'],
                        'rejected_frame_ids': rejected, 'failure_flag_counts': dict(flags), 'prefix_max_abs_difference': prefix_max,
                        'raw_GT_matrix_max_abs_difference': gt_diff, 'independent_metric_max_abs_differences': differences,
                        'all_compact_frame_fields_exact': True, 'recalculated_scalar_metrics': {key: value for key, value in recalculated.items() if not isinstance(value, list)}})
    for artifact in comparison['public_compact_summaries']:
        path = ROOT / artifact['path']
        require(sha(path) == artifact['sha256'] and path.stat().st_size == artifact['bytes'] < 1024*1024, 'all6 compact file SHAs')
    result = {'status': 'passed', 'audit_source_sha256': sha(Path(__file__)), 'protocol_sha256': PROTOCOL_SHA,
              'batch_metrics_sha256': sha(BATCH / 'metrics.json'), 'comparison_metrics_sha256': sha(COMPARISON / 'metrics.json'),
              'actual_input_checks': input_checks, 'actual_rounds': 4, 'complete_frame_rows': 400,
              'tracked_nonanchor_rows': 396, 'rejected_rows': 4, 'methods': methods,
              'limits': ['Read-only CPU audit; no registration, model optimization, threshold tuning or frame selection.',
                         'Independent raw-GT SciPy xyzw, generic relative solves, align_vectors and rotation magnitudes.',
                         'GT isolation is supported by frozen pipeline source review and reported execution order; hashes alone cannot prove runtime causality.',
                         'Actual sensor bundles are checked bytewise against protocol. Later correspondences can differ between photo0/photo1.',
                         'The raw source, protocol and full local iteration ledger remain authoritative; portable website JSON can transform paths/formatting.']}
    target = COMPARISON / 'independent_result_audit.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'actual_input_checks': input_checks, 'rows': 400, 'rejected_rows': 4, 'artifact': str(target)}))


if __name__ == '__main__':
    main()
