"""Reusable independent CPU evidence checks for a future recovery collector.

No collector, trainer, loader, renderer or solver is imported or run. Trajectory
formulas reuse the explicitly locked independent raw-GT/SciPy audit helper;
this is not a third independent evaluator. Actual artifacts are checked only
when a separate lineage-aware wrapper calls these functions after collection.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import audit_rgbd_photogeometric_results as metric_helper

METRIC_HELPER_SHA = 'e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53'
INPUT_KEYS = ('image', 'depth_m', 'valid_depth_mask', 'K', 'initial_w2c',
              'prior_reference_w2c', 'icp_prefix_c2w_float64')
PARAMETER_KEYS = ('means', 'log_scales', 'quats', 'color_logits', 'opacity_logits')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def raw_rows(path, width):
    rows = [s.split() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    require(rows and all(len(r) == width for r in rows), 'Raw TUM row width')
    times = [Decimal(r[0]) for r in rows]
    require(all(t.is_finite() for t in times) and all(a < b for a, b in zip(times, times[1:])), 'Strict raw Decimal timestamps')
    return rows, times


def verify_full_raw_association(dataset, sensors):
    """Independent replication of declared float-time association, no loader."""
    dataset = Path(dataset)
    rgb, rgb_decimal = raw_rows(dataset / 'rgb.txt', 2)
    depth, depth_decimal = raw_rows(dataset / 'depth.txt', 2)
    gt, gt_decimal = raw_rows(dataset / 'groundtruth.txt', 8)
    rt, dt, gt_t = [np.asarray([float(r[0]) for r in rows]) for rows in (rgb, depth, gt)]
    def nearest(times, value):
        j = int(np.searchsorted(times, value))
        return min((i for i in (j - 1, j) if 0 <= i < len(times)), key=lambda i: (abs(times[i] - value), i))
    selected = []; used_depth = set(); last = -math.inf
    for i, timestamp in enumerate(rt):
        j, k = nearest(dt, timestamp), nearest(gt_t, timestamp)
        if abs(dt[j] - timestamp) >= .02 or abs(gt_t[k] - timestamp) >= .02: continue
        if timestamp - last <= 1 / 32 or j in used_depth: continue
        selected.append((i, j, k)); used_depth.add(j); last = timestamp
    require(len(selected) == len(sensors), 'All associated rows / no selected-frame truncation')
    for index, ((i, j, k), sensor) in enumerate(zip(selected, sensors)):
        require([sensor[x] for x in ('frame_id', 'rgb_index', 'depth_index', 'gt_index')] == [index, i, j, k], 'Actual associated indices')
        require([sensor[x] for x in ('rgb_timestamp', 'depth_timestamp', 'gt_timestamp')] == [rt[i], dt[j], gt_t[k]], 'Actual metadata timestamps')
        require(sensor['rgb_path'] == rgb[i][1] and sensor['depth_path'] == depth[j][1], 'Actual sensor paths')
        require(sensor['rgb_depth_abs_dt_s'] == abs(rt[i] - dt[j]) and sensor['rgb_gt_abs_dt_s'] == abs(rt[i] - gt_t[k]), 'Actual float association deltas')
        # Independently check selected GT nearest by exact text timestamps. This
        # does not silently substitute a new Decimal association policy.
        decimals = [(abs(gt_decimal[q] - rgb_decimal[i]), q) for q in (k - 1, k, k + 1) if 0 <= q < len(gt)]
        require(min(decimals)[1] == k, 'Selected nearest GT also agrees with raw Decimal time')
    truth = metric_helper.truth_from_raw({'dataset': str(dataset), 'selected_sensor_metadata': sensors})
    return truth, {'raw_counts': {'RGB': len(rgb), 'depth': len(depth), 'GT': len(gt)},
                   'associated_frames': len(sensors), 'all_indices_paths_float_times_and_Decimal_GT_nearest_verified': True,
                   'index_sha256': {name: sha(dataset / name) for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt')}}


def independent_trajectory(estimated, truth):
    require(sha(Path(metric_helper.__file__)) == METRIC_HELPER_SHA, 'Locked independent metric helper')
    estimated, truth = np.asarray(estimated, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    require(estimated.shape == truth.shape and estimated.ndim == 3 and estimated.shape[1:] == (4, 4), 'Paired trajectory matrices')
    require(np.isfinite(estimated).all() and np.isfinite(truth).all(), 'Finite trajectory matrices')
    require(np.max(np.abs(estimated[:, 3] - [0, 0, 0, 1])) < 1e-12, 'Homogeneous bottom row')
    R = estimated[:, :3, :3]
    structure = {'rotation_orthogonality_max_abs': float(np.max(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)))),
                 'rotation_determinant_max_abs_from_one': float(np.max(np.abs(np.linalg.det(R) - 1)))}
    require(max(structure.values()) < 2e-5, 'Persisted camera rigidity')
    values = metric_helper.independent_metrics(estimated, truth)
    values['raw_endpoint_translation_error_m'] = values['per_frame_raw_translation_error_m'][-1]
    values['raw_endpoint_rotation_error_degrees'] = float(math.degrees(Rotation.from_matrix(truth[-1, :3, :3].T @ estimated[-1, :3, :3]).magnitude()))
    return values, structure


def check_tum_text(path, sensors, poses):
    rows, times = raw_rows(path, 8)
    require(len(rows) == len(sensors) == len(poses), 'All text frames retained')
    maximum_position = 0.; maximum_rotation = 0.
    for row, timestamp, sensor, pose in zip(rows, times, sensors, poses):
        require(timestamp == Decimal(f'{sensor["rgb_timestamp"]:.9f}'), 'Exact recorded nine-decimal RGB timestamp')
        vector = np.asarray(row[1:], dtype=np.float64)
        require(np.isfinite(vector).all() and np.linalg.norm(vector[3:]) > 0, 'Finite xyzw text pose')
        rotation = Rotation.from_quat(vector[3:]).as_matrix()
        maximum_position = max(maximum_position, float(np.max(np.abs(vector[:3] - pose[:3, 3]))))
        maximum_rotation = max(maximum_rotation, float(np.max(np.abs(rotation - pose[:3, :3]))))
    require(maximum_position < 5.1e-9 and maximum_rotation < 2e-6, 'Text position and canonical quaternion serialization')
    return {'all_frame_count': len(rows), 'Decimal_timestamps_exact': True,
            'position_matrix_max_abs_difference_m': maximum_position,
            'rotation_matrix_max_abs_difference_including_float32_SO3_roundoff': maximum_rotation,
            'sha256': sha(path)}


def check_chunks(folder, original, compact):
    import json
    folder = Path(folder).resolve(); hashes = {}; counts = {}
    require(compact['summary_provenance']['full_metrics_sha256'] == sha(folder / 'metrics.json'), 'Compact original metric SHA')
    for kind in ('frame_metrics', 'selected_sensor_metadata'):
        rows = []
        for descriptor in compact[kind + '_chunks']:
            path = (folder / descriptor['path']).resolve()
            require(path.is_relative_to(folder / 'summary_chunks'), 'Safe summary chunk path')
            require(sha(path) == descriptor['sha256'] and path.stat().st_size == descriptor['bytes'] <= 1024 * 1024, 'Chunk byte count/SHA')
            data = json.loads(path.read_bytes())
            require(data['kind'] == kind and data['experiment_id'] == original['experiment_id'], 'Chunk identity')
            require(data['start_local_index'] == descriptor['start_local_index'] == len(rows), 'Chronological chunk range')
            require(data['row_count'] == descriptor['row_count'] == len(data['rows']), 'Exact chunk row count')
            rows.extend(data['rows'])
            require(data['stop_local_index_exclusive'] == descriptor['stop_local_index_exclusive'] == len(rows), 'No gap/overlap chunk endpoint')
            hashes[descriptor['path']] = sha(path)
        require(len(rows) == original['frames_processed'] and [r['frame_id'] for r in rows] == original['selected_frame_ids'], 'All complete-sequence rows')
        for reduced, full in zip(rows, original[kind]):
            if kind == 'selected_sensor_metadata': require(reduced == full, 'Full original sensor metadata exact')
            else:
                require(reduced['failure_flags'] == full['failure_flags'], 'Original failed flags retained')
                for key, value in reduced.items():
                    if key == 'icp': require(all(full['icp'][sub] == v for sub, v in value.items()), 'Original ICP compact fields exact')
                    else: require(value == full[key], 'Original compact frame fields exact')
        counts[kind] = len(rows)
    return {'all_frame_and_sensor_row_counts': counts, 'chunk_sha256': hashes,
            'all_retained_fields_and_failure_flags_exact': True}


def tensor_record(tensor):
    require(tensor.device.type == 'cpu' and not tensor.requires_grad, 'Detached CPU-only tensor')
    array = tensor.numpy()
    require(np.isfinite(array).all(), 'Finite saved tensor')
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'raw_C_order_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def compare_contexts(left, right):
    inputs = {}; parameters = {}
    for key in INPUT_KEYS:
        a, b = tensor_record(left[key]), tensor_record(right[key])
        inputs[key] = {'prior0': a, 'prior001': b, 'shape_dtype_and_raw_bytes_equal': a == b}
    require(set(left['parameters']) == set(right['parameters']) == set(PARAMETER_KEYS), 'Five known map parameters')
    for key in PARAMETER_KEYS:
        a, b = tensor_record(left['parameters'][key]), tensor_record(right['parameters'][key])
        require(a['shape'] == b['shape'] and a['dtype'] == b['dtype'], 'Paired parameter shape/dtype')
        difference = float(np.max(np.abs(left['parameters'][key].numpy().astype(np.float64) - right['parameters'][key].numpy().astype(np.float64))))
        parameters[key] = {'prior0': a, 'prior001': b, 'raw_bytes_equal': a == b, 'max_absolute_element_difference': difference}
    require(left['source_frame_ids_per_gaussian'] == right['source_frame_ids_per_gaussian'] and
            set(left['source_frame_ids_per_gaussian']) == {0}, 'Actual first-map source IDs/order')
    return {'input_tensors': inputs, 'warmup_map_parameters': parameters,
            'seven_common_sensor_initial_reference_tensors_byte_identical': all(r['shape_dtype_and_raw_bytes_equal'] for r in inputs.values()),
            'all_five_warmup_map_parameter_tensors_byte_identical': all(r['raw_bytes_equal'] for r in parameters.values()),
            'warmup_point_count': len(left['source_frame_ids_per_gaussian']), 'first_map_source_ids_and_order_equal': True}


def camera_source_provenance(final_map, causal_last, original):
    count = original['frames_processed']; rows = original['frame_metrics']
    require(causal_last['ground_truth_included'] is False and 'evaluation_only_gt_c2w' not in causal_last, 'Causal checkpoint has no GT pose')
    ids = final_map['source_frame_ids_per_gaussian']; R = final_map['parameters']['means']
    require(len(ids) == len(R) == original['final_point_count'] and ids == causal_last['source_frame_ids_per_gaussian'], 'Final Gaussian source IDs/count')
    histogram = Counter(ids)
    require(histogram[0] == original['initial_point_count'], 'Actual initial-map source count')
    for frame in range(1, count):
        require(histogram[frame] == rows[frame]['new_points_added'], 'Actual added Gaussian source frame counts')
        if rows[frame].get('pose_rejected', False): require(histogram[frame] == 0 and rows[frame]['new_points_added'] == 0, 'Rejected frame cannot introduce points')
    require(all(isinstance(i, int) and 0 <= i < count for i in ids), 'No future/out-of-range Gaussian source')
    windows = 0
    for event in original['mapping_events']:
        if 'current_local_index' not in event: continue
        current = event['current_local_index']; selected = event['window_local_indices']
        require(all(0 <= i <= current for i in selected) and event['window_frame_ids'] == selected, 'Causal mapping windows')
        require(all(i == 0 or not rows[i].get('pose_rejected', False) for i in selected), 'Window excludes rejected cameras')
        windows += 1
    return {'per_gaussian_source_frame_counts': {str(k): v for k, v in sorted(histogram.items())},
            'causal_mapping_events_checked': windows, 'no_future_or_rejected_frame_points_or_window_cameras': True}


def synthetic_controls():
    def pose(w, t):
        a = np.eye(4); a[:3, :3] = Rotation.from_rotvec(w).as_matrix(); a[:3, 3] = t
        return a
    truth = np.array([pose([.05 * i, -.04 * i, .025 * i], [np.sin(.4 * i), .12 * i, .18 * np.cos(.7 * i)]) for i in range(8)])
    gauge = pose([.21, -.14, .28], [.4, -.2, .13])
    estimate = np.stack([np.linalg.solve(gauge, g) for g in truth])
    rigid, _ = independent_trajectory(estimate, truth)
    require(rigid['se3_aligned_ate_translation_rmse_m'] < 1e-14 and rigid['adjacent_rpe_translation_rmse_m'] < 1e-14, 'Rigid gauge invariance')
    scale2 = truth.copy(); scale2[:, :3, 3] *= 2
    scaled, _ = independent_trajectory(scale2, truth)
    require(scaled['se3_aligned_ate_translation_rmse_m'] > .2, 'SE3 alignment cannot hide scale2')
    dg = pose([.4, -.2, .1], [.3, -.1, .2]); de = pose([-.2, .3, .4], [-.2, .15, .1])
    correct, inverse, wrong_side = np.linalg.solve(dg, de), np.linalg.solve(de, dg), de @ np.linalg.inv(dg)
    t = float(np.linalg.norm(correct[:3, 3])); inverse_t = float(np.linalg.norm(inverse[:3, 3])); wrong_t = float(np.linalg.norm(wrong_side[:3, 3]))
    angle = math.degrees(Rotation.from_matrix(correct[:3, :3]).magnitude())
    inverse_angle = math.degrees(Rotation.from_matrix(inverse[:3, :3]).magnitude())
    require(abs(t - inverse_t) < 1e-14 and abs(angle - inverse_angle) < 1e-12 and abs(t - wrong_t) > .001, 'Whole inverse insensitive / wrong-side control')
    return {'status': 'passed_CPU_synthetic_only', 'rigid_gauge_ATE_m': rigid['se3_aligned_ate_translation_rmse_m'],
            'rigid_gauge_RPE_translation_m': rigid['adjacent_rpe_translation_rmse_m'], 'scale2_SE3_ATE_m': scaled['se3_aligned_ate_translation_rmse_m'],
            'RPE_whole_inverse': {'translation_norm_m': t, 'inverse_translation_norm_m': inverse_t,
                                  'rotation_degrees': angle, 'inverse_rotation_degrees': inverse_angle,
                                  'norm_angle_invariance_does_not_validate_residual_direction': True},
            'wrong_side_translation_norm_m': wrong_t, 'actual_real_summary_audits': 0, 'GPU_calls': 0}
