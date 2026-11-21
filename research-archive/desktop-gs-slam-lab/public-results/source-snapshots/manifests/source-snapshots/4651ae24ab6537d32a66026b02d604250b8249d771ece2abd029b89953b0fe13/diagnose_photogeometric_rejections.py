"""Read-only diagnosis of recorded photo/geometric pair rejections.

Reconstructs exactly the saved transforms and line-search scales against the
frozen, GT-free sensor bundles. It never solves a new update, registers a pair,
changes a gate, retries a run, or loads a GT trajectory. Objective/mask checks
are independent NumPy calculations; residual/Jacobian and base correspondence
construction reuse the frozen library and are identified as such.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import rgbd_photogeometric_odometry as odom

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'outputs/rgbd-photogeometric-real-cpu-preparation-003/protocol.json'
PROTOCOL_SHA = '81325bceecd336b678e8cadda1af4bf926fd3c7e5788acbb0db128524ccc265b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def forbid(*args, **kwargs):
    raise RuntimeError('Read-only diagnosis forbids registration or solving new steps')


def cost_components(r_geo, r_photo, config):
    def mean_huber(values, scale):
        x = np.asarray(values, dtype=np.float64) / scale
        a = np.abs(x)
        d = config.huber_delta_normalized
        return float(np.mean(np.where(a <= d, .5 * x * x, d * (a - .5 * d))))
    geo = mean_huber(r_geo, config.sigma_geo_m)
    photo = mean_huber(r_photo, config.sigma_intensity)
    return {'geometry': geo, 'photo_unweighted': photo,
            'combined': geo + config.lambda_photo * photo}


def objective_gradient(r_geo, j_geo, r_photo, j_photo, config):
    # Exact Huber derivative at the recorded base query, no least-squares solve.
    delta = config.huber_delta_normalized
    geo = np.clip(r_geo / config.sigma_geo_m, -delta, delta)
    photo = np.clip(r_photo / config.sigma_intensity, -delta, delta)
    return (np.mean(geo[:, None] * j_geo / config.sigma_geo_m, axis=0) +
            config.lambda_photo * np.mean(photo[:, None] * j_photo / config.sigma_intensity, axis=0))


def support_and_cells(points, target, base_uv):
    # Independent reconstruction of the *trial* 2x2 support rule. Base pair
    # selection additionally has the frozen 4x4 one-cell guard.
    p = np.asarray(points, dtype=np.float64)
    K = target.geometry.K
    uv = np.column_stack((K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2],
                          K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]))
    h, w = target.gray.shape
    finite = np.isfinite(p).all(axis=1) & np.isfinite(uv).all(axis=1)
    positive = p[:, 2] > 0
    inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) &
              (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
    supported = np.zeros(len(p), dtype=bool)
    ids = np.flatnonzero(inside)
    cells = np.floor(uv[ids]).astype(np.int64)
    u, v = cells[:, 0], cells[:, 1]
    valid = target.geometry.valid
    supported[ids] = valid[v, u] & valid[v, u + 1] & valid[v + 1, u] & valid[v + 1, u + 1]
    base_cells = np.floor(base_uv).astype(np.int64)
    changed = int(np.count_nonzero(np.any(cells != base_cells[ids], axis=1)))
    return {'nonfinite_points_or_queries': int(np.count_nonzero(~finite)),
            'nonpositive_depth': int(np.count_nonzero(~positive)),
            'out_of_bilinear_domain': int(np.count_nonzero(~inside)),
            'invalid_2x2_target_support_inside': int(np.count_nonzero(inside & ~supported)),
            'changed_bilinear_cells': changed,
            'maximum_query_displacement_pixels': float(np.max(np.linalg.norm(uv - base_uv, axis=1))),
            'valid_for_fixed_denominator_trial': bool(finite.all() and positive.all() and inside.all() and supported.all())}


def rotation_degrees(R):
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def diagnose_pair(row, previous_row, source_pyramid, target_pyramid, config):
    accumulated = np.eye(4)
    steps = []
    max_cost_difference = 0.
    max_pose_difference = 0.
    classifications = Counter()
    for saved in row['iteration_history']:
        level = saved['level']
        source, target = source_pyramid[level], target_pyramid[level]
        before = np.asarray(saved['T_previous_from_current_before'], dtype=np.float64)
        pose_diff = float(np.max(np.abs(accumulated - before)))
        max_pose_difference = max(max_pose_difference, pose_diff)
        require(pose_diff < 2e-14, 'Recorded update/level continuity')
        pair = odom.pair_correspondences(before, source, target,
                                        config.geometry.distance_gates_m[level], config)
        require(pair['counts'] == saved['counts'], 'Exact recorded correspondence counts')
        require(abs(pair['ratio'] - saved['correspondence_ratio']) < 1e-15, 'Exact ratio')
        rg, jg, rp, jp, base_uv = odom.terms(before, pair, target)
        base = cost_components(rg, rp, config)
        base_diff = abs(base['combined'] - saved['normalized_robust_objective'])
        max_cost_difference = max(max_cost_difference, base_diff)
        require(base_diff < 2e-13, 'Recorded base objective')
        gradient = objective_gradient(rg, jg, rp, jp, config)
        proposed = np.asarray(saved['proposed_delta_twist_m_rad'])
        limited = proposed * saved['step_limit_factor']
        step = {'level': level, 'iteration': saved['iteration'], 'counts': saved['counts'],
                'accepted': saved['accepted'], 'stop_reason': saved.get('stop_reason', 'continued'),
                'base_objective': base, 'joint_rank': saved['joint_rank'],
                'condition': saved['column_normalized_hessian_condition'],
                'plane_rmse_m': saved['plane_rmse_m'], 'intensity_rmse': saved['intensity_rmse'],
                'intensity_mean_residual': saved['intensity_mean_residual'],
                'limited_translation_step_norm_m': float(np.linalg.norm(limited[:3])),
                'limited_rotation_step_norm_rad': float(np.linalg.norm(limited[3:])),
                'base_directional_derivative_along_limited_step': float(gradient @ limited),
                'base_minimum_distance_to_integer_cell_boundary_pixels':
                    float(np.min(np.abs(base_uv - np.rint(base_uv)))),
                'line_search_trials': []}
        for trial_record in saved['line_search_trials']:
            scale = trial_record['scale']
            trial = odom.icp.se3_exp(limited * scale) @ before
            points = odom.icp.transform_points(trial, pair['source'])
            mask = support_and_cells(points, target, base_uv)
            recorded_cost = trial_record['objective']
            require(trial_record['same_correspondence_denominator'] == len(pair['source']), 'Fixed denominator')
            require((recorded_cost is not None) == mask['valid_for_fixed_denominator_trial'], 'Recorded support/None decision')
            recomputed = None
            components = None
            if mask['valid_for_fixed_denominator_trial']:
                tr_geo, _, tr_photo, _, _ = odom.terms(trial, pair, target)
                components = cost_components(tr_geo, tr_photo, config)
                recomputed = components['combined']
                difference = abs(recomputed - recorded_cost)
                max_cost_difference = max(max_cost_difference, difference)
                require(difference < 2e-13, 'Recorded fixed-correspondence trial cost')
                classification = 'finite_strict_descent' if recorded_cost < saved['normalized_robust_objective'] else 'finite_non_descent'
            else:
                classification = 'trial_support_invalid'
            classifications[classification] += 1
            item = {'scale': scale, 'recorded_objective': recorded_cost,
                    'recomputed_objective_components': components,
                    'recorded_difference_from_base': None if recorded_cost is None else recorded_cost - saved['normalized_robust_objective'],
                    'same_correspondence_denominator': trial_record['same_correspondence_denominator'],
                    'classification': classification, 'support': mask}
            step['line_search_trials'].append(item)
            if classification == 'finite_strict_descent':
                require(saved['accepted'] and scale == saved['line_search_scale'], 'Accepted first descent')
                update_diff = float(np.max(np.abs(trial - np.asarray(saved['T_previous_from_current_after']))))
                max_pose_difference = max(max_pose_difference, update_diff)
                require(update_diff < 2e-14, 'Saved left-composed accepted update')
                accumulated = trial
        if saved.get('stop_reason') == 'small_update':
            require(not saved['line_search_trials'] and np.linalg.norm(limited[:3]) < config.geometry.convergence_translation_m and
                    np.linalg.norm(limited[3:]) < config.geometry.convergence_rotation_radians, 'Recorded two-part small-update condition')
        steps.append(step)
    candidate = np.asarray(row['candidate_previous_from_current'])
    require(np.max(np.abs(candidate - accumulated)) < 2e-14, 'Candidate equals recorded last accepted transform')
    previous = np.asarray(previous_row['estimated_c2w'])
    propagated = np.asarray(row['estimated_c2w'])
    require(np.array_equal(previous, propagated), 'Rejected pair propagates exactly previous pose')
    hypothetical = previous @ candidate
    terminal = steps[-1]
    require(terminal['level'] == 2 and terminal['stop_reason'] == 'line_search_no_descent', 'Actual finest stop')
    require(row['failure_flags'] == ['finest_level_line_search_no_descent'], 'Complete actual sole rejection flag')
    require(all(t['classification'] == 'finite_non_descent' for t in terminal['line_search_trials']), 'Terminal failure is finite non-descent')
    return {'frame_id': row['frame_id'], 'previous_frame_id': previous_row['frame_id'],
            'local_index': row['local_index'], 'failure_flags': row['failure_flags'],
            'levels': row['levels'], 'steps': steps, 'trial_classification_counts': dict(classifications),
            'max_absolute_recomputed_objective_difference': max_cost_difference,
            'max_absolute_recorded_pose_reconstruction_difference': max_pose_difference,
            'candidate_relative_translation_m': row['relative_translation_m'],
            'candidate_relative_rotation_degrees': row['relative_rotation_degrees'],
            'rejected_prefix_exactly_previous_pose': True,
            'counterfactual_candidate_not_accepted': {
                'world_camera_center_distance_from_propagated_m': float(np.linalg.norm(hypothetical[:3, 3] - propagated[:3, 3])),
                'rotation_distance_from_propagated_degrees': rotation_degrees(propagated[:3, :3].T @ hypothetical[:3, :3])},
            'terminal_rejection': {
                'actual_criterion': 'All six finite fixed-denominator trial objectives were >= base; strict < was required.',
                'support_invalid_trials': 0,
                'smallest_recorded_objective_increase': min(t['recorded_difference_from_base'] for t in terminal['line_search_trials']),
                'translation_step_norm_m': terminal['limited_translation_step_norm_m'],
                'rotation_step_norm_rad': terminal['limited_rotation_step_norm_rad'],
                'convergence_translation_threshold_m': config.geometry.convergence_translation_m,
                'convergence_rotation_threshold_rad': config.geometry.convergence_rotation_radians,
                'rank_and_residual_gates_passed': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/rgbd-photogeometric-comparison-001/rejection-diagnostic.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'New artifact required; existing evidence is never overwritten')
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'Reviewed protocol exact bytes')
    protocol = read(PROTOCOL)
    watched = {PROTOCOL: sha(PROTOCOL)}
    for name, digest in protocol['source_sha256'].items():
        path = ROOT / 'scripts' / name
        require(sha(path) == digest, 'Frozen source ' + name)
        watched[path] = digest
    # Fail loudly if a reconstruction accidentally invokes an optimizer.
    odom.register_pair = forbid
    odom.weighted_system = forbid
    pairs = []
    bundles = {}
    all_run_failure_counts = {}
    source_metrics = {}
    for frozen in protocol['rounds']:
        folder = ROOT / 'outputs' / frozen['run']
        metric = folder / 'metrics.json'
        data = read(metric)
        source_metrics[frozen['round_id']] = {'path': metric.relative_to(ROOT).as_posix(), 'sha256': sha(metric)}
        for path in folder.iterdir():
            if path.is_file():
                watched[path] = sha(path)
        require(data['configuration'] == frozen['configuration'] and data['source_sha256'] == protocol['source_sha256'], 'Frozen run config/source')
        require(data['GT_pose_used_in_optimizer'] is False and data['GPU_runs'] == 0 and data['frames_processed'] == 100, 'Actual run scope')
        rows = data['frame_metrics']
        rejected = [r for r in rows if r['failure_flags']]
        require(len(rejected) == data['failure_frame_count'], 'All failures retained')
        all_run_failure_counts[frozen['round_id']] = len(rejected)
        if not rejected:
            continue
        scene = frozen['scene']
        bundle_record = protocol['sensor_bundles'][scene]
        bundle_path = ROOT / bundle_record['path']
        require(sha(bundle_path) == bundle_record['sha256'] and bundle_record['contains_GT_poses'] is False, 'GT-free original bundle')
        watched[bundle_path] = sha(bundle_path)
        with np.load(bundle_path, allow_pickle=False) as archive:
            bundle = {k: archive[k].copy() for k in archive.files}
        require(set(bundle) == {f'level{l}_{field}' for l in range(3) for field in ('rgb', 'depth', 'valid', 'gray', 'K')}, 'Only sensor arrays in bundle')
        bundles[scene] = {'path': bundle_record['path'], 'sha256': bundle_record['sha256'], 'GT_trajectory_read': False}
        c = frozen['configuration']['odometry']
        config = odom.OdometryConfig(**{k: v for k, v in c.items() if k != 'geometry'},
                                     geometry=odom.icp.ICPConfig(**c['geometry']))
        require(json.loads(json.dumps(asdict(config))) == c, 'No configuration change')
        for row in rejected:
            local = row['local_index']
            pyramids = []
            for index in (local, local - 1):
                pyramid = []
                for level in range(3):
                    prefix = f'level{level}_'
                    frame = odom.make_frame(bundle[prefix + 'depth'][index], bundle[prefix + 'valid'][index],
                                            bundle[prefix + 'K'], bundle[prefix + 'rgb'][index], config)
                    require(np.array_equal(frame.gray, bundle[prefix + 'gray'][index]), 'Exact prepared grayscale')
                    pyramid.append(frame)
                pyramids.append(pyramid)
            diagnosed = diagnose_pair(row, rows[local - 1], pyramids[0], pyramids[1], config)
            diagnosed['scene'] = scene
            diagnosed['round_id'] = frozen['round_id']
            pairs.append(diagnosed)
    require([(p['scene'], p['frame_id']) for p in pairs] == [('desk', 215), ('desk', 265), ('desk', 280), ('xyz', 201)], 'Exactly all four recorded failures')
    require(all(sha(path) == digest for path, digest in watched.items()), 'Original sources/inputs/artifacts unchanged after diagnosis')
    report = {'status': 'passed_read_only_recorded_trial_reconstruction', 'diagnostic_source_sha256': sha(Path(__file__)),
              'protocol_sha256': PROTOCOL_SHA, 'original_metrics': source_metrics, 'sensor_bundles': bundles,
              'actual_failure_frame_counts': all_run_failure_counts, 'rejected_pairs': pairs,
              'original_sources_inputs_artifacts_unchanged_files': len(watched), 'new_registration_runs': 0,
              'new_optimizer_solves': 0, 'GPU_runs': 0, 'GT_trajectory_loaded': False,
              'limits': [
                  'All saved trials of the four rejected pairs were reconstructed; other 392 accepted/nonanchor pair histories were not replayed.',
                  'Base correspondence, residual/Jacobian and SE3 routines reuse the frozen library. Independent NumPy formulas recompute trial support and Huber objectives; this is not a second independent odometry algorithm.',
                  'Rank/residual gates passed for all four rejected pairs. The sole final flag comes from strict line-search no-descent, even though earlier updates produced a nonidentity candidate.',
                  'Tiny positive loss differences alone do not prove floating-point error, incorrect derivatives, illumination failure or a better alternative convergence rule. No threshold was changed.',
                  'Bilinear cell crossings and directional derivatives are measured diagnostic quantities; no additional finite-difference optimization or GT-based candidate selection was performed.',
                  'Rejected candidates are counterfactual diagnostics only; the saved trajectory continues at the exact preceding pose. No speed comparison is made.']}
    payload = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(payload) < 1024 * 1024, 'Small diagnostic JSON under 1MiB')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps({'status': report['status'], 'path': str(args.output), 'bytes': len(payload),
                      'sha256': sha(args.output), 'pairs': len(pairs), 'failure_counts': all_run_failure_counts}))


if __name__ == '__main__':
    main()
