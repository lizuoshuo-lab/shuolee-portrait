"""New CPU prototype: joint SE3/log-gain/bias RGB-D pair registration.

r_photo = exp(a)*I_reference(project(T_reference_from_current*p_current))
          + b - I_current.
State order is [v_m(3), w_rad(3), a_log_relative_gain, b_gray_intensity].
Pose updates are exp(delta_xi)@T; a and b receive additive updates. This model
estimates a relative image-to-image correction, not radiometric calibration or
linear-light RGB. There is no brightness prior/clamp or rank-hiding regularizer.
No dataset, GT trajectory, torch, renderer or GPU is used by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path

import numpy as np

import rgbd_photogeometric_odometry as frozen

FROZEN_LIBRARY_SHA = '80160ea921a40363b9143d6c2a11c1954063aa5fe610397e4f8dcc336261b125'


@dataclass(frozen=True)
class AffineConfig:
    base: frozen.OdometryConfig = field(default_factory=frozen.OdometryConfig)
    convergence_log_gain: float = 1e-6
    convergence_bias_intensity: float = 1e-6

    def validate(self):
        self.base.validate()
        if self.base.lambda_photo != 1.:
            raise ValueError('This synthetic prototype fixes photo weight1, no weight sweep')
        if not all(math.isfinite(v) and v > 0 for v in (self.convergence_log_gain, self.convergence_bias_intensity)):
            raise ValueError('Finite positive nuisance convergence tolerances')


def frozen_source_check():
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != FROZEN_LIBRARY_SHA:
        raise RuntimeError('Frozen photo/geometry library changed')
    frozen.frozen_source_check()


def affine_terms(transform, log_gain, bias, pair, target):
    gain = math.exp(float(log_gain))
    if not math.isfinite(gain) or not math.isfinite(bias):
        raise FloatingPointError('Nonfinite unconstrained affine brightness state')
    p = frozen.icp.transform_points(transform, pair['source'])
    r_geo, j_geo6 = frozen.icp.residual_and_jacobian(p, pair['target'], pair['normals'])
    sampled, gradient6, uv = frozen.intensity_residual_jacobian(
        p, target.geometry.K, target.gray, np.zeros(len(p)))
    residual = gain * sampled + bias - pair['source_intensity']
    j_photo = np.column_stack((gain * gradient6, gain * sampled, np.ones(len(p))))
    j_geo = np.column_stack((j_geo6, np.zeros((len(p), 2))))
    if not np.isfinite(residual).all() or not np.isfinite(j_photo).all():
        raise FloatingPointError('Nonfinite affine residual/Jacobian')
    return r_geo, j_geo, residual, j_photo, uv


def objective(r_geo, r_photo, config):
    return frozen.objective(r_geo, r_photo, config.base)


def objective_gradient(r_geo, j_geo, r_photo, j_photo, config):
    c = config.base; delta = c.huber_delta_normalized
    return (np.mean(np.clip(r_geo / c.sigma_geo_m, -delta, delta)[:, None] * j_geo / c.sigma_geo_m, axis=0) +
            np.mean(np.clip(r_photo / c.sigma_intensity, -delta, delta)[:, None] * j_photo / c.sigma_intensity, axis=0))


def normalized_system(r_geo, j_geo, r_photo, j_photo, config):
    """8-column IRLS + normalization; reject deficient rank without damping."""
    c = config.base
    geo, photo = r_geo / c.sigma_geo_m, r_photo / c.sigma_intensity
    wg = np.minimum(1., c.huber_delta_normalized / np.maximum(np.abs(geo), 1e-15))
    wp = np.minimum(1., c.huber_delta_normalized / np.maximum(np.abs(photo), 1e-15))
    A = np.vstack((j_geo / c.sigma_geo_m * np.sqrt(wg)[:, None],
                   j_photo / c.sigma_intensity * np.sqrt(wp)[:, None]))
    rhs = -np.r_[geo * np.sqrt(wg), photo * np.sqrt(wp)]
    column_norm = np.linalg.norm(A, axis=0)
    safe = np.where(column_norm > 1e-15, column_norm, 1.)
    normalized = A / safe
    singular = np.linalg.svd(normalized, compute_uv=False)
    threshold = max((singular[0] if len(singular) else 0) * c.geometry.rank_rcond, 1e-15)
    rank = int(np.count_nonzero(singular > threshold))
    condition = None if len(singular) < 8 or singular[-1] <= 1e-15 else float((singular[0] / singular[-1]) ** 2)
    stats = {'joint_state_dimension': 8, 'joint_rank': rank, 'column_normalized_hessian_condition': condition,
             'column_norms': column_norm.tolist(), 'normalized_singular_values': singular.tolist(),
             'rank_threshold': float(threshold), 'geometry_only_rank': int(np.linalg.matrix_rank(j_geo, tol=1e-8)),
             'photo_only_rank': int(np.linalg.matrix_rank(j_photo, tol=1e-8)),
             'plane_rmse_m': float(np.sqrt(np.mean(r_geo ** 2))),
             'affine_corrected_intensity_rmse': float(np.sqrt(np.mean(r_photo ** 2))),
             'affine_corrected_intensity_mean_residual': float(np.mean(r_photo)),
             'normalized_robust_objective': objective(r_geo, r_photo, config),
             'covariance_or_radiometric_noise_information_estimated': False,
             'brightness_prior_or_clamp_or_damping_applied': False}
    if rank < 8 or condition is None or condition > c.geometry.max_normalized_hessian_condition:
        return None, stats
    solution, _, solved_rank, _ = np.linalg.lstsq(normalized, rhs, rcond=c.geometry.rank_rcond)
    if solved_rank != 8 or not np.isfinite(solution).all(): return None, stats
    return solution / safe, stats


def trial_objective(transform, log_gain, bias, pair, target, config):
    p = frozen.icp.transform_points(transform, pair['source'])
    uv = frozen.project(p, target.geometry.K)
    _, _, inside = frozen.bilinear(target.gray, uv)
    if not inside.all() or not (p[:, 2] > 0).all(): return None, 'projection_or_bilinear_domain_invalid'
    cells = np.floor(uv).astype(np.int64); u, v = cells[:, 0], cells[:, 1]
    valid = target.geometry.valid
    if not (valid[v, u] & valid[v, u + 1] & valid[v + 1, u] & valid[v + 1, u + 1]).all():
        return None, 'target_2x2_support_invalid'
    try:
        rg, _, rp, _, _ = affine_terms(transform, log_gain, bias, pair, target)
        value = objective(rg, rp, config)
    except (OverflowError, FloatingPointError): return None, 'unconstrained_affine_state_nonfinite'
    return (value, None) if math.isfinite(value) else (None, 'nonfinite_objective')


def register_pair_affine(source_pyramid, target_pyramid, config=AffineConfig()):
    """CPU synthetic-call API; callers select data scope, no automatic execute."""
    config.validate(); frozen_source_check()
    if len(source_pyramid) != 3 or len(target_pyramid) != 3: raise ValueError('Three fixed levels required')
    transform = np.eye(4); a = 0.; b = 0.; history = []; levels = []; c = config.base; g = c.geometry
    for level, (source, target) in enumerate(zip(source_pyramid, target_pyramid)):
        updates = 0; stop = 'iteration_budget'
        for iteration in range(g.iterations[level]):
            pair = frozen.pair_correspondences(transform, source, target, g.distance_gates_m[level], c)
            row = {'level': level, 'iteration': iteration, 'counts': pair['counts'], 'correspondence_ratio': pair['ratio'],
                   'accepted': False, 'transform_before': transform.tolist(), 'log_gain_before': a, 'bias_before': b,
                   'line_search_trials': []}
            if len(pair['source']) < g.min_correspondences or pair['ratio'] < g.min_correspondence_ratio:
                stop = 'insufficient_joint_correspondences'; row['stop_reason'] = stop; history.append(row); break
            rg, jg, rp, jp, _ = affine_terms(transform, a, b, pair, target)
            proposed, stats = normalized_system(rg, jg, rp, jp, config); row.update(stats)
            if proposed is None:
                stop = 'rank_deficient_or_ill_conditioned_8d'; row['stop_reason'] = stop; history.append(row); break
            row['proposed_delta_8d'] = proposed.tolist()
            # Only the frozen physical pose step limits apply; the shared scalar
            # scales the full direction. No nuisance bound/prior hides rank.
            limit = min(1., g.max_iteration_translation_m / max(np.linalg.norm(proposed[:3]), 1e-15),
                        math.radians(g.max_iteration_rotation_degrees) / max(np.linalg.norm(proposed[3:6]), 1e-15))
            delta = proposed * limit; row['shared_pose_limit_factor'] = float(limit)
            if (np.linalg.norm(delta[:3]) < g.convergence_translation_m and np.linalg.norm(delta[3:6]) < g.convergence_rotation_radians and
                    abs(delta[6]) < config.convergence_log_gain and abs(delta[7]) < config.convergence_bias_intensity):
                stop = 'small_joint_update'; row['stop_reason'] = stop; history.append(row); break
            for scale in g.line_search_scales:
                trial_T = frozen.icp.se3_exp(delta[:6] * scale) @ transform
                trial_a, trial_b = float(a + delta[6] * scale), float(b + delta[7] * scale)
                cost, reason = trial_objective(trial_T, trial_a, trial_b, pair, target, config)
                row['line_search_trials'].append({'scale': scale, 'candidate_transform': trial_T.tolist(),
                    'candidate_log_gain': trial_a, 'candidate_bias': trial_b, 'objective': cost, 'invalid_reason': reason,
                    'same_correspondence_denominator': len(pair['source'])})
                if cost is not None and cost < stats['normalized_robust_objective']:
                    transform, a, b = trial_T, trial_a, trial_b; updates += 1
                    row.update(accepted=True, accepted_scale=scale, accepted_delta_8d=(delta * scale).tolist(),
                               fixed_correspondence_objective_after=cost, transform_after=transform.tolist(), log_gain_after=a, bias_after=b)
                    break
            if not row['accepted']: stop = 'line_search_no_descent'; row['stop_reason'] = stop
            history.append(row)
            if not row['accepted']: break
        levels.append({'level': level, 'iterations_recorded': iteration + 1, 'accepted_updates': updates, 'stop_reason': stop})
    final = frozen.pair_correspondences(transform, source_pyramid[-1], target_pyramid[-1], g.distance_gates_m[-1], c)
    flags = []; stats = {}
    if len(final['source']) < g.min_correspondences or final['ratio'] < g.min_correspondence_ratio:
        flags.append('final_insufficient_joint_correspondences')
    else:
        rg, jg, rp, jp, _ = affine_terms(transform, a, b, final, target_pyramid[-1])
        _, stats = normalized_system(rg, jg, rp, jp, config)
        if stats['joint_rank'] < 8: flags.append('final_unobservable_8d_pose_brightness')
        if stats['column_normalized_hessian_condition'] is None or stats['column_normalized_hessian_condition'] > g.max_normalized_hessian_condition:
            flags.append('final_ill_conditioned_joint_system')
        if stats['plane_rmse_m'] > g.max_final_plane_rmse_m: flags.append('final_plane_residual_exceeds_engineering_gate')
        if stats['affine_corrected_intensity_rmse'] > c.max_final_intensity_rmse:
            flags.append('final_affine_intensity_residual_exceeds_engineering_gate')
    translation = float(np.linalg.norm(transform[:3, 3])); rotation = frozen.icp.rotation_degrees(transform[:3, :3])
    if translation > g.max_pair_translation_m: flags.append('relative_translation_exceeds_threshold')
    if rotation > g.max_pair_rotation_degrees: flags.append('relative_rotation_exceeds_threshold')
    orthogonality = float(np.max(np.abs(transform[:3, :3].T @ transform[:3, :3] - np.eye(3))))
    determinant_error = float(abs(np.linalg.det(transform[:3, :3]) - 1))
    if (not np.isfinite(transform).all() or orthogonality > 1e-8 or determinant_error > 1e-8 or
            np.max(np.abs(transform[3] - [0, 0, 0, 1])) > 1e-12): flags.append('nonfinite_or_nonrigid_transform')
    if levels[-1]['stop_reason'] == 'line_search_no_descent': flags.append('finest_level_line_search_no_descent')
    return {'accepted': not flags, 'failure_flags': flags, 'candidate_reference_from_current': transform.tolist(),
            'candidate_log_relative_gain': a, 'candidate_relative_gain': math.exp(a), 'candidate_bias_intensity': b,
            'relative_translation_m': translation, 'relative_rotation_degrees': rotation,
            'rotation_orthogonality_max_abs': orthogonality, 'rotation_determinant_error_from_one': determinant_error,
            'final_counts': final['counts'], 'final_statistics': stats, 'levels': levels, 'iteration_history': history,
            'configuration': asdict(config), 'state_order': ['vx_m', 'vy_m', 'vz_m', 'wx_rad', 'wy_rad', 'wz_rad', 'log_gain', 'bias_intensity'],
            'pose_direction': 'current -> reference', 'GT_used_in_optimizer': False,
            'brightness_interpretation': 'relative exp(a)*reference+b approximately current; no radiometric calibration or affine prior/clamp'}
