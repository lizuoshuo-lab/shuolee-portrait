"""Synthetic CPU controls for ONE frozen 8D weighted least-squares linearization.

No image, TUM, ground truth, torch, solver prototype or renderer is imported.
These deliberately constructed matrices are not Jacobians sampled from a real
registration problem. Support, residual normalization and IRLS weights remain
fixed throughout each comparison. State order: [v_m(3), w_rad(3), a_log_gain,
b_intensity]. The system minimizes ||A delta - rhs||^2 with rhs=-sqrt(W) r/sigma.

Run directory is exclusively created; source, inputs, all intermediate matrices,
expected negative controls and any unexpected failure remain on disk. This does
not implement a change to the existing affine registration solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import scipy
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
RCOND = 1e-11
STATE_UNITS = ['m', 'm', 'm', 'rad', 'rad', 'rad', 'log-relative-gain', 'gray-intensity']


def write_json(path, obj):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=scalar) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(x):
    return np.asarray(x, dtype=np.float64).tolist()


def maxabs(x):
    return float(np.max(np.abs(x)))


def rank_spectrum(a):
    s = np.linalg.svd(a, compute_uv=False)
    tol = float(s[0] * RCOND) if len(s) else 0.
    rank = int(np.count_nonzero(s > tol))
    columns = a.shape[1]
    return {'rank': rank, 'singular_values': array(s), 'rank_threshold': tol,
            'full_column_rank': rank == columns,
            'condition_2': float(s[0] / s[-1]) if rank == columns else None}


def close(actual, expected, atol=1e-9, rtol=1e-9, label=''):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: maximum absolute difference {maxabs(np.asarray(actual)-np.asarray(expected))}')


def dct_basis(n):
    rows = np.arange(n, dtype=np.float64)[:, None]
    cols = np.arange(n, dtype=np.float64)[None, :]
    q = np.cos(math.pi * (rows + .5) * cols / n) * math.sqrt(2. / n)
    q[:, 0] /= math.sqrt(2.)
    return q


def make_case(kind):
    q = dct_basis(8)
    g, h = np.vstack((q, np.zeros_like(q))), np.vstack((np.zeros_like(q), q))
    ap = g[:, :6] + h[:, :6] * np.array([.3, -.4, .2, -.15, .1, -.2])[None, :]
    an = h[:, :2].copy()
    if kind == 'singular-nuisance':
        an[:, 1] = 2. * an[:, 0]
    elif kind == 'pose-absorbed-by-full-rank-nuisance':
        ap[:, 0] = h[:, 0]
    elif kind == 'near-pose-nuisance-coupling':
        ap[:, 0] = h[:, 0] + 1e-4 * g[:, 0]
    elif kind != 'full-rank':
        raise ValueError(kind)
    desired_a = np.column_stack((ap, an)) * np.array([100., 50., 80., 20., 35., 15., 4., 2.])[None, :]
    # The first 8 rows have geometric residual units m; the last 8 are intensity.
    # Nuisance columns in the geometric rows are zero by construction.
    sigma = np.r_[np.full(8, .01), np.full(8, .1)]
    normalized_r = np.array([.25, -2., .5, 1.4, -.7, .9, -1.6, .3,
                             -.8, 1.7, -.4, .6, -2.2, .2, 1.2, -.5])
    weights = np.minimum(1., 1. / np.maximum(np.abs(normalized_r), 1e-15))
    raw_r = normalized_r * sigma
    raw_j = desired_a * (sigma / np.sqrt(weights))[:, None]
    a = np.sqrt(weights)[:, None] * raw_j / sigma[:, None]
    rhs = -np.sqrt(weights) * raw_r / sigma
    return {'raw_J': raw_j, 'raw_residual': raw_r, 'sigma': sigma,
            'normalized_residual': normalized_r, 'fixed_IRLS_weights': weights,
            'A': a, 'rhs': rhs}


def projection_system(a, rhs):
    ap, an = a[:, :6], a[:, 6:]
    u, sn, vt = np.linalg.svd(an, full_matrices=False)
    rank_n = int(np.count_nonzero(sn > sn[0] * RCOND))
    ur = u[:, :rank_n]
    pinv_n = np.linalg.pinv(an, rcond=RCOND)
    p_svd = np.eye(len(a)) - ur @ ur.T
    p_pinv = np.eye(len(a)) - an @ pinv_n
    # Independent orthogonal basis construction with column-pivoted full QR.
    q, r, pivots = qr(an, mode='full', pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank_qr = int(np.count_nonzero(diagonal > diagonal[0] * RCOND))
    q_perp = q[:, rank_qr:]
    p_qr = q_perp @ q_perp.T
    projected_ap, projected_rhs = p_svd @ ap, p_svd @ rhs
    dp, _, rank_p, _ = np.linalg.lstsq(projected_ap, projected_rhs, rcond=RCOND)
    dn = pinv_n @ (rhs - ap @ dp)
    projection_delta = np.r_[dp, dn]
    dp_qr, _, _, _ = np.linalg.lstsq(q_perp.T @ ap, q_perp.T @ rhs, rcond=RCOND)
    dn_qr, _, _, _ = np.linalg.lstsq(an, rhs - ap @ dp_qr, rcond=RCOND)
    delta_qr = np.r_[dp_qr, dn_qr]
    h, gradient_rhs = a.T @ a, a.T @ rhs
    hpp, hpn, hnn = h[:6, :6], h[:6, 6:], h[6:, 6:]
    s = ap.T @ p_svd @ ap
    schur_rhs = ap.T @ p_svd @ rhs
    hnn_pinv = np.linalg.pinv(hnn, rcond=RCOND)
    s_pinv = hpp - hpn @ hnn_pinv @ hpn.T
    rhs_pinv = gradient_rhs[:6] - hpn @ hnn_pinv @ gradient_rhs[6:]
    return locals()


def check_case(kind, destination):
    inp = make_case(kind)
    a, rhs = inp['A'], inp['rhs']
    m = projection_system(a, rhs)
    destination.mkdir()
    # Save all constructed inputs and base projection matrices before assertions.
    # An unexpected assertion must not erase the system that triggered it.
    matrices = {k: array(v) for k, v in inp.items()}
    for k in ['ap', 'an', 'u', 'sn', 'vt', 'ur', 'pinv_n', 'p_svd', 'p_pinv',
              'q', 'r', 'pivots', 'q_perp', 'p_qr', 'projected_ap', 'projected_rhs',
              'h', 'gradient_rhs', 'hpp', 'hpn', 'hnn', 's', 'schur_rhs', 'hnn_pinv', 's_pinv', 'rhs_pinv']:
        matrices[k] = array(m[k])
    write_json(destination/'matrices.json', matrices)
    ap, an = m['ap'], m['an']
    joint, _, rank_joint, _ = np.linalg.lstsq(a, rhs, rcond=RCOND)
    projected, delta_qr = m['projection_delta'], m['delta_qr']
    nuisance_rank, pose_rank = m['rank_n'], int(m['rank_p'])
    expected = {'full-rank': (8, 2, 6), 'singular-nuisance': (7, 1, 6),
                'pose-absorbed-by-full-rank-nuisance': (7, 2, 5),
                'near-pose-nuisance-coupling': (8, 2, 6)}[kind]
    assert (int(rank_joint), nuisance_rank, pose_rank) == expected
    assert rank_joint == nuisance_rank + pose_rank
    assert m['rank_qr'] == nuisance_rank
    close(m['p_svd'], m['p_qr'], label='SVD versus pivoted QR projector')
    close(m['p_svd'], m['p_pinv'], label='U basis versus An pseudoinverse projector')
    close(m['p_svd'] @ m['p_svd'], m['p_svd'], label='projector idempotence')
    close(m['p_svd'] @ an, np.zeros_like(an), label='nuisance annihilation')
    close(m['s'], m['s_pinv'], atol=2e-10, label='projector versus generalized Schur')
    close(m['schur_rhs'], m['rhs_pinv'], atol=2e-10, label='Schur RHS minus sign')
    close(a @ joint, a @ projected, atol=1e-7, label='joint and projected fitted values')
    close(a @ joint, a @ delta_qr, atol=1e-7, label='joint and QR fitted values')
    normal_residual = a.T @ (a @ joint - rhs)
    close(normal_residual, np.zeros(8), atol=1e-7, label='joint stationary weighted gradient')
    safe_norm = np.linalg.norm(a, axis=0)
    assert np.all(safe_norm > 0)
    ahat = a / safe_norm
    z, _, rank_normalized, _ = np.linalg.lstsq(ahat, rhs, rcond=RCOND)
    restored = z / safe_norm
    close(a @ restored, a @ joint, atol=1e-7, label='column normalization fitted values')
    if rank_joint == 8:
        close(restored, joint, atol=1e-7, rtol=1e-8, label='unique physical delta recovered')
    initial_cost = float(rhs @ rhs)
    cost = lambda d: float(np.sum((a @ d - rhs) ** 2))
    result = {
        'status': 'passed', 'case': kind,
        'rank_A': int(rank_joint), 'rank_Ap_before_elimination': rank_spectrum(ap)['rank'],
        'rank_An': nuisance_rank, 'rank_PAp': pose_rank,
        'rank_identity_A_equals_An_plus_PAp': True,
        'state_units': STATE_UNITS,
        'joint_minimum_norm_physical_delta': array(joint),
        'projected_minimum_norm_pose_then_minimum_norm_nuisance_delta': array(projected),
        'pivoted_QR_elimination_delta': array(delta_qr),
        'column_norm_inverse_state_units': array(safe_norm),
        'column_normalized_z': array(z), 'restored_physical_delta': array(restored),
        'unique_joint_solution': rank_joint == 8,
        'unique_pose_after_nuisance_elimination': pose_rank == 6,
        'minimum_norm_caveat': 'In deficient systems sequential or column-scaled minimum-norm solutions may differ; fitted values and minimum residual are the invariant comparison.',
        'S_eigenvalues_raw_mixed_parameter_units': array(np.linalg.eigvalsh(m['s'])),
        'A_spectrum_raw_physical_coordinates': rank_spectrum(a),
        'A_spectrum_column_normalized_coordinates': rank_spectrum(ahat),
        'PAp_spectrum_raw_physical_coordinates': rank_spectrum(m['projected_ap']),
        'initial_weighted_squared_residual': initial_cost,
        'joint_weighted_squared_residual': cost(joint),
        'projected_weighted_squared_residual': cost(projected),
        'QR_weighted_squared_residual': cost(delta_qr),
        'joint_vs_projected_delta_max_abs_mixed_units': maxabs(joint - projected),
        'joint_vs_projected_fitted_value_max_abs_normalized_residual': maxabs(a @ (joint - projected)),
        'joint_vs_QR_fitted_value_max_abs_normalized_residual': maxabs(a @ (joint - delta_qr)),
        'column_normalization_changes_minimum_norm_branch': bool(maxabs(joint-restored) > 1e-8),
        'projector_SVD_QR_max_abs': maxabs(m['p_svd']-m['p_qr']),
        'projected_S_vs_generalized_block_S_max_abs_mixed_units': maxabs(m['s']-m['s_pinv']),
        'covariance_estimated': False,
    }
    extra = {'A_column_normalized': ahat, 'column_norm': safe_norm,
             'joint_delta': joint, 'projected_delta': projected, 'QR_delta': delta_qr,
             'normalized_z': z, 'restored_physical_delta': restored}
    if nuisance_rank == 2 and pose_rank == 6:
        solved_hnp = np.linalg.solve(m['hnn'], m['hpn'].T)
        solved_gn = np.linalg.solve(m['hnn'], m['gradient_rhs'][6:])
        s_direct = m['hpp'] - m['hpn'] @ solved_hnp
        g_direct = m['gradient_rhs'][:6] - m['hpn'] @ solved_gn
        dp_direct = np.linalg.solve(s_direct, g_direct)
        dn_direct = np.linalg.solve(m['hnn'], m['gradient_rhs'][6:] - m['hpn'].T @ dp_direct)
        delta_direct = np.r_[dp_direct, dn_direct]
        close(a @ delta_direct, a @ joint, atol=2e-7, rtol=2e-7, label='legal full-rank normal Schur fit')
        result['direct_normal_block_Schur'] = {
            'status': 'executed_full_rank_Hnn_and_S', 'delta': array(delta_direct),
            'fitted_value_max_abs_difference': maxabs(a @ (delta_direct-joint)),
            'delta_max_abs_difference_mixed_units': maxabs(delta_direct-joint),
            'weighted_squared_residual': cost(delta_direct)}
        extra.update(S_direct=s_direct, rhs_direct=g_direct, delta_direct=delta_direct)
    else:
        result['direct_normal_block_Schur'] = {
            'status': 'not_executed',
            'reason': 'Hnn singular: an ordinary inverse is not legitimate' if nuisance_rank < 2 else 'Hnn is invertible but S is singular: no unique 6D pose inverse',
            'legal_path_used': 'rank-aware SVD/pivoted QR projection and nuisance pseudoinverse'}
    if kind == 'full-rank':
        wrong_sign = -joint
        wrong_unscale = z
        assert cost(wrong_sign) > initial_cost
        assert cost(wrong_unscale) > 10. * cost(joint)
        wrong_g = m['gradient_rhs'][:6] + m['hpn'] @ np.linalg.solve(m['hnn'], m['gradient_rhs'][6:])
        wrong_dp = np.linalg.solve(m['s'], wrong_g)
        wrong_dn = np.linalg.lstsq(an, rhs-ap@wrong_dp, rcond=RCOND)[0]
        wrong_block = np.r_[wrong_dp, wrong_dn]
        # Use the exact quadratic cost difference, not an arbitrary minimum
        # error magnitude: even a small wrong step is a meaningful negative control.
        wrong_dp_difference = wrong_dp-joint[:6]
        expected_wrong_cost_increase = float(wrong_dp_difference @ m['s'] @ wrong_dp_difference)
        wrong_cost_increase = cost(wrong_block)-cost(joint)
        assert wrong_cost_increase > 100.*np.finfo(float).eps*initial_cost
        close(wrong_cost_increase, expected_wrong_cost_increase, atol=1e-12,
              label='wrong Schur sign analytic quadratic cost increase')
        # delta_physical = T u: translations are expressed in mm in u.
        t_units = np.diag([.001, .001, .001, 1., 1., 1., 1., 1.])
        u_units = np.linalg.lstsq(a @ t_units, rhs, rcond=RCOND)[0]
        close(t_units @ u_units, joint, label='metres versus millimetres roundtrip')
        t_signed = np.diag([-1., 1., 1., 1., -1., 1., -1., 1.])
        u_signed = np.linalg.lstsq(a @ t_signed, rhs, rcond=RCOND)[0]
        close(t_signed @ u_signed, joint, label='signed coordinate roundtrip')
        result['negative_controls'] = {
            'wrong_rhs_sign_rejected_by_higher_cost': cost(wrong_sign),
            'using_z_as_physical_delta_without_dividing_column_norm_rejected_cost': cost(wrong_unscale),
            'wrong_plus_sign_in_Schur_rhs_rejected_cost': cost(wrong_block),
            'wrong_Schur_sign_cost_increase': wrong_cost_increase,
            'wrong_Schur_sign_analytic_quadratic_cost_increase': expected_wrong_cost_increase}
        result['units_controls'] = {'metres_mm_physical_delta_max_abs': maxabs(t_units @ u_units-joint),
                                    'signed_axis_physical_delta_max_abs': maxabs(t_signed @ u_signed-joint),
                                    'A_condition_with_translation_mm': rank_spectrum(a @ t_units)['condition_2'],
                                    'A_condition_in_original_units': rank_spectrum(a)['condition_2'],
                                    'H_spectrum_depends_on_parameter_units': True}
        extra.update(T_units=t_units, units_solution=u_units, T_signed=t_signed,
                     signed_solution=u_signed, wrong_rhs_delta=wrong_sign,
                     wrong_unscale_delta=wrong_unscale, wrong_Schur_rhs=wrong_g,
                     wrong_Schur_delta=wrong_block)
    if kind == 'pose-absorbed-by-full-rank-nuisance':
        null_direction = np.zeros(8)
        null_direction[0], null_direction[6] = 1., -25.
        close(a @ null_direction, np.zeros(len(a)), label='explicit pose-nuisance cancelling null direction')
        assert rank_spectrum(ap)['rank'] == 6 and nuisance_rank == 2 and pose_rank == 5
        assert maxabs(joint-projected) > 1e-3
        close(cost(joint+null_direction*.3), cost(joint), label='same cost along unobservable direction')
        result['negative_control_full_rank_nuisance_is_not_full_rank_pose'] = {
            'rank_An': 2, 'rank_Ap': 6, 'rank_PAp': 5, 'rank_joint': 7,
            'null_direction_in_physical_coordinates': array(null_direction),
            'null_direction_units_caveat': 'Components use their state units; this is not a homogeneous Euclidean distance.',
            'minimum_cost_unchanged_after_explicit_null_shift': True}
        extra['physical_null_direction'] = null_direction
    matrices.update({k: array(v) for k, v in extra.items()})
    write_json(destination/'matrices.json', matrices)
    write_json(destination/'result.json', result)
    return result, inp


def regularization_control(inp, destination):
    a, rhs = inp['A'], inp['rhs']
    norms = np.linalg.norm(a, axis=0)
    ahat = a / norms
    # alpha is predeclared in column-normalized z coordinates, not a data-estimated covariance.
    alpha = .25
    unregularized = np.linalg.lstsq(ahat, rhs, rcond=RCOND)[0]
    nuisance_rows = np.column_stack((np.zeros((2, 6)), np.eye(2))) * math.sqrt(alpha)
    a_prior = np.vstack((ahat, nuisance_rows))
    rhs_prior = np.r_[rhs, np.zeros(2)]
    z_prior = np.linalg.lstsq(a_prior, rhs_prior, rcond=RCOND)[0]
    h, g = ahat.T @ ahat, ahat.T @ rhs
    hnn_prior = h[6:, 6:] + alpha*np.eye(2)
    s_prior = h[:6, :6] - h[:6, 6:] @ np.linalg.solve(hnn_prior, h[6:, :6])
    b_prior = g[:6] - h[:6, 6:] @ np.linalg.solve(hnn_prior, g[6:])
    p_prior = np.linalg.solve(s_prior, b_prior)
    n_prior = np.linalg.solve(hnn_prior, g[6:] - h[6:, :6] @ p_prior)
    close(z_prior, np.r_[p_prior, n_prior], label='nuisance prior augmented LS versus Schur')
    a_damp = np.vstack((ahat, math.sqrt(alpha)*np.eye(8)))
    b_damp = np.r_[rhs, np.zeros(8)]
    z_damp = np.linalg.lstsq(a_damp, b_damp, rcond=RCOND)[0]
    close(z_damp, np.linalg.solve(h + alpha*np.eye(8), g), label='damping augmented LS versus normal solve')
    assert rank_spectrum(ahat)['rank'] == 7 and rank_spectrum(a_prior)['rank'] == 8
    assert np.linalg.eigvalsh(s_prior)[0] > .01
    assert maxabs(z_prior-unregularized) > 1e-3 and maxabs(z_damp-unregularized) > 1e-3
    data_cost = lambda z: float(np.sum((ahat @ z-rhs)**2))
    result = {
        'status': 'passed', 'alpha_fixed': alpha, 'coordinates': 'z = column_norm * physical_delta',
        'data_rank_before_prior': 7, 'augmented_rank_with_nuisance_prior': 8,
        'meaning': 'The positive added curvature is supplied by a new assumption, not recovered data observability.',
        'nuisance_prior_cost': 'alpha * (z[6]^2 + z[7]^2)',
        'damping_cost': 'alpha * sum(z^2)',
        'physical_coordinate_damping_cost': 'alpha * sum((column_norm * physical_delta)^2)',
        'unregularized_data_cost': data_cost(unregularized),
        'nuisance_prior_data_cost': data_cost(z_prior),
        'nuisance_prior_penalty': float(alpha*np.sum(z_prior[6:]**2)),
        'damped_data_cost': data_cost(z_damp), 'damping_penalty': float(alpha*np.sum(z_damp**2)),
        'prior_S_eigenvalues_normalized_coordinates': array(np.linalg.eigvalsh(s_prior)),
        'unregularized_physical_delta': array(unregularized/norms),
        'nuisance_prior_physical_delta': array(z_prior/norms),
        'damped_physical_delta': array(z_damp/norms),
        'these_are_three_different_optimization_problems': True,
        'no_change_to_existing_odometry_solver': True, 'covariance_estimated': False}
    destination.mkdir()
    write_json(destination/'matrices.json', {k: array(v) for k, v in {
        'A': a, 'rhs': rhs, 'column_norm': norms, 'Ahat': ahat, 'A_prior': a_prior,
        'rhs_prior': rhs_prior, 'A_damp': a_damp, 'rhs_damp': b_damp, 'H': h,
        'normal_rhs': g, 'Hnn_prior': hnn_prior, 'S_prior': s_prior, 'Schur_rhs_prior': b_prior,
        'z_unregularized': unregularized, 'z_prior': z_prior, 'z_damp': z_damp}.items()})
    write_json(destination/'result.json', result)
    return result


def normal_condition_control(destination):
    u, v = dct_basis(16)[:, :8], dct_basis(8)
    specified_s = np.geomspace(1., 1e-6, 8)
    a = u @ np.diag(specified_s) @ v.T
    known_delta = np.array([.8, -.4, .2, -.6, .3, -.7, .5, -.1])
    rhs = a @ known_delta
    solution_svd = np.linalg.lstsq(a, rhs, rcond=1e-13)[0]
    h = a.T @ a
    solution_normal = np.linalg.solve(h, a.T @ rhs)
    measured_s = np.linalg.svd(a, compute_uv=False)
    measured_h_s = np.linalg.svd(h, compute_uv=False)
    cond_a = float(measured_s[0]/measured_s[-1])
    cond_h = float(measured_h_s[0]/measured_h_s[-1])
    ratio = cond_h/cond_a**2
    assert abs(ratio-1.) < 2e-4
    assert maxabs(solution_svd-known_delta) < 1e-8
    # This records observed roundoff; the measured ordering of two solvers is not asserted universal.
    destination.mkdir()
    result = {'status': 'passed', 'constructed_singular_values': array(specified_s),
              'measured_condition_A': cond_a, 'measured_condition_A_transpose_A': cond_h,
              'normal_condition_divided_by_A_condition_squared': ratio,
              'SVD_delta_max_abs_error_synthetic_dimensionless_coordinates': maxabs(solution_svd-known_delta),
              'normal_solve_delta_max_abs_error_synthetic_dimensionless_coordinates': maxabs(solution_normal-known_delta),
              'meaning': 'In exact arithmetic kappa2(A.T A)=kappa2(A)^2 for full column rank. Float64 formation and solving of normal equations may lose accuracy; this toy is not a universal error-rate claim.',
              'state_units': 'This separate numerical conditioning matrix is dimensionless, not a camera Jacobian.',
              'covariance_estimated': False}
    write_json(destination/'matrices.json', {k: array(x) for k, x in {
        'U': u, 'V': v, 'specified_singular_values': specified_s, 'A': a,
        'H': h, 'known_delta': known_delta, 'rhs': rhs, 'solution_SVD': solution_svd,
        'solution_normal': solution_normal, 'measured_A_singular_values': measured_s,
        'measured_H_singular_values': measured_h_s}.items()})
    write_json(destination/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='affine-schur-observability-cpu-controls-001')
    args = parser.parse_args()
    if not args.run or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.run):
        raise ValueError('run must be a simple lowercase directory name')
    output = ROOT/'outputs'/args.run
    output.mkdir(exist_ok=False)
    source = Path(__file__).resolve()
    (output/'source_script.py').write_bytes(source.read_bytes())
    config = {'status': 'fixed_before_controls', 'source_sha256': sha(source),
              'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
              'rank_rcond': RCOND, 'fixed_support_rows': 16, 'pose_dimensions': 6, 'nuisance_dimensions': 2,
              'state_units': STATE_UNITS, 'sigma_geometry_m': .01, 'sigma_photo_intensity': .1,
              'huber_delta_normalized': 1., 'weights_fixed_for_all_comparisons': True,
              'support_fixed_for_all_comparisons': True,
              'residual_convention': 'r + J delta; rhs=-sqrt(W)*r/sigma; cost=||A delta-rhs||^2',
              'engineering_scales_not_calibrated_noise': True,
              'toy_matrix_is_not_an_actual_image_or_pose_Jacobian': True,
              'data_GT_solver_renderer_or_GPU_used': False,
              'notes': 'No nonlinear iteration or Huber reweighting is performed. Costs describe only this frozen weighted linear model, not the nonlinear robust registration objective.'}
    write_json(output/'configuration.json', config)
    try:
        cases, inputs = {}, {}
        for kind in ['full-rank', 'singular-nuisance', 'pose-absorbed-by-full-rank-nuisance', 'near-pose-nuisance-coupling']:
            cases[kind], inputs[kind] = check_case(kind, output/kind)
        regularization = regularization_control(inputs['pose-absorbed-by-full-rank-nuisance'], output/'prior-and-damping')
        conditioning = normal_condition_control(output/'normal-equation-conditioning')
        assert 'torch' not in sys.modules and 'gsplat' not in sys.modules
        prior_development_failure_files = [
            ROOT/'outputs/affine-schur-observability-development-failures-001/failure.json',
            ROOT/'outputs/affine-schur-observability-development-failures-001/source_before_syntax_repair.py']
        for old_failure in sorted((ROOT/'outputs').glob('affine-schur-observability-cpu-controls-*/failure.json')):
            prior_development_failure_files.extend([old_failure, old_failure.parent/'source_script.py'])
        development_history = {str(p.relative_to(ROOT)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                               for p in prior_development_failure_files if p.exists() and output not in p.parents}
        negative_history = {'status': 'expected_controls_preserved', 'unexpected_runtime_failures_this_run': 0,
            'prior_development_failure_evidence': development_history,
            'prior_development_failure_explanation': 'A parse error before execution was retained. Formal CPU run001 then failed an unnecessarily large 1e-3 minimum-cost-increase assertion: the actual wrong-sign increase was positive 0.000378437485075267. The repaired check verifies the analytic quadratic difference and a floating-point resolution floor without changing toy inputs. Run002 preserved a NumPy bool JSON serialization failure; the scalar serializer was fixed without changing inputs or mathematics.',
            'expected_rejected_claims': [
                'Singular Hnn was not ordinarily inverted.',
                'Full rank An and full rank Ap did not imply full rank projected pose system.',
                'Sequential pseudoinverse and joint minimum-norm physical deltas were not treated as identical in coupled rank deficiency.',
                'Column normalized z was not accepted as an unscaled physical delta.',
                'Wrong RHS sign and wrong Schur block RHS plus sign gave higher fixed weighted cost.',
                'Added nuisance prior or damping was not described as recovered data observability.',
                'Condition numbers and mixed-unit H eigenvalues were not called pose covariance.'],
            'scope': 'Expected negative controls are findings, not hidden run failures. Any unexpected exception is separately written to failure.json and re-raised.'}
        write_json(output/'failure_history.json', negative_history)
        metrics = {'status': 'passed_all_declared_synthetic_CPU_controls', 'source_sha256': sha(source),
                   'cases': cases, 'regularization_control': regularization,
                   'normal_equation_conditioning': conditioning, 'GPU_calls': 0,
                   'torch_imported': False, 'existing_affine_solver_changed_or_run': False,
                   'scope': 'One fixed-support fixed-weight synthetic linearization; not real TUM results, nonlinear optimization improvement or covariance calibration.'}
        write_json(output/'metrics.json', metrics)
        files = {str(p.relative_to(output)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(output.rglob('*')) if p.is_file()}
        write_json(output/'completion.json', {'status': 'completed', 'source_sha256': sha(source),
                                            'files': files, 'all_matrix_inputs_and_results_preserved': True,
                                            'GPU_calls': 0})
        print(json.dumps({'status': metrics['status'], 'run': args.run, 'source_sha256': sha(source),
                          'ranks': {k: [v['rank_A'], v['rank_An'], v['rank_PAp']] for k,v in cases.items()}}))
    except BaseException as exc:
        write_json(output/'failure.json', {'status': 'failed_preserved', 'source_sha256': sha(source),
                   'exception_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                   'partial_outputs_retained': True, 'GPU_calls': 0})
        raise


if __name__ == '__main__':
    main()
