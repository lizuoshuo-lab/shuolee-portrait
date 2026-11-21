"""Independent CPU audit of saved synthetic frozen8D least-squares systems.

Never imports the generator or its projection routine. SciPy pivoted full QR
supplies a nuisance orthogonal complement; rank-revealing gelsy solves operate
on rectangular matrices. Singular branches compare fits/residuals, not equal
mixed-unit minimum-norm deltas. No images, TUM, nonlinear solver, torch or GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/affine-schur-observability-cpu-controls-003'
GENERATOR = ROOT / 'scripts/check_affine_schur_observability.py'
GENERATOR_SHA = '74343301e91be0b424466b3402224f56f0962575defb7381a2c0d6c152ead27b'
METRICS_SHA = 'adf2c8e8728d0921e079a4c51fc0492f3a0d133715a3769bf4bf5a299a9e5fd8'
COMPLETION_SHA = 'db7fd3ad46207c43f40c2e4a8eb43db8fcdaf1c07a141e0ca16897914e406065'
CASES = {'full-rank': (8, 2, 6), 'singular-nuisance': (7, 1, 6),
         'pose-absorbed-by-full-rank-nuisance': (7, 2, 5), 'near-pose-nuisance-coupling': (8, 2, 6)}
RCOND = 1e-11


def require(ok, message):
    if not ok: raise AssertionError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def maximum(value): return float(np.max(np.abs(value)))


def save(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream: stream.write(raw)


def close(a, b, label, atol=3e-10, rtol=3e-11):
    a, b = np.asarray(a), np.asarray(b); require(np.allclose(a, b, atol=atol, rtol=rtol), label)
    return maximum(a - b)


def rank(a):
    s = linalg.svdvals(a); return int(np.count_nonzero(s > s[0] * RCOND)), s


def fit(a, rhs):
    delta, _, rank_a, _ = linalg.lstsq(a, rhs, cond=RCOND, lapack_driver='gelsy')
    return delta, int(rank_a)


def cost(a, rhs, delta): return math.fsum(float(v) ** 2 for v in a @ delta - rhs)


def check_case(name, saved, reported):
    raw_j, raw_r, sigma, weights = [np.asarray(saved[k], dtype=np.float64) for k in ('raw_J', 'raw_residual', 'sigma', 'fixed_IRLS_weights')]
    require(raw_j.shape == (16, 8) and raw_r.shape == sigma.shape == weights.shape == (16,), 'Fixed16x8 toy support')
    require(np.array_equal(raw_j[:8, 6:], np.zeros((8, 2))) and np.array_equal(sigma, np.r_[np.full(8, .01), np.full(8, .1)]), 'Geometric nuisance zeros and m/intensity engineering scales')
    require(np.array_equal(weights, np.minimum(1., 1 / np.maximum(np.abs(raw_r / sigma), 1e-15))), 'Declared fixed Huber IRLS weights')
    a = np.sqrt(weights)[:, None] * raw_j / sigma[:, None]; rhs = -np.sqrt(weights) * raw_r / sigma
    close(a, saved['A'], 'Rebuilt A from rawJ/sigma/weights'); close(rhs, saved['rhs'], 'Negative raw residual RHS')
    close(raw_r / sigma, saved['normalized_residual'], 'Normalized raw residual')
    ap, an = a[:, :6], a[:, 6:]; close(ap, saved['ap'], 'Pose block'); close(an, saved['an'], 'Nuisance block')
    q, r, permutation = linalg.qr(an, mode='full', pivoting=True)
    diagonals = np.abs(np.diag(r)); rank_n_qr = int(np.count_nonzero(diagonals > diagonals[0] * RCOND))
    rank_n, sn = rank(an); require(rank_n == rank_n_qr, 'Rank-revealing QR/SciPy SVD nuisance rank agrees')
    q_perp = q[:, rank_n:]; p = q_perp @ q_perp.T; reduced_a, reduced_rhs = q_perp.T @ ap, q_perp.T @ rhs
    joint, rank_a = fit(a, rhs); dp, rank_p = fit(reduced_a, reduced_rhs); dn, _ = fit(an, rhs - ap @ dp); sequential = np.r_[dp, dn]
    require((rank_a, rank_n, rank_p) == CASES[name] and rank_a == rank_n + rank_p, 'Declared independent ranks and direct-sum identity')
    require((rank_a, rank_n, rank_p) == (reported['rank_A'], reported['rank_An'], reported['rank_PAp']), 'Actual reported rank tuple')
    projector_errors = {key: close(saved[key], p, 'Independent orthogonal complement ' + key) for key in ('p_svd', 'p_pinv', 'p_qr')}
    close(p @ p, p, 'Idempotent orthogonal complement'); close(p @ an, np.zeros_like(an), 'Nuisance annihilation')
    old_q, old_r = np.asarray(saved['q']), np.asarray(saved['r']); old_permutation = np.asarray(saved['pivots'], dtype=int)
    close(old_q.T @ old_q, np.eye(16), 'Saved fullQR orthogonality'); close(old_q @ old_r, an[:, old_permutation], 'Saved pivoted QR reconstructs nuisance')
    close(np.asarray(saved['u']) @ np.diag(saved['sn']) @ np.asarray(saved['vt']), an, 'Saved SVD reconstructs nuisance')
    close(saved['projected_ap'], p @ ap, 'Projected pose matrix'); close(saved['projected_rhs'], p @ rhs, 'Projected RHS')
    h = a.T @ a; g = a.T @ rhs; s = reduced_a.T @ reduced_a; schur_rhs = reduced_a.T @ reduced_rhs
    for key, value in {'h': h, 'gradient_rhs': g, 'hpp': h[:6, :6], 'hpn': h[:6, 6:], 'hnn': h[6:, 6:], 's': s, 'schur_rhs': schur_rhs}.items(): close(saved[key], value, 'Rebuilt block ' + key)
    hnn_pinv = linalg.pinv(h[6:, 6:], atol=0., rtol=RCOND)
    block_s = h[:6, :6] - h[:6, 6:] @ hnn_pinv @ h[6:, :6]
    block_rhs = g[:6] - h[:6, 6:] @ hnn_pinv @ g[6:]
    close(saved['s_pinv'], block_s, 'Generalized Schur block'); close(saved['rhs_pinv'], block_rhs, 'Minus-sign generalized Schur RHS')
    close(s, block_s, 'QR complement versus generalized block Schur'); close(schur_rhs, block_rhs, 'QR complement versus block RHS')
    norms = linalg.norm(a, axis=0); normalized = a / norms; z, normalized_rank = fit(normalized, rhs); restored = z / norms
    close(saved['column_norm'], norms, 'Column normalization scale'); close(saved['A_column_normalized'], normalized, 'Normalized matrix')
    require(normalized_rank == rank_a, 'Observed rank invariant under this nonzero column scaling')
    variants = {k: np.asarray(saved[k]) for k in ('joint_delta', 'projected_delta', 'QR_delta', 'restored_physical_delta')}
    variants.update(independent_joint=joint, independent_QR_sequential=sequential, independent_column_normalized_restored=restored)
    fit_diffs = {key: close(a @ delta, a @ joint, 'Same fit of variant ' + key, atol=2e-7, rtol=2e-8) for key, delta in variants.items()}
    if rank_a == 8:
        delta_diffs = {key: close(delta, joint, 'Unique full-rank delta of variant ' + key, atol=1e-7, rtol=2e-8) for key, delta in variants.items()}
    else: delta_diffs = {key: maximum(delta - joint) for key, delta in variants.items()}
    costs = {key: cost(a, rhs, delta) for key, delta in variants.items()}
    for field, key in [('joint_weighted_squared_residual', 'joint_delta'), ('projected_weighted_squared_residual', 'projected_delta'), ('QR_weighted_squared_residual', 'QR_delta')]: close(costs[key], reported[field], 'Saved weighted fixed cost ' + field)
    close(a.T @ (a @ joint - rhs), np.zeros(8), 'Independent joint stationary gradient', atol=2e-7)
    if rank_a == 8:
        require(reported['direct_normal_block_Schur']['status'] == 'executed_full_rank_Hnn_and_S', 'Legal normal Schur branch')
        close(a @ np.asarray(saved['delta_direct']), a @ joint, 'Saved direct normal Schur fit', atol=2e-7)
    else: require(reported['direct_normal_block_Schur']['status'] == 'not_executed', 'Singular blocks not illegally ordinarily inverted')
    extras = {}
    if name == 'full-rank':
        wrong_rhs = np.asarray(saved['wrong_rhs_delta']); wrong_z = np.asarray(saved['wrong_unscale_delta']); wrong_block = np.asarray(saved['wrong_Schur_delta'])
        wrong_costs = [cost(a, rhs, x) for x in (wrong_rhs, wrong_z, wrong_block)]
        require(wrong_costs[0] > cost(a, rhs, np.zeros(8)) and wrong_costs[1] > 10 * costs['independent_joint'] and wrong_costs[2] > costs['independent_joint'], 'Meaningful wrong RHS/unscale/Schur signs')
        shifted = wrong_block[:6] - joint[:6]; analytic = float(shifted @ s @ shifted); increase = wrong_costs[2] - costs['independent_joint']
        close(increase, analytic, 'Independent quadratic wrong Schur sign increase', atol=3e-12)
        for key in ('T_units', 'T_signed'):
            transform = np.asarray(saved[key]); coordinate, _ = fit(a @ transform, rhs); close(transform @ coordinate, joint, 'Correct physical unit/sign roundtrip')
        extras['wrong_Schur_rhs_cost_increase'] = increase; extras['analytic_quadratic_cost_increase'] = analytic
    if name == 'pose-absorbed-by-full-rank-nuisance':
        null = np.asarray(saved['physical_null_direction']); close(a @ null, np.zeros(16), 'Explicit physical mixed-unit null cancellation')
        close(cost(a, rhs, joint + .3 * null), costs['independent_joint'], 'Cost unchanged under null shift')
        require(maximum(joint - sequential) > .001 and maximum(joint - restored) > .001, 'Rankdef sequential and normalized minimum-norm branches really differ')
        extras['sequential_vs_joint_mixed_unit_delta_difference'] = maximum(joint - sequential)
        extras['column_scaled_vs_joint_mixed_unit_delta_difference'] = maximum(joint - restored)
    return {'case': name, 'rank_A_An_projected': [rank_a, rank_n, rank_p], 'column_normalized_rank': normalized_rank,
            'independent_SciPy_QR_projector_max_differences': projector_errors, 'saved_and_independent_fitted_value_max_differences': fit_diffs,
            'mixed_unit_delta_max_differences': delta_diffs, 'recomputed_fixed_weight_costs': costs, 'negative_controls': extras,
            'comparison_boundary': 'Approximate float64 fit/delta agreement in ill-conditioned rank8; direct normal Schur stored delta differs visibly; not exact equivalence.' if name.startswith('near-') else
                                   'Rankdef fit/minimum residual equivalence only; mixed-unit minimum-norm branch can differ.' if rank_a < 8 else 'Unique rank8 physical delta agrees to measured floating-point tolerance.'}, a, rhs


def regularization(saved, reported):
    a = np.asarray(saved['A']); rhs = np.asarray(saved['rhs']); norms = linalg.norm(a, axis=0); normalized = a / norms; alpha = .25
    prior_rows = np.column_stack((np.zeros((2, 6)), np.eye(2))) * math.sqrt(alpha)
    prior_a = np.vstack((normalized, prior_rows)); prior_rhs = np.r_[rhs, np.zeros(2)]
    damp_a = np.vstack((normalized, math.sqrt(alpha) * np.eye(8))); damp_rhs = np.r_[rhs, np.zeros(8)]
    for key, actual in {'column_norm': norms, 'Ahat': normalized, 'A_prior': prior_a, 'rhs_prior': prior_rhs, 'A_damp': damp_a, 'rhs_damp': damp_rhs}.items(): close(saved[key], actual, 'Actual regularization rows ' + key)
    baseline, rank0 = fit(normalized, rhs); prior, rankprior = fit(prior_a, prior_rhs); damp, rankdamp = fit(damp_a, damp_rhs)
    require((rank0, rankprior, rankdamp) == (7, 8, 8), 'Prior/damping rank comes from new rows')
    h, g = normalized.T @ normalized, normalized.T @ rhs; hnn = h[6:, 6:] + alpha * np.eye(2)
    s = h[:6, :6] - h[:6, 6:] @ linalg.solve(hnn, h[6:, :6]); reduced_rhs = g[:6] - h[:6, 6:] @ linalg.solve(hnn, g[6:])
    close(saved['S_prior'], s, 'New-assumption Schur curvature'); close(saved['Schur_rhs_prior'], reduced_rhs, 'Prior Schur minus RHS')
    close(prior, saved['z_prior'], 'Unique augmented prior fit'); close(damp, saved['z_damp'], 'Unique augmented damping fit')
    data_costs = {'unregularized_data_cost': cost(normalized, rhs, baseline), 'nuisance_prior_data_cost': cost(normalized, rhs, prior), 'damped_data_cost': cost(normalized, rhs, damp)}
    penalties = {'nuisance_prior_penalty': alpha * float(prior[6:] @ prior[6:]), 'damping_penalty': alpha * float(damp @ damp)}
    for key, value in {**data_costs, **penalties}.items(): close(value, reported[key], 'Changed objective actual value ' + key)
    close(cost(prior_a, prior_rhs, prior), data_costs['nuisance_prior_data_cost'] + penalties['nuisance_prior_penalty'], 'Prior explicitly adds cost')
    close(cost(damp_a, damp_rhs, damp), data_costs['damped_data_cost'] + penalties['damping_penalty'], 'Damping explicitly adds cost')
    close(alpha * float((norms * (damp / norms)) @ (norms * (damp / norms))), penalties['damping_penalty'], 'Normalized damping restored to physical coordinates')
    require(penalties['nuisance_prior_penalty'] > 0 and penalties['damping_penalty'] > 0 and reported['covariance_estimated'] is False, 'New cost, no recovered observability/no covariance')
    return {'ranks_unregularized_prior_damp': [rank0, rankprior, rankdamp], 'recomputed_data_costs': data_costs, 'recomputed_added_penalties': penalties,
            'prior_minimum_S_eigenvalue_in_normalized_coordinates': float(linalg.eigvalsh(s)[0]), 'different_problems_not_numerical_reexpression': True}


def conditioning(saved, reported):
    a, rhs, known = [np.asarray(saved[key]) for key in ('A', 'rhs', 'known_delta')]; h = a.T @ a
    close(h, saved['H'], 'Stored normal matrix'); close(a @ known, rhs, 'Known synthetic RHS')
    s, hs = linalg.svdvals(a), linalg.svdvals(h); ca, ch = float(s[0] / s[-1]), float(hs[0] / hs[-1]); ratio = ch / ca**2
    delta, _, _, _ = linalg.lstsq(a, rhs, cond=1e-13, lapack_driver='gelsd'); normal = linalg.solve(h, a.T @ rhs, assume_a='gen')
    require(abs(ratio - 1.) < 2e-4, 'Float64 condition-square control')
    close(ca, reported['measured_condition_A'], 'A condition', atol=.001, rtol=1e-10)
    close(ch, reported['measured_condition_A_transpose_A'], 'H condition', atol=1e5, rtol=1e-7)
    return {'independent_condition_A': ca, 'independent_condition_H': ch, 'ratio_H_over_A_squared': ratio,
            'independent_gelsd_delta_error': maximum(delta - known), 'independent_generic_normal_solve_delta_error': maximum(normal - known),
            'stored_normal_solve_delta_error': maximum(np.asarray(saved['solution_normal']) - known),
            'boundary': 'Synthetic dimensionless matrix; exact arithmetic kappa(H)=kappa(A)^2, float64 measurement approximate; no universal solver error ratio/camera covariance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', default='affine-schur-observability-independent-audit-001'); args = parser.parse_args()
    require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique simple output')
    output = ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False); (output / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    state = {'status': 'running_CPU_readonly_synthetic_audit', 'audit_source_sha256': sha(Path(__file__)), 'GPU_calls': 0}; watched = {}
    try:
        def watch(path, expected=None, byte_count=None):
            path = path.resolve(); require(path.is_relative_to(ROOT), 'Safe local evidence'); digest = sha(path)
            require(expected is None or digest == expected, 'Actual SHA ' + str(path)); require(byte_count is None or path.stat().st_size == byte_count, 'Actual bytes ' + str(path)); watched[path] = digest
        watch(GENERATOR, GENERATOR_SHA); watch(INPUT / 'metrics.json', METRICS_SHA); watch(INPUT / 'completion.json', COMPLETION_SHA)
        metrics, completion, config = [read(INPUT / f) for f in ('metrics.json', 'completion.json', 'configuration.json')]
        require(completion['status'] == 'completed' and len(completion['files']) == 16 and metrics['status'] == 'passed_all_declared_synthetic_CPU_controls', 'Actual completed16 artifacts')
        for name, record in completion['files'].items(): watch(INPUT / name, record['sha256'], record['bytes'])
        failures = read(INPUT / 'failure_history.json')['prior_development_failure_evidence']; require(len(failures) == 6, 'All6 old development failure artifacts retained')
        for name, record in failures.items(): watch(ROOT / name, record['sha256'], record['bytes'])
        require(config['rank_rcond'] == RCOND and config['weights_fixed_for_all_comparisons'] and config['support_fixed_for_all_comparisons'] and
                config['state_units'] == ['m', 'm', 'm', 'rad', 'rad', 'rad', 'log-relative-gain', 'gray-intensity'] and not metrics['existing_affine_solver_changed_or_run'], 'Fixed engineering units/weights/support, no solver modification')
        findings = []; source_inputs = {}
        for name in CASES:
            state['current_case'] = name; saved, reported = read(INPUT / name / 'matrices.json'), read(INPUT / name / 'result.json')
            require(reported == metrics['cases'][name], 'Actual per-case and top metrics exact')
            result, a, rhs = check_case(name, saved, reported); findings.append(result); source_inputs[name] = a, rhs
        reg = regularization(read(INPUT / 'prior-and-damping/matrices.json'), read(INPUT / 'prior-and-damping/result.json'))
        cond = conditioning(read(INPUT / 'normal-equation-conditioning/matrices.json'), read(INPUT / 'normal-equation-conditioning/result.json'))
        require(not any(name in sys.modules for name in ('check_affine_schur_observability', 'torch', 'gsplat', 'rgbd_affine_photometric_odometry', 'tum_rgbd')), 'No target generator/data/solver/GPU imports')
        require(all(sha(path) == digest for path, digest in watched.items()), 'All consumed artifacts/source/history unchanged')
        result = {'status': 'passed_independent_synthetic_Schur_audit', 'audit_source_sha256': sha(Path(__file__)), 'generator_sha256': GENERATOR_SHA,
                  'original_metrics_sha256': METRICS_SHA, 'original_completion_sha256': COMPLETION_SHA, 'completed_artifacts_verified': 16, 'historical_failure_artifacts_verified': 6,
                  'consumed_files_rehashed_unchanged': len(watched), 'cases': findings, 'regularization': reg, 'conditioning': cond, 'GPU_calls': 0,
                  'target_generator_or_projection_imported': False, 'limits': ['One constructed fixed-support/fixed-IRLS linear model, not image Jacobians/nonlinear TUM improvement.',
                            'Rectangular rank-revealing SciPy gelsy plus pivoted fullQR complement independently rebuilt from raw J/r/sigma/weights; generator/projector never imported.',
                            'Only rank8 supports unique-delta comparisons; deficient cases preserve fitted values and minimum residual, not a common mixed-unit minimum-norm branch.',
                            'Near-coupling equivalence is approximate at measured float64 conditioning; direct normal block cancellation yields visibly different delta.',
                            'Priors/damping add assumptions/cost in declared normalized coordinates; engineering scales and H eigenvalues are not calibrated covariance.',
                            'All16 outputs and6 old failures rehashed; no GPU/TUM/solver/source/site changes.']}
        save(output / 'metrics.json', result); print(json.dumps({'status': result['status'], 'source_sha256': result['audit_source_sha256'], 'output': str(output), 'ranks': {r['case']: r['rank_A_An_projected'] for r in findings}}))
    except BaseException as error:
        state.update(status='failed_independent_audit_preserved', exception_type=type(error).__name__, exception_message=str(error), traceback=traceback.format_exc())
        save(output / 'failure.json', state); raise


if __name__ == '__main__': main()
