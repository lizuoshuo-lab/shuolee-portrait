"""Synthetic CPU controls for relative affine brightness + six-DOF pose.

All images/depth/true poses are constructed locally. No real TUM optimization,
GPU or model-selection sweep occurs. Eight-column FD uses an independent scalar
bilinear interpolator and SciPy expm, including nonidentity/nonzero pose state.
Every toy candidate, rejection, step and line search remains in its artifact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import rgbd_affine_photometric_odometry as affine

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition: raise AssertionError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def array_record(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def exp_se3(delta):
    x, y, z = delta[3:]; matrix = np.zeros((4, 4))
    matrix[:3, :3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
    matrix[:3, 3] = delta[:3]
    return expm(matrix)


def scalar_bilinear(image, u, v):
    i, j = math.floor(u), math.floor(v); x, y = u - i, v - j
    return ((1 - x) * (1 - y) * image[j, i] + x * (1 - y) * image[j, i + 1] +
            (1 - x) * y * image[j + 1, i] + x * y * image[j + 1, i + 1])


def independent_residuals(T, a, b, pair, image, K):
    p = pair['source'] @ T[:3, :3].T + T[:3, 3]
    uv = np.column_stack((K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]))
    intensity = np.array([scalar_bilinear(image, u, v) for u, v in uv])
    geometric = np.sum((p - pair['target']) * pair['normals'], axis=1)
    return geometric, math.exp(a) * intensity + b - pair['source_intensity'], uv


def independent_objective(geometric, photo, config):
    c = config.base
    def term(values, scale):
        x = values / scale; d = c.huber_delta_normalized
        return float(np.mean(np.where(np.abs(x) <= d, .5 * x ** 2, d * (np.abs(x) - .5 * d))))
    return term(geometric, c.sigma_geo_m) + term(photo, c.sigma_intensity)


def differential_controls(config):
    h, w = 60, 80; v, u = np.mgrid[:h, :w]
    image = .3 + .002 * u + .003 * v + .00004 * u * v + .02 * np.sin(u * .11) * np.cos(v * .09)
    K = np.array([[60., 0., 39.5], [0., 60., 29.5], [0., 0., 1.]])
    uv = np.array([[12.27, 13.68], [30.41, 25.23], [44.76, 33.34], [65.19, 42.81]])
    z = np.array([1.7, 2.2, 2.6, 3.1])
    p = np.c_[(uv[:, 0] - K[0, 2]) * z / K[0, 0], (uv[:, 1] - K[1, 2]) * z / K[1, 1], z]
    T = exp_se3(np.array([.035, -.023, .012, .025, -.018, .011]))
    source = np.linalg.solve(T, np.c_[p, np.ones(len(p))].T).T[:, :3]
    normals = np.array([[.2, .4, .8], [-.3, .7, .4], [.5, -.2, .7], [.3, .8, -.2]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    pair = {'source': source, 'target': p + normals * np.array([.004, -.021, .015, -.003])[:, None],
            'normals': normals, 'source_intensity': np.array([.2, .4, .5, .6])}
    target = SimpleNamespace(gray=image, geometry=SimpleNamespace(K=K))
    a, b = .16, -.027; epsilon = 1e-7
    rg, jg, rp, jp, base_uv = affine.affine_terms(T, a, b, pair, target)
    independent_geo, independent_photo, _ = independent_residuals(T, a, b, pair, image, K)
    require(np.max(np.abs(rg - independent_geo)) < 1e-14 and np.max(np.abs(rp - independent_photo)) < 1e-14, 'Independent scalar residuals')
    require(float(np.min(np.abs(base_uv - np.rint(base_uv)))) > .1, 'FD queries avoid pixel-cell boundaries')
    require(min(np.min(np.abs(np.abs(rg / .01) - 1)), np.min(np.abs(np.abs(rp / .1) - 1))) > .01, 'FD avoids Huber corner')
    numeric_geo, numeric_photo = np.zeros_like(jg), np.zeros_like(jp)
    numeric_loss = np.zeros(8); wrong_right_photo = np.zeros_like(jp)
    for axis in range(8):
        delta = np.zeros(8); delta[axis] = epsilon
        plus = independent_residuals(exp_se3(delta[:6]) @ T, a + delta[6], b + delta[7], pair, image, K)
        minus = independent_residuals(exp_se3(-delta[:6]) @ T, a - delta[6], b - delta[7], pair, image, K)
        numeric_geo[:, axis] = (plus[0] - minus[0]) / (2 * epsilon)
        numeric_photo[:, axis] = (plus[1] - minus[1]) / (2 * epsilon)
        numeric_loss[axis] = (independent_objective(*plus[:2], config) - independent_objective(*minus[:2], config)) / (2 * epsilon)
        require(np.array_equal(np.floor(plus[2]), np.floor(base_uv)) and np.array_equal(np.floor(minus[2]), np.floor(base_uv)), 'All8 FD remain in initial cells')
        if axis < 6:
            right_plus = independent_residuals(T @ exp_se3(delta[:6]), a, b, pair, image, K)
            right_minus = independent_residuals(T @ exp_se3(-delta[:6]), a, b, pair, image, K)
            wrong_right_photo[:, axis] = (right_plus[1] - right_minus[1]) / (2 * epsilon)
    analytic_loss = affine.objective_gradient(rg, jg, rp, jp, config)
    geometric_error = float(np.max(np.abs(numeric_geo - jg)))
    photo_error = float(np.max(np.abs(numeric_photo - jp)))
    loss_error = float(np.max(np.abs(numeric_loss - analytic_loss)))
    wrong_side = float(np.max(np.abs(wrong_right_photo[:, :6] - jp[:, :6])))
    omit_gain = float(np.max(np.abs(jp[:, :6] / math.exp(a) - jp[:, :6])))
    require(geometric_error < 1e-8 and photo_error < 1e-8 and loss_error < 1e-6, 'All8 independent finite differences')
    require(wrong_side > .001 and omit_gain > .001, 'Wrong right perturbation and omitted gain are detected')
    require(np.count_nonzero(jg[:, 6:]) == 0, 'Geometric nuisance columns exactly zero')
    return {'status': 'passed', 'epsilon': epsilon, 'state_order': ['v3_m', 'w3_rad', 'log_gain', 'bias_intensity'],
            'base_transform_nonidentity': T.tolist(), 'base_log_gain_nonzero': a, 'base_bias_nonzero': b,
            'geometry_J8_max_absolute_FD_error': geometric_error, 'photo_J8_max_absolute_FD_error': photo_error,
            'normalized_Huber_objective_gradient8_max_absolute_FD_error': loss_error,
            'wrong_right_perturbation_max_photo_J_error': wrong_side, 'omitted_gain_on_pose_J_max_error': omit_gain,
            'geometric_nuisance_columns_exact_zero': True, 'all_8_FD_stay_in_original_noninteger_cells': True,
            'analytic_photo_J8': jp.tolist(), 'numeric_photo_J8': numeric_photo.tolist(),
            'analytic_normalized_loss_gradient8': analytic_loss.tolist(), 'numeric_normalized_loss_gradient8': numeric_loss.tolist()}


def plane_arrays(width, height, c2w, texture, gain, bias):
    K = np.array([[.75 * width, 0., width / 2 - .5], [0., .75 * height, height / 2 - .5], [0., 0., 1.]])
    v, u = np.mgrid[:height, :width]
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), axis=-1)
    directions = rays @ c2w[:3, :3].T
    depth = (2.2 - c2w[2, 3]) / directions[..., 2]
    world = c2w[:3, 3] + directions * depth[..., None]
    if texture == 'constant': value = np.full(depth.shape, .35)
    elif texture == 'linear': value = .35 + .07 * world[..., 0] + .09 * world[..., 1]
    elif texture == 'bilinear': value = .35 + .07 * world[..., 0] + .09 * world[..., 1] + .08 * world[..., 0] * world[..., 1]
    else: raise ValueError('Unknown synthetic texture')
    intensity = gain * value + bias
    require(intensity.min() >= 0 and intensity.max() <= 1, 'Synthetic RGB is finite in[0,1] without clipping')
    rgb = np.repeat(intensity[..., None], 3, axis=-1)
    return {'depth': depth, 'valid': np.isfinite(depth) & (depth > 0), 'K': K, 'rgb': rgb}


def run_case(name, texture, true_T, gain, bias, config, output, expect_observable):
    folder = output / name; folder.mkdir()
    source_pyramid, target_pyramid, arrays = [], [], {}
    for level, (width, height) in enumerate(affine.frozen.icp.LEVELS):
        current = plane_arrays(width, height, true_T, texture, gain, bias)
        reference = plane_arrays(width, height, np.eye(4), texture, 1., 0.)
        for prefix, data, frames in (('current', current, source_pyramid), ('reference', reference, target_pyramid)):
            frames.append(affine.frozen.make_frame(data['depth'], data['valid'], data['K'], data['rgb'], config.base))
            arrays.update({f'{prefix}_level{level}_{key}': value for key, value in data.items()})
    np.savez_compressed(folder / 'synthetic_inputs.npz', **arrays)
    write(folder / 'input_manifest.json', {'scope': 'Only procedurally constructed toy sensor arrays, not TUM',
          'arrays': {key: array_record(value) for key, value in arrays.items()}, 'NPZ_sha256': sha(folder / 'synthetic_inputs.npz'),
          'toy_truth_for_construction_and_posthoc_evaluation_only': {'reference_from_current': true_T.tolist(), 'relative_gain': gain, 'bias_intensity': bias}})
    result = affine.register_pair_affine(source_pyramid, target_pyramid, config)
    # Final GT comparisons are after optimizer returns; candidate is never
    # selected using these quantities.
    candidate = np.asarray(result['candidate_reference_from_current'])
    translation_error = float(np.linalg.norm(candidate[:3, 3] - true_T[:3, 3]))
    rotation_error = math.degrees(Rotation.from_matrix(true_T[:3, :3].T @ candidate[:3, :3]).magnitude())
    gain_error = abs(result['candidate_relative_gain'] - gain); bias_error = abs(result['candidate_bias_intensity'] - bias)
    verdict = {'translation_error_m': translation_error, 'rotation_error_degrees': rotation_error,
               'relative_gain_error': gain_error, 'bias_intensity_error': bias_error,
               'observable_expected': expect_observable, 'actual_final_joint_rank': result['final_statistics']['joint_rank'],
               'actual_final_condition': result['final_statistics']['column_normalized_hessian_condition'],
               'GT_used_to_choose_candidate_or_stop': False}
    write(folder / 'result.json', {**result, 'toy_posthoc_evaluation': verdict})
    if expect_observable:
        require(result['accepted'] and result['final_statistics']['joint_rank'] == 8, name + ': accepted full-rank toy')
        require(translation_error < 2e-5 and rotation_error < .002 and gain_error < 2e-5 and bias_error < 2e-5, name + ': known pose and affine state recovery')
    else:
        require(not result['accepted'] and result['final_statistics']['joint_rank'] < 8 and
                'final_unobservable_8d_pose_brightness' in result['failure_flags'], name + ': nuisance/texture degeneracy must reject')
        # Rejection must retain previous pose, even when a diagnostic candidate
        # exists; there is no nuisance prior/clamp to convert failure to success.
        propagated = candidate if result['accepted'] else np.eye(4)
        require(np.array_equal(propagated, np.eye(4)), 'Degenerate pair previous-pose retention')
    return {'case': name, 'accepted': result['accepted'], 'failure_flags': result['failure_flags'],
            'candidate_log_gain': result['candidate_log_relative_gain'], 'candidate_relative_gain': result['candidate_relative_gain'],
            'candidate_bias_intensity': result['candidate_bias_intensity'], 'posthoc': verdict,
            'result_relative_path': (folder / 'result.json').relative_to(ROOT).as_posix(), 'result_sha256': sha(folder / 'result.json'),
            'input_manifest_sha256': sha(folder / 'input_manifest.json'), 'all_iteration_and_trial_candidates_retained': True}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', default='rgbd-affine-photometric-cpu-controls-001')
    args = p.parse_args()
    require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique simple run name')
    output = ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False)
    config = affine.AffineConfig(); config.validate(); affine.frozen_source_check()
    sources = {'new_library': sha(Path(affine.__file__)), 'new_checker': sha(Path(__file__)),
               'frozen_photo_geometry': affine.FROZEN_LIBRARY_SHA, 'frozen_ICP': affine.frozen.FROZEN_ICP_SHA}
    for source in (Path(affine.__file__), Path(__file__)): shutil.copyfile(source, output / source.name)
    write(output / 'configuration.json', {'source_sha256': sources, 'affine_configuration': asdict(config),
          'real_TUM_registration_runs': 0, 'GPU_runs': 0,
          'model': 'exp(a)*reference(project(T_current_to_reference*p_current))+b-current'})
    try:
        differential = differential_controls(config); write(output / 'differential_checks.json', differential)
        true_T = exp_se3(np.array([.018, -.012, .006, .012, -.008, .009]))
        cases = [run_case('bilinear-known-pose-gain-bias', 'bilinear', true_T, 1.12, .045, config, output, True),
                 run_case('bilinear-brightness-plus-point2', 'bilinear', np.eye(4), 1., .2, config, output, True),
                 run_case('constant-texture-rank-rejection', 'constant', np.eye(4), 1.12, .045, config, output, False),
                 run_case('linear-texture-nuisance-coupling', 'linear', np.eye(4), 1.12, .045, config, output, False)]
        require('torch' not in sys.modules and 'gsplat' not in sys.modules, 'CPU controls imported no GPU stack')
        require(sha(Path(affine.__file__)) == sources['new_library'] and sha(Path(__file__)) == sources['new_checker'], 'Sources unchanged during synthetic controls')
        affine.frozen_source_check()
        write(output / 'metrics.json', {'status': 'passed_synthetic_8d_affine_brightness_controls', 'source_sha256': sources,
              'differential_checks': differential, 'cases': cases, 'real_TUM_registration_runs': 0, 'GPU_runs': 0,
              'limits': ['Relative image-pair gain/bias only, not exposure calibration, inverse-sRGB conversion or a linear-light model.',
                         'Engineering0.01m/0.1intensity scales and Huber1 are unchanged; no covariance/noise calibration is claimed.',
                         'Full-rank8 is measured, not assumed. Linear/constant texture permits nuisance/pose coupling and must remain rejected.',
                         'No brightness prior, clamp or damping hides rank. Only frozen physical pose step limits share-scale the full8D direction.',
                         'Known affine brightness and pose results are toy-only mathematical controls, not evidence TUM rejections were illumination-driven or fixed.',
                         'All candidate states, final flags, convergence and per-trial histories remain in each result.json.']})
        print(json.dumps({'status': 'passed_synthetic_8d_affine_brightness_controls', 'output': str(output), 'source_sha256': sources,
                          'cases': cases}, indent=2))
    except BaseException as error:
        import traceback
        write(output / 'failure.json', {'status': 'synthetic_control_failed_retained', 'source_sha256': sources,
              'exception_type': type(error).__name__, 'exception_message': str(error), 'traceback': traceback.format_exc(),
              'real_TUM_registration_runs': 0, 'GPU_runs': 0})
        raise


if __name__ == '__main__': main()
