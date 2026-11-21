"""Independent SOURCE review and small SciPy construction, never author controls.

No real Gaussian map, trajectory, GT, image, renderer or registration is loaded.
Fixed numerical values below are mathematical witnesses, not SLAM results.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value: raise AssertionError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def camera(rotvec, translation):
    result = np.eye(4); result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix(); result[:3, 3] = translation
    return result


def wxyz(rotation): return rotation.as_quat()[[3, 0, 1, 2]]
def rotation_wxyz(q): return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def projected_covariance(camera_mean, camera_covariance, K):
    x, y, z = camera_mean
    require(z > 0., 'Positive camera-space depth for pinhole witness')
    J = np.array([[K[0, 0] / z, 0., -K[0, 0] * x / z ** 2],
                  [0., K[1, 1] / z, -K[1, 1] * y / z ** 2]])
    pixel = (K @ camera_mean)[:2] / z
    return pixel, J @ camera_covariance @ J.T


def independent_math():
    K = np.array([[132., 0., 79.5], [0., 128., 59.5], [0., 0., 1.]])
    scale_m = np.array([.01, .035, .075])
    old = camera([.12, -.08, .2], [.4, -.3, .1])
    new = camera([-.1, .18, .27], [.6, -.25, -.04])
    local_mean = np.array([.12, -.07, 2.3])
    world_mean = (old @ np.r_[local_mean, 1.])[:3]
    principal = Rotation.from_rotvec([.31, -.19, .14])
    world_cov = principal.as_matrix() @ np.diag(scale_m ** 2) @ principal.as_matrix().T
    # Generic inverse independently reconstructs the old-to-new world transform.
    correction = new @ np.linalg.inv(old)
    Rdelta = Rotation.from_matrix(correction[:3, :3])
    mean_corrected = (correction @ np.r_[world_mean, 1.])[:3]
    cov_corrected = correction[:3, :3] @ world_cov @ correction[:3, :3].T
    q_original = wxyz(principal)
    q_delta = wxyz(Rdelta)
    q_corrected = wxyz(Rdelta * principal)  # SciPy composition independently fixes Hamilton left order.
    reconstructed_cov = rotation_wxyz(q_corrected).as_matrix() @ np.diag(scale_m ** 2) @ rotation_wxyz(q_corrected).as_matrix().T
    old_inverse, new_inverse = np.linalg.inv(old), np.linalg.inv(new)
    before_mean = (old_inverse @ np.r_[world_mean, 1.])[:3]
    after_mean = (new_inverse @ np.r_[mean_corrected, 1.])[:3]
    before_cov = old_inverse[:3, :3] @ world_cov @ old_inverse[:3, :3].T
    after_cov = new_inverse[:3, :3] @ cov_corrected @ new_inverse[:3, :3].T
    before_pixel, before_pixel_cov = projected_covariance(before_mean, before_cov, K)
    after_pixel, after_pixel_cov = projected_covariance(after_mean, after_cov, K)
    residuals = {'camera_mean_m': float(np.max(np.abs(before_mean - after_mean))),
                 'camera_covariance_m_squared': float(np.max(np.abs(before_cov - after_cov))),
                 'pinhole_mean_pixels': float(np.max(np.abs(before_pixel - after_pixel))),
                 'pinhole_covariance_pixels_squared': float(np.max(np.abs(before_pixel_cov - after_pixel_cov))),
                 'quaternion_covariance_m_squared': float(np.max(np.abs(reconstructed_cov - cov_corrected)))}
    require(residuals['camera_mean_m'] < 1e-12 and residuals['camera_covariance_m_squared'] < 1e-14 and residuals['pinhole_mean_pixels'] < 1e-10 and residuals['pinhole_covariance_pixels_squared'] < 1e-10 and residuals['quaternion_covariance_m_squared'] < 1e-14, 'Mean, covariance, units and Hamilton left-order invariants')
    pose_only_mean = (new_inverse @ np.r_[world_mean, 1.])[:3]
    omitted_cov = new_inverse[:3, :3] @ world_cov @ new_inverse[:3, :3].T
    right_quat = wxyz(principal * Rdelta)
    right_cov = rotation_wxyz(right_quat).as_matrix() @ np.diag(scale_m ** 2) @ rotation_wxyz(right_quat).as_matrix().T
    right_camera_cov = new_inverse[:3, :3] @ right_cov @ new_inverse[:3, :3].T
    inverse_mean = (new_inverse @ np.linalg.inv(correction) @ np.r_[world_mean, 1.])[:3]
    negative_differences = {'pose_only_camera_mean_m': float(np.linalg.norm(pose_only_mean - before_mean)),
        'mean_transport_without_cov_camera_covariance_m_squared': float(np.max(np.abs(omitted_cov - before_cov))),
        'right_quaternion_composition_camera_covariance_m_squared': float(np.max(np.abs(right_camera_cov - before_cov))),
        'inverted_transport_camera_mean_m': float(np.linalg.norm(inverse_mean - before_mean))}
    require(negative_differences['pose_only_camera_mean_m'] > .01 and negative_differences['mean_transport_without_cov_camera_covariance_m_squared'] > 1e-5 and negative_differences['right_quaternion_composition_camera_covariance_m_squared'] > 1e-5 and negative_differences['inverted_transport_camera_mean_m'] > .01, 'Fixed physically effective non-isotropic, non-commuting negatives')
    # Two anchors agree at an old world point, but their rigid corrections differ.
    old_b = camera([-.03, .17, -.11], [-.3, .2, .05])
    new_b = camera([.09, -.12, -.07], [-.1, .15, .15])
    correction_b = new_b @ np.linalg.inv(old_b)
    shared_old_world = np.r_[world_mean, 1.]
    transported_a, transported_b = correction @ shared_old_world, correction_b @ shared_old_world
    local_b = np.linalg.solve(old_b, shared_old_world)
    after_local_b = np.linalg.solve(new_b, transported_b)
    seam_distance = float(np.linalg.norm(transported_a[:3] - transported_b[:3]))
    require(seam_distance > .01 and np.max(np.abs(after_local_b - local_b)) < 1e-12, 'Per-anchor invariance coexists with a seam and no single global transform')
    # World-frame view directions change. Degree0 RGB stays fixed; directional SH needs transport.
    old_view_direction = world_mean - old[:3, 3]; old_view_direction /= np.linalg.norm(old_view_direction)
    new_view_direction = mean_corrected - new[:3, 3]; new_view_direction /= np.linalg.norm(new_view_direction)
    restored_direction = correction[:3, :3].T @ new_view_direction
    require(np.max(np.abs(restored_direction - old_view_direction)) < 1e-12, 'Inverse rotation recovers original world-frame SH query direction')
    return {'status': 'passed_fixed_independent_SciPy_math_witnesses', 'residuals': residuals, 'negative_differences': negative_differences,
        'two_anchor_seam_distance_m': seam_distance, 'fixed_configuration': {'K_pixels': K.tolist(), 'principal_scales_m': scale_m.tolist(), 'old_c2w': old.tolist(), 'new_c2w': new.tolist(), 'original_wxyz': q_original.tolist(), 'correction_wxyz': q_delta.tolist(), 'corrected_wxyz': q_corrected.tolist()},
        'units': {'mean': 'meters', 'covariance3D': 'meters squared', 'K': 'pixels', 'pinhole_J': 'pixels per meter', 'covariance2D': 'pixels squared'},
        'scope': 'Small synthetic mathematical construction only. No author control execution, actual map, sensor/GT, solver, renderer or loop closure.'}


def author_fixture_math(config):
    """Read authored constants, reconstruct with SciPy without author functions."""
    def components(spec):
        return Rotation.from_euler('xyz', spec['rotation_xyz_degrees'], degrees=True), np.asarray(spec['translation_m'])
    rold, told = components(config['anchor_a_old_c2w'])
    rd, td = components(config['anchor_a_world_left_correction'])
    rnew, tnew = rd * rold, rd.apply(told) + td
    p = np.asarray(config['gaussian_camera_mean_m']); scales = np.asarray(config['gaussian_scales_m'])
    rl = Rotation.from_euler('xyz', config['gaussian_camera_rotation_xyz_degrees'], degrees=True)
    local_sigma = rl.as_matrix() @ np.diag(scales ** 2) @ rl.as_matrix().T
    mu = rold.apply(p) + told; corrected_mu = rd.apply(mu) + td
    sigma = rold.as_matrix() @ local_sigma @ rold.as_matrix().T
    corrected_sigma = rd.as_matrix() @ sigma @ rd.as_matrix().T
    camera_after = rnew.inv().apply(corrected_mu - tnew)
    sigma_after = rnew.inv().as_matrix() @ corrected_sigma @ rnew.as_matrix()
    fx, fy, cx, cy = config['pinhole_fx_fy_cx_cy_pixels']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    u0, S0 = projected_covariance(p, local_sigma, K)
    u1, S1 = projected_covariance(camera_after, sigma_after, K)
    qrotation = rd * (rold * rl)
    q_sigma = qrotation.as_matrix() @ np.diag(scales ** 2) @ qrotation.as_matrix().T
    pose_only = rnew.inv().apply(mu - tnew)
    omitted = rnew.inv().as_matrix() @ sigma @ rnew.as_matrix()
    _, omitted_pixels = projected_covariance(camera_after, omitted, K)
    _, tb = components(config['anchor_b_old_c2w'])
    rdb, tdb = components(config['anchor_b_world_left_correction'])
    mu_b = rdb.apply(mu) + tdb
    values = {
        'camera_mean_max_abs_m': float(np.max(np.abs(camera_after - p))),
        'camera_covariance_max_abs_m2': float(np.max(np.abs(sigma_after - local_sigma))),
        'pinhole_mean_max_abs_pixel': float(np.max(np.abs(u1 - u0))),
        'pinhole_covariance_max_abs_pixel2': float(np.max(np.abs(S1 - S0))),
        'quaternion_covariance_max_abs_m2': float(np.max(np.abs(q_sigma - corrected_sigma))),
        'pose_only_camera_mean_change_m': float(np.linalg.norm(pose_only - p)),
        'omitted_covariance_camera_change_Frobenius_m2': float(np.linalg.norm(omitted - local_sigma)),
        'omitted_covariance_pixel_change_Frobenius_pixel2': float(np.linalg.norm(omitted_pixels - S0)),
        'two_anchor_center_separation_m': float(np.linalg.norm(mu_b - corrected_mu)),
    }
    # Boundaries copied from the predeclared file; no adaptation or tuning.
    limits = config['tolerances']
    require(values['camera_mean_max_abs_m'] <= limits['mean_abs_m'] and values['camera_covariance_max_abs_m2'] <= limits['covariance_abs_m2'] and values['pinhole_mean_max_abs_pixel'] <= limits['pixel_abs'] and values['pinhole_covariance_max_abs_pixel2'] <= limits['pixel_covariance_abs'] and values['quaternion_covariance_max_abs_m2'] <= limits['covariance_abs_m2'], 'Independent SciPy same authored fixture identities meet predeclared numerical boundaries')
    require(values['pose_only_camera_mean_change_m'] > limits['negative_mean_m'] and values['omitted_covariance_camera_change_Frobenius_m2'] > limits['negative_covariance_m2'] and values['omitted_covariance_pixel_change_Frobenius_pixel2'] > limits['negative_pixel_covariance'] and values['two_anchor_center_separation_m'] > limits['seam_center_min_m'], 'Authored constants give physically effective independent negatives without running author controls')
    return {'status': 'passed_independent_SciPy_reconstruction_of_authored_fixed_constants', 'values': values, 'author_controls_executed': 0, 'projection_scope': 'analytic first-order pinhole, no actual raster or SH'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--author-source', type=Path, required=True); parser.add_argument('--author-sha', required=True)
    parser.add_argument('--author-fixtures', type=Path, required=True); parser.add_argument('--fixtures-sha', required=True)
    args = parser.parse_args(); require(sha(args.author_source) == args.author_sha and sha(args.author_fixtures) == args.fixtures_sha, 'Exact author SOURCE/fixture locks')
    output = ROOT / 'outputs/gaussian-pose-transport-independent-source-review-001'; output.mkdir(exist_ok=False)
    (output / 'review_source.py').write_bytes(Path(__file__).read_bytes()); (output / 'author_source.py').write_bytes(args.author_source.read_bytes()); (output / 'author_fixtures.json').write_bytes(args.author_fixtures.read_bytes())
    try:
        text = args.author_source.read_text(encoding='utf-8'); compile(text, str(args.author_source), 'exec')
        tree = ast.parse(text); imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        require(not any(any(bad in value for bad in ['track_rgbd', 'train_tum', 'tum_rgbd', 'torch', 'gsplat', 'requests', 'subprocess']) for value in imports), 'Author control has no solver/renderer/loader/network/subprocess import')
        math = independent_math(); fixture_math = author_fixture_math(json.loads(args.author_fixtures.read_text(encoding='utf-8')))
        require(sha(args.author_source) == args.author_sha and sha(args.author_fixtures) == args.fixtures_sha and not any(n in sys.modules for n in ['torch', 'gsplat', 'tum_rgbd', 'track_rgbd_icp']), 'Author bytes unchanged and no GT/GPU/solver import')
        result = {'status': 'passed_independent_gaussian_transport_SOURCE_review_and_fixed_math_witnesses', 'review_source_sha256': sha(Path(__file__)), 'author_source_relative_path': str(args.author_source.resolve().relative_to(ROOT)).replace('\\', '/'), 'author_source_sha256': args.author_sha,
            'author_fixture_sha256': args.fixtures_sha, 'author_source_imports': imports, 'independent_math': math, 'independent_authored_fixture_math': fixture_math, 'author_controls_executed': 0, 'actual_maps_loaded': 0, 'new_GT_inputs': 0, 'solver_calls': 0, 'GPU_calls': 0, 'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'manual_code_review': ['c2w maps each anchor-local point to its old/new world pose; Delta=Cnew inv(Cold) must act on the left.', 'Mean and anisotropic covariance need synchronized rigid transport; principal scales stay unchanged and quaternion composition is active Hamilton wxyz on the left.', 'Isotropic covariance cannot expose omitted covariance rotation; unequal principal scales and noncommuting rotations give effective negatives.', 'Different anchor corrections can split formerly coincident world points even while each anchor camera-space remains invariant; this is not global loop closure.', 'World-frame directional SH is outside constant-color controls; its coefficients or directional query must rotate consistently.'],
            'limits': ['This independent script does not run or import the author control or establish real rendering/loop-closure performance.', 'Pinhole covariance is a first-order analytic projection with metric/pixel units, not a gsplat renderer result.', 'Cold/Cnew must refer to the same physical anchor and use compatible world gauge/units; merged multi-anchor Gaussians need an explicit ownership policy.', 'No frozen experimental trainer, map, real output, acceptance gate or threshold is modified.']}
        write(output / 'metrics.json', result)
        print(json.dumps({'status': result['status'], 'review_source_sha256': result['review_source_sha256'], 'author_source_sha256': args.author_sha, 'metrics_sha256': sha(output / 'metrics.json'), 'author_controls_executed': 0, 'GPU_calls': 0}))
    except BaseException as error:
        write(output / 'failure.json', {'status': 'independent_transport_review_failure_retained', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'author_controls_executed': 0, 'GPU_calls': 0}); raise


if __name__ == '__main__': main()
