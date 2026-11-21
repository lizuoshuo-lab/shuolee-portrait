"""Deterministic matrix controls for anchor-local Gaussian pose correction.

SOURCE preparation is separate from execution. Only --check runs these controls.
No renderer, dataset, optimization, random inputs, SH transport, or map repair.
OpenCV column vectors; c2w cameras; active Hamilton wxyz Gaussian rotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "outputs/gaussian-map-pose-correction-source-preparation-001/fixtures.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class Evidence:
    def __init__(self, tolerances):
        self.tolerances = tolerances
        self.checks = []
        self.actual = {}

    def below(self, name, actual, limit_key, units, scope):
        actual = float(actual)
        limit = float(self.tolerances[limit_key])
        self.checks.append(dict(name=name, actual=actual, comparison="<=", limit=limit,
                                units=units, scope=scope, passed=bool(np.isfinite(actual) and actual <= limit)))
        if not self.checks[-1]["passed"]:
            raise AssertionError(name)

    def above(self, name, actual, limit_key, units, scope):
        actual = float(actual)
        limit = float(self.tolerances[limit_key])
        self.checks.append(dict(name=name, actual=actual, comparison=">", limit=limit,
                                units=units, scope=scope, passed=bool(np.isfinite(actual) and actual > limit)))
        if not self.checks[-1]["passed"]:
            raise AssertionError(name)

    def exact(self, name, actual, scope):
        self.checks.append(dict(name=name, actual=bool(actual), comparison="is True",
                                units="boolean", scope=scope, passed=bool(actual)))
        if not actual:
            raise AssertionError(name)


def rotation_xyz_degrees(values):
    """Rz @ Ry @ Rx active rotation; angles describe those fixed axis factors."""
    x, y, z = np.deg2rad(np.asarray(values, dtype=np.float64))
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


def quat_product(a, b):
    """Hamilton wxyz product: R(a product b) = R(a) @ R(b)."""
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v - x*i - y*j - z*k,
                     w*i + x*v + y*k - z*j,
                     w*j - x*k + y*v + z*i,
                     w*k + x*j - y*i + z*v], dtype=np.float64)


def quat_xyz_degrees(values):
    """Independent half-angle axis quaternions for the Rz @ Ry @ Rx factors."""
    x, y, z = np.deg2rad(np.asarray(values, dtype=np.float64)) / 2.
    qx = np.array([np.cos(x), np.sin(x), 0., 0.])
    qy = np.array([np.cos(y), 0., np.sin(y), 0.])
    qz = np.array([np.cos(z), 0., 0., np.sin(z)])
    return quat_product(qz, quat_product(qy, qx))


def rotation_from_quat(q):
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) == 0:
        raise ValueError("Quaternion must be finite and nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1.-2.*(y*y+z*z), 2.*(x*y-w*z), 2.*(x*z+w*y)],
                     [2.*(x*y+w*z), 1.-2.*(x*x+z*z), 2.*(y*z-w*x)],
                     [2.*(x*z-w*y), 2.*(y*z+w*x), 1.-2.*(x*x+y*y)]])


def camera(spec):
    c = np.eye(4, dtype=np.float64)
    c[:3, :3] = rotation_xyz_degrees(spec["rotation_xyz_degrees"])
    c[:3, 3] = np.asarray(spec["translation_m"], dtype=np.float64)
    return c


def rigid_inverse(c):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = c[:3, :3].T
    result[:3, 3] = -c[:3, :3].T @ c[:3, 3]
    return result


def point(c, p):
    return c[:3, :3] @ p + c[:3, 3]


def camera_gaussian(c2w, mean_world, covariance_world):
    # Generic solve is intentionally a separate route from rigid_inverse.
    mean_camera = np.linalg.solve(c2w, np.r_[mean_world, 1.])[:3]
    r = c2w[:3, :3]
    return mean_camera, r.T @ covariance_world @ r


def project(mean_camera, covariance_camera, intrinsics):
    x, y, z = mean_camera
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Synthetic perspective control requires positive depth")
    fx, fy, cx, cy = intrinsics
    uv = np.array([fx*x/z+cx, fy*y/z+cy])
    j = np.array([[fx/z, 0., -fx*x/(z*z)], [0., fy/z, -fy*y/(z*z)]])
    # Linearized pinhole covariance, not a renderer's EWA/conic implementation.
    return uv, j @ covariance_camera @ j.T


def proper(name, c, evidence):
    r = c[:3, :3]
    evidence.exact(name+"_finite", np.all(np.isfinite(c)), "4x4 float64 matrix")
    evidence.below(name+"_bottom_row", np.max(np.abs(c[3]-[0., 0., 0., 1.])), "matrix_abs", "dimensionless", "exact homogeneous structure")
    evidence.below(name+"_R_orthogonality", np.linalg.norm(r.T@r-np.eye(3)), "matrix_abs", "dimensionless Frobenius", "proper SE3 prerequisite")
    evidence.below(name+"_det_positive_one", abs(np.linalg.det(r)-1.), "matrix_abs", "dimensionless", "proper rotation prerequisite")


def invariant(name, old_c, new_c, old_mu, new_mu, old_cov, new_cov, k, evidence):
    old_p, old_s = camera_gaussian(old_c, old_mu, old_cov)
    new_p, new_s = camera_gaussian(new_c, new_mu, new_cov)
    old_uv, old_u = project(old_p, old_s, k)
    new_uv, new_u = project(new_p, new_s, k)
    evidence.below(name+"_camera_mean", np.max(np.abs(new_p-old_p)), "mean_abs_m", "m", "same sensor anchor, corrected camera and Gaussian")
    evidence.below(name+"_camera_covariance", np.max(np.abs(new_s-old_s)), "covariance_abs_m2", "m^2", "same anchor, full covariance congruence")
    evidence.below(name+"_pinhole_mean", np.max(np.abs(new_uv-old_uv)), "pixel_abs", "pixel", "positive-depth deterministic pinhole projection")
    evidence.below(name+"_linearized_pinhole_covariance", np.max(np.abs(new_u-old_u)), "pixel_covariance_abs", "pixel^2", "first-order J Sigma J^T only; no renderer")
    evidence.actual[name] = dict(old_c2w=old_c, new_c2w=new_c, old_mean_world_m=old_mu,
        new_mean_world_m=new_mu, old_covariance_world_m2=old_cov, new_covariance_world_m2=new_cov,
        old_mean_camera_m=old_p, new_mean_camera_m=new_p, old_covariance_camera_m2=old_s,
        new_covariance_camera_m2=new_s, old_pixel_mean=old_uv, new_pixel_mean=new_uv,
        old_linearized_pixel_covariance=old_u, new_linearized_pixel_covariance=new_u)
    return old_p, old_s, old_uv, old_u


def run_controls(config, e):
    old = camera(config["anchor_a_old_c2w"])
    specified_delta = camera(config["anchor_a_world_left_correction"])
    new = specified_delta @ old
    delta = new @ rigid_inverse(old)
    for name, c in (("Cold_a", old), ("Cnew_a", new), ("delta_a", delta)):
        proper(name, c, e)
    e.below("recover_specified_world_left_rotation", np.max(np.abs(delta[:3,:3]-specified_delta[:3,:3])), "matrix_abs", "dimensionless", "Cnew @ inv(Cold), both same anchor")
    e.below("recover_specified_world_left_translation", np.max(np.abs(delta[:3,3]-specified_delta[:3,3])), "mean_abs_m", "m", "translation compared separately from rotation")
    e.above("noncommuting_rotation_factors", np.linalg.norm((specified_delta@old)[:3,:3]-(old@specified_delta)[:3,:3]), "negative_rotation_frobenius", "dimensionless Frobenius", "fixture is not a commuting transform")
    e.above("noncommuting_translation_composition", np.linalg.norm((specified_delta@old)[:3,3]-(old@specified_delta)[:3,3]), "negative_mean_m", "m", "translation composition has a meaningful direction control")

    p_local = np.asarray(config["gaussian_camera_mean_m"], dtype=np.float64)
    scales = np.asarray(config["gaussian_scales_m"], dtype=np.float64)
    e.exact("scales_finite_and_strictly_positive", np.all(np.isfinite(scales)) and np.all(scales > 0.), "standard-deviation axis scales in meters")
    r_local = rotation_xyz_degrees(config["gaussian_camera_rotation_xyz_degrees"])
    s_local = r_local @ np.diag(scales**2) @ r_local.T
    old_mu = point(old, p_local)
    old_cov = old[:3,:3] @ s_local @ old[:3,:3].T
    new_mu = point(delta, old_mu)
    new_cov = delta[:3,:3] @ old_cov @ delta[:3,:3].T
    # A physical local Gaussian gives independent expected world parameters.
    expected_new_mu = point(new, p_local)
    expected_new_cov = new[:3,:3] @ s_local @ new[:3,:3].T
    e.below("local_gaussian_expected_new_world_mean", np.max(np.abs(new_mu-expected_new_mu)), "mean_abs_m", "m", "independent fixed camera-space Gaussian construction")
    e.below("local_gaussian_expected_new_world_covariance", np.max(np.abs(new_cov-expected_new_cov)), "covariance_abs_m2", "m^2", "independent fixed camera-space Gaussian construction")
    e.above("strict_anisotropy", np.min(np.diff(np.sort(scales**2))), "anisotropy_min_variance_gap_m2", "m^2", "three unequal variances make a missing covariance rotation detectable")
    e.above("positive_depth_margin", p_local[2], "positive_depth_min_m", "m", "avoid near-zero-depth amplification")
    k = np.asarray(config["pinhole_fx_fy_cx_cy_pixels"], dtype=np.float64)
    old_p, old_s, old_uv, old_u = invariant("anchor_a", old, new, old_mu, new_mu, old_cov, new_cov, k, e)
    e.below("eigenvalues_scales_preserved", np.max(np.abs(np.linalg.eigvalsh(new_cov)-np.sort(scales**2))), "covariance_abs_m2", "m^2", "rigid congruence preserves eigenvalues")

    q_old = quat_product(quat_xyz_degrees(config["anchor_a_old_c2w"]["rotation_xyz_degrees"]),
                         quat_xyz_degrees(config["gaussian_camera_rotation_xyz_degrees"]))
    q_delta = quat_xyz_degrees(config["anchor_a_world_left_correction"]["rotation_xyz_degrees"])
    q_new = quat_product(q_delta, q_old)
    q_new /= np.linalg.norm(q_new)
    q_cov = rotation_from_quat(q_new) @ np.diag(scales**2) @ rotation_from_quat(q_new).T
    e.below("quaternion_left_product_rotation", np.max(np.abs(rotation_from_quat(q_new)-delta[:3,:3]@rotation_from_quat(q_old))), "matrix_abs", "dimensionless", "Hamilton wxyz q_delta product q_old")
    e.below("quaternion_left_product_covariance", np.max(np.abs(q_cov-new_cov)), "covariance_abs_m2", "m^2", "quaternion route agrees with covariance congruence")
    e.below("quaternion_sign_equivalent", np.max(np.abs(rotation_from_quat(-q_new)-rotation_from_quat(q_new))), "matrix_abs", "dimensionless", "q and -q describe the same rotation")
    q_wrong = quat_product(q_old, q_delta)
    wrong_q_cov = rotation_from_quat(q_wrong) @ np.diag(scales**2) @ rotation_from_quat(q_wrong).T
    e.above("right_product_wrong_in_world_frame_covariance", np.linalg.norm(wrong_q_cov-new_cov), "negative_covariance_m2", "m^2 Frobenius", "same declared world-left correction; right product uses a different frame")

    # Negative 1: update camera while leaving the world Gaussian unchanged.
    pose_only_p, pose_only_s = camera_gaussian(new, old_mu, old_cov)
    pose_only_uv, pose_only_u = project(pose_only_p, pose_only_s, k)
    e.above("pose_only_camera_mean_changes", np.linalg.norm(pose_only_p-old_p), "negative_mean_m", "m", "same physical Gaussian cannot stay fixed in world under this anchor correction")
    e.above("pose_only_pixel_mean_changes", np.linalg.norm(pose_only_uv-old_uv), "negative_pixel", "pixel", "deterministic nontrivial correction; not a claim for every special-case point")
    # Negative 2: update mean correctly but leave an anisotropic covariance behind.
    no_cov_p, no_cov_s = camera_gaussian(new, new_mu, old_cov)
    no_cov_uv, no_cov_u = project(no_cov_p, no_cov_s, k)
    e.below("mean_only_pixel_mean_still_matches", np.max(np.abs(no_cov_uv-old_uv)), "pixel_abs", "pixel", "a center-only check would miss this covariance defect")
    e.above("mean_only_camera_covariance_changes", np.linalg.norm(no_cov_s-old_s), "negative_covariance_m2", "m^2 Frobenius", "anisotropic axes must rotate with the world correction")
    e.above("mean_only_pixel_covariance_changes", np.linalg.norm(no_cov_u-old_u), "negative_pixel_covariance", "pixel^2 Frobenius", "linearized projected ellipse changes despite identical projected mean")
    # Spherical covariance is invariant to rotation and is not a valid negative fixture.
    isotropic = np.eye(3) * float(config["isotropic_scale_m"])**2
    _, iso_old = camera_gaussian(old, old_mu, isotropic)
    _, iso_new = camera_gaussian(new, new_mu, isotropic)
    e.below("isotropic_missing_rotation_is_not_detectable", np.max(np.abs(iso_new-iso_old)), "covariance_abs_m2", "m^2", "explains why the covariance negative control must be anisotropic")
    wrong_delta = rigid_inverse(old) @ new
    wrong_mu = point(wrong_delta, old_mu)
    e.above("camera_local_delta_misused_as_world_delta", np.linalg.norm(wrong_mu-new_mu), "negative_mean_m", "m", "inv(Cold) @ Cnew is camera-local; applying it to world means is a frame error")
    e.actual["negative_controls"] = dict(pose_only_mean_camera_m=pose_only_p, pose_only_covariance_camera_m2=pose_only_s,
        pose_only_pixel_mean=pose_only_uv, pose_only_linearized_pixel_covariance=pose_only_u,
        mean_only_mean_camera_m=no_cov_p, mean_only_covariance_camera_m2=no_cov_s,
        mean_only_pixel_mean=no_cov_uv, mean_only_linearized_pixel_covariance=no_cov_u,
        q_old_wxyz=q_old, q_delta_wxyz=q_delta, q_new_wxyz=q_new, q_wrong_right_product_wxyz=q_wrong,
        wrong_frame_delta=wrong_delta, wrong_frame_new_world_mean_m=wrong_mu)

    # Two copies initially represent the same world Gaussian but have distinct anchors.
    old_b = camera(config["anchor_b_old_c2w"])
    specified_delta_b = camera(config["anchor_b_world_left_correction"])
    new_b = specified_delta_b @ old_b
    delta_b = new_b @ rigid_inverse(old_b)
    for name, c in (("Cold_b", old_b), ("Cnew_b", new_b), ("delta_b", delta_b)):
        proper(name, c, e)
    b_mu = point(delta_b, old_mu)
    b_cov = delta_b[:3,:3] @ old_cov @ delta_b[:3,:3].T
    invariant("anchor_b", old_b, new_b, old_mu, b_mu, old_cov, b_cov, k, e)
    separation = np.linalg.norm(new_mu-b_mu)
    e.above("different_anchor_corrections_split_overlapping_centers", separation, "seam_center_min_m", "m", "piecewise rigid updates do not guarantee overlap consistency")
    b_in_a, b_s_in_a = camera_gaussian(new, b_mu, b_cov)
    b_uv_in_a, _ = project(b_in_a, b_s_in_a, k)
    e.above("different_anchor_corrections_split_in_anchor_a_pixels", np.linalg.norm(b_uv_in_a-old_uv), "negative_pixel", "pixel", "potential seam geometry; no raster or surface-quality claim")
    common_b_mu = point(delta, old_mu)
    e.below("one_common_world_correction_keeps_overlap", np.linalg.norm(common_b_mu-new_mu), "mean_abs_m", "m", "single global gauge action is different from per-anchor corrections")
    e.actual["two_anchor_control"] = dict(initial_shared_world_mean_m=old_mu, initial_shared_world_covariance_m2=old_cov,
        delta_a=delta, delta_b=delta_b, corrected_anchor_a_world_mean_m=new_mu, corrected_anchor_b_world_mean_m=b_mu,
        corrected_center_separation_m=separation, b_copy_pixel_in_new_anchor_a=b_uv_in_a,
        own_anchor_invariance_does_not_imply_cross_anchor_consistency=True)
    e.exact("no_renderer_or_training_modules", not any(n in sys.modules for n in ("torch", "gsplat")), "NumPy matrix controls only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True, help="explicitly execute reviewed synthetic controls")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixtures-sha", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run", default="gaussian-map-pose-correction-cpu-controls-001")
    args = parser.parse_args()
    if not args.run or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in args.run):
        raise ValueError("Safe unique run name required")
    source = Path(__file__).resolve()
    if digest(source) != args.source_sha or digest(args.fixtures) != args.fixtures_sha:
        raise ValueError("Reviewed source and deterministic fixture bytes must match")
    config = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if config["status"] != "prepared_deterministic_inputs_not_executed" or config["dtype"] != "float64":
        raise ValueError("Exact reviewed float64 synthetic fixture required")
    output = ROOT / "outputs" / args.run
    output.mkdir(exist_ok=False)
    (output / "source_script.py").write_bytes(source.read_bytes())
    (output / "fixtures.json").write_bytes(args.fixtures.read_bytes())
    e = Evidence(config["tolerances"])
    scope = dict(source_sha256=args.source_sha, fixture_sha256=args.fixtures_sha,
        matrix_dtype="float64", renderer_calls=0, optimizer_steps=0, real_dataset_reads=0, GPU_calls=0,
        scope="Deterministic same-anchor matrix/projection identities and negative controls; not map repair or SH transport")
    try:
        run_controls(config, e)
        e.exact("reviewed_input_bytes_unchanged", digest(source) == args.source_sha and digest(args.fixtures) == args.fixtures_sha, "source/fixture postguard")
        write_new(output / "metrics.json", json_value(dict(status="passed_synthetic_anchor_gaussian_correction_controls", **scope,
            all_checks=e.checks, actual_matrices_and_projections=e.actual, limits=config["limits"])))
        files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output.iterdir()) if p.is_file()}
        write_new(output / "completion.json", dict(status="completed", **scope, consumed_and_saved_files=files))
    except BaseException as error:
        write_new(output / "failure.json", json_value(dict(status="failed_synthetic_controls_preserved", **scope,
            exception_type=type(error).__name__, exception_message=str(error), traceback=traceback.format_exc(),
            checks_completed=e.checks, actual_matrices_and_projections=e.actual)))
        raise
    print(json.dumps(dict(status="passed_synthetic_anchor_gaussian_correction_controls", check_count=len(e.checks), GPU_calls=0)))


if __name__ == "__main__":
    main()
