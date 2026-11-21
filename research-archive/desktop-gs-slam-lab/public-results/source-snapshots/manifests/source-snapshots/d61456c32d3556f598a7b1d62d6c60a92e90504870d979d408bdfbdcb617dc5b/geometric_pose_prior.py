"""Differentiable engineering pose prior around a fixed geometric reference.

Inputs are rigid world-to-camera matrices (OpenCV w2c), in meters. Analytic
rigid inversion gives camera centers and c2w rotations. sigma_t_m and
sigma_r_rad are engineering scales, not calibrated covariance or Hessian
information. The rotation residual is chordal, with a small-angle approximation
to squared rotation angle; it is not an SO(3) logarithm.

The reference is detached. This helper does not optimize or select poses, does
not access GT, and does not change the locked V3 or diagnostic implementation.
Run this file for float64 CPU mathematical controls and a fresh JSON manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _shape_and_dtype(camera: torch.Tensor) -> None:
    if not isinstance(camera, torch.Tensor) or camera.shape != (4, 4) or not camera.is_floating_point():
        raise ValueError("Expected one floating-point rigid 4x4 w2c tensor")


@torch.no_grad()
def validate_rigid_camera(camera: torch.Tensor, tolerance: float | None = None) -> None:
    """Optional boundary validation, separate from the differentiable loss graph."""
    _shape_and_dtype(camera)
    tolerance = tolerance if tolerance is not None else (1e-8 if camera.dtype == torch.float64 else 1e-5)
    if not torch.isfinite(camera).all():
        raise ValueError("Camera must be finite")
    rotation = camera[:3, :3]
    identity = torch.eye(3, dtype=camera.dtype, device=camera.device)
    bottom = camera.new_tensor([0., 0., 0., 1.])
    if (camera[3]-bottom).abs().max() > tolerance or (rotation.T@rotation-identity).abs().max() > tolerance or abs(float(torch.linalg.det(rotation))-1.) > tolerance:
        raise ValueError("Analytic rigid inverse requires a proper SO(3) rotation and affine bottom row")


def rigid_inverse_w2c(w2c: torch.Tensor, *, validate: bool = False) -> torch.Tensor:
    """Return c2w=[R.T,-R.T@t;0,1], preserving candidate autograd."""
    _shape_and_dtype(w2c)
    if validate:
        validate_rigid_camera(w2c)
    rotation_transpose = w2c[:3, :3].T
    center = -rotation_transpose@w2c[:3, 3]
    upper = torch.cat((rotation_transpose, center[:, None]), dim=1)
    return torch.cat((upper, w2c.new_tensor([[0., 0., 0., 1.]])), dim=0)


def geometric_pose_prior(candidate_w2c: torch.Tensor, reference_w2c: torch.Tensor, *,
                         sigma_t_m: float, sigma_r_rad: float, weight: float = 1.,
                         validate: bool = False) -> torch.Tensor:
    """Return weight*(||C-Cref||²/sigma_t² + ||R-Rref||F²/(2*sigma_r²)).

    R and C are c2w rotation and camera center. The fixed reference is detached
    and cast to the candidate dtype/device. At weight=0 the prior is off: exact
    differentiable zero, without evaluating a reference residual. Valid positive
    scales are still required to catch malformed configuration. Optional camera
    validation is intended at boundaries, not required on every optimizer step.
    """
    _shape_and_dtype(candidate_w2c)
    if not all(math.isfinite(float(value)) for value in (sigma_t_m, sigma_r_rad, weight)) or sigma_t_m <= 0 or sigma_r_rad <= 0 or weight < 0:
        raise ValueError("Positive finite meter/radian engineering scales and nonnegative finite weight required")
    if validate:
        validate_rigid_camera(candidate_w2c)
    if weight == 0:
        return candidate_w2c.sum()*0.
    _shape_and_dtype(reference_w2c)
    reference = reference_w2c.detach().to(dtype=candidate_w2c.dtype, device=candidate_w2c.device)
    candidate_c2w = rigid_inverse_w2c(candidate_w2c)
    reference_c2w = rigid_inverse_w2c(reference, validate=validate)
    translation_squared = (candidate_c2w[:3, 3]-reference_c2w[:3, 3]).square().sum()
    rotation_squared = (candidate_c2w[:3, :3]-reference_c2w[:3, :3]).square().sum()
    return float(weight)*(translation_squared/float(sigma_t_m)**2 + rotation_squared/(2*float(sigma_r_rad)**2))


def _finite_difference_control(name, base, reference, twist_values, scales, weight=1.):
    from track_fixed_map import se3_exp
    twist = torch.tensor(twist_values, dtype=torch.float64, requires_grad=True)

    def loss_at(vector):
        return geometric_pose_prior(se3_exp(vector)@base, reference, sigma_t_m=scales[0], sigma_r_rad=scales[1], weight=weight, validate=True)

    loss = loss_at(twist)
    gradient = torch.autograd.grad(loss, twist)[0]
    epsilon = 1e-6
    numeric = []
    with torch.no_grad():
        for axis in range(6):
            delta = torch.zeros(6, dtype=torch.float64); delta[axis] = epsilon
            numeric.append(float((loss_at(twist+delta)-loss_at(twist-delta))/(2*epsilon)))
    finite_difference = torch.tensor(numeric, dtype=torch.float64)
    absolute = float((gradient-finite_difference).abs().max())
    relative = float((gradient-finite_difference).norm()/finite_difference.norm().clamp_min(1e-12))
    if not torch.isfinite(gradient).all() or not torch.isfinite(finite_difference).all() or absolute > 2e-7 or (finite_difference.norm() > 1e-7 and relative > 1e-7):
        raise AssertionError(f"Six-dimensional CPU prior derivative mismatch: {name}")
    return {"name": name, "left_SE3_twist_rho_m_omega_rad": twist_values,
            "scalar_prior": float(loss.detach()), "analytic_six_dof": gradient.tolist(),
            "central_finite_difference_six_dof": numeric, "epsilon": epsilon,
            "gradient_max_absolute_error": absolute, "gradient_relative_l2_error": relative,
            "all_gradients_finite": True}


def cpu_checks() -> dict:
    from track_fixed_map import se3_exp
    identity = torch.eye(4, dtype=torch.float64)
    base_twist = torch.tensor([.3, -.1, .2, .2, -.1, .15], dtype=torch.float64)
    reference_twist = torch.tensor([.25, -.16, .15, -.13, .08, .21], dtype=torch.float64)
    base, reference = se3_exp(base_twist), se3_exp(reference_twist)
    scales = (.05, math.radians(5.))
    cases = [
        _finite_difference_control("translation_at_identity", identity, identity, [.02, -.03, .01, 0., 0., 0.], scales),
        _finite_difference_control("rotation_at_identity", identity, identity, [0., 0., 0., .03, -.02, .04], scales),
        _finite_difference_control("off_identity_base_nonzero_reference_at_zero_twist", base, reference, [0.]*6, scales),
        _finite_difference_control("off_identity_nonzero_left_perturbation", base, reference, [.01, -.02, .015, .02, -.03, .01], scales, weight=.7),
        _finite_difference_control("candidate_equals_fixed_reference", reference, reference, [0.]*6, scales),
        _finite_difference_control("weight_zero_off", base, reference, [.01, -.02, .015, .02, -.03, .01], scales, weight=0.),
    ]
    # Hand-computable examples do not use the implementation as their oracle.
    translation = identity.clone(); translation[0, 3] = .1
    translation_expected = .1**2/scales[0]**2
    translation_actual = float(geometric_pose_prior(translation, identity, sigma_t_m=scales[0], sigma_r_rad=scales[1]))
    theta = .1
    rotation = identity.clone()
    rotation[:3, :3] = torch.tensor([[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]], dtype=torch.float64)
    rotation_expected = 2*(1-math.cos(theta))/scales[1]**2
    rotation_actual = float(geometric_pose_prior(rotation, identity, sigma_t_m=scales[0], sigma_r_rad=scales[1]))
    if abs(translation_actual-translation_expected) > 1e-12 or abs(rotation_actual-rotation_expected) > 1e-12:
        raise AssertionError("Hand-computed translation or chordal-angle prior mismatch")
    candidate = se3_exp(torch.tensor([.01, -.02, .015, .02, -.03, .01], dtype=torch.float64))@base
    c2w = rigid_inverse_w2c(candidate, validate=True)
    inverse_error = float((c2w-torch.linalg.inv(candidate)).abs().max())
    if inverse_error > 1e-12:
        raise AssertionError("Analytic inverse differs from independent general inverse")
    # Common world gauge: c2w'=G@c2w, hence w2c'=w2c@inverse(G).
    gauge = se3_exp(torch.tensor([1., -.4, .7, -.3, .2, .4], dtype=torch.float64))
    transformed_candidate = candidate@torch.linalg.inv(gauge)
    transformed_reference = reference@torch.linalg.inv(gauge)
    original_loss = float(geometric_pose_prior(candidate, reference, sigma_t_m=scales[0], sigma_r_rad=scales[1]))
    transformed_loss = float(geometric_pose_prior(transformed_candidate, transformed_reference, sigma_t_m=scales[0], sigma_r_rad=scales[1], validate=True))
    gauge_error = abs(original_loss-transformed_loss)
    if gauge_error > 1e-10:
        raise AssertionError("Prior changed under common SE3 world gauge")
    # Deliberately incorrect camera center: use w2c translation t directly.
    true_translation_squared = float((rigid_inverse_w2c(candidate)[:3, 3]-rigid_inverse_w2c(reference)[:3, 3]).square().sum())
    wrong_translation_squared = float((candidate[:3, 3]-reference[:3, 3]).square().sum())
    wrong_before = wrong_translation_squared/scales[0]**2
    wrong_after = float((transformed_candidate[:3, 3]-transformed_reference[:3, 3]).square().sum())/scales[0]**2
    if abs(true_translation_squared-wrong_translation_squared) < 1e-4 or abs(wrong_before-wrong_after) < .1:
        raise AssertionError("Wrong-w2c-center negative control failed to detect the convention error")
    # A fixed reference must not receive gradients, even if caller marked it trainable.
    independent_candidate = candidate.detach().requires_grad_(True)
    trainable_reference = reference.detach().requires_grad_(True)
    fixed_loss = geometric_pose_prior(independent_candidate, trainable_reference, sigma_t_m=scales[0], sigma_r_rad=scales[1])
    fixed_loss.backward()
    if trainable_reference.grad is not None or independent_candidate.grad is None or not torch.isfinite(independent_candidate.grad).all():
        raise AssertionError("Fixed reference detachment or candidate graph is incorrect")
    if cases[-2]["scalar_prior"] != 0 or any(value != 0 for value in cases[-2]["analytic_six_dof"]):
        raise AssertionError("Equal pose must have exact zero finite gradient")
    if cases[-1]["scalar_prior"] != 0 or any(value != 0 for value in cases[-1]["analytic_six_dof"]):
        raise AssertionError("Weight zero must disable the prior with exact zero gradient")
    return {"status": "passed_cpu_float64_mathematical_controls", "device": "CPU", "dtype": "torch.float64",
            "formula": "weight*(||C-Cref||^2/sigma_t_m^2 + ||R_c2w-Rref_c2w||_F^2/(2*sigma_r_rad^2))",
            "conventions": "Inputs w2c; analytic inverse to c2w; C=-R_w2c.T@t_w2c; translation meters, rotation radians; left SE3 twist [rho_m,omega_rad]",
            "fixed_reference_detached": True, "engineering_scales": {"sigma_t_m": scales[0], "sigma_r_rad": scales[1]},
            "scale_interpretation": "Engineering normalization and weight only, not calibrated covariance, uncertainty, or Hessian information",
            "six_dof_controls": cases,
            "hand_controls": {"translation_actual": translation_actual, "translation_expected": translation_expected,
                              "rotation_theta_rad": theta, "rotation_chordal_actual": rotation_actual,
                              "rotation_chordal_expected_2_one_minus_cos_over_sigma_squared": rotation_expected,
                              "exact_geodesic_theta_squared_over_sigma_squared": theta**2/scales[1]**2},
            "analytic_vs_general_inverse_max_abs_error": inverse_error,
            "common_world_gauge": {"prior_before": original_loss, "prior_after": transformed_loss, "absolute_difference": gauge_error},
            "wrong_w2c_translation_as_center_negative_control": {"true_camera_center_squared_distance_m2": true_translation_squared,
                                                                "wrong_w2c_translation_squared_distance_m2": wrong_translation_squared,
                                                                "wrong_normalized_translation_term_before_world_gauge": wrong_before,
                                                                "wrong_term_after_world_gauge": wrong_after,
                                                                "convention_error_detected": True},
            "limitations": ["Chordal rotation is a small-angle approximation to squared SO3 angle, not exact logarithm", "Synthetic CPU math only; no V4 integration, dataset accuracy, parameter calibration or GPU run", "Engineering test scales are not a chosen training configuration", "Rigid inverse assumes proper SO3 and affine bottom row; optional boundary validation provided"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"manifests/geometric-pose-prior-cpu-checks.json")
    args = parser.parse_args()
    result = cpu_checks()
    result["source"] = {"geometric_pose_prior.py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "se3_helper_sha256": hashlib.sha256((ROOT/"scripts/track_fixed_map.py").read_bytes()).hexdigest(),
                        "torch": torch.__version__, "fixed_SE3_source": "scripts/track_fixed_map.py:se3_exp",
                        "derivation": "Analytic inverse [R.T,-R.T@t;0,1]; Frobenius orthogonal invariance; ||R-Rref||F^2=4*(1-cos(theta))"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
