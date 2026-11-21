"""CPU projective point-to-plane ICP control for two fixed TUM clips.

T_previous_from_current maps the current camera's depth points to the previous
camera. With p=Tq and target normal n, residual r=n.(p-v) has the left-SE3
Jacobian [n, p cross n]. Solve J delta=-r and update T=exp(delta)@T.
The first c2w is identity; c2w_current=c2w_previous@T. Every pair starts from
identity and uses depth only. GT pose values enter only final evaluation.

Usage:
  python scripts/track_rgbd_icp.py --self-test
  python scripts/track_rgbd_icp.py --clip short20 --run <new-run-name>
  python scripts/track_rgbd_icp.py --clip long100 --run <new-run-name>

No torch, CUDA, Open3D or model renderer is imported or used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from tum_rgbd import associate_frames, load_frames, scaled_intrinsics
from audit_tum_trajectory import audit_pose_arrays

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ((40, 30), (80, 60), (160, 120))
CLIPS = {"short20": (30, 20), "long100": (200, 100)}


@dataclass(frozen=True)
class ICPConfig:
    iterations: tuple[int, int, int] = (15, 12, 10)
    distance_gates_m: tuple[float, float, float] = (.10, .07, .05)
    normal_angle_degrees: float = 45.
    edge_absolute_m: float = .04
    edge_relative: float = .02
    huber_delta_m: float = .01
    min_correspondences: int = 50
    min_correspondence_ratio: float = .10
    rank_rcond: float = 1e-6
    max_normalized_hessian_condition: float = 1e8
    max_iteration_translation_m: float = .05
    max_iteration_rotation_degrees: float = 5.
    max_pair_translation_m: float = .20
    max_pair_rotation_degrees: float = 20.
    max_final_plane_rmse_m: float = .03
    convergence_translation_m: float = 1e-6
    convergence_rotation_radians: float = 1e-6
    line_search_scales: tuple[float, ...] = (1., .5, .25, .125, .0625, .03125)


@dataclass
class Geometry:
    points: np.ndarray
    normals: np.ndarray
    valid: np.ndarray
    K: np.ndarray


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                    encoding="utf-8")


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """SE3 exponential for [translation tangent meters, rotation radians]."""
    tangent = np.asarray(twist, dtype=np.float64)
    if tangent.shape != (6,) or not np.isfinite(tangent).all():
        raise ValueError("Finite six-vector required")
    velocity, omega = tangent[:3], tangent[3:]
    theta2 = float(omega @ omega)
    if theta2 < 1e-10:
        a = 1-theta2/6+theta2*theta2/120
        b = .5-theta2/24+theta2*theta2/720
        c = 1/6-theta2/120+theta2*theta2/5040
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta)/theta
        b = (1-math.cos(theta))/theta2
        c = (theta-math.sin(theta))/(theta2*theta)
    W = skew(omega)
    W2 = W @ W
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3)+a*W+b*W2
    transform[:3, 3] = (np.eye(3)+b*W+c*W2) @ velocity
    return transform


def inverse_rigid(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T+transform[:3, 3]


def rotation_degrees(rotation: np.ndarray) -> float:
    sine = np.linalg.norm([rotation[2, 1]-rotation[1, 2],
                           rotation[0, 2]-rotation[2, 0],
                           rotation[1, 0]-rotation[0, 1]])/2
    cosine = np.clip((np.trace(rotation)-1)/2, -1., 1.)
    return math.degrees(math.atan2(float(sine), float(cosine)))


def depth_geometry(depth: np.ndarray, mask: np.ndarray, K: np.ndarray,
                   config: ICPConfig) -> Geometry:
    """Central-difference normals, rejecting holes and adjacent depth jumps.

    Image coordinates use integer pixel centers and +Y down, +Z forward.
    Normals are oriented toward the camera. Only center+four valid neighbors
    with depth differences below absolute+relative limits are retained.
    """
    z = np.asarray(depth, dtype=np.float64)
    valid_depth = np.asarray(mask, dtype=bool) & np.isfinite(z) & (z > 0)
    height, width = z.shape
    v, u = np.mgrid[:height, :width]
    points = np.stack(((u-K[0, 2])*z/K[0, 0],
                       (v-K[1, 2])*z/K[1, 1], z), axis=-1)
    dx = points[1:-1, 2:]-points[1:-1, :-2]
    dy = points[2:, 1:-1]-points[:-2, 1:-1]
    interior_normals = np.cross(dx, dy)
    lengths = np.linalg.norm(interior_normals, axis=-1)
    center = z[1:-1, 1:-1]
    tolerance = config.edge_absolute_m+config.edge_relative*center
    interior_valid = valid_depth[1:-1, 1:-1].copy()
    for neighbor, neighbor_valid in [
        (z[1:-1, 2:], valid_depth[1:-1, 2:]),
        (z[1:-1, :-2], valid_depth[1:-1, :-2]),
        (z[2:, 1:-1], valid_depth[2:, 1:-1]),
        (z[:-2, 1:-1], valid_depth[:-2, 1:-1]),
    ]:
        interior_valid &= neighbor_valid & (np.abs(neighbor-center) <= tolerance)
    interior_valid &= np.isfinite(lengths) & (lengths > 1e-12)
    normalized = np.zeros_like(interior_normals)
    normalized[interior_valid] = interior_normals[interior_valid]/lengths[interior_valid, None]
    facing_away = np.sum(normalized*points[1:-1, 1:-1], axis=-1) > 0
    normalized[facing_away] *= -1
    normals = np.zeros_like(points)
    normals[1:-1, 1:-1] = normalized
    valid = np.zeros_like(valid_depth)
    valid[1:-1, 1:-1] = interior_valid
    return Geometry(points, normals, valid, np.asarray(K, dtype=np.float64))


def correspondences(transform: np.ndarray, source: Geometry, target: Geometry,
                    distance_gate: float, config: ICPConfig) -> dict:
    """Project current points into previous depth; no GT or color input."""
    source_points = source.points[source.valid]
    source_normals = source.normals[source.valid]
    p = transform_points(transform, source_points)
    rotated_normals = source_normals @ transform[:3, :3].T
    counts = {"valid_source_normals": len(p)}
    positive = np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-8)
    ids = np.flatnonzero(positive)
    counts["positive_projectable"] = len(ids)
    height, width = target.valid.shape
    projected = p[ids]
    uv = np.column_stack((target.K[0, 0]*projected[:, 0]/projected[:, 2]+target.K[0, 2],
                          target.K[1, 1]*projected[:, 1]/projected[:, 2]+target.K[1, 2]))
    # Test bounds before integer conversion to avoid overflow for extreme poses.
    inside = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -.5) & (uv[:, 0] < width-.5)
    inside &= (uv[:, 1] >= -.5) & (uv[:, 1] < height-.5)
    ids, uv = ids[inside], uv[inside]
    pixel = np.floor(uv+.5).astype(np.int64)
    counts["projected_inside"] = len(ids)
    target_ok = target.valid[pixel[:, 1], pixel[:, 0]]
    ids, pixel = ids[target_ok], pixel[target_ok]
    counts["valid_target_normals"] = len(ids)
    target_points = target.points[pixel[:, 1], pixel[:, 0]]
    target_normals = target.normals[pixel[:, 1], pixel[:, 0]]
    normal_ok = np.sum(rotated_normals[ids]*target_normals, axis=1) >= math.cos(math.radians(config.normal_angle_degrees))
    counts["normal_consistent"] = int(normal_ok.sum())
    distances = np.linalg.norm(p[ids]-target_points, axis=1)
    geometric_ok = normal_ok & (distances <= distance_gate)
    counts["distance_consistent"] = int(geometric_ok.sum())
    ids, pixel = ids[geometric_ok], pixel[geometric_ok]
    target_points, target_normals = target_points[geometric_ok], target_normals[geometric_ok]
    distances = distances[geometric_ok]
    # Prevent many source pixels from overweighting a single target pixel.
    target_ids = pixel[:, 1]*width+pixel[:, 0]
    order = np.lexsort((distances, target_ids))
    ordered_ids = target_ids[order]
    unique = np.r_[True, np.diff(ordered_ids) != 0] if len(order) else np.zeros(0, dtype=bool)
    keep = order[unique]
    ids, target_points, target_normals = ids[keep], target_points[keep], target_normals[keep]
    counts["unique_correspondences"] = len(ids)
    return {"source": source_points[ids], "transformed": p[ids],
            "target": target_points, "normals": target_normals,
            "counts": counts,
            "ratio": len(ids)/max(len(source_points), 1)}


def residual_and_jacobian(p: np.ndarray, target: np.ndarray,
                          normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.sum(normals*(p-target), axis=1)
    jacobian = np.column_stack((normals, np.cross(p, normals)))
    return residual, jacobian


def huber_objective(residual: np.ndarray, delta: float) -> float:
    absolute = np.abs(residual)
    return float(np.mean(np.where(absolute <= delta, .5*residual**2,
                                  delta*(absolute-.5*delta))))


def normal_equations(residual: np.ndarray, jacobian: np.ndarray,
                     config: ICPConfig) -> tuple[np.ndarray | None, dict]:
    """IRLS weighted least squares with explicit, undamped degeneracy checks.

    Column normalization only changes solver coordinates. The returned delta is
    rescaled to meters/radians. Raw Hessian mixes units; normalized spectrum is
    used for the declared numerical rank/condition rule, not pose covariance.
    """
    weights = np.minimum(1., config.huber_delta_m/np.maximum(np.abs(residual), 1e-15))
    weighted = jacobian*np.sqrt(weights)[:, None]
    rhs = -residual*np.sqrt(weights)
    hessian = weighted.T @ weighted
    raw_eigenvalues = np.maximum(np.linalg.eigvalsh(hessian), 0.)
    column_norms = np.linalg.norm(weighted, axis=0)
    safe_norms = np.where(column_norms > 1e-15, column_norms, 1.)
    normalized = weighted/safe_norms[None]
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(singular > (singular[0]*config.rank_rcond if len(singular) and singular[0] else 1e-15)))
    normalized_eigenvalues = np.maximum(np.linalg.eigvalsh(normalized.T @ normalized), 0.)
    condition = None if len(singular) < 6 or singular[-1] <= 1e-15 else float((singular[0]/singular[-1])**2)
    stats = {"rank": rank,
             "hessian_eigenvalues_ascending": raw_eigenvalues.tolist(),
             "column_normalized_hessian_eigenvalues_ascending": normalized_eigenvalues.tolist(),
             "column_normalized_hessian_condition": condition,
             "jacobian_column_norms": column_norms.tolist(),
             "plane_rmse_m": float(np.sqrt(np.mean(residual**2))),
             "plane_median_absolute_m": float(np.median(np.abs(residual))),
             "robust_objective": huber_objective(residual, config.huber_delta_m)}
    if rank < 6 or condition is None or condition > config.max_normalized_hessian_condition:
        return None, stats
    normalized_delta, _, solved_rank, _ = np.linalg.lstsq(normalized, rhs, rcond=config.rank_rcond)
    if solved_rank != 6 or not np.isfinite(normalized_delta).all():
        return None, stats
    return normalized_delta/safe_norms, stats


def register_pair(source_pyramid: list[Geometry], target_pyramid: list[Geometry],
                  config: ICPConfig) -> dict:
    if len(source_pyramid) != 3 or len(target_pyramid) != 3:
        raise ValueError("Exactly three coarse-to-fine levels required")
    transform = np.eye(4)
    history, levels = [], []
    for level, (source, target) in enumerate(zip(source_pyramid, target_pyramid)):
        stop_reason = "iteration_budget"
        accepted_updates = 0
        for iteration in range(config.iterations[level]):
            pairs = correspondences(transform, source, target, config.distance_gates_m[level], config)
            row = {"level": level, "width": target.valid.shape[1], "height": target.valid.shape[0],
                   "iteration": iteration, "counts": pairs["counts"],
                   "correspondence_ratio": pairs["ratio"], "accepted": False}
            if len(pairs["source"]) < config.min_correspondences or pairs["ratio"] < config.min_correspondence_ratio:
                stop_reason = "insufficient_correspondences"
                row["stop_reason"] = stop_reason
                history.append(row)
                break
            residual, jacobian = residual_and_jacobian(pairs["transformed"], pairs["target"], pairs["normals"])
            delta, spectrum = normal_equations(residual, jacobian, config)
            row.update(spectrum)
            if delta is None:
                stop_reason = "rank_deficient_or_ill_conditioned"
                row["stop_reason"] = stop_reason
                history.append(row)
                break
            row["proposed_delta_translation_m"] = float(np.linalg.norm(delta[:3]))
            row["proposed_delta_rotation_rad"] = float(np.linalg.norm(delta[3:]))
            row["proposed_delta_twist"] = delta.tolist()
            limit = min(1., config.max_iteration_translation_m/max(np.linalg.norm(delta[:3]), 1e-15),
                        math.radians(config.max_iteration_rotation_degrees)/max(np.linalg.norm(delta[3:]), 1e-15))
            delta *= limit
            row["step_limit_factor"] = float(limit)
            if np.linalg.norm(delta[:3]) < config.convergence_translation_m and np.linalg.norm(delta[3:]) < config.convergence_rotation_radians:
                stop_reason = "small_update"
                row.update({"stop_reason": stop_reason, "accepted_delta_twist": [0.]*6})
                history.append(row)
                break
            before = spectrum["robust_objective"]
            for scale in config.line_search_scales:
                trial = se3_exp(delta*scale) @ transform
                trial_points = transform_points(trial, pairs["source"])
                trial_residual = np.sum(pairs["normals"]*(trial_points-pairs["target"]), axis=1)
                cost = huber_objective(trial_residual, config.huber_delta_m)
                # Same correspondences and normals for this local line-search test.
                if math.isfinite(cost) and cost < before:
                    transform = trial
                    row.update({"accepted": True, "line_search_scale": scale,
                                "accepted_delta_twist": (delta*scale).tolist(),
                                "fixed_correspondence_objective_after": cost,
                                "fixed_correspondence_plane_rmse_after_m": float(np.sqrt(np.mean(trial_residual**2)))})
                    accepted_updates += 1
                    break
            if not row["accepted"]:
                stop_reason = "line_search_no_descent"
                row["stop_reason"] = stop_reason
                history.append(row)
                break
            history.append(row)
        levels.append({"level": level, "resolution": [target.valid.shape[1], target.valid.shape[0]],
                       "iterations_recorded": iteration+1, "accepted_updates": accepted_updates,
                       "stop_reason": stop_reason})

    final_pairs = correspondences(transform, source_pyramid[-1], target_pyramid[-1], config.distance_gates_m[-1], config)
    failure_flags = []
    final_stats = {}
    if len(final_pairs["source"]) < config.min_correspondences or final_pairs["ratio"] < config.min_correspondence_ratio:
        failure_flags.append("final_insufficient_correspondences")
    else:
        residual, jacobian = residual_and_jacobian(final_pairs["transformed"], final_pairs["target"], final_pairs["normals"])
        _, final_stats = normal_equations(residual, jacobian, config)
        if final_stats["rank"] < 6:
            failure_flags.append("final_unobservable_6d_pose")
        condition = final_stats["column_normalized_hessian_condition"]
        if condition is None or condition > config.max_normalized_hessian_condition:
            failure_flags.append("final_ill_conditioned_normal_equations")
        if final_stats["plane_rmse_m"] > config.max_final_plane_rmse_m:
            failure_flags.append("final_plane_residual_exceeds_threshold")
    translation = float(np.linalg.norm(transform[:3, 3]))
    rotation = rotation_degrees(transform[:3, :3])
    if translation > config.max_pair_translation_m:
        failure_flags.append("relative_translation_exceeds_threshold")
    if rotation > config.max_pair_rotation_degrees:
        failure_flags.append("relative_rotation_exceeds_threshold")
    rigid_error = float(np.max(np.abs(transform[:3, :3].T @ transform[:3, :3]-np.eye(3))))
    if not np.isfinite(transform).all() or rigid_error > 1e-8 or abs(np.linalg.det(transform[:3, :3])-1) > 1e-8:
        failure_flags.append("nonfinite_or_nonrigid_transform")
    # Failed coarse levels are retained as diagnostics; a valid fine level may recover.
    if levels[-1]["stop_reason"] == "line_search_no_descent":
        failure_flags.append("finest_level_line_search_no_descent")
    return {"accepted": not failure_flags, "failure_flags": failure_flags,
            "candidate_previous_from_current": transform.tolist(),
            "relative_translation_m": translation, "relative_rotation_degrees": rotation,
            "rotation_orthogonality_max_abs": rigid_error,
            "final_counts": final_pairs["counts"], "final_correspondence_ratio": final_pairs["ratio"],
            "final_statistics": final_stats, "levels": levels, "iteration_history": history}


def integrate_relative(previous_c2w: np.ndarray, result: dict) -> np.ndarray:
    # A failed frame remains in the trajectory, with a frozen previous estimate.
    relative = np.asarray(result["candidate_previous_from_current"]) if result["accepted"] else np.eye(4)
    return previous_c2w @ relative


def synthetic_room_depth(width: int, height: int, camera_c2w: np.ndarray,
                         only_front_plane: bool = False) -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[.75*width, 0., width/2-.5],
                  [0., .75*width, height/2-.5], [0., 0., 1.]])
    v, u = np.mgrid[:height, :width]
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)), axis=-1)
    directions = rays @ camera_c2w[:3, :3].T
    origin = camera_c2w[:3, 3]
    # Camera lies inside a box: closest positive ray/plane hit is a visible wall.
    planes = [(np.array([0., 0., 1.]), 2.2)]
    if not only_front_plane:
        planes += [(np.array([1., 0., 0.]), .8), (np.array([-1., 0., 0.]), .7),
                   (np.array([0., 1., 0.]), .55), (np.array([0., -1., 0.]), .65)]
    depths = []
    for normal, offset in planes:
        denominator = directions @ normal
        depth = np.full((height, width), np.inf)
        forward = denominator > 1e-10
        depth[forward] = (offset-origin @ normal)/denominator[forward]
        depth[depth <= .1] = np.inf
        depths.append(depth)
    output = np.min(depths, axis=0)
    return output, K


def self_test() -> dict:
    """Analytical controls, not a measured real-data accuracy claim."""
    config = ICPConfig()
    exponential_error = 0.
    for twist in [np.zeros(6), np.array([.03, -.02, .01, .02, -.03, .01]),
                  np.array([-.3, .2, .4, 1e-10, -2e-10, 3e-10])]:
        algebra = np.zeros((4, 4))
        algebra[:3, :3], algebra[:3, 3] = skew(twist[3:]), twist[:3]
        actual = se3_exp(twist)
        exponential_error = max(exponential_error, float(np.max(np.abs(actual-expm(algebra)))))
        assert np.max(np.abs(actual @ inverse_rigid(actual)-np.eye(4))) < 1e-12
    assert exponential_error < 1e-12

    rng = np.random.default_rng(90210)
    q = rng.normal(size=(80, 3))*.3+np.array([0., 0., 2.])
    target = rng.normal(size=(80, 3))*.3+np.array([0., 0., 2.])
    normals = rng.normal(size=(80, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    transform = se3_exp(np.array([.03, -.04, .02, .02, -.01, .03]))
    p = transform_points(transform, q)
    _, analytical = residual_and_jacobian(p, target, normals)
    numeric = np.zeros_like(analytical)
    for axis in range(6):
        perturbation = np.zeros(6)
        perturbation[axis] = 1e-7
        plus = transform_points(se3_exp(perturbation) @ transform, q)
        minus = transform_points(se3_exp(-perturbation) @ transform, q)
        numeric[:, axis] = np.sum(normals*(plus-minus), axis=1)/(2e-7)
    jacobian_error = float(np.max(np.abs(analytical-numeric)))
    assert jacobian_error < 1e-7

    truth = se3_exp(np.array([.025, -.015, .012, .012, -.018, .009]))
    # Exact corresponding points isolate left-update algebra from raster
    # sampling and finite-difference surface normals used by projective ICP.
    exact_target = transform_points(truth, q)
    exact_estimate = np.eye(4)
    exact_residual_history = []
    for _ in range(8):
        exact_p = transform_points(exact_estimate, q)
        exact_r, exact_j = residual_and_jacobian(exact_p, exact_target, normals)
        exact_delta, exact_stats = normal_equations(exact_r, exact_j, config)
        assert exact_delta is not None and exact_stats["rank"] == 6
        exact_residual_history.append(exact_stats["plane_rmse_m"])
        exact_estimate = se3_exp(exact_delta) @ exact_estimate
    exact_transform_error = float(np.max(np.abs(exact_estimate-truth)))
    assert exact_transform_error < 1e-10
    assert exact_residual_history[-1] < 1e-12

    source, target_pyramid = [], []
    plane_source, plane_target = [], []
    for width, height in LEVELS:
        for camera, destination in [(np.eye(4), target_pyramid), (truth, source)]:
            depth, K = synthetic_room_depth(width, height, camera)
            destination.append(depth_geometry(depth, np.isfinite(depth), K, config))
        # A lateral displacement is unobservable for an infinite fronto-parallel plane.
        lateral = np.eye(4)
        lateral[0, 3] = .05
        for camera, destination in [(np.eye(4), plane_target), (lateral, plane_source)]:
            depth, K = synthetic_room_depth(width, height, camera, only_front_plane=True)
            destination.append(depth_geometry(depth, np.isfinite(depth), K, config))
    registration = register_pair(source, target_pyramid, config)
    assert registration["accepted"], registration["failure_flags"]
    estimate = np.asarray(registration["candidate_previous_from_current"])
    error = inverse_rigid(truth) @ estimate
    translation_error = float(np.linalg.norm(error[:3, 3]))
    rotation_error = rotation_degrees(error[:3, :3])
    # Room corners can have mixed central-difference normals even when their
    # depth jump is below the declared edge threshold. This separate raster
    # test allows sub-millimeter/sub-tenth-degree error, unlike exact algebra.
    assert translation_error < .0005 and rotation_error < .05, (translation_error, rotation_error)
    assert estimate[0, 3] > 0  # Source camera moved right: current->previous translation is positive.
    assert registration["final_statistics"]["rank"] == 6

    degenerate = register_pair(plane_source, plane_target, config)
    assert not degenerate["accepted"]
    assert degenerate["final_statistics"]["rank"] <= 3
    assert "final_unobservable_6d_pose" in degenerate["failure_flags"]
    sequence = [np.eye(4)]
    sequence.append(integrate_relative(sequence[-1], registration))
    sequence.append(integrate_relative(sequence[-1], degenerate))
    assert len(sequence) == 3 and np.array_equal(sequence[1], sequence[2])
    world_point = np.array([[.1, .2, 2.]])
    previous_point = transform_points(inverse_rigid(sequence[0]), world_point)
    current_point = transform_points(inverse_rigid(sequence[1]), world_point)
    assert np.max(np.abs(transform_points(estimate, current_point)-previous_point)) < 1e-12

    edge_depth = np.full((30, 40), 2.)
    edge_depth[:, 20:] = 4.
    edge_depth[10, 10] = 0.
    _, edge_K = synthetic_room_depth(40, 30, np.eye(4))
    geometry = depth_geometry(edge_depth, edge_depth > 0, edge_K, config)
    assert not geometry.valid[:, 19:21].any()
    assert not geometry.valid[10, 10] and not geometry.valid[10, 9] and not geometry.valid[9, 10]
    assert np.isfinite(geometry.normals).all()
    blank = [depth_geometry(np.zeros_like(level.valid, dtype=float), np.zeros_like(level.valid), level.K, config)
             for level in target_pyramid]
    no_data = register_pair(blank, target_pyramid, config)
    assert not no_data["accepted"] and "final_insufficient_correspondences" in no_data["failure_flags"]
    return {"status": "passed", "configuration": asdict(config), "levels": LEVELS,
            "se3_exponential_vs_scipy_expm_max_abs": exponential_error,
            "left_perturbation_jacobian_finite_difference_max_abs": jacobian_error,
            "exact_correspondence_rigid_motion": {
                "transform_max_abs_error": exact_transform_error,
                "plane_rmse_history_m": exact_residual_history,
                "purpose": "Isolate 6D left-update solve/sign from raster association and normal estimation"},
            "synthetic_room": {"truth_previous_from_current": truth.tolist(),
                               "estimated_previous_from_current": estimate.tolist(),
                               "translation_error_m": translation_error,
                               "rotation_error_degrees": rotation_error,
                               "assertion_translation_tolerance_m": .0005,
                               "assertion_rotation_tolerance_degrees": .05,
                               "discretization_note": "Nearest-pixel projective matching and central-difference normals at wall intersections are not exact analytic correspondences",
                               "final_normal_equations": registration["final_statistics"],
                               "final_correspondences": registration["final_counts"],
                               "levels": registration["levels"]},
            "single_plane": {"rank": degenerate["final_statistics"]["rank"],
                             "failure_flags": degenerate["failure_flags"],
                             "lateral_translation_not_claimed_observable": True},
            "source_to_target_sign_and_c2w_composition": "passed",
            "failed_frame_retained_with_previous_pose": True,
            "depth_hole_and_edge_normal_rejection": True,
            "empty_depth_rejected": True,
            "scope": "Synthetic metric-room pose recovery, SE3/Jacobian algebra and failure controls only; no real TUM accuracy measured"}


def source_hashes() -> dict:
    return {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ["track_rgbd_icp.py", "tum_rgbd.py", "audit_tum_trajectory.py"]}


def write_trajectory(path: Path, timestamps: list[float], poses: np.ndarray) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw; c2w; meters\n")
        for timestamp, pose in zip(timestamps, poses):
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            stream.write(" ".join(f"{number:.9f}" for number in [timestamp, *pose[:3, 3], *quaternion])+"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--clip", choices=CLIPS, default="short20")
    parser.add_argument("--run", help="Required new output name for real data; never overwritten")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/tum/rgbd_dataset_freiburg1_desk")
    args = parser.parse_args()
    config = ICPConfig()
    if args.self_test:
        if args.run:
            parser.error("--self-test cannot be combined with a real-data --run")
        checks = self_test()
        checks.update({"source_sha256": source_hashes(), "implementation": "NumPy/SciPy CPU-only projective point-to-plane ICP",
                       "real_data_runs_in_this_check": 0,
                       "runtime": {"python": sys.version.split()[0],
                                   "numpy": importlib.metadata.version("numpy"),
                                   "scipy": importlib.metadata.version("scipy"),
                                   "device": "CPU"}})
        destination = ROOT / "manifests/track-rgbd-icp-cpu-checks.json"
        write_json(destination, checks)
        print(json.dumps({"status": checks["status"], "manifest": destination.name,
                          "synthetic_translation_error_m": checks["synthetic_room"]["translation_error_m"],
                          "synthetic_rotation_error_degrees": checks["synthetic_room"]["rotation_error_degrees"],
                          "plane_rank": checks["single_plane"]["rank"],
                          "source_sha256": checks["source_sha256"]}, indent=2))
        return
    if not args.run or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("A new --run containing letters/numbers/-/_ is required for a real-data experiment")
    output = ROOT / "outputs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Keep the same association and PNG/K convention as the GS teaching prototype.
    frames, association = associate_frames(args.dataset, .02, .02)
    start, count = CLIPS[args.clip]
    selected = frames[start:start+count]
    if len(selected) != count or [frame.frame_id for frame in selected] != list(range(start, start+count)):
        raise ValueError("The requested fixed clip is unavailable")
    configuration = {"clip": args.clip, "start_frame": start, "frames": count,
                     "dataset": args.dataset.resolve().as_posix(), "run": args.run,
                     "pyramid_sizes": LEVELS, "icp": asdict(config),
                     "pair_initialization": "identity current-to-previous transform for every pair",
                     "normal_orientation": "toward camera; central differences with valid-depth/edge rejection",
                     "correspondence": "nearest pixel projective association; 3D distance and normal rejection; one-to-one target pixels",
                     "hessian_definition": "Huber-IRLS J.T W J; raw columns use meters/radians; normalized-column spectrum is a numerical condition diagnostic, not pose covariance",
                     "line_search": "Huber objective evaluated on fixed correspondences for each proposed left increment",
                     "failure_policy": "retain frame and previous c2w; retain rejected candidate and diagnostics",
                     "source_sha256": source_hashes()}
    write_json(output / "configuration.json", configuration)
    pyramid_data = []
    for width, height in LEVELS:
        # The loader reads the same raw PNG with nearest depth sampling at each scale.
        _, depths, masks = load_frames(args.dataset, selected, width, height, .2, 5.)
        pyramid_data.append((depths, masks, scaled_intrinsics(width, height)))

    def geometry_for_frame(index: int) -> list[Geometry]:
        return [depth_geometry(depths[index], masks[index], K, config) for depths, masks, K in pyramid_data]

    estimated = [np.eye(4)]
    frame_records = [{"local_index": 0, "frame_id": selected[0].frame_id,
                      "accepted": True, "role": "identity world gauge", "failure_flags": [],
                      "estimated_c2w": np.eye(4).tolist()}]
    target_pyramid = geometry_for_frame(0)
    loop_started = time.perf_counter()
    csv_columns = ["local_index", "frame_id", "level", "iteration", "correspondences",
                   "rank", "plane_rmse_m", "robust_objective", "accepted", "stop_reason",
                   "delta_translation_m", "delta_rotation_rad", "normalized_hessian_condition"]
    with (output / "iterations.jsonl").open("x", encoding="utf-8") as detailed, \
         (output / "loss.csv").open("x", newline="", encoding="utf-8") as loss_file:
        loss_writer = csv.DictWriter(loss_file, fieldnames=csv_columns)
        loss_writer.writeheader()
        for index in range(1, count):
            frame_started = time.perf_counter()
            source_pyramid = geometry_for_frame(index)
            result = register_pair(source_pyramid, target_pyramid, config)
            history = result.pop("iteration_history")
            for row in history:
                row.update({"local_index": index, "frame_id": selected[index].frame_id})
                detailed.write(json.dumps(row, allow_nan=False)+"\n")
                accepted_delta = np.asarray(row.get("accepted_delta_twist", [0.]*6))
                loss_writer.writerow({"local_index": index, "frame_id": selected[index].frame_id,
                                      "level": row["level"], "iteration": row["iteration"],
                                      "correspondences": row["counts"]["unique_correspondences"],
                                      "rank": row.get("rank"), "plane_rmse_m": row.get("plane_rmse_m"),
                                      "robust_objective": row.get("robust_objective"), "accepted": row["accepted"],
                                      "stop_reason": row.get("stop_reason"),
                                      "delta_translation_m": np.linalg.norm(accepted_delta[:3]),
                                      "delta_rotation_rad": np.linalg.norm(accepted_delta[3:]),
                                      "normalized_hessian_condition": row.get("column_normalized_hessian_condition")})
            pose = integrate_relative(estimated[-1], result)
            estimated.append(pose)
            result.update({"local_index": index, "frame_id": selected[index].frame_id,
                           "rgb_timestamp": selected[index].rgb_timestamp,
                           "estimated_c2w": pose.tolist(), "cpu_seconds": time.perf_counter()-frame_started})
            frame_records.append(result)
            target_pyramid = source_pyramid
            detailed.flush()
            loss_file.flush()
            print(f'frame={index+1}/{count} accepted={result["accepted"]} '
                  f'corr={result["final_counts"]["unique_correspondences"]} '
                  f'failures={result["failure_flags"]}', flush=True)
    cpu_loop_seconds = time.perf_counter()-loop_started
    # GT pose values are first consumed here, after every estimated pose is final.
    truth = np.stack([frame.c2w for frame in selected])
    anchored_truth = inverse_rigid(truth[0])[None] @ truth
    estimated_array = np.stack(estimated)
    evaluation = audit_pose_arrays(estimated_array, anchored_truth)
    timestamps = [frame.rgb_timestamp for frame in selected]
    write_trajectory(output / "trajectory_estimated.txt", timestamps, estimated_array)
    write_trajectory(output / "trajectory_groundtruth_anchored.txt", timestamps, anchored_truth)
    write_json(output / "trajectory_evaluation.json", evaluation)
    failures = sum(bool(row["failure_flags"]) for row in frame_records)
    metrics = {"experiment_id": args.run, "status": "completed_with_tracking_failures" if failures else "completed",
               "implementation": "CPU previous-frame projective point-to-plane ICP control",
               "configuration": configuration, "association": association,
               "frames_processed": len(estimated), "failure_frame_count": failures,
               "ground_truth_usage": "GT timestamps used by shared association filter; GT poses exclusively for final evaluation, never initialization, correspondences, loss, normal equations or candidate selection",
               "frame_metrics": frame_records, "trajectory": evaluation,
               "cpu_loop_seconds": cpu_loop_seconds,
               "total_seconds_before_metrics_export": time.perf_counter()-started,
               "limitations": ["Depth-only frame-to-frame geometry control; no map, loop closure, relocalization or BA",
                               "Projective ICP is local and requires overlapping, sufficiently non-planar depth",
                               "Failures remain in all metrics with previous pose propagated; failed motion is not recovered later",
                               "No GT pose values in optimization, but association uses GT timestamps",
                               "Same TUM loader assumes registered RGB/depth and does not interpolate GT, undistort or compensate temporal offset",
                               "Rank/condition/residual flags are heuristics, not proof of accurate tracking",
                               "Final audit computes trajectory metrics; it does not independently prove the ICP algorithm is correct",
                               "No relative accuracy or speed claim against the GS prototype is implied before matched measurements"],
               "artifacts": ["configuration.json", "metrics.json", "iterations.jsonl", "loss.csv",
                             "trajectory_estimated.txt", "trajectory_groundtruth_anchored.txt", "trajectory_evaluation.json"]}
    write_json(output / "metrics.json", metrics)
    print(json.dumps({"status": metrics["status"], "frames": len(estimated), "failed_frames": failures,
                      "raw_translation_rmse_m": evaluation["raw_anchored_translation_rmse_m"],
                      "se3_ate_rmse_m": evaluation["se3_aligned_ate_translation_rmse_m"],
                      "rpe_translation_rmse_m": evaluation["adjacent_rpe_translation_rmse_m"],
                      "cpu_loop_seconds": cpu_loop_seconds}, indent=2))


if __name__ == "__main__":
    main()
