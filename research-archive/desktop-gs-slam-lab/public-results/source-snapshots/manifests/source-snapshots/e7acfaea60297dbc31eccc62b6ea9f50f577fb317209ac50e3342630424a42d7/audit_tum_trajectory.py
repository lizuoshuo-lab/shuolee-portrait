"""Independently audit paired TUM c2w trajectories with NumPy only.

Each row is timestamp tx ty tz qx qy qz qw, with translation in meters.
No nearest-time association, interpolation, scale fitting, or frame removal is
performed. Raw anchored error requires both input files to already share their
first-camera gauge (use trajectory_groundtruth_anchored.txt for the lab online
prototype). SE3 ATE aligns one global rigid transform, without scale.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def quaternion_xyzw_to_rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def parse_trajectory_text(text, source="<memory>", quaternion_norm_tolerance=1e-3):
    """Parse exact Decimal timestamps and validated scalar-last c2w quaternions."""
    timestamps, poses, norm_errors = [], [], []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{source}:{line_number}: expected exactly eight TUM columns")
        try:
            timestamp = Decimal(fields[0])
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{source}:{line_number}: invalid numeric TUM row") from exc
        if not timestamp.is_finite() or not np.isfinite(values).all():
            raise ValueError(f"{source}:{line_number}: timestamp/pose must be finite")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError(f"{source}:{line_number}: timestamps must be strictly increasing")
        quaternion = values[3:]
        norm = np.linalg.norm(quaternion)
        norm_error = abs(float(norm)-1.0)
        if norm < 1e-12 or norm_error > quaternion_norm_tolerance:
            raise ValueError(f"{source}:{line_number}: xyzw quaternion norm {norm} exceeds tolerance")
        quaternion = quaternion / norm
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quaternion_xyzw_to_rotation(quaternion)
        pose[:3, 3] = values[:3]
        if abs(np.linalg.det(pose[:3, :3])-1.0) > 1e-12:
            raise ValueError(f"{source}:{line_number}: rotation is not proper SO3")
        timestamps.append(timestamp)
        poses.append(pose)
        norm_errors.append(norm_error)
    if len(poses) < 2:
        raise ValueError(f"{source}: at least two poses are required for adjacent-frame RPE")
    return {"timestamps": timestamps, "poses": np.stack(poses),
            "quaternion_norm_max_error_before_normalization": max(norm_errors),
            "row_count": len(poses)}


def validate_time_pairing(estimated, ground_truth, timestamp_tolerance=Decimal("0")):
    if estimated["row_count"] != ground_truth["row_count"]:
        raise ValueError("Trajectories have different row counts; frames cannot be silently dropped")
    if not timestamp_tolerance.is_finite() or timestamp_tolerance < 0:
        raise ValueError("Timestamp tolerance must be finite and nonnegative")
    differences = [abs(left-right) for left, right in
                   zip(estimated["timestamps"], ground_truth["timestamps"])]
    maximum = max(differences)
    if maximum > timestamp_tolerance:
        index = next(i for i, value in enumerate(differences) if value > timestamp_tolerance)
        raise ValueError(f"Timestamp mismatch at paired row {index}: "
                         f"{estimated['timestamps'][index]} versus {ground_truth['timestamps'][index]}")
    return {"pairing": "strict rowwise Decimal timestamps; no association/interpolation",
            "timestamp_tolerance_seconds": str(timestamp_tolerance),
            "maximum_timestamp_difference_seconds": str(maximum),
            "frame_count": estimated["row_count"],
            "adjacent_interval_count": estimated["row_count"]-1}


def inverse_rigid(transform):
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def rotation_angle_degrees(rotation):
    cosine = np.clip((np.trace(rotation)-1.0)/2.0, -1.0, 1.0)
    skew = np.array([rotation[2, 1]-rotation[1, 2],
                     rotation[0, 2]-rotation[2, 0],
                     rotation[1, 0]-rotation[0, 1]], dtype=np.float64)
    sine = np.linalg.norm(skew)/2.0
    return math.degrees(math.atan2(float(sine), float(cosine)))


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64)**2)))


def se3_alignment_from_positions(estimated_positions, ground_truth_positions):
    """Kabsch alignment (rotation plus translation only), explicitly no scale."""
    estimated_center = estimated_positions.mean(axis=0)
    truth_center = ground_truth_positions.mean(axis=0)
    centered_estimated = estimated_positions-estimated_center
    centered_truth = ground_truth_positions-truth_center
    covariance = centered_estimated.T @ centered_truth
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = -1.0 if np.linalg.det(right_transposed.T @ left.T) < 0 else 1.0
    rotation = right_transposed.T @ correction @ left.T
    translation = truth_center-rotation @ estimated_center
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    return transformation, singular_values


def audit_pose_arrays(estimated_poses, ground_truth_poses):
    estimated = np.asarray(estimated_poses, dtype=np.float64)
    truth = np.asarray(ground_truth_poses, dtype=np.float64)
    if estimated.shape != truth.shape or estimated.ndim != 3 or estimated.shape[1:] != (4, 4):
        raise ValueError("Matching [N,4,4] c2w pose arrays required")
    if len(estimated) < 2 or not np.isfinite(estimated).all() or not np.isfinite(truth).all():
        raise ValueError("Finite trajectories with at least two poses required")
    estimated_positions, truth_positions = estimated[:, :3, 3], truth[:, :3, 3]
    raw_translation = np.linalg.norm(estimated_positions-truth_positions, axis=1)
    raw_rotation = [rotation_angle_degrees(gt[:3, :3].T @ est[:3, :3])
                    for est, gt in zip(estimated, truth)]
    alignment, singular_values = se3_alignment_from_positions(estimated_positions, truth_positions)
    aligned_positions = estimated_positions @ alignment[:3, :3].T + alignment[:3, 3]
    aligned_translation = np.linalg.norm(aligned_positions-truth_positions, axis=1)
    rpe_translation, rpe_rotation = [], []
    for index in range(len(estimated)-1):
        relative_estimated = inverse_rigid(estimated[index]) @ estimated[index+1]
        relative_truth = inverse_rigid(truth[index]) @ truth[index+1]
        error = inverse_rigid(relative_truth) @ relative_estimated
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        rpe_rotation.append(rotation_angle_degrees(error[:3, :3]))
    first_translation = float(raw_translation[0])
    first_rotation = float(raw_rotation[0])
    alignment_rank = int(np.linalg.matrix_rank(estimated_positions-estimated_positions.mean(axis=0)))
    return {
        "raw_anchored_translation_rmse_m": rmse(raw_translation),
        "raw_anchored_translation_mean_m": float(raw_translation.mean()),
        "raw_anchored_translation_max_m": float(raw_translation.max()),
        "raw_rotation_rmse_deg": rmse(raw_rotation),
        "se3_aligned_ate_translation_rmse_m": rmse(aligned_translation),
        "se3_aligned_ate_translation_mean_m": float(aligned_translation.mean()),
        "se3_aligned_ate_translation_max_m": float(aligned_translation.max()),
        "adjacent_rpe_translation_rmse_m": rmse(rpe_translation),
        "adjacent_rpe_rotation_rmse_deg": rmse(rpe_rotation),
        "first_pose_translation_difference_m": first_translation,
        "first_pose_rotation_difference_deg": first_rotation,
        "first_pose_anchor_matches": first_translation < 1e-6 and first_rotation < 1e-5,
        "raw_definition": "direct camera-center difference in provided c2w gauge, without fitting; meaningful as anchored error only when input gauges are shared",
        "ate_alignment": "one Kabsch SE3 rotation+translation from paired camera centers; scale fixed at 1",
        "rpe_definition": "inv(inv(GT_i)@GT_i+1) @ (inv(EST_i)@EST_i+1), adjacent paired frames",
        "rpe_delta_frames": 1,
        "alignment_scale": 1.0,
        "alignment_estimated_to_ground_truth": alignment.tolist(),
        "centered_estimated_positions_rank": alignment_rank,
        "alignment_singular_values": singular_values.tolist(),
        "alignment_degeneracy_note": "collinear or stationary camera-center paths do not uniquely determine the fitted rotation" if alignment_rank < 2 else None,
        "per_frame_raw_translation_error_m": raw_translation.tolist(),
        "per_frame_se3_aligned_ate_translation_error_m": aligned_translation.tolist(),
        "per_interval_rpe_translation_error_m": rpe_translation,
        "per_interval_rpe_rotation_error_deg": rpe_rotation,
    }


def self_test():
    """Physics/metric invariants plus expected-error and strict-parser controls."""
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [2., 1., 1.]])
    truth = np.tile(np.eye(4), (4, 1, 1))
    truth[:, :3, 3] = positions
    perfect = audit_pose_arrays(truth, truth)
    assert perfect["raw_anchored_translation_rmse_m"] < 1e-12
    assert perfect["se3_aligned_ate_translation_rmse_m"] < 1e-12
    assert perfect["adjacent_rpe_translation_rmse_m"] < 1e-12
    transform = np.eye(4)
    theta = math.radians(37)
    transform[:3, :3] = [[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
    transform[:3, 3] = [3., -2., .7]
    transformed = audit_pose_arrays(transform[None] @ truth, truth)
    assert transformed["raw_anchored_translation_rmse_m"] > 1
    assert transformed["se3_aligned_ate_translation_rmse_m"] < 1e-12
    assert transformed["adjacent_rpe_translation_rmse_m"] < 1e-12
    assert transformed["adjacent_rpe_rotation_rmse_deg"] < 1e-10
    scaled_poses = truth.copy()
    scaled_poses[:, :3, 3] *= 2
    scaled = audit_pose_arrays(scaled_poses, truth)
    expected_scale_error = rmse(np.linalg.norm(positions-positions.mean(axis=0), axis=1))
    assert abs(scaled["se3_aligned_ate_translation_rmse_m"]-expected_scale_error) < 1e-12
    assert scaled["se3_aligned_ate_translation_rmse_m"] > .9
    translation_truth = np.tile(np.eye(4), (3, 1, 1))
    translation_truth[:, 0, 3] = [0., 1., 2.]
    translation_estimated = translation_truth.copy()
    translation_estimated[:, 0, 3] = [0., 1.25, 2.5]
    translation = audit_pose_arrays(translation_estimated, translation_truth)
    assert abs(translation["adjacent_rpe_translation_rmse_m"]-.25) < 1e-12
    assert translation["adjacent_rpe_rotation_rmse_deg"] < 1e-12
    rotation_truth = np.tile(np.eye(4), (3, 1, 1))
    rotation_estimated = rotation_truth.copy()
    for index, angle in enumerate((0., 10., 20.)):
        half_angle = math.radians(angle)/2
        rotation_estimated[index, :3, :3] = quaternion_xyzw_to_rotation([0, 0, math.sin(half_angle), math.cos(half_angle)])
    rotation = audit_pose_arrays(rotation_estimated, rotation_truth)
    assert abs(rotation["adjacent_rpe_rotation_rmse_deg"]-10) < 1e-12
    assert rotation["adjacent_rpe_translation_rmse_m"] < 1e-12
    ninety = quaternion_xyzw_to_rotation([0, 0, math.sqrt(.5), math.sqrt(.5)])
    assert np.max(np.abs(ninety @ np.array([1., 0., 0.])-np.array([0., 1., 0.]))) < 1e-12
    text = "1305031453.000000001 0 0 0 0 0 0 1\n1305031453.000000002 1 0 0 0 0 0 -1\n"
    parsed = parse_trajectory_text(text)
    assert np.max(np.abs(parsed["poses"][:, :3, :3]-np.eye(3))) < 1e-12
    validate_time_pairing(parsed, parsed)
    # Decimal keeps nanosecond differences at an epoch where float64 is too coarse.
    modified = parse_trajectory_text(text.replace("1305031453.000000002", "1305031453.000000003"))
    try:
        validate_time_pairing(parsed, modified)
    except ValueError:
        pass
    else:
        raise AssertionError("One-nanosecond time mismatch was not rejected")
    malformed_cases = {
        "duplicate_timestamp": text.replace("1305031453.000000002", "1305031453.000000001"),
        "nonunit_quaternion": text.replace("0 0 0 -1", "0 0 0 2"),
        "nonfinite_position": text.replace("1 0 0 0 0 0 -1", "nan 0 0 0 0 0 -1"),
    }
    for name, malformed in malformed_cases.items():
        try:
            parse_trajectory_text(malformed)
        except ValueError:
            continue
        raise AssertionError(f"Malformed control not rejected: {name}")
    return {"status": "passed", "perfect_trajectory": True,
            "global_se3_transform_invariance": True,
            "scale_2_not_removed_by_se3": True,
            "scale_2_expected_ate_rmse_m": expected_scale_error,
            "pure_translation_expected_rpe_rmse_m": .25,
            "pure_rotation_expected_rpe_rmse_deg": 10.,
            "scalar_last_quaternion_axis_and_sign_controls": True,
            "exact_nanosecond_timestamp_pairing": True,
            "malformed_timestamp_quaternion_and_finite_controls": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimated", nargs="?", type=Path)
    parser.add_argument("ground_truth", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timestamp-tolerance", default="0",
                        help="Decimal seconds; default exact rowwise timestamp equality")
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=1e-3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.quaternion_norm_tolerance < 0 or not math.isfinite(args.quaternion_norm_tolerance):
        parser.error("Quaternion norm tolerance must be finite and nonnegative")
    if args.self_test:
        result = {"independent_metric_self_tests": self_test()}
    else:
        result = {}
    if args.estimated is not None or args.ground_truth is not None:
        if args.estimated is None or args.ground_truth is None:
            parser.error("Both estimated and ground-truth trajectory files are required")
        try:
            tolerance = Decimal(args.timestamp_tolerance)
        except InvalidOperation:
            parser.error("Timestamp tolerance must be a decimal number")
        estimated_bytes = args.estimated.read_bytes()
        truth_bytes = args.ground_truth.read_bytes()
        estimated = parse_trajectory_text(estimated_bytes.decode("utf-8-sig"), str(args.estimated), args.quaternion_norm_tolerance)
        truth = parse_trajectory_text(truth_bytes.decode("utf-8-sig"), str(args.ground_truth), args.quaternion_norm_tolerance)
        pairing = validate_time_pairing(estimated, truth, tolerance)
        result.update({"status": "completed", "implementation": "independent NumPy-only TUM trajectory audit",
                       "camera_convention": "c2w; translation meters; quaternion scalar-last xyzw",
                       "estimated_input": str(args.estimated), "ground_truth_input": str(args.ground_truth),
                       "estimated_sha256": hashlib.sha256(estimated_bytes).hexdigest(),
                       "ground_truth_sha256": hashlib.sha256(truth_bytes).hexdigest(),
                       "time_pairing": pairing, "quaternion_norm_tolerance": args.quaternion_norm_tolerance,
                       "estimated_quaternion_norm_max_error": estimated["quaternion_norm_max_error_before_normalization"],
                       "ground_truth_quaternion_norm_max_error": truth["quaternion_norm_max_error_before_normalization"],
                       "metrics": audit_pose_arrays(estimated["poses"], truth["poses"])})
    elif not args.self_test:
        parser.error("Provide two trajectory files or use --self-test")
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # A fresh audit never overwrites an earlier result silently.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized+"\n")
    print(serialized)


if __name__ == "__main__":
    main()
