"""NTU VIRAL position-only evaluation, with a rotating IMU-to-prism lever arm.

The Leica reference has no orientation truth. World-position increment error
here is intentionally not called full SE3 RPE. No estimator is executed.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

LEVER = np.array([-.293656, -.012288, -.273095])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_estimate(path, point_frame):
    times, positions, quaternions = [], [], []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Expected TUM: seconds x y z qx qy qz qw")
        times.append(int(Decimal(fields[0]) * Decimal(1_000_000_000)))
        positions.append([float(x) for x in fields[1:4]])
        quaternions.append([float(x) for x in fields[4:8]])
    t = np.asarray(times, dtype=np.int64)
    p, q = np.asarray(positions), np.asarray(quaternions)
    if t.size < 3 or not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Estimate must contain finite, strictly increasing poses")
    if np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 1e-3):
        raise ValueError("Estimate quaternion is not normalized within 1e-3")
    if point_frame == "imu":
        p = p + Rotation.from_quat(q).apply(np.broadcast_to(LEVER, p.shape))
    return t, p


def read_gt(path, clock):
    times, positions = [], []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            times.append(int(row["field.header.stamp" if clock == "header" else "%time"]))
            positions.append([float(row["field.pose.position." + axis]) for axis in "xyz"])
    t, p = np.asarray(times, dtype=np.int64), np.asarray(positions)
    if t.size < 3 or np.any(np.diff(t) <= 0) or not np.isfinite(p).all():
        raise ValueError("GT must have finite, strictly increasing position samples")
    return t, p


def match_positions(query, gt_t, gt_p, max_gap_ns):
    insertion = np.searchsorted(gt_t, query)
    indices, points = [], []
    rejected = {"outside_reference_span": 0, "reference_gap_exceeded": 0}
    for i, (t, k) in enumerate(zip(query, insertion)):
        if k < len(gt_t) and gt_t[k] == t:
            indices.append(i); points.append(gt_p[k]); continue
        if k == 0 or k == len(gt_t):
            rejected["outside_reference_span"] += 1; continue
        width = int(gt_t[k]) - int(gt_t[k-1])
        if width > max_gap_ns:
            rejected["reference_gap_exceeded"] += 1; continue
        weight = (int(t) - int(gt_t[k-1])) / width
        indices.append(i); points.append((1-weight)*gt_p[k-1] + weight*gt_p[k])
    return np.asarray(indices, dtype=int), np.asarray(points), rejected


def fit_rigid(source, target):
    if len(source) < 3:
        raise ValueError("Insufficient matched points")
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    if np.count_nonzero(singular > singular[0] * 1e-10) < 2:
        raise ValueError("Position-only alignment rotation is underconstrained by the path")
    guard = np.eye(3); guard[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ guard @ u.T
    translation = target.mean(0) - r @ source.mean(0)
    return r, translation


def evaluate(est_t, est_p, gt_t, gt_p, max_gap_ns, max_increment_ns):
    indices, target, rejected = match_positions(est_t, gt_t, gt_p, max_gap_ns)
    r, translation = fit_rigid(est_p[indices], target)
    aligned = est_p[indices] @ r.T + translation
    error = np.linalg.norm(aligned - target, axis=1)
    adjacent = (np.diff(indices) == 1) & (np.diff(est_t[indices]) <= max_increment_ns)
    # Both ends must be matched consecutive estimates; no bridge across missing truth.
    adjacent &= np.diff(est_t[indices]) <= max_gap_ns
    increment = np.linalg.norm(np.diff(aligned, axis=0)[adjacent] - np.diff(target, axis=0)[adjacent], axis=1)
    return {"estimate_rows": len(est_t), "matched_rows": len(indices), "rejected_rows": rejected,
            "matched_fraction": len(indices)/len(est_t), "GT_rows": len(gt_t),
            "GT_max_interval_s": int(np.diff(gt_t).max())/1e9,
            "scale": 1.0, "alignment": "single posthoc SE3 fit to matched positions; no scale optimization",
            "alignment_R": r.tolist(), "alignment_translation_m": translation.tolist(),
            "aligned_position_APE_RMSE_m": float(np.sqrt(np.mean(error**2))),
            "aligned_position_APE_median_m": float(np.median(error)),
            "aligned_position_APE_max_m": float(error.max()),
            "last_matched_aligned_position_error_m": float(error[-1]),
            "position_increment_error_pairs": int(adjacent.sum()),
            "position_increment_error_RMSE_m": float(np.sqrt(np.mean(increment**2))) if increment.size else None,
            "orientation_GT_available": False, "rotation_error_computed": False,
            "full_SE3_RPE_computed": False,
            "max_GT_interpolation_gap_s": max_gap_ns/1e9,
            "max_estimate_increment_s": max_increment_ns/1e9}, indices, target, aligned, error


def controls(output):
    # Closed-form rigid relation, rotating lever arm, and a known absent interval.
    t = np.arange(12, dtype=np.int64) * 100_000_000 + 10_000_000_000
    p = np.column_stack((np.linspace(0, 1, 12), np.sin(np.arange(12)*.7), np.cos(np.arange(12)*.4)))
    r = Rotation.from_euler("xyz", [17, -11, 23], degrees=True).as_matrix()
    truth = p @ r.T + [2, -3, .5]
    report, _, _, _, _ = evaluate(t, p, t, truth, 200_000_000, 200_000_000)
    orientations = Rotation.from_euler("z", np.linspace(0, 90, 12), degrees=True)
    prism = p + orientations.apply(np.broadcast_to(LEVER, p.shape))
    naive = p + LEVER
    short_gt_t = np.array([0, 10, 1000, 1010], dtype=np.int64)
    indices, _, rejected = match_positions(np.array([-1, 5, 500, 1005, 1011]), short_gt_t,
                                           np.column_stack((short_gt_t, short_gt_t*0, short_gt_t*0)), 20)
    drifted = p.copy(); drifted[:, 0] += np.linspace(0, .2, len(p))**2
    drift, _, _, _, _ = evaluate(t, drifted, t, truth, 200_000_000, 200_000_000)
    gates = {"known_rigid_alignment_error_below_1e_12": report["aligned_position_APE_RMSE_m"] < 1e-12,
             "scale_fixed_one": report["scale"] == 1,
             "known_rigid_increment_error_below_1e_12": report["position_increment_error_RMSE_m"] < 1e-12,
             "rotating_lever_arm_is_not_fixed_world_offset": np.max(np.linalg.norm(prism-naive, axis=1)) > .2,
             "GT_gap_and_outside_samples_excluded": indices.tolist() == [1, 3] and rejected == {"outside_reference_span": 2, "reference_gap_exceeded": 1},
             "nonrigid_drift_not_removed_by_rigid_fit": drift["aligned_position_APE_RMSE_m"] > .005,
             "no_orientation_or_full_pose_RPE": not report["rotation_error_computed"] and not report["full_SE3_RPE_computed"]}
    gates = {name: bool(value) for name, value in gates.items()}
    result = {"status": "passed_synthetic_position_evaluator_controls" if all(gates.values()) else "failed_synthetic_controls",
              "source_sha256": sha(__file__), "gates": gates, "rigid_case": report, "known_drift_case": drift,
              "scope": "Constructed numerical controls, no FAST-LIVO2 run or real sensor accuracy measured."}
    (output/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if not all(gates.values()):
        raise AssertionError(gates)
    print(json.dumps({"status": result["status"], "gates": gates}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--point-frame", choices=("imu", "prism"))
    parser.add_argument("--provenance", choices=("local_run", "author_saved"))
    parser.add_argument("--gt-clock", choices=("header", "record"), default="header")
    parser.add_argument("--max-gt-gap", type=float, default=.2)
    parser.add_argument("--max-increment", type=float, default=.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-controls", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    if args.synthetic_controls:
        controls(args.output); return
    if not all((args.estimate, args.gt, args.point_frame, args.provenance)):
        raise ValueError("Explicit trajectory, GT, point frame and provenance are required")
    if args.max_gt_gap <= 0 or args.max_increment <= 0:
        raise ValueError("Time thresholds must be positive")
    est_t, est_p = read_estimate(args.estimate, args.point_frame)
    gt_t, gt_p = read_gt(args.gt, args.gt_clock)
    report, indices, target, aligned, error = evaluate(est_t, est_p, gt_t, gt_p,
                    round(args.max_gt_gap*1e9), round(args.max_increment*1e9))
    report.update({"status": "position_only_saved_trajectory_reanalysis", "provenance": args.provenance,
        "estimate_sha256": sha(args.estimate), "GT_sha256": sha(args.gt), "source_sha256": sha(__file__),
        "GT_clock_field": "field.header.stamp" if args.gt_clock == "header" else "%time",
        "input_estimate_point_frame": args.point_frame, "rotating_lever_arm_applied": args.point_frame == "imu",
        "lever_arm_IMU_to_prism_m": LEVER.tolist(),
        "scope": "Saved trajectory position analysis. Author-saved results are not local FAST-LIVO2 replay; GT has no orientation; aligned world-position increments are not full SE3 RPE."})
    rows = np.column_stack((indices, (est_t[indices]-est_t[0])/1e9, target, aligned, error))
    np.savetxt(args.output/"matched-position-errors.csv", rows, delimiter=",", fmt="%.12g",
               header="estimate_index,relative_time_s,gt_x,gt_y,gt_z,aligned_x,aligned_y,aligned_z,error_m", comments="")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(target[:,0], target[:,1], label="Leica prism GT")
    axes[0].plot(aligned[:,0], aligned[:,1], label="Saved trajectory, SE3 aligned")
    axes[0].set(xlabel="x / m", ylabel="y / m", title="Matched positions; fixed metric scale")
    axes[0].axis("equal"); axes[0].legend()
    axes[1].plot((est_t[indices]-est_t[0])/1e9, error)
    axes[1].set(xlabel="time since first estimate / s", ylabel="position error / m", title="All matched APE; no GT-gap extrapolation")
    for axis in axes: axis.grid(alpha=.25)
    fig.suptitle(f"{args.provenance}: position-only reanalysis, not a new estimator run")
    fig.savefig(args.output/"position-reanalysis.png", dpi=160); plt.close(fig)
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
