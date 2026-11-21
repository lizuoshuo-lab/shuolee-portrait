"""Compare accepted local LIO/LIVO poses on the same original Hilti dense GT rows.

Dense IMU-frame reference is a development reference, not surveyed millimetre GT.
No lever arm, fitted time shift, scale fit, ROS odometry wall-time, or sparse-frame guess.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "outputs/fast-livo2-hilti-gt-001/evaluation"
GT = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\fast-livo2-data\hilti2022-recovery-003-exp14\ground_truth\exp14_basement_2_imu.txt")
GT_SHA = "dec7b20cc11002482ccb416a6e1a1c410d385b843504d3aedf7b5956746af1b6"
MAX_ESTIMATE_GAP_NS = 200_000_000
MAX_REFERENCE_PAIR_NS = 200_000_000
MIN_COMMON_REFERENCE_FRACTION = .90


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_poses(path):
    stamps, values = [], []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Expected sensor timestamp xyz qx qy qz qw")
        stamps.append(int(Decimal(fields[0]) * Decimal(1_000_000_000)))
        values.append([float(v) for v in fields[1:]])
    times = np.array(stamps, dtype=np.int64)
    array = np.array(values, dtype=float)
    if len(times) < 3 or not np.isfinite(array).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite, strictly increasing actual poses required; no sorting or deduplication")
    if np.any(np.abs(np.linalg.norm(array[:, 3:], axis=1) - 1) > 1e-3):
        raise ValueError("Non-unit quaternion or placeholder orientation")
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, :3, 3] = array[:, :3]
    poses[:, :3, :3] = Rotation.from_quat(array[:, 3:]).as_matrix()
    return times, poses


def interpolate(times, poses, queries, max_gap_ns=MAX_ESTIMATE_GAP_NS):
    """Estimate interpolation only; original reference query points are unchanged."""
    output = np.repeat(np.eye(4)[None], len(queries), axis=0)
    valid = np.zeros(len(queries), dtype=bool)
    reasons = {"outside_estimate_span": 0, "estimate_bracket_gap_exceeded": 0}
    for i, (query, k) in enumerate(zip(queries, np.searchsorted(times, queries))):
        if k < len(times) and times[k] == query:
            output[i], valid[i] = poses[k], True
            continue
        if k == 0 or k == len(times):
            reasons["outside_estimate_span"] += 1
            continue
        gap = int(times[k]) - int(times[k - 1])
        if gap > max_gap_ns:
            reasons["estimate_bracket_gap_exceeded"] += 1
            continue
        fraction = (int(query) - int(times[k - 1])) / gap
        output[i, :3, 3] = (1 - fraction) * poses[k - 1, :3, 3] + fraction * poses[k, :3, 3]
        output[i, :3, :3] = Slerp([0., 1.], Rotation.from_matrix(poses[k - 1:k + 1, :3, :3]))([fraction]).as_matrix()[0]
        valid[i] = True
    return output, valid, reasons


def rigid_align(poses, reference):
    source, target = poses[:, :3, 3], reference[:, :3, 3]
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    if singular[0] <= 0 or np.count_nonzero(singular > singular[0] * 1e-10) < 2:
        raise ValueError("Position trajectory does not constrain rigid alignment")
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    transform = np.eye(4)
    transform[:3, :3] = vt.T @ correction @ u.T
    transform[:3, 3] = target.mean(0) - transform[:3, :3] @ source.mean(0)
    return transform[None] @ poses, transform


def stats(values, unit):
    return {"count": len(values), "unit": unit, "rmse": float(np.sqrt(np.mean(values ** 2))),
            "mean": float(values.mean()), "median": float(np.median(values)), "max": float(values.max())}


def errors(poses, reference, times, original_indices):
    aligned, transform = rigid_align(poses, reference)
    first_transform = reference[0] @ np.linalg.inv(poses[0])
    anchored = first_transform[None] @ poses
    aligned_error = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    anchored_error = np.linalg.norm(anchored[:, :3, 3] - reference[:, :3, 3], axis=1)
    pair_mask = (np.diff(original_indices) == 1) & (np.diff(times) <= MAX_REFERENCE_PAIR_NS)
    delta_ref = np.linalg.inv(reference[:-1]) @ reference[1:]
    delta_est = np.linalg.inv(poses[:-1]) @ poses[1:]
    relative = (np.linalg.inv(delta_ref) @ delta_est)[pair_mask]
    if not len(relative):
        raise ValueError("No unbridged common-reference RPE pairs")
    trans = np.linalg.norm(relative[:, :3, 3], axis=1)
    rot = np.rad2deg(Rotation.from_matrix(relative[:, :3, :3]).magnitude())
    result = {"ate_position_se3": stats(aligned_error, "m"),
              "first_matched_pose_gauge_position": stats(anchored_error, "m"),
              "common_reference_adjacent_rpe_translation": stats(trans, "m"),
              "common_reference_adjacent_rpe_rotation": stats(rot, "deg"),
              "scale": 1.0, "SE3_alignment_estimated_to_reference": transform.tolist(),
              "first_matched_pose_alignment_estimated_to_reference": first_transform.tolist(),
              "last_common_point_aligned_error_m": float(aligned_error[-1]),
              "last_common_point_first_matched_pose_gauge_error_m": float(anchored_error[-1]),
              "reference_adjacent_pairs_excluded": int(len(pair_mask) - pair_mask.sum())}
    return result, aligned, aligned_error, anchored_error


def load_accepted(run, mode):
    status_path = run / "status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    assert state["status"] == "actual_local_functional_replay_evidence_passed"
    assert state["mode"] == mode and all(state["runtime_evidence_checks"].values())
    assert state["fusion_child_exit_code"] == 0 and state["bag_playback_completed"]
    assert state["child_exit_evidence"]["all_required_children_exit_zero"]
    assert state["source_commit"] == "0d2c0346107b75b59934975adec9a6eeeb913c64"
    assert state["rate"] == .5 and state["fusion_started"]
    prep_path = run / "preflight.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    assert prep == state["preflight"] and prep["status"] == "read_only_preflight_passed"
    assert len(prep["checks"]) == 13 and all(prep["checks"].values())
    assert all(row["sha256"] == row["expected_sha256"] for row in prep["compiled_files"])
    assert len(prep["compiled_files"]) == 6
    build_receipt = LAB / "outputs/fast-livo2-shutdown-fix-001/build-after-verification.json"
    assert prep["accepted_build_receipt_SHA256"] == sha(build_receipt)
    build = json.loads(build_receipt.read_text(encoding="utf-8"))
    assert state["binary"]["sha256"] == build["binary_after_sha256"] == prep["compiled_files"][0]["sha256"]
    sources = [json.loads((run / filename).read_text(encoding="utf-8")) for filename in ("source-before.json", "source-after.json")]
    assert sources[0] == sources[1] and sources[0]["selected_receipt_files_verified"] == 75
    assert sha(run / "runner-source.py") == "c0b8f3417ee455a3ac9f11bcd8e15d06596b70c85c9b54a91478a3e357f6b683"
    parameter_checks = json.loads((run / "actual-parameter-checks.json").read_text(encoding="utf-8"))
    assert all(row["passed"] for row in parameter_checks.values())
    planned = yaml.safe_load((run / "planned-parameters.yaml").read_text(encoding="utf-8"))
    actual_parameters = flatten_parameters(yaml.safe_load((run / "actual-parameters.yaml").read_text(encoding="utf-8")))
    assert all(key in actual_parameters and actual_parameters[key] == value for key, value in planned.items()), "Full planned camera/extrinsic/algorithm parameter binding failed"
    assert state["bag"]["sha256_after"] == "18eea013076b8263ababf4f36eb68d331564a0a493262b006dc0dddc2aa85b82"
    pose_summary = state["pre_vio_trajectory"]
    assert "IMU body" in pose_summary["phase"] and not pose_summary["negative_timestamp_steps"]
    candidates = [p for p in (run / "official-log-after/result").glob("*.txt") if sha(p) == pose_summary["sha256"]]
    assert len(candidates) == 1, "Bind the primary trajectory to the actual runner hash"
    times, poses = read_poses(candidates[0])
    assert len(times) == pose_summary["rows"]
    recovery = LAB / "outputs/fast-livo2-gt-input-recovery-003"
    input_seal = json.loads((recovery / "receipt.json").read_text(encoding="utf-8"))
    audit_relative = "offline-audit-runs/20261010T094728Z_ae2afedf/receipt.json"
    binding = next(row for key, row in input_seal["evidence"].items() if key.replace("\\", "/") == audit_relative)
    audit_path = recovery / audit_relative
    assert sha(audit_path) == binding["sha256"]
    sensor_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lidar_times = sensor_audit["observed_topics"]["/hesai/pandar"]
    assert times[0] <= lidar_times["first_header_ns"] + 5_000_000_000, "Excessive primary trajectory startup loss"
    assert times[-1] >= lidar_times["last_header_ns"] - 500_000_000, "Primary trajectory misses last sensor segment"
    assert times[-1] <= lidar_times["last_header_ns"] + 500_000_000, "Primary trajectory extends outside recorded sensor interval"
    if mode == "lio":
        markers = state["visual_log_evidence"]["markers"]
        assert markers["handle_vio_raw_features"]["lines"] == markers["retrieved_visual_candidates"]["lines"] == 0
        assert markers["jacobian_stage"]["timing_lines"] == markers["ekf_stage"]["timing_lines"] == 0
    return state, times, poses, {"run_directory": str(run), "actual_status_sha256": sha(status_path),
                               "preflight_sha256": sha(prep_path), "source_before_sha256": sha(run / "source-before.json"),
                               "source_after_sha256": sha(run / "source-after.json"), "runner_sha256": sha(run / "runner-source.py"),
                               "compiled_binary_sha256": state["binary"]["sha256"],
                               "full_planned_parameters_bound_to_actual": True,
                               "input_offline_audit_sha256": sha(audit_path),
                               "primary_startup_delay_from_first_lidar_header_s": int(times[0] - lidar_times["first_header_ns"]) / 1e9,
                               "primary_end_delay_from_last_lidar_header_s": int(times[-1] - lidar_times["last_header_ns"]) / 1e9,
                               "actual_parameters_sha256": sha(run / "actual-parameters.yaml"),
                               "trajectory_relative_path": candidates[0].relative_to(run).as_posix(),
                               "trajectory_sha256": sha(candidates[0]), "original_pose_rows": len(times),
                               "phase": pose_summary["phase"]}


def flatten_parameters(parameters, prefix=""):
    result = {}
    for key, value in parameters.items():
        name = prefix + "/" + str(key).strip("/")
        if isinstance(value, dict):
            result.update(flatten_parameters(value, name))
        else:
            result[name] = value
    return result


def comparable_parameters(run):
    parameters = yaml.safe_load((run / "actual-parameters.yaml").read_text(encoding="utf-8"))
    assert parameters["common"]["img_en"] in (0, 1)
    assert parameters["image_save"]["interval"] == 20
    assert parameters["time_offset"]["img_time_offset"] == 0.
    assert parameters["time_offset"]["imu_time_offset"] == 0.
    assert parameters["time_offset"]["lidar_time_offset"] == 0.
    parameters["common"].pop("img_en")
    parameters["image_save"].pop("img_save_en")
    parameters["evo"].pop("seq_name")
    parameters.pop("run_id", None)
    parameters.pop("roslaunch", None)
    return parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lio-run", type=Path, required=True)
    parser.add_argument("--livo-run", type=Path, required=True)
    args = parser.parse_args()
    assert comparable_parameters(args.lio_run) == comparable_parameters(args.livo_run), "Unregistered actual algorithm parameter difference"
    assert sha(GT) == GT_SHA
    ref_t, reference = read_poses(GT)
    assert len(ref_t) == 689
    prepared, receipts = {}, {}
    common = np.ones(len(ref_t), dtype=bool)
    for mode, path in (("lio", args.lio_run), ("livo", args.livo_run)):
        state, stamps, poses, receipt = load_accepted(path.resolve(), mode)
        sampled, valid, rejected = interpolate(stamps, poses, ref_t)
        receipt.update(rejected_reference_queries=rejected, individually_supported_reference_rows=int(valid.sum()))
        receipts[mode], prepared[mode] = receipt, sampled
        common &= valid
    indices = np.flatnonzero(common)
    if len(indices) < 100 or len(indices) / len(ref_t) < MIN_COMMON_REFERENCE_FRACTION:
        raise ValueError("Insufficient paired reference coverage")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, errors_by_mode = {}, {}
    for mode in ("lio", "livo"):
        metric, aligned, aligned_error, anchored_error = errors(prepared[mode][common], reference[common], ref_t[common], indices)
        metrics[mode] = metric
        errors_by_mode[mode] = (aligned_error, anchored_error)
        np.savetxt(OUT / f"{mode}-aligned-common-position.txt", np.column_stack((ref_t[common] / 1e9, aligned[:, :3, 3])), fmt="%.9f")
    with (OUT / "paired-errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["original_GT_index", "timestamp_ns", "lio_aligned_position_error_m", "livo_aligned_position_error_m", "lio_first_matched_pose_error_m", "livo_first_matched_pose_error_m"])
        for local, index in enumerate(indices):
            writer.writerow([int(index), int(ref_t[index]), errors_by_mode["lio"][0][local], errors_by_mode["livo"][0][local], errors_by_mode["lio"][1][local], errors_by_mode["livo"][1][local]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    elapsed = (ref_t[common] - ref_t[common][0]) / 1e9
    for mode, color in (("lio", "#3463b3"), ("livo", "#da7540")):
        axes[0].plot(elapsed, errors_by_mode[mode][0], label=mode.upper(), color=color)
        axes[1].plot(elapsed, errors_by_mode[mode][1], label=mode.upper(), color=color)
    axes[0].set(title="Independent SE(3) alignment; scale=1", xlabel="Common reference elapsed time (s)", ylabel="IMU position error (m)")
    axes[1].set(title="First common pose gauge", xlabel="Common reference elapsed time (s)", ylabel="IMU position error (m)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Local FAST-LIVO2 exp14: same dense development-reference timestamps; pre-VIO output phase")
    fig.savefig(OUT / "paired-reference-errors.png", dpi=160)
    plt.close(fig)
    result = {"status": "accepted_local_two_mode_dense_reference_comparison", "dense_GT_sha256": GT_SHA,
              "dense_reference_rows": len(ref_t), "common_reference_rows": len(indices),
              "common_reference_fraction": len(indices) / len(ref_t), "common_reference_indices": indices.tolist(),
              "minimum_preregistered_common_reference_fraction": MIN_COMMON_REFERENCE_FRACTION,
              "common_start_timestamp_ns": int(ref_t[indices[0]]), "common_end_timestamp_ns": int(ref_t[indices[-1]]),
              "maximum_estimate_interpolation_gap_s": MAX_ESTIMATE_GAP_NS / 1e9,
              "maximum_RPE_reference_pair_interval_s": MAX_REFERENCE_PAIR_NS / 1e9,
              "input_receipts": receipts, "metrics": metrics,
              "actual_parameters_comparable_except_mode_and_output_identity": True,
              "evaluator_source_sha256": sha(Path(__file__)),
              "protocol": "Original dense GT queries unchanged; both sensor-time IMU-body estimates linearly interpolate position and SLERP orientation, intersect support. Separate posthoc rigid SE3 per mode, no scale or temporal fit; no bridging rejected rows or long reference gaps for RPE.",
              "limits": ["Publisher dense 6DoF IMU reference is for development/debugging; no dense millimetre-accuracy claim.",
                         "Sparse four checkpoints and evaluator suffix/frame ambiguity remain separate diagnostics; no sparse score generated here.",
                         "Primary output is after LiDAR and before same-cycle VIO; LIVO can include previous visual update history.",
                         "ROS odometry clock headers and sampled post-VIO image poses are not substituted for this common primary trajectory.",
                         "Default fixed publisher FAST configuration, including zero image offset, is not a physical synchronization or new calibration validation.",
                         "Single sequence and one functional replay per mode; no broad superiority or real-time claim."]}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "common_rows": len(indices), "lio_ATE_m": metrics["lio"]["ate_position_se3"]["rmse"], "livo_ATE_m": metrics["livo"]["ate_position_se3"]["rmse"]}))


if __name__ == "__main__":
    main()
