"""Read-only collector for the four predeclared complete-sequence V4 controls.

--check-inputs reports readiness and observed hashes, without trajectory analysis
or writing a summary. --execute requires all four terminal exit-0 runs, their
chunked summaries and CPU artifact checks. It creates one exclusive new output.
No trainer, loader, solver, trajectory auditor or renderer is imported. The
separate locked context checker only loads detached tensors on CPU after gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "manifests/tum-online-v4-full-sequence-protocol-001.json"
PROTOCOL_SHA = "ee635a6d65cc02dd26ff16e3f83721afa745a56b9f7a7464453833651a4963af"
BATCH = "outputs/tum-online-v4-full-sequence-001"
TERMINAL = "all_four_predeclared_full_sequence_runs_completed"
MAX_BYTES = 1024 * 1024
LOCKED = {
    "scripts/run_v4_full_sequence.py": "bfda7d3388ec7683a6e64f17415f11302beac9f64fdfb375f2278094d27493d0",
    "scripts/summarize_tum_online_v4_full.py": "4203a4aadb6547157e27ba160a9ae36e4c028f68d55670834411259c44cb4eac",
    "scripts/check_tum_v4_full_artifacts.py": "68f2e360558f81f4cf3dc040be99c94d0390fea590bb4ebc2d446712834405d3",
    "scripts/check_tum_v4_full_context_pairs.py": "56ef706ded769ca1e9fc12c20acc585b1fba45973639e2cbfb62cabec8138a49",
    "scripts/audit_tum_xyz_v4_artifacts.py": "88ff3a75db6d9afc5b90756e92e5572c1549ebd3bb346612cd6faf4537b53d6e",
    "scripts/audit_tum_trajectory.py": "e7acfaea60297dbc31eccc62b6ea9f50f577fb317209ac50e3342630424a42d7",
    "scripts/train_tum_online_gs.py": "45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def safe_path(base, relative):
    path = (Path(base) / relative).resolve()
    require(path.is_relative_to(Path(base).resolve()), "Path escapes evidence root: " + str(relative))
    return path


def read_json(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def bounded(value):
    data = encoded(value)
    require(len(data) <= MAX_BYTES, "JSON exceeds 1 MiB; retain failure, never truncate flags or frames")
    return data


def exclusive_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(bounded(value))


def verify_hash(base, relative, expected, observed):
    path = safe_path(base, relative)
    actual = sha(path)
    observed[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(relative)] = actual
    require(actual == expected, "SHA mismatch: " + str(relative))
    return path


def stamp(value):
    instant = datetime.fromisoformat(value["utc"])
    require(instant.tzinfo is not None, "Timestamp lacks timezone")
    require(instant == datetime.fromisoformat(value["beijing"]), "UTC/Beijing timestamp mismatch")
    return instant


def validate_ledger(protocol, ledger):
    """Pure gate, separately exercised with deliberately invalid CPU fixtures."""
    require(ledger["status"] == TERMINAL, "Batch is not the exact all-four terminal state")
    require(ledger["protocol_sha256"] == PROTOCOL_SHA, "Ledger protocol SHA")
    require(ledger["driver_sha256"] == protocol["driver_sha256"], "Ledger driver SHA")
    require(ledger["source_sha256"] == protocol["source_sha256"], "Ledger source table")
    require(ledger["GPU_runs_started"] == 4 and ledger["current_run"] is None, "Batch not fully collected")
    records = ledger["run_records"]
    require(len(records) == len(protocol["new_runs"]) == 4, "Exactly four records required")
    deadline = datetime.fromisoformat(protocol["deadline_beijing"])
    require(ledger["deadline_beijing"] == protocol["deadline_beijing"], "Deadline changed")
    previous_end = stamp(ledger["started_at"])
    durations = {}
    for expected, actual in zip(protocol["new_runs"], records):
        require(actual["order"] == expected["order"] and actual["run"] == expected["run"], "Run ordering differs")
        require(actual["configuration"] == expected["configuration"], "Run configuration differs")
        require(actual["returncode"] == 0 and actual["status"] == "completed", "Run did not complete with exit0")
        require(actual["frames"] == expected["configuration"]["frames"], "Record incomplete frames")
        require(actual["log"] == expected["log_relative_path"], "Unexpected log path")
        ready, guard, popped, ended = [stamp(actual[key]) for key in
            ("ready_at", "spawn_guard_at", "Popen_returned_at", "ended_at")]
        require(previous_end <= ready <= guard <= popped <= ended, "Overlap or nonchronological serial run")
        require(guard < deadline, "New GPU child was started at/after deadline")
        for key in ("metrics_sha256", "log_sha256"):
            require(isinstance(actual[key], str) and len(actual[key]) == 64 and
                    all(c in "0123456789abcdef" for c in actual[key]), "Missing or malformed artifact SHA")
        durations[actual["run"]] = (ended - guard).total_seconds()
        previous_end = ended
    require(stamp(ledger["ended_at"]) >= previous_end, "Batch ended before collected final child")
    return durations


def check_chunks(folder, metrics, summary, audit, observed):
    require(summary["summary_provenance"]["full_metrics_sha256"] == sha(folder / "metrics.json"), "Summary metrics SHA")
    require(summary["summary_provenance"]["summarizer_sha256"] == LOCKED["scripts/summarize_tum_online_v4_full.py"], "Summary source")
    require(summary["configuration"] == metrics["configuration"], "Summary configuration")
    require(summary["mapping_events"] == metrics["mapping_events"], "Summary map events")
    require(summary["pose_gradient_checks"] == metrics["pose_gradient_checks"], "Summary probe")
    require(summary["first_tracking_context"] == metrics["first_tracking_context"], "Summary actual first context")
    require((folder / "summary.json").stat().st_size <= MAX_BYTES, "Original summary oversized")
    require(audit["summary_chunks"]["status"] == "passed", "Chunk audit not passed")
    for relative, expected in audit["summary_chunks"]["input_sha256"].items():
        verify_hash(folder, relative, expected, observed)
    for kind in ("frame_metrics", "selected_sensor_metadata"):
        rows = []
        for item in summary[kind + "_chunks"]:
            path = verify_hash(folder, item["path"], item["sha256"], observed)
            require(path.is_relative_to((folder / "summary_chunks").resolve()), "Chunk outside summary folder")
            require(path.stat().st_size == item["bytes"] <= MAX_BYTES, "Chunk byte count")
            part = read_json(path)
            require(part["kind"] == kind and part["experiment_id"] == metrics["experiment_id"], "Chunk identity")
            require(part["start_local_index"] == item["start_local_index"] == len(rows), "Chunk gap or overlap")
            require(part["row_count"] == item["row_count"] == len(part["rows"]), "Chunk row count")
            rows.extend(part["rows"])
            require(part["stop_local_index_exclusive"] == item["stop_local_index_exclusive"] == len(rows), "Chunk endpoint")
        original = metrics[kind]
        require(len(rows) == len(original) == metrics["frames_processed"], "Dropped frame or sensor")
        require([r["frame_id"] for r in rows] == metrics["selected_frame_ids"], "Chunk frame IDs")
        if kind == "selected_sensor_metadata":
            require(rows == original, "Sensor chunk differs")
        else:
            for reduced, full in zip(rows, original):
                require(reduced["failure_flags"] == full["failure_flags"], "Lost failure flags")
                for key, value in reduced.items():
                    if key == "icp":
                        require(all(full[key][sub] == item for sub, item in value.items()), "Chunk ICP field")
                    else:
                        require(value == full[key], "Chunk frame field")


def validate_run(row, protocol, record, observed):
    folder = safe_path(ROOT, row["output_relative_path"])
    metric_path = folder / "metrics.json"
    original = read_json(metric_path)
    observed[str(metric_path.relative_to(ROOT))] = sha(metric_path)
    config, count = row["configuration"], row["configuration"]["frames"]
    require(original["status"] == "completed" and original["experiment_id"] == row["run"], "Run status/identity")
    require(original["configuration"] == config and original["implementation_version"] == 4, "Run config/version")
    require(original["frames_processed"] == count and original["selected_frame_ids"] == list(range(count)), "Incomplete frame selection")
    for field in ("frame_metrics", "selected_sensor_metadata"):
        require(len(original[field]) == count and [r["frame_id"] for r in original[field]] == list(range(count)), "Incomplete frame rows")
    for field, relative in (("script_sha256", "scripts/train_tum_online_gs_v4.py"), ("loader_sha256", "scripts/tum_rgbd.py"),
                            ("pose_helper_sha256", "scripts/track_fixed_map.py"), ("pose_prior_helper_sha256", "scripts/geometric_pose_prior.py")):
        require(original[field] == protocol["source_sha256"][relative], "Run source field " + field)
    require(original["icp_initializer"]["helper_sha256"] == protocol["source_sha256"]["scripts/track_rgbd_icp.py"], "ICP helper SHA")
    if record is not None:
        verify_hash(folder, "metrics.json", record["metrics_sha256"], observed)
        verify_hash(ROOT, record["log"], record["log_sha256"], observed)
        require(record["pose_gradient_checks"] == original["pose_gradient_checks"], "Ledger probe mismatch")
        require(record["failure_frame_count"] == original["failure_frame_count"], "Ledger flags mismatch")
    audit_path = folder / "cpu_full_v4_artifact_checks.json"
    audit = read_json(audit_path)
    observed[str(audit_path.relative_to(ROOT))] = sha(audit_path)
    require(audit["status"] == "passed" and audit["run"] == row["run"], "Full CPU artifact audit not passed")
    require(audit["full_checker_sha256"] == LOCKED["scripts/check_tum_v4_full_artifacts.py"] and
            audit["protocol_sha256"] == PROTOCOL_SHA and audit["GPU_calls"] == 0, "Unknown CPU full checker")
    require(audit["frame_count"] == count and audit["all_requested_frames_and_original_flags_retained"], "Audit incomplete frames")
    require(audit["configuration_36fields_exact"] and audit["entire_sensor_metadata_matches_full_CPU_baseline"], "Audit config/sensors")
    require(audit["source_and_window_causality_passed"] and not audit["GT_pose_fields_in_causal_progress_and_last_checkpoint"], "Causal artifact audit")
    for relative, expected in audit["input_hashes"].items():
        verify_hash(folder, relative, expected, observed)
    context = original["first_tracking_context"]
    require(read_json(folder / "first_tracking_context.json") == context, "Context metadata linkage")
    verify_hash(folder, "first_tracking_context.pt", context["checkpoint_sha256"], observed)
    verify_hash(folder, "first_tracking_context.json", audit["first_tracking_context_full"]["json_sha256"], observed)
    require(context["saved_before_first_pose_gradient_probe"] and context["tracking_optimizer_steps_executed_before_save"] == 0, "Context not saved before probe")
    require(context["sensor_metadata"]["frame_id"] == 1 and context["map_source_frame_ids"] == [0], "Actual first context scope")
    require(all(context[key] is False for key in ("ground_truth_included", "future_sensor_observations_included", "future_maps_or_trajectories_included")), "Context has future/GT")
    check_chunks(folder, original, read_json(folder / "summary.json"), audit, observed)
    return original, audit


def check_inputs():
    """No aggregation, raw GT consumption, tensor loading or output mutation."""
    observed, issues, runs, baselines = {}, [], [], {}
    try:
        verify_hash(ROOT, PROTOCOL, PROTOCOL_SHA, observed)
        protocol = read_json(ROOT / PROTOCOL)
        require(protocol["driver_sha256"] == LOCKED["scripts/run_v4_full_sequence.py"], "Protocol driver")
        for relative, expected in {**protocol["source_sha256"], **LOCKED}.items():
            verify_hash(ROOT, relative, expected, observed)
        for relative, expected in protocol["CPU_check_artifact_sha256"].items():
            verify_hash(ROOT, relative, expected, observed)
        for item in protocol["dataset_archives"].values():
            path = verify_hash(ROOT, item["relative_path"], item["sha256"], observed)
            require(path.stat().st_size == item["bytes"], "Archive byte size")
    except (ValueError, KeyError, OSError) as error:
        return {"status": "blocked_invalid_protocol_or_source", "ready": False, "issues": [str(error)], "observed_sha256": observed}
    ledger_path = ROOT / BATCH / "metrics.json"
    ledger = None
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        observed[str(ledger_path.relative_to(ROOT))] = sha(ledger_path)
        try:
            validate_ledger(protocol, ledger)
            require(read_json(ROOT / BATCH / "progress.json") == ledger, "Terminal progress/metrics differ")
            observed[BATCH + "/progress.json"] = sha(ROOT / BATCH / "progress.json")
        except (KeyError, ValueError, OSError) as error:
            issues.append("terminal_ledger: " + str(error))
    else:
        issues.append("terminal_ledger: metrics.json absent; batch has not been collected as all-four terminal")
    progress = read_json(ROOT / BATCH / "progress.json")
    observed[BATCH + "/progress.json"] = sha(ROOT / BATCH / "progress.json")
    record_map = {r["run"]: r for r in (ledger or progress).get("run_records", [])}
    for scene, reference in protocol["full_CPU_baseline_references"].items():
        try:
            for relative, expected in reference["artifact_sha256"].items():
                verify_hash(ROOT, relative, expected, observed)
            folder = ROOT / "outputs" / reference["run"]
            for relative, expected in reference["validation_evidence_sha256"].items():
                verify_hash(folder, relative, expected, observed)
            for name, expected in reference["source_sha256"].items():
                verify_hash(ROOT, "scripts/" + name, expected, observed)
            require(read_json(folder / "validation.json")["status"] == "passed", "ICP validation not passed")
            baselines[scene] = {"status": "ready", "run": reference["run"], "frames": reference["frames"]}
        except (KeyError, ValueError, OSError) as error:
            baselines[scene] = {"status": "blocked", "issue": str(error)}
            issues.append(scene + " baseline: " + str(error))
    for row in protocol["new_runs"]:
        item = {"run": row["run"], "frames_required": row["configuration"]["frames"]}
        try:
            record = record_map.get(row["run"])
            require(record is not None and record.get("returncode") == 0 and record.get("status") == "completed", "terminal exit0 record pending")
            validate_run(row, protocol, record, observed)
            item["status"] = "ready_artifacts"
        except (KeyError, ValueError, OSError) as error:
            item.update(status="pending_or_invalid", issue=str(error))
            issues.append(row["run"] + ": " + str(error))
        runs.append(item)
    return {"status": "ready_all_four_complete_audited_inputs" if not issues else "not_ready_no_aggregation",
        "ready": not issues, "batch_status": progress["status"], "run_inputs": runs, "baselines": baselines,
        "issues": issues, "observed_sha256": observed, "collector_sha256": sha(Path(__file__)),
        "trajectory_metrics_recomputed": False, "GT_poses_read": False, "torch_imported": "torch" in sys.modules,
        "GPU_calls": 0, "output_written_by_check_inputs": False}


def rigid_inverse(matrix):
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def check_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    require(poses.ndim == 3 and poses.shape[1:] == (4, 4) and np.isfinite(poses).all(), "Finite Nx4x4 poses required")
    require(np.max(np.abs(poses[:, 3] - [0, 0, 0, 1])) < 1e-9, "Pose bottom row")
    rotations = poses[:, :3, :3]
    error = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
    det_error = float(np.max(np.abs(np.linalg.det(rotations) - 1)))
    require(error < 2e-5 and det_error < 2e-5, "Nonrigid camera matrix")
    return {"orthogonality_max_abs": error, "determinant_error_max": det_error}


def angle_degrees(matrix):
    # SciPy's proper-rotation conversion tolerates persisted float32 roundoff.
    return float(np.linalg.norm(Rotation.from_matrix(matrix).as_rotvec()) * 180 / math.pi)


def mean(values):
    values = list(values)
    return math.fsum(float(v) for v in values) / len(values)


def rmse(values):
    values = list(values)
    return math.sqrt(math.fsum(float(v) ** 2 for v in values) / len(values))


def trajectory(estimated, truth):
    estimated, truth = np.asarray(estimated, float), np.asarray(truth, float)
    require(len(estimated) == len(truth) >= 2, "Trajectory pair length")
    structure = check_poses(estimated)
    check_poses(truth)
    x, y = estimated[:, :3, 3], truth[:, :3, 3]
    mx, my = np.array([mean(x[:, j]) for j in range(3)]), np.array([mean(y[:, j]) for j in range(3)])
    xc, yc = x - mx, y - my
    cross = np.array([[math.fsum(xc[:, i] * yc[:, j]) for j in range(3)] for i in range(3)])
    u, singular, vt = np.linalg.svd(cross)
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = my - rotation @ mx
    alignment = np.eye(4); alignment[:3, :3] = rotation; alignment[:3, 3] = translation
    raw = np.linalg.norm(x - y, axis=1)
    aligned = np.linalg.norm(x @ rotation.T + translation - y, axis=1)
    raw_rot = np.array([angle_degrees(g[:3, :3].T @ e[:3, :3]) for e, g in zip(estimated, truth)])
    rt, rr = [], []
    for i in range(1, len(estimated)):
        dg = rigid_inverse(truth[i-1]) @ truth[i]
        de = rigid_inverse(estimated[i-1]) @ estimated[i]
        error = rigid_inverse(dg) @ de
        rt.append(float(np.linalg.norm(error[:3, 3]))); rr.append(angle_degrees(error[:3, :3]))
    stats = {"raw_anchored_translation_rmse_m": rmse(raw), "se3_aligned_ate_translation_rmse_m": rmse(aligned),
        "raw_anchored_rotation_rmse_degrees": rmse(raw_rot), "adjacent_rpe_translation_rmse_m": rmse(rt),
        "adjacent_rpe_rotation_rmse_degrees": rmse(rr), "raw_endpoint_translation_error_m": float(raw[-1]),
        "raw_endpoint_rotation_error_degrees": float(raw_rot[-1]), "raw_translation_max_m": float(np.max(raw)),
        "raw_translation_mean_m": mean(raw), "alignment_scale": 1., "alignment_estimated_to_truth": alignment.tolist(),
        "alignment_singular_values": singular.tolist(), "centered_estimated_positions_rank": int(np.linalg.matrix_rank(xc)),
        "camera_structure": structure, "frame_count": len(estimated), "adjacent_pair_count": len(rt)}
    arrays = {"raw_error_m": raw.tolist(), "raw_rotation_deg": raw_rot.tolist(), "ate_error_m": aligned.tolist(),
        "rpe_translation_m": [None] + rt, "rpe_rotation_deg": [None] + rr, "position_m": x.tolist()}
    return stats, arrays


def text_rows(path, width):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    require(all(len(r) == width for r in rows), "Raw TUM row width")
    times = np.array([float(r[0]) for r in rows])
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0), "Raw TUM time order")
    return rows, times


def nearest(times, value):
    right = int(np.searchsorted(times, value))
    options = [i for i in (right - 1, right) if 0 <= i < len(times)]
    return min(options, key=lambda i: (abs(float(times[i]) - value), i))


def reconstruct_truth(dataset, sensors, observed):
    """Independent raw text association and SciPy xyzw; no frozen loader calls."""
    rgb, rgb_t = text_rows(dataset / "rgb.txt", 2)
    depth, depth_t = text_rows(dataset / "depth.txt", 2)
    gt, gt_t = text_rows(dataset / "groundtruth.txt", 8)
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        observed[str((dataset / name).relative_to(ROOT))] = sha(dataset / name)
    selected, used, last = [], set(), -math.inf
    for i, timestamp in enumerate(rgb_t):
        j, k = nearest(depth_t, timestamp), nearest(gt_t, timestamp)
        if abs(depth_t[j] - timestamp) >= .02 or abs(gt_t[k] - timestamp) >= .02:
            continue
        if timestamp - last <= 1 / 32 or j in used:
            continue
        selected.append((i, j, k)); used.add(j); last = timestamp
    require(len(selected) == len(sensors), "Stored selection is not the complete associated sequence")
    absolute = []
    for fid, ((i, j, k), sensor) in enumerate(zip(selected, sensors)):
        require([sensor[x] for x in ("frame_id", "rgb_index", "depth_index", "gt_index")] == [fid, i, j, k], "Stored sensor indices differ from raw association")
        require([sensor[x] for x in ("rgb_timestamp", "depth_timestamp", "gt_timestamp")] == [rgb_t[i], depth_t[j], gt_t[k]], "Stored timestamps")
        require(sensor["rgb_path"] == rgb[i][1] and sensor["depth_path"] == depth[j][1], "Stored sensor paths")
        require(sensor["rgb_depth_abs_dt_s"] == abs(rgb_t[i] - depth_t[j]) and sensor["rgb_gt_abs_dt_s"] == abs(rgb_t[i] - gt_t[k]), "Stored time differences")
        vector = np.array(gt[k][1:], dtype=float)
        require(np.isfinite(vector).all() and np.linalg.norm(vector[3:]) > 0, "GT xyzw finite/nonzero")
        require(abs(np.linalg.norm(vector[3:]) - sensor["gt_quaternion_norm_before_normalization"]) < 1e-12, "GT raw quaternion norm")
        pose = np.eye(4); pose[:3, :3] = Rotation.from_quat(vector[3:]).as_matrix(); pose[:3, 3] = vector[:3]
        absolute.append(pose)
    absolute = np.array(absolute)
    anchored = rigid_inverse(absolute[0]) @ absolute
    check_poses(anchored)
    return anchored, {"raw_counts": {"RGB": len(rgb), "depth": len(depth), "GT": len(gt)},
        "associated_frames": len(selected), "gauge": "inverse(first selected raw GT c2w) @ each raw GT c2w; no new fit",
        "GT_xyzw": True, "quaternions_normalized_by_SciPy": True, "nearest_GT_no_interpolation_or_motion_compensation": True,
        "all_stored_indices_paths_timestamps_verified": True,
        "sensor_span_s": float(rgb_t[selected[-1][0]] - rgb_t[selected[0][0]]),
        "RGB_depth_max_abs_dt_s": max(s["rgb_depth_abs_dt_s"] for s in sensors),
        "RGB_GT_max_abs_dt_s": max(s["rgb_gt_abs_dt_s"] for s in sensors)}


def compare_reported(stats, original, icp=False):
    mapping = {"raw_anchored_translation_rmse_m": "raw_anchored_translation_rmse_m",
        "se3_aligned_ate_translation_rmse_m": "se3_aligned_ate_translation_rmse_m" if icp else "ate_rmse_m_se3_no_scale",
        "raw_anchored_rotation_rmse_degrees": "raw_rotation_rmse_deg" if icp else "raw_anchored_rotation_rmse_degrees",
        "adjacent_rpe_translation_rmse_m": "adjacent_rpe_translation_rmse_m",
        "adjacent_rpe_rotation_rmse_degrees": "adjacent_rpe_rotation_rmse_deg" if icp else "adjacent_rpe_rotation_rmse_degrees"}
    errors = {key: abs(stats[key] - original[value]) for key, value in mapping.items()}
    # Stored GS matrices have float32 rigid roundoff; metric angle implementations
    # differ (SciPy rotvec here, online helper atan2). Keep the discrepancy visible.
    require(all(error < (1e-4 if "rotation" in key else 2e-6) for key, error in errors.items()), "Independent reconstructed metric disagrees beyond declared serialization tolerance")
    return errors


def column_table(records):
    require(records and all(list(r) == list(records[0]) for r in records), "Column schema varies")
    return {"columns": list(records[0]), "rows": [list(r.values()) for r in records]}


def summarize_scene(scene, protocol, ledger, observed, context_checker):
    choices = [row for row in protocol["new_runs"] if row["scene"] == scene]
    require([r["configuration"]["pose_prior_weight"] for r in choices] == [0., .01], "Scene weight order")
    reference = protocol["full_CPU_baseline_references"][scene]
    folder = ROOT / "outputs" / reference["run"]
    baseline = read_json(folder / "metrics.json")
    sensors = read_json(folder / "selected_sensor_metadata.json")["selected_frames"]
    truth, sensor_info = reconstruct_truth(Path(choices[0]["configuration"]["dataset"]), sensors, observed)
    icp = np.array([r["estimated_c2w"] for r in baseline["frame_metrics"]], dtype=float)
    icp_stats, icp_arrays = trajectory(icp, truth)
    icp_stats["reported_metric_absolute_differences"] = compare_reported(icp_stats, baseline["trajectory"], True)
    require(len(icp) == reference["frames"] and np.array_equal(icp[0], np.eye(4)), "ICP full count / gauge")
    icp_rows = []
    for i, r in enumerate(baseline["frame_metrics"]):
        icp_rows.append({"frame_id": i, "accepted": r["accepted"], "failure_flags": r["failure_flags"],
            **{key: value[i] for key, value in icp_arrays.items()}})
    result = {"scene": scene, "frame_count": len(icp), "seed": 0, "sensor_evaluation": sensor_info,
        "baseline": {"run": reference["run"], "metrics": icp_stats, "failure_frame_count": baseline["failure_frame_count"],
            "timing_s": {key: baseline[key] for key in ("cpu_loop_seconds", "total_seconds_before_metrics_export")}}, "arms": {}}
    details = {"scene": scene, "format": "ordered_columns_and_rows_v1", "rows_not_dropped": True,
        "sensor_columns": ["frame_id", "rgb_index", "depth_index", "gt_index", "rgb_timestamp", "rgb_depth_abs_dt_s", "rgb_gt_abs_dt_s"],
        "sensors": [[s[k] for k in ("frame_id", "rgb_index", "depth_index", "gt_index", "rgb_timestamp", "rgb_depth_abs_dt_s", "rgb_gt_abs_dt_s")] for s in sensors],
        "truth_camera_centers_m": truth[:, :3, 3].tolist(), "ICP": column_table(icp_rows), "arms": {}}
    plot_arrays = {"GT": truth[:, :3, 3], "ICP": icp_arrays}
    for choice in choices:
        record = next(r for r in ledger["run_records"] if r["run"] == choice["run"])
        metrics, audit = validate_run(choice, protocol, record, observed)
        require(metrics["selected_sensor_metadata"] == sensors, "V4/ICP sensor pairing mismatch")
        require(metrics["icp_initializer"]["configuration"] == baseline["configuration"]["icp"], "V4/ICP solver config mismatch")
        rows = metrics["frame_metrics"]
        estimated = np.array([r["estimated_c2w"] for r in rows], dtype=float)
        prefix = np.array([r["icp"]["icp_prefix_c2w_float64"] for r in rows], dtype=float)
        require(np.array_equal(prefix, icp), "Actual float64 ICP prefix is not exact full baseline")
        stats, arrays = trajectory(estimated, truth)
        stats["reported_metric_absolute_differences"] = compare_reported(stats, metrics["trajectory"])
        correction = np.linalg.norm(estimated[:, :3, 3] - icp[:, :3, 3], axis=1)
        correction_rotation = [angle_degrees(p[:3, :3].T @ e[:3, :3]) for p, e in zip(icp, estimated)]
        diff = np.array(arrays["raw_error_m"]) - np.array(icp_arrays["raw_error_m"])
        frame_rows = []
        for i, r in enumerate(rows):
            b = baseline["frame_metrics"][i]
            require(r["icp"]["accepted"] == b["accepted"] and r["icp"]["failure_flags"] == b["failure_flags"], "V4/ICP accept/flag mismatch")
            if i:
                require(np.array_equal(r["icp"]["registration"]["candidate_previous_from_current"], b["candidate_previous_from_current"]), "ICP relative candidate differs")
            rejected = r.get("pose_rejected", False)
            if rejected:
                require(np.array_equal(estimated[i], estimated[i-1]) and r["new_points_added"] == 0, "Rejected frame was moved or mapped")
            selected = r.get("optimized_loss_before_mapping", {})
            initial = r.get("initial_loss", {})
            frame_rows.append({"frame_id": i, "accepted": not rejected and r["icp"]["accepted"],
                "pose_rejected": rejected, "failure_flags": r["failure_flags"], "map_coverage_warning_flags": r.get("map_coverage_warning_flags", []),
                "icp_accepted": r["icp"]["accepted"], "icp_failure_flags": r["icp"]["failure_flags"],
                "icp_propagated_failure_flags": r["icp"].get("propagated_failure_flags", []),
                "selected_best_step": r.get("best_step_selected_by_tracking_objective_only"),
                "tracking_steps_completed": r.get("gs_refinement_steps_completed", 0),
                **{key: value[i] for key, value in arrays.items()}, "ICP_raw_error_m": icp_arrays["raw_error_m"][i],
                "GS_minus_ICP_raw_error_m": float(diff[i]), "correction_from_ICP_center_m": float(correction[i]),
                "correction_from_ICP_rotation_deg": correction_rotation[i], "point_count": r["point_count"], "new_points_added": r["new_points_added"],
                "initial_sensor_prior_combined": [initial.get(k) for k in ("sensor_objective", "prior_objective", "objective")],
                "selected_sensor_prior_combined": [selected.get(k) for k in ("sensor_objective", "prior_objective", "objective")]})
        require(sum(bool(r["failure_flags"]) for r in frame_rows) == metrics["failure_frame_count"], "Failure count mismatch")
        label = "prior0" if choice["configuration"]["pose_prior_weight"] == 0 else "prior001"
        result["arms"][label] = {"run": choice["run"], "pose_prior_weight": choice["configuration"]["pose_prior_weight"],
            "metrics": stats, "posthoc_translation_vs_full_ICP_excluding_gauge": {
                "denominator": len(diff)-1, "strictly_better_count": int(np.sum(diff[1:] < 0)),
                "strictly_worse_count": int(np.sum(diff[1:] > 0)), "exactly_equal_count": int(np.sum(diff[1:] == 0)),
                "mean_error_change_m": mean(diff[1:]), "not_used_for_candidate_or_weight_selection": True},
            "correction_from_full_ICP": {"center_mean_m": mean(correction[1:]), "center_max_m": float(np.max(correction[1:])),
                "rotation_mean_degrees": mean(correction_rotation[1:]), "rotation_max_degrees": max(correction_rotation[1:])},
            "all_float64_prefix_matrices_and_relative_candidates_exactly_match_full_ICP": True,
            "failure_frame_count": metrics["failure_frame_count"], "pose_rejected_count": sum(r["pose_rejected"] for r in frame_rows),
            "icp_failure_frame_count": metrics["icp_failed_frame_count"],
            "failure_flags_histogram": dict(Counter(flag for r in frame_rows for flag in r["failure_flags"])),
            "pose_gradient_checks": metrics["pose_gradient_checks"], "actual_first_tracking_context": metrics["first_tracking_context"],
            "initial_point_count": metrics["initial_point_count"], "final_point_count": metrics["final_point_count"],
            "all_mapping_events": metrics["mapping_events"], "per_gaussian_source_frame_counts": audit["per_gaussian_source_frame_counts"],
            "timing_s": {key: metrics[key] for key in ("initialization_seconds", "online_loop_seconds_excludes_warmup", "total_seconds_before_metrics_export")},
            "sensor_prior_combined_scope": "Persisted selected-best pre-map objectives, not final raw Adam iteration; GT used only in this posthoc collector"}
        details["arms"][label] = column_table(frame_rows)
        plot_arrays[label] = arrays
    context = context_checker.compare(scene)
    require(context["checker_sha256"] == LOCKED["scripts/check_tum_v4_full_context_pairs.py"], "Context checker source drift")
    require(len(context["input_tensors"]) == 7 and len(context["warmup_map_parameters"]) == 5, "Context tensor counts")
    result["actual_first_context_pair_comparison"] = context
    return result, details, plot_arrays


def make_plot(path, scene, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    truth = np.asarray(arrays["GT"])
    axes[0, 0].plot(truth[:, 0], truth[:, 2], color="black", label="GT reference")
    specs = [(axes[0, 1], "raw_error_m", 1000, "Raw center error (mm)"),
        (axes[0, 2], "ate_error_m", 1000, "SE3 aligned position error (mm)"),
        (axes[1, 0], "rpe_translation_m", 1000, "Adjacent RPE translation (mm)"),
        (axes[1, 1], "rpe_rotation_deg", 1, "Adjacent RPE rotation (deg)"),
        (axes[1, 2], "raw_rotation_deg", 1, "Raw anchored rotation error (deg)")]
    for name, color in (("ICP", "#4c78a8"), ("prior0", "#f58518"), ("prior001", "#54a24b")):
        points = np.asarray(arrays[name]["position_m"])
        axes[0, 0].plot(points[:, 0], points[:, 2], color=color, label=name, linewidth=1)
        for ax, key, scale, title in specs:
            values = np.asarray([np.nan if v is None else v for v in arrays[name][key]])
            ax.plot(np.arange(len(values)), values * scale, color=color, label=name, linewidth=.9)
            ax.set(title=title, xlabel="Associated frame ID")
    axes[0, 0].set(title="XZ projection; metrics use full 3D", xlabel="X (m)", ylabel="Z (m)")
    axes[0, 0].axis("equal")
    for ax in axes.flat:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(f"{scene}: complete associated sequence, seed 0, all frames retained\nIndependent scene axes; timings include I/O and are not FPS")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def execute(run):
    require(run == "tum-online-v4-full-sequence-summary-001", "Only the predeclared exclusive collector output is allowed")
    output = ROOT / "outputs" / run
    output.mkdir(exist_ok=False)
    try:
        preflight = check_inputs()
        exclusive_json(output / "input_preflight.json", preflight)
        require(preflight["ready"], "All-four completed/audited input gate is not ready; no metrics aggregated")
        protocol = read_json(ROOT / PROTOCOL); ledger = read_json(ROOT / BATCH / "metrics.json")
        durations = validate_ledger(protocol, ledger)
        observed = dict(preflight["observed_sha256"])
        spec = importlib.util.spec_from_file_location("locked_full_context_checker", ROOT / "scripts/check_tum_v4_full_context_pairs.py")
        checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
        result = {"status": "completed_read_only_all_four_full_sequence_controls", "collector_sha256": sha(Path(__file__)),
            "protocol_sha256": PROTOCOL_SHA, "scope": "Two same-domain Freiburg1 complete associated sequences, one seed per arm; no cross-scene mean or single-seed SD",
            "metric_implementation": "Raw stored estimated_c2w and raw TUM xyzw; SciPy Rotation, analytical rigid inverse, Kabsch SVD scale1, math.fsum; no loader/solver/auditor metric imported",
            "GT_usage": "Completed trajectories only; first raw GT anchor, posthoc metric alignment only; no trajectory, map, threshold or weight modification",
            "RPE_definition": "inv(inv(GT[i-1])@GT[i]) @ (inv(EST[i-1])@EST[i]); delta1, all adjacent frames",
            "timing_scope": "Observed durations include I/O and serialization according to original field boundaries, not isolated CPU/GPU kernel time or SLAM FPS",
            "driver_child_wall_seconds": durations, "scenes": {}, "GPU_calls": 0}
        pending_payloads, plots = {}, {}
        for scene in ("desk", "xyz"):
            summary, frames, arrays = summarize_scene(scene, protocol, ledger, observed, checker)
            result["scenes"][scene] = summary
            pending_payloads[scene + "_frames.json"] = bounded(frames)
            plots[scene] = arrays
        require(not checker.torch.cuda.is_initialized(), "CPU collector initialized CUDA")
        # Re-hash every consumed artifact immediately before publishing any result.
        for relative, expected in observed.items():
            require(sha(safe_path(ROOT, relative)) == expected, "Input changed during read-only collection: " + relative)
        result["input_sha256"] = observed
        result["all_inputs_rehashed_unchanged"] = True
        result["torch_CUDA_initialized"] = False
        pending_payloads["summary.json"] = bounded(result)
        for name, data in pending_payloads.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
        for scene, arrays in plots.items():
            make_plot(output / (scene + "_comparison.png"), scene, arrays)
        with (output / "collector_source.py").open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
        exclusive_json(output / "completion.json", {"status": "completed", "source_sha256": sha(Path(__file__)),
            "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()},
            "four_runs_two_baselines_all_frames_retained": True, "GPU_calls": 0})
        print(json.dumps({"status": result["status"], "output": str(output), "scene_frames": {s: result["scenes"][s]["frame_count"] for s in result["scenes"]}}))
    except BaseException as error:
        exclusive_json(output / "failure.json", {"status": "failed_partial_evidence_retained", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(), "GPU_calls": 0,
            "original_inputs_modified": False, "no_frame_or_flag_truncation": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--run", default="tum-online-v4-full-sequence-summary-001")
    args = parser.parse_args()
    if args.check_inputs:
        print(json.dumps(check_inputs(), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        execute(args.run)


if __name__ == "__main__":
    main()
