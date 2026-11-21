"""CPU-only full fr1/desk wrapper around the frozen projective ICP baseline.

The original solver and loader are not edited. A preflight replays the recorded
short20 inputs through the same library API and compares every estimated pose,
accept/reject decision, correspondence count and final normal-equation statistic.
Only after that passes is full549 added to the imported module's CLIPS in this
process and its unchanged main() invoked. GT poses remain final-evaluation only.

Usage: python scripts/track_rgbd_icp_full.py --run tum-icp-full549-001
       python scripts/track_rgbd_icp_full.py --check-only
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np

import track_rgbd_icp as icp

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "track_rgbd_icp.py": "21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a",
    "tum_rgbd.py": "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11",
    "audit_tum_trajectory.py": "e7acfaea60297dbc31eccc62b6ea9f50f577fb317209ac50e3342630424a42d7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_sources() -> dict:
    observed = {name: sha256(ROOT / "scripts" / name) for name in EXPECTED_HASHES}
    if observed != EXPECTED_HASHES:
        raise RuntimeError("Frozen ICP/loader/auditor hash changed; refusing to run")
    return {**observed, Path(__file__).name: sha256(Path(__file__))}


def sensor_metadata(frame) -> dict:
    # Do not serialize or inspect GT matrices before final evaluation.
    return {"frame_id": frame.frame_id, "rgb_index": frame.rgb_index,
            "depth_index": frame.depth_index, "gt_index": frame.gt_index,
            "rgb_timestamp": frame.rgb_timestamp, "depth_timestamp": frame.depth_timestamp,
            "gt_timestamp": frame.gt_timestamp,
            "rgb_depth_abs_dt_s": abs(frame.rgb_timestamp-frame.depth_timestamp),
            "rgb_gt_abs_dt_s": abs(frame.rgb_timestamp-frame.gt_timestamp),
            "rgb_path": frame.rgb_path, "depth_path": frame.depth_path,
            "gt_quaternion_norm_before_normalization": frame.gt_quaternion_norm_before_normalization}


def preflight(dataset: Path) -> dict:
    started = time.perf_counter()
    hashes = frozen_sources()
    reference_dir = ROOT / "outputs/tum-icp-short20-001"
    reference = read_json(reference_dir / "metrics.json")
    reference_validation = read_json(reference_dir / "validation.json")
    config = icp.ICPConfig()
    if json.loads(json.dumps(asdict(config))) != reference["configuration"]["icp"]:
        raise RuntimeError("Default ICP parameters differ from recorded short20 baseline")
    if reference["configuration"]["source_sha256"] != EXPECTED_HASHES:
        raise RuntimeError("Short20 baseline was produced by a different implementation")
    frames, association = icp.associate_frames(dataset, .02, .02)
    if len(frames) != 549 or [frame.frame_id for frame in frames] != list(range(549)):
        raise RuntimeError("This wrapper is fixed to all 549 associated fr1/desk frames")
    if association != reference["association"]:
        raise RuntimeError("Sensor association changed from the recorded baseline")
    selected = frames[30:50]
    if [sensor_metadata(frame) for frame in selected] != reference_validation["selected_sensor_metadata"]:
        raise RuntimeError("Short20 sensor inputs differ from recorded baseline")
    pyramid_data = []
    for width, height in icp.LEVELS:
        _, depths, masks = icp.load_frames(dataset, selected, width, height, .2, 5.)
        pyramid_data.append((depths, masks, icp.scaled_intrinsics(width, height)))

    def geometry(index):
        return [icp.depth_geometry(depth[index], valid[index], K, config)
                for depth, valid, K in pyramid_data]

    poses = [np.eye(4)]
    target = geometry(0)
    max_pose_error = 0.
    max_candidate_error = 0.
    compared_numeric_statistics = 0
    for index in range(1, 20):
        source = geometry(index)
        result = icp.register_pair(source, target, config)
        old = reference["frame_metrics"][index]
        for key in ["accepted", "failure_flags", "final_counts", "levels"]:
            if result[key] != old[key]:
                raise RuntimeError(f"Short20 replay mismatch frame {index}: {key}")
        for key, value in result["final_statistics"].items():
            expected = old["final_statistics"][key]
            if value is None or expected is None:
                if value != expected:
                    raise RuntimeError(f"Short20 statistic mismatch: {key}")
            elif not np.allclose(value, expected, rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"Short20 statistic mismatch frame {index}: {key}")
            compared_numeric_statistics += 1
        candidate_error = float(np.max(np.abs(np.asarray(result["candidate_previous_from_current"])
                                                -np.asarray(old["candidate_previous_from_current"]))))
        poses.append(icp.integrate_relative(poses[-1], result))
        pose_error = float(np.max(np.abs(poses[-1]-np.asarray(old["estimated_c2w"]))))
        max_candidate_error = max(max_candidate_error, candidate_error)
        max_pose_error = max(max_pose_error, pose_error)
        if max_pose_error > 1e-12 or max_candidate_error > 1e-12:
            raise RuntimeError(f"Short20 pose replay mismatch at frame {index}")
        target = source
    return {"status": "passed", "reference_run": "tum-icp-short20-001",
            "replayed_frame_ids": list(range(30, 50)), "replayed_pairs": 19,
            "pose_max_absolute_difference": max_pose_error,
            "relative_candidate_max_absolute_difference": max_candidate_error,
            "decisions_correspondences_and_level_stops_identical": True,
            "compared_final_statistic_fields": compared_numeric_statistics,
            "numeric_comparison_rtol": 1e-12, "numeric_comparison_atol": 1e-12,
            "default_parameters_unchanged": True, "sensor_metadata_unchanged": True,
            "source_sha256": hashes,
            "reference_metrics_sha256": sha256(reference_dir / "metrics.json"),
            "runtime": {"python": sys.version.split()[0],
                        "numpy": importlib.metadata.version("numpy"),
                        "scipy": importlib.metadata.version("scipy"), "device": "CPU"},
            "gt_pose_used_for_replay_optimization_or_selection": False,
            "wall_seconds": time.perf_counter()-started,
            "scope": "Reuses the same implementation; regression consistency, not independent algorithm validation"}


def finite_summary(values: list) -> dict:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return {"finite_count": len(finite), "missing_or_nonfinite_count": len(values)-len(finite),
            "min": float(finite.min()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None}


def failure_segments(records: list[dict]) -> list[dict]:
    segments = []
    for row in records[1:]:
        if not row["failure_flags"]:
            continue
        frame_id = row["frame_id"]
        if not segments or frame_id != segments[-1]["last_frame_id"]+1:
            segments.append({"first_frame_id": frame_id, "last_frame_id": frame_id,
                             "frame_count": 0, "flags": []})
        segment = segments[-1]
        segment["last_frame_id"] = frame_id
        segment["frame_count"] += 1
        segment["flags"] = sorted(set(segment["flags"]) | set(row["failure_flags"]))
    return segments


def finalize(output: Path, dataset: Path, precheck: dict, wall_seconds: float) -> dict:
    hashes = frozen_sources()
    metrics = read_json(output / "metrics.json")
    configuration = read_json(output / "configuration.json")
    frames, association = icp.associate_frames(dataset, .02, .02)
    records = metrics["frame_metrics"]
    if len(records) != 549 or [row["frame_id"] for row in records] != list(range(549)):
        raise RuntimeError("Full result did not retain every associated frame")
    if association != metrics["association"] or configuration["source_sha256"] != EXPECTED_HASHES:
        raise RuntimeError("Full-run association or frozen-source inconsistency")
    if configuration["icp"] != json.loads(json.dumps(asdict(icp.ICPConfig()))):
        raise RuntimeError("Unexpected ICP parameter change")
    estimated = np.asarray([row["estimated_c2w"] for row in records])
    if not np.isfinite(estimated).all() or not np.array_equal(estimated[0], np.eye(4)):
        raise RuntimeError("Nonfinite poses or first camera not identity")
    for row in records[1:]:
        if row["failure_flags"] and not np.array_equal(estimated[row["local_index"]],
                                                       estimated[row["local_index"]-1]):
            raise RuntimeError("Rejected frame did not preserve previous pose")
    trajectory_file = np.loadtxt(output / "trajectory_estimated.txt")
    truth_file = np.loadtxt(output / "trajectory_groundtruth_anchored.txt")
    if trajectory_file.shape != (549, 8) or truth_file.shape != (549, 8):
        raise RuntimeError("Incomplete trajectory files")
    if not np.array_equal(trajectory_file[:, 0], [frame.rgb_timestamp for frame in frames]):
        raise RuntimeError("Trajectory timestamps do not match sensor associations")
    if not np.array_equal(trajectory_file[:, 0], truth_file[:, 0]):
        raise RuntimeError("Estimated/evaluation timestamps differ")
    wrapper_metadata = {"entrypoint": "scripts/track_rgbd_icp_full.py",
                        "source_sha256": hashes,
                        "process_only_change": "icp.CLIPS['full549']=(0,549); original main and all solver functions unchanged",
                        "selection_rule": "Every chronological associated frame 0..548 of TUM fr1/desk; no frame or rejected candidate removed",
                        "first_camera": "identity world gauge",
                        "pair_initialization": "identity current-to-previous for every pair; no GT reset",
                        "ground_truth_usage": metrics["ground_truth_usage"],
                        "preflight": "preflight.json",
                        "runtime": precheck["runtime"],
                        "execution_scope": "Single-dataset complete-sequence control; not cross-scene generalization"}
    configuration["wrapper"] = wrapper_metadata
    metrics["configuration"] = configuration
    metrics["wrapper"] = wrapper_metadata
    metrics["artifacts"] += ["preflight.json", "source_manifest.json", "selected_sensor_metadata.json",
                             "validation.json", "summary.json", "source_script.py", "source_snapshot/"]
    icp.write_json(output / "configuration.json", configuration)
    icp.write_json(output / "metrics.json", metrics)
    icp.write_json(output / "preflight.json", precheck)
    icp.write_json(output / "source_manifest.json", wrapper_metadata)
    icp.write_json(output / "selected_sensor_metadata.json",
                   {"association": association, "selected_frames": [sensor_metadata(frame) for frame in frames]})
    shutil.copyfile(Path(__file__), output / "source_script.py")
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for name in EXPECTED_HASHES:
        shutil.copyfile(ROOT / "scripts" / name, snapshot / name)
    tracked = records[1:]
    final_stats = [row["final_statistics"] for row in tracked]
    flags = Counter(flag for row in tracked for flag in row["failure_flags"])
    stops = Counter(level["stop_reason"] for row in tracked for level in row["levels"])
    iteration_count = 0
    iteration_stops = Counter()
    with (output / "iterations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            iteration_count += 1
            row = json.loads(line)
            if row.get("stop_reason"):
                iteration_stops[row["stop_reason"]] += 1
    trajectory = metrics["trajectory"]
    segments = failure_segments(records)
    summary = {"experiment_id": output.name, "status": metrics["status"],
               "implementation": metrics["implementation"], "device": "CPU",
               "frames_processed": 549, "tracked_pairs": 548,
               "frame_range_inclusive": [0, 548],
               "sensor_time_span_seconds": frames[-1].rgb_timestamp-frames[0].rgb_timestamp,
               "first_rgb_timestamp": frames[0].rgb_timestamp,
               "last_rgb_timestamp": frames[-1].rgb_timestamp,
               "failure_frame_count": metrics["failure_frame_count"],
               "failure_flags_counts": dict(flags), "failure_segments": segments,
               "longest_failure_segment_frames": max((item["frame_count"] for item in segments), default=0),
               "raw_anchored_translation_rmse_m": trajectory["raw_anchored_translation_rmse_m"],
               "se3_aligned_ate_translation_rmse_m": trajectory["se3_aligned_ate_translation_rmse_m"],
               "adjacent_rpe_translation_rmse_m": trajectory["adjacent_rpe_translation_rmse_m"],
               "adjacent_rpe_rotation_rmse_deg": trajectory["adjacent_rpe_rotation_rmse_deg"],
               "final_raw_translation_error_m": trajectory["per_frame_raw_translation_error_m"][-1],
               "max_raw_translation_error_m": trajectory["raw_anchored_translation_max_m"],
               "ate_alignment": trajectory["ate_alignment"], "rpe_delta_frames": 1,
               "final_rank_counts": dict(Counter(str(value.get("rank", "unavailable")) for value in final_stats)),
               "final_correspondences": finite_summary([row["final_counts"]["unique_correspondences"] for row in tracked]),
               "final_plane_rmse_m": finite_summary([value.get("plane_rmse_m") for value in final_stats]),
               "final_column_normalized_hessian_condition": finite_summary([value.get("column_normalized_hessian_condition") for value in final_stats]),
               "level_stop_reason_counts": dict(stops), "iteration_records": iteration_count,
               "iteration_stop_reason_counts": dict(iteration_stops),
               "tracking_loop_wall_seconds": metrics["cpu_loop_seconds"],
               "full_entrypoint_wall_seconds": wall_seconds,
               "timing_note": "Observed wall duration on this host; not process CPU time, isolated performance benchmark or SLAM FPS",
               "preflight_reference_pose_max_absolute_difference": precheck["pose_max_absolute_difference"],
               "configuration": {"pyramid_sizes": configuration["pyramid_sizes"], "icp": configuration["icp"]},
               "source_sha256": hashes,
               "ground_truth_usage": metrics["ground_truth_usage"],
               "limitations": metrics["limitations"]+["Complete fr1/desk sequence only; not evidence of cross-scene generalization"],
               "evidence": {"metrics": "metrics.json", "configuration": "configuration.json",
                            "validation": "validation.json", "preflight": "preflight.json"}}
    icp.write_json(output / "summary.json", summary)
    validation = {"status": "passed", "frames_in_metrics_and_both_trajectories": 549,
                  "all_frame_ids_chronological_and_complete": True, "all_poses_finite": True,
                  "first_pose_exact_identity": True, "rejected_frames_propagate_previous_pose": True,
                  "no_rejected_frames_removed": True, "trajectory_timestamps_match_association": True,
                  "default_parameters_unchanged": True, "frozen_sources_unchanged": True,
                  "preflight_reference_run": precheck["reference_run"],
                  "preflight_pose_max_absolute_difference": precheck["pose_max_absolute_difference"],
                  "evidence_sha256": {name: sha256(output / name) for name in [
                      "metrics.json", "configuration.json", "iterations.jsonl", "loss.csv",
                      "trajectory_estimated.txt", "trajectory_groundtruth_anchored.txt",
                      "trajectory_evaluation.json", "source_script.py", "summary.json",
                      "selected_sensor_metadata.json", "preflight.json", "source_manifest.json"]}}
    icp.write_json(output / "validation.json", validation)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tum-icp-full549-001")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Run name must contain only letters/numbers/-/_")
    output = ROOT / "outputs" / args.run
    if not args.check_only and output.exists():
        raise FileExistsError(f"Never overwrite a previous run: {output}")
    precheck = preflight(args.dataset)
    icp.write_json(ROOT / "manifests/track-rgbd-icp-full-preflight.json", precheck)
    print(json.dumps({"preflight": "passed", "pose_max_abs_difference": precheck["pose_max_absolute_difference"],
                      "candidate_max_abs_difference": precheck["relative_candidate_max_absolute_difference"],
                      "wrapper_sha256": precheck["source_sha256"][Path(__file__).name]}, indent=2), flush=True)
    if args.check_only:
        return
    old_argv, old_clips = sys.argv, dict(icp.CLIPS)
    icp.CLIPS["full549"] = (0, 549)
    sys.argv = [str(ROOT / "scripts/track_rgbd_icp.py"), "--clip", "full549",
                "--run", args.run, "--dataset", str(args.dataset)]
    started = time.perf_counter()
    try:
        icp.main()
    finally:
        sys.argv = old_argv
        icp.CLIPS.clear()
        icp.CLIPS.update(old_clips)
    summary = finalize(output, args.dataset, precheck, time.perf_counter()-started)
    print(json.dumps({key: summary[key] for key in ["experiment_id", "frames_processed", "failure_frame_count",
                      "raw_anchored_translation_rmse_m", "se3_aligned_ate_translation_rmse_m",
                      "adjacent_rpe_translation_rmse_m", "adjacent_rpe_rotation_rmse_deg",
                      "tracking_loop_wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
