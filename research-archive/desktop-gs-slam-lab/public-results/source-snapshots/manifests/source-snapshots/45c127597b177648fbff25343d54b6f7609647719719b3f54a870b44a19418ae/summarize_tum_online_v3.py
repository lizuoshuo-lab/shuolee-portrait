"""Export small, traceable V3 summaries without copying iterative ICP logs.

The full metrics/checkpoints remain in the lab. Summaries select original fields,
retain every frame and failure flag, and identify the complete metrics by SHA256.
No training, pose selection, loss modification or GT alignment is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["tum-online-v3-icpprefix-map20-001", "tum-online-v3-icpprefix-refine20-001",
                "tum-online-v3-icpprefix-map100-001", "tum-online-v3-icpprefix-refine100-001"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def summarize(run):
    directory = ROOT / "outputs" / run
    full_bytes = (directory / "metrics.json").read_bytes()
    original = json.loads(full_bytes)
    if original["experiment_id"] != run or original["implementation_version"] != 3:
        raise ValueError("Mismatched experiment/version")
    if not original["status"].startswith("completed"):
        raise ValueError("Only finished runs may be summarized")
    keys = ["experiment_id", "status", "implementation_version", "configuration", "association",
            "selected_frame_ids", "selected_sensor_metadata", "scope", "localization_source",
            "ground_truth_usage", "world_gauge", "camera_canonicalization", "camera",
            "script_sha256", "loader_sha256", "pose_helper_sha256", "dataset_attribution",
            "created_utc", "frames_processed", "failure_frame_count", "icp_failed_frame_count",
            "initial_point_count", "final_point_count", "pose_gradient_checks",
            "online_loop_seconds_excludes_warmup", "limitations"]
    result = {key: original[key] for key in keys}
    result["icp_initializer"] = {key: original["icp_initializer"][key]
                                 for key in ("enabled", "configuration", "helper_sha256", "prefix_policy", "failure_policy")}
    # Large lists of full matrices and GT per frame remain in the original files.
    excluded_trajectory = {"per_view", "adjacent_rpe", "anchored_truth_c2w", "aligned_estimated_c2w"}
    for name in ("trajectory", "icp_prefix_trajectory_evaluation"):
        result[name] = {key: value for key, value in original[name].items() if key not in excluded_trajectory}
    result["frame_metrics"] = []
    icp_keys = {"enabled", "mode", "accepted", "failure_flags", "propagated_failure_flags",
                "source_current_frame_id", "target_previous_frame_id", "relative_transform_direction",
                "registration_seconds", "total_cpu_seconds", "fallback",
                "gs_accepted_delta_from_icp_prefix", "gs_accepted_delta_from_initializer",
                "gs_initial_sensor_loss", "gs_selected_sensor_loss_before_mapping",
                "gs_accepted_sensor_loss_after_mapping", "gs_pose_refinement_applied"}
    frame_keys = {"local_index", "frame_id", "rgb_timestamp", "depth_timestamp", "rgb_depth_abs_dt_s",
                  "initialization", "tracking_mode", "gs_pose_refinement_applied",
                  "gs_refinement_steps_requested", "gs_refinement_steps_completed",
                  "map_coverage_warning_flags", "initial_loss", "optimized_loss", "optimized_loss_before_mapping",
                  "accepted_pose_loss_after_mapping", "best_step_selected_by_sensor_loss_only",
                  "initial_to_candidate_delta", "previous_to_candidate_delta", "pose_rejected", "failure_flags",
                  "new_points_added", "point_count", "tracking_seconds", "frame_total_seconds"}
    for frame in original["frame_metrics"]:
        copy = {key: value for key, value in frame.items() if key in frame_keys}
        copy["icp"] = {key: value for key, value in (frame.get("icp") or {}).items() if key in icp_keys}
        result["frame_metrics"].append(copy)
    result["mapping_events"] = original["mapping_events"]
    result["summary_provenance"] = {
        "full_metrics_filename": "metrics.json", "full_metrics_sha256": hashlib.sha256(full_bytes).hexdigest(),
        "full_metrics_bytes": len(full_bytes), "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "omitted": "detailed ICP iteration/geometry records and per-frame camera matrices; complete originals retained locally",
        "all_frames_and_failure_flags_preserved": True,
    }
    if len(result["frame_metrics"]) != result["frames_processed"]:
        raise ValueError("A failed or accepted frame was dropped")
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    if len(encoded) > 1024*1024:
        raise ValueError("Summary exceeds reviewed website limit")
    (directory / "summary.json").write_bytes(encoded)
    trajectory = result["trajectory"]
    return {"run": run, "status": result["status"], "frames": result["frames_processed"],
            "failures": result["failure_frame_count"], "raw_m": trajectory["raw_anchored_translation_rmse_m"],
            "ate_m": trajectory["ate_rmse_m_se3_no_scale"], "summary_bytes": len(encoded)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    args = parser.parse_args()
    if any(Path(run).name != run or run in ("", ".", "..") for run in args.runs):
        raise ValueError("Run names must be single directory names")
    print(json.dumps([summarize(run) for run in args.runs], indent=2))


if __name__ == "__main__":
    main()
