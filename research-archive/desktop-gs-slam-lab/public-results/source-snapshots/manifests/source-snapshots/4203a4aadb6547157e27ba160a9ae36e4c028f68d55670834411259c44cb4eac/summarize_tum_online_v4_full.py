"""CPU-only, non-overwriting chunked summaries of the four full V4 controls.

The selected fields follow the existing V4 summarizer. Frame and sensor rows
are split into chronological chunks of at most 100 rows and at most 1 MiB.
All frame IDs and failure flags are retained. Original metrics/checkpoints and
the old summarizer are never modified; no training or new evaluation occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"manifests/tum-online-v4-full-sequence-protocol-001.json"
PROTOCOL_SHA = "ee635a6d65cc02dd26ff16e3f83721afa745a56b9f7a7464453833651a4963af"
MAX_BYTES = 1024*1024
CHUNK_ROWS = 100
METADATA_KEYS = ["experiment_id", "status", "implementation_version", "configuration", "association",
    "selected_frame_ids", "scope", "localization_source", "ground_truth_usage", "world_gauge",
    "camera_canonicalization", "camera", "script_sha256", "loader_sha256", "pose_helper_sha256",
    "dataset_attribution", "pose_prior", "pose_prior_helper_sha256", "v3_source_sha256",
    "first_tracking_context", "created_utc", "frames_processed", "failure_frame_count", "icp_failed_frame_count",
    "initial_point_count", "final_point_count", "pose_gradient_checks", "online_loop_seconds_excludes_warmup", "limitations"]
ICP_KEYS = {"enabled", "mode", "accepted", "failure_flags", "propagated_failure_flags",
    "source_current_frame_id", "target_previous_frame_id", "relative_transform_direction", "registration_seconds",
    "total_cpu_seconds", "fallback", "gs_accepted_delta_from_icp_prefix", "gs_accepted_delta_from_initializer",
    "gs_initial_sensor_loss", "gs_selected_sensor_loss_before_mapping", "gs_accepted_sensor_loss_after_mapping",
    "gs_pose_refinement_applied", "gs_initial_tracking_loss", "gs_selected_tracking_loss_before_mapping",
    "gs_selected_prior_objective_before_mapping", "gs_selected_combined_objective_before_mapping"}
FRAME_KEYS = {"local_index", "frame_id", "rgb_timestamp", "depth_timestamp", "rgb_depth_abs_dt_s",
    "initialization", "tracking_mode", "gs_pose_refinement_applied", "gs_refinement_steps_requested",
    "gs_refinement_steps_completed", "map_coverage_warning_flags", "initial_loss", "optimized_loss",
    "optimized_loss_before_mapping", "accepted_pose_loss_after_mapping", "best_step_selected_by_tracking_objective_only",
    "initial_sensor_loss", "selected_sensor_loss_before_mapping", "tracking_objective_scope", "initial_to_candidate_delta",
    "previous_to_candidate_delta", "pose_rejected", "failure_flags", "new_points_added", "point_count",
    "tracking_seconds", "frame_total_seconds"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")


def validate_original(original, row, locked_sources):
    count = row["configuration"]["frames"]
    if (original["configuration"] != row["configuration"] or
            original["frames_processed"] != count or
            original["selected_frame_ids"] != list(range(count)) or
            len(original["frame_metrics"]) != len(original["selected_sensor_metadata"]) or
            len(original["frame_metrics"]) != count or
            [frame["frame_id"] for frame in original["frame_metrics"]] != list(range(count)) or
            [sensor["frame_id"] for sensor in original["selected_sensor_metadata"]] != list(range(count))):
        raise ValueError("Incomplete or nonchronological full protocol frames/configuration")
    fields = {"script_sha256": "scripts/train_tum_online_gs_v4.py", "loader_sha256": "scripts/tum_rgbd.py",
              "pose_helper_sha256": "scripts/track_fixed_map.py", "pose_prior_helper_sha256": "scripts/geometric_pose_prior.py"}
    for field, relative in fields.items():
        if original[field] != locked_sources[relative] or sha(ROOT/relative) != locked_sources[relative]:
            raise ValueError("Unknown completed artifact/current source hash: "+field)
    if (original["icp_initializer"]["helper_sha256"] != locked_sources["scripts/track_rgbd_icp.py"] or
            sha(ROOT/"scripts/track_rgbd_icp.py") != locked_sources["scripts/track_rgbd_icp.py"]):
        raise ValueError("Unknown completed ICP helper hash")


def build_payloads(run, original, metrics_bytes):
    if original["experiment_id"] != run or original["implementation_version"] != 4 or not original["status"].startswith("completed"):
        raise ValueError("Only matching completed V4 experiments may be summarized")
    count = original["frames_processed"]
    rows = []
    for frame in original["frame_metrics"]:
        row = {key: value for key, value in frame.items() if key in FRAME_KEYS}
        row["icp"] = {key: value for key, value in (frame.get("icp") or {}).items() if key in ICP_KEYS}
        rows.append(row)
    sensor = original["selected_sensor_metadata"]
    if len(rows) != len(sensor) or len(rows) != count or [row["frame_id"] for row in rows] != original["selected_frame_ids"]:
        raise ValueError("A frame/sensor row or failed observation was dropped")
    if sum(bool(row["failure_flags"]) for row in rows) != original["failure_frame_count"]:
        raise ValueError("Failure count differs from original rows")
    payloads = {}
    result = {key: original[key] for key in METADATA_KEYS}
    result["icp_initializer"] = {key: original["icp_initializer"][key] for key in
        ("enabled", "configuration", "helper_sha256", "prefix_policy", "failure_policy")}
    for kind, values in (("frame_metrics", rows), ("selected_sensor_metadata", sensor)):
        descriptors = []
        for start in range(0, count, CHUNK_ROWS):
            end = min(count, start+CHUNK_ROWS)
            name = f"{kind}_{start:04d}_{end-1:04d}.json"
            data = encode({"experiment_id": run, "kind": kind, "start_local_index": start,
                "stop_local_index_exclusive": end, "row_count": end-start, "rows": values[start:end]})
            if len(data) > MAX_BYTES:
                raise ValueError("Chunk exceeds byte ceiling; retain originals without partial summary: "+name)
            payloads[name] = data
            descriptors.append({"path": "summary_chunks/"+name, "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data), "start_local_index": start, "stop_local_index_exclusive": end,
                "row_count": end-start, "first_frame_id": values[start]["frame_id"], "last_frame_id": values[end-1]["frame_id"]})
        result[kind+"_chunks"] = descriptors
    excluded = {"per_view", "adjacent_rpe", "anchored_truth_c2w", "aligned_estimated_c2w"}
    for name in ("trajectory", "icp_prefix_trajectory_evaluation"):
        result[name] = {key: value for key, value in original[name].items() if key not in excluded}
    result["mapping_events"] = original["mapping_events"]
    result["all_frame_status"] = [{key: row[key] for key in ("local_index", "frame_id", "failure_flags", "point_count")}
                                  for row in rows]
    result["summary_provenance"] = {
        "format": "v4_full_chunked_v1", "full_metrics_filename": "metrics.json",
        "full_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(), "full_metrics_bytes": len(metrics_bytes),
        "summarizer_sha256": sha(Path(__file__)), "field_selection_reference": "scripts/summarize_tum_online_v4.py",
        "field_selection_reference_sha256": sha(ROOT/"scripts/summarize_tum_online_v4.py"),
        "all_frames_sensors_and_failure_flags_preserved_across_chronological_chunks": True,
        "all_chunks_are_indexed_by_relative_path_bytecount_and_SHA256": True,
        "row_limit_per_chunk": CHUNK_ROWS, "single_JSON_byte_limit": MAX_BYTES,
        "omitted": "Detailed ICP iterations/geometry and camera matrices remain in the complete original local metrics/checkpoints",
        "no_training_or_GT_pose_evaluation_performed_by_summarizer": True}
    summary_bytes = encode(result)
    if len(summary_bytes) > MAX_BYTES:
        raise ValueError("Primary summary exceeds byte ceiling; retain originals")
    return result, summary_bytes, payloads


def summarize(run):
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Unexpected full-sequence protocol")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    allowed = {row["run"]: row for row in protocol["new_runs"]}
    if run not in allowed:
        raise ValueError("Run is not one of the four reviewed full controls")
    folder = ROOT/"outputs"/run
    full_bytes = (folder/"metrics.json").read_bytes()
    original = json.loads(full_bytes)
    validate_original(original, allowed[run], protocol["source_sha256"])
    result, data, chunks = build_payloads(run, original, full_bytes)
    destination = folder/"summary_chunks"
    if (folder/"summary.json").exists() or destination.exists():
        raise FileExistsError("Refuse to overwrite summary or chunk evidence")
    destination.mkdir(exist_ok=False)
    for name, payload in chunks.items():
        with (destination/name).open("xb") as stream:
            stream.write(payload)
    with (folder/"summary.json").open("xb") as stream:
        stream.write(data)
    if sha(folder/"metrics.json") != result["summary_provenance"]["full_metrics_sha256"]:
        raise RuntimeError("Original metrics changed during export")
    return {"run": run, "status": result["status"], "frames": result["frames_processed"],
        "failure_frame_count": result["failure_frame_count"], "summary_bytes": len(data),
        "chunk_count": len(chunks), "max_chunk_bytes": max(map(len, chunks.values())),
        "summary_sha256": sha(folder/"summary.json"), "metrics_sha256": result["summary_provenance"]["full_metrics_sha256"],
        "summarizer_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(run) for run in args.runs], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
