"""CPU artifact checks for the four protocol-locked full-sequence V4 runs.

Uses the existing separate NumPy trajectory auditor and generic artifact checks.
This is a serialization/causality/objective check, not an independent rerun of
tracking or proof of SLAM accuracy. No renderer, training entrypoint or CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import audit_tum_xyz_v4_artifacts as shared
from tum_rgbd import Frame, load_frames, scaled_intrinsics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"manifests/tum-online-v4-full-sequence-protocol-001.json"
PROTOCOL_SHA = "ee635a6d65cc02dd26ff16e3f83721afa745a56b9f7a7464453833651a4963af"
SHARED_SHA = "88ff3a75db6d9afc5b90756e92e5572c1549ebd3bb346612cd6faf4537b53d6e"
require, read_json, sha, no_gt = shared.require, shared.read_json, shared.sha256, shared.no_gt_pose_fields


def validate_context(folder, metrics, checkpoint):
    context_path, json_path = folder/"first_tracking_context.pt", folder/"first_tracking_context.json"
    context = torch.load(context_path, map_location="cpu", weights_only=True)
    meta = read_json(json_path)
    config, rows = metrics["configuration"], metrics["frame_metrics"]
    current = meta["sensor_metadata"]["local_index"]
    first_id, current_id = rows[0]["frame_id"], rows[current]["frame_id"]
    require(meta == metrics["first_tracking_context"] == checkpoint["first_tracking_context"], "context metadata linkage")
    require(meta["checkpoint"] == context_path.name and meta["checkpoint_sha256"] == sha(context_path), "context checkpoint hash")
    require(context["configuration"] == meta["configuration"] == config, "context configuration")
    require(context["implementation_version"] == meta["implementation_version"] == 4, "context version")
    require(not no_gt(context) and not no_gt(meta), "GT payload in context")
    for key in ("ground_truth_included", "future_sensor_observations_included", "future_maps_or_trajectories_included"):
        require(context[key] is meta[key] is False, "context future/GT flag")
    allowed = {"status", "implementation_version", "configuration", "ground_truth_included", "future_sensor_observations_included",
        "future_maps_or_trajectories_included", "source", "parameters", "source_frame_ids_per_gaussian", "map_role", "image",
        "depth_m", "valid_depth_mask", "K", "initial_w2c", "prior_reference_w2c", "icp_prefix_c2w_float64",
        "sensor_metadata", "initial_tracking_loss", "tracking_objective"}
    require(set(context) == allowed, "unexpected context payload")
    require(meta["saved_before_first_pose_gradient_probe"] is True and meta["tracking_optimizer_steps_executed_before_save"] == 0, "context ordering")
    require(current == 1 and first_id == 0 and current_id == 1, "full frame0 gauge and first frame1 tracking context")
    require(context["sensor_metadata"] == meta["sensor_metadata"] and context["sensor_metadata"]["frame_id"] == current_id, "context sensor frame")
    require(set(context["source_frame_ids_per_gaussian"]) == {first_id} and meta["map_source_frame_ids"] == [first_id], "warmup only first sensor")
    require(meta["point_count"] == metrics["initial_point_count"] == len(context["source_frame_ids_per_gaussian"]), "dynamic initial count")
    parameters = context["parameters"]
    require(set(parameters) == set(checkpoint["parameters"]), "context parameter fields")
    require(all(not value.requires_grad and value.device.type == "cpu" and torch.isfinite(value).all() and len(value) == meta["point_count"] for value in parameters.values()), "context parameters frozen/finite/CPU")
    require(context["image"].shape == (config["height"], config["width"], 3), "current RGB shape")
    require(context["depth_m"].shape == context["valid_depth_mask"].shape == (config["height"], config["width"]), "current depth shape")
    require(context["valid_depth_mask"].dtype == torch.bool and int(context["valid_depth_mask"].sum()) == meta["valid_depth_pixels"], "mask count/type")
    require(list(context["image"].shape) == meta["image_shape"] and list(context["depth_m"].shape) == meta["depth_shape"], "shape metadata")
    require(context["initial_tracking_loss"] == meta["initial_tracking_loss"] == rows[current]["initial_loss"], "initial loss linkage")
    for key in ("initial_w2c", "prior_reference_w2c"):
        require(np.array_equal(context[key].numpy(), np.array(meta[key])) and np.array_equal(context[key].numpy(), np.array(rows[current][key])), "context camera linkage")
    require(torch.equal(context["initial_w2c"], context["prior_reference_w2c"]), "context reference must be initial")
    require(context["icp_prefix_c2w_float64"].dtype == torch.float64 and torch.equal(context["icp_prefix_c2w_float64"], checkpoint["icp_prefix_c2w_float64"][current]), "context prefix linkage")
    require(np.array_equal(context["icp_prefix_c2w_float64"].numpy(), np.array(meta["icp_prefix_c2w_float64"])), "prefix metadata")
    require(np.array_equal(context["K"].numpy(), np.array(meta["K"])) and np.allclose(context["K"].numpy(), scaled_intrinsics(config["width"], config["height"]), rtol=0, atol=1e-5), "context K")
    source_paths = {"v3_script_sha256": "train_tum_online_gs.py", "script_sha256": "train_tum_online_gs_v4.py",
        "pose_prior_helper_sha256": "geometric_pose_prior.py", "pose_helper_sha256": "track_fixed_map.py",
        "loader_sha256": "tum_rgbd.py", "icp_helper_sha256": "track_rgbd_icp.py"}
    require(context["source"] == meta["source"], "context source metadata")
    for field, name in source_paths.items():
        require(context["source"][field] == sha(ROOT/"scripts"/name), "context source SHA "+name)
    def unknown_gt_frame(sensor):
        return Frame(sensor["frame_id"], -1, -1, -1, sensor["rgb_timestamp"], sensor["depth_timestamp"], float("nan"),
                     sensor["rgb_path"], sensor["depth_path"], np.full((4, 4), np.nan), float("nan"))
    sensor = context["sensor_metadata"]
    images, depths, masks = load_frames(Path(config["dataset"]), [unknown_gt_frame(sensor)], config["width"], config["height"], config["min_depth"], config["max_depth"])
    require(np.array_equal(images[0], context["image"].numpy()) and np.array_equal(depths[0], context["depth_m"].numpy()) and np.array_equal(masks[0], context["valid_depth_mask"].numpy()), "context current raw PNG parity with NaN GT")
    first = metrics["selected_sensor_metadata"][0]
    _, _, first_mask = load_frames(Path(config["dataset"]), [unknown_gt_frame(first)], config["width"], config["height"], config["min_depth"], config["max_depth"])
    eligible = int(first_mask[0, ::config["initial_pixel_stride"], ::config["initial_pixel_stride"]].sum())
    require(meta["point_count"] == min(config["initial_points"], eligible), "initial count differs from first-depth mask, not a hardcoded desk/xyz count")
    return {"pt_sha256": sha(context_path), "json_sha256": sha(json_path), "current_sensor_frame_id": current_id,
        "warmup_source_frame_ids": [first_id], "warmup_point_count": meta["point_count"], "first_depth_eligible_pixels": eligible,
        "current_sensor_tensors_reloaded_exact_with_NaN_GT": True, "all_future_GT_flags_false": True,
        "first_tracking_before_probe_optimizer_steps": 0, "context_payload_schema_current_only": True}


def check_chunks(folder, metrics):
    summary = read_json(folder/"summary.json")
    provenance = summary["summary_provenance"]
    require(provenance["full_metrics_sha256"] == sha(folder/"metrics.json"), "summary original hash")
    require(provenance["summarizer_sha256"] == sha(ROOT/"scripts/summarize_tum_online_v4_full.py"), "summary source hash")
    require(summary["configuration"] == metrics["configuration"] and summary["failure_frame_count"] == metrics["failure_frame_count"], "summary configuration/flags")
    totals, hashes = {}, {"summary.json": sha(folder/"summary.json")}
    for kind in ("frame_metrics", "selected_sensor_metadata"):
        rows = []
        for item in summary[kind+"_chunks"]:
            target = (folder/item["path"]).resolve()
            require(target.is_relative_to((folder/"summary_chunks").resolve()), "unsafe chunk path")
            require(sha(target) == item["sha256"] and target.stat().st_size == item["bytes"] <= 1024*1024, "chunk bytes/SHA")
            chunk = read_json(target)
            require(chunk["experiment_id"] == metrics["experiment_id"] and chunk["kind"] == kind and chunk["start_local_index"] == len(rows), "chunk IDs/order")
            require(chunk["row_count"] == item["row_count"] == len(chunk["rows"]) and chunk["stop_local_index_exclusive"] == len(rows)+len(chunk["rows"]), "chunk counts/gaps")
            rows.extend(chunk["rows"]); hashes[item["path"]] = sha(target)
        expected = metrics[kind]
        require(len(rows) == len(expected) == metrics["frames_processed"] and [r["frame_id"] for r in rows] == [r["frame_id"] for r in expected], "full summary retained every sensor/frame")
        if kind == "frame_metrics":
            for actual, original in zip(rows, expected):
                for key, value in actual.items():
                    if key == "icp":
                        require(all(original["icp"][field] == item for field, item in value.items()), "summary ICP scalar mismatch")
                    else:
                        require(original[key] == value, "summary original field mismatch")
                require(actual["failure_flags"] == original["failure_flags"], "summary failed flags lost")
        else:
            require(rows == expected, "summary sensor rows differ")
        totals[kind] = len(rows)
    require(summary["mapping_events"] == metrics["mapping_events"], "summary mapping events changed")
    require([r["failure_flags"] for r in summary["all_frame_status"]] == [r["failure_flags"] for r in metrics["frame_metrics"]], "main summary failure list mismatch")
    return {"status": "passed", "reassembled_row_counts": totals, "input_sha256": hashes,
            "all_chunked_fields_identical_to_original_subset": True, "all_failure_flags_retained": True}


def check_run(run):
    require(sha(PROTOCOL) == PROTOCOL_SHA, "unknown full protocol SHA")
    require(sha(ROOT/"scripts/audit_tum_xyz_v4_artifacts.py") == SHARED_SHA, "shared audit source changed")
    protocol = read_json(PROTOCOL)
    candidates = {row["run"]: row for row in protocol["new_runs"]}
    require(run in candidates, "run outside full protocol")
    row = candidates[run]
    folder, baseline = ROOT/row["output_relative_path"], row["full_ICP_reference"]
    metrics = read_json(folder/"metrics.json")
    config = metrics["configuration"]
    require(config == row["configuration"], "36field protocol configuration mismatch")
    require(config["start_frame"] == 0 and config["frame_stride"] == 1 and metrics["selected_frame_ids"] == list(range(config["frames"])), "full frame coverage/gauge")
    for relative, expected in protocol["source_sha256"].items():
        require(sha(ROOT/relative) == expected, "frozen training/helper source changed")
    baseline_rows = read_json(ROOT/"outputs"/baseline/"selected_sensor_metadata.json")
    require(metrics["selected_sensor_metadata"] == baseline_rows["selected_frames"], "full CPU baseline sensor metadata differs")
    result = shared.audit_base_run(run, baseline)
    checkpoint = torch.load(folder/"online_map.pt", map_location="cpu", weights_only=True)
    rows = metrics["frame_metrics"]
    weight, sigma_t, sigma_r = config["pose_prior_weight"], config["pose_prior_translation_sigma_m"], math.radians(config["pose_prior_rotation_sigma_deg"])
    maxima = {"sensor_channels": 0., "combined_sum": 0., "selected_numpy_prior": 0., "raw_numpy_prior": 0., "initializer_prefix_camera": 0.}
    checked = 0
    for frame in rows[1:]:
        reference = np.array(frame["prior_reference_w2c"])
        require(np.array_equal(reference, np.array(frame["initial_w2c"])), "fixed prior reference moved")
        if frame["icp"]["accepted"]:
            ref_c2w = np.eye(4); ref_c2w[:3, :3] = reference[:3, :3].T; ref_c2w[:3, 3] = -reference[:3, :3].T@reference[:3, 3]
            delta = float(np.max(np.abs(ref_c2w-checkpoint["icp_prefix_c2w_float64"][frame["local_index"]].numpy())))
            require(delta < 2e-6, "accepted ICP-prefix initializer differs")
            maxima["initializer_prefix_camera"] = max(maxima["initializer_prefix_camera"], delta)
        for field, pose_field, max_key in (("initial_loss", "initial_w2c", None), ("optimized_loss_before_mapping", "candidate_optimized_w2c", "selected_numpy_prior"), ("selected_raw_candidate_loss_before_projection", "selected_raw_w2c_before_projection", "raw_numpy_prior")):
            loss = frame[field]
            errors = shared.check_loss(loss, config, True)
            maxima["sensor_channels"] = max(maxima["sensor_channels"], errors[0]); maxima["combined_sum"] = max(maxima["combined_sum"], errors[1])
            expected = shared.numpy_prior(frame[pose_field], reference, weight, sigma_t, sigma_r)
            discrepancy = abs(expected-loss["prior_objective"])
            require(discrepancy <= 1e-6+1e-5*abs(expected), "persisted NumPy/GPU prior mismatch")
            if max_key: maxima[max_key] = max(maxima[max_key], discrepancy)
            checked += 1
        require(frame["initial_loss"]["prior_objective"] == 0. and frame["optimized_loss_before_mapping"]["objective"] <= frame["initial_loss"]["objective"]+1e-8, "initial prior/best combined selection")
        require(0 <= frame["best_step_selected_by_tracking_objective_only"] <= frame["gs_refinement_steps_requested"], "selected best step")
        for loss_name, sensor_name in (("initial_loss", "initial_sensor_loss"), ("optimized_loss_before_mapping", "selected_sensor_loss_before_mapping")):
            loss = frame[loss_name]
            sensor = {key: value for key, value in loss.items() if key not in ("sensor_objective", "prior_objective")}
            sensor["objective"] = loss["sensor_objective"]
            require(sensor == frame[sensor_name], "sensor scalar extraction")
        shared.check_loss(frame["accepted_pose_loss_after_mapping"], config, False)
        icp = frame["icp"]
        require(icp["gs_initial_tracking_loss"] == frame["initial_loss"] and icp["gs_selected_tracking_loss_before_mapping"] == frame["optimized_loss_before_mapping"], "ICP combined loss linkage")
        require(icp["gs_selected_prior_objective_before_mapping"] == frame["optimized_loss_before_mapping"]["prior_objective"] and icp["gs_selected_combined_objective_before_mapping"] == frame["optimized_loss_before_mapping"]["objective"], "ICP prior/combined linkage")
        require(icp["gs_initial_sensor_loss"] == frame["initial_sensor_loss"] and icp["gs_selected_sensor_loss_before_mapping"] == frame["selected_sensor_loss_before_mapping"] and icp["gs_accepted_sensor_loss_after_mapping"] == frame["accepted_pose_loss_after_mapping"], "ICP sensor linkage")
        require(np.array_equal(np.array(icp["gs_selected_w2c"]), np.array(frame["estimated_w2c"])), "ICP accepted pose linkage")
        if frame["pose_rejected"]:
            require(frame["new_points_added"] == 0 and frame["estimated_w2c"] == rows[frame["local_index"]-1]["estimated_w2c"], "rejected pose not frozen")
    phases = Counter()
    with (folder/"loss.csv").open(newline="", encoding="utf-8-sig") as stream:
        for item in csv.DictReader(stream):
            phase = item["phase"]; phases[phase] += 1
            current, observed = int(item["current_local_index"]), int(item["observation_local_index"])
            require(0 <= observed <= current < config["frames"], "CSV future/invalid observation")
            values = {key: float(item[key]) for key in ("objective", "rgb_mse", "depth_mae_m", "alpha_loss")}
            if phase in ("tracking", "icp_failure_sensor_evaluation"):
                require(current == observed and current > 0 and item["sensor_objective"] != "" and item["prior_objective"] != "", "tracking CSV scope")
                values.update({key: float(item[key]) for key in ("sensor_objective", "prior_objective")})
                shared.check_loss(values, config, True)
            else:
                require(phase in ("warmup_mapping", "online_mapping") and item["sensor_objective"] == item["prior_objective"] == "", "map CSV prior leakage")
                shared.check_loss(values, config, False)
    require(phases["warmup_mapping"] > 0 and phases["tracking"] > 0, "CSV missing phases")
    result.update({"check_scope": "CPU full-length artifact/objective checks; no training/rendering rerun or claimed independent optimization replication",
        "full_checker_sha256": sha(Path(__file__)), "shared_audit_sha256": SHARED_SHA, "protocol_sha256": PROTOCOL_SHA,
        "configuration_36fields_exact": True, "entire_sensor_metadata_matches_full_CPU_baseline": True,
        "full_length_prior_checks_count": checked, "prior_consistency_max_abs": maxima,
        "first_tracking_context_full": validate_context(folder, metrics, checkpoint),
        "summary_chunks": check_chunks(folder, metrics), "CSV_rows_by_phase": dict(phases),
        "map_CSV_prior_columns_empty_tracking_columns_explicit": True,
        "GPU_calls": 0})
    require(not torch.cuda.is_initialized(), "CPU checker unexpectedly initialized CUDA")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        result = {"status": "passed", "CPU_checks": shared.self_check(), "checker_sha256": sha(Path(__file__)),
                  "shared_auditor_sha256": sha(ROOT/"scripts/audit_tum_xyz_v4_artifacts.py"), "GPU_calls": 0}
        require(result["shared_auditor_sha256"] == SHARED_SHA and not torch.cuda.is_initialized(), "locked CPU helper")
    elif args.run:
        result = check_run(args.run)
    else:
        parser.error("--run or --self-check required")
    output = args.output or (ROOT/"outputs"/args.run/"cpu_full_v4_artifact_checks.json" if args.run else None)
    if output:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in result if key in ("status", "run", "frame_count", "failure_frame_count", "metrics", "checker_sha256", "full_checker_sha256", "GPU_calls")}, indent=2))


if __name__ == "__main__":
    main()
