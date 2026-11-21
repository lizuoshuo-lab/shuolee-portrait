"""CPU audit of frozen-parameter V4 transfer artifacts on explicit TUM fr1_xyz.

This never imports the training script or uses CUDA. Online trajectory metrics
are recomputed with the separate NumPy auditor. ICP and this audit share that
metric implementation, so ICP comparison is artifact/serialization consistency,
not a second independent ATE/RPE formula.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_tum_trajectory import (audit_pose_arrays, parse_trajectory_text,
                                  validate_time_pairing)

ROOT = Path(__file__).resolve().parents[1]
LOCKED_SOURCE = "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f"
LOCKED_V3 = "45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff"
LOCKED_PRIOR = "d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b"
METRIC_PAIRS = {
    "raw_anchored_translation_rmse_m": "raw_anchored_translation_rmse_m",
    "se3_aligned_ate_translation_rmse_m": "ate_rmse_m_se3_no_scale",
    "adjacent_rpe_translation_rmse_m": "adjacent_rpe_translation_rmse_m",
    "adjacent_rpe_rotation_rmse_deg": "adjacent_rpe_rotation_rmse_degrees",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_gt_pose_fields(value, path=""):
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            here, name = path+"/"+str(key), str(key).lower()
            if name == "ground_truth_included":
                require(item is False, here+" must explicitly be false")
            elif name == "max_gt_dt":
                require(isinstance(item, (float, int)), here+" must be threshold metadata")
            elif "ground_truth" in name or name.startswith("gt_") or "_gt_" in name or name == "evaluation_only_gt_c2w":
                findings.append(here)
            findings.extend(no_gt_pose_fields(item, here))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(no_gt_pose_fields(item, path+"/"+str(index)))
    return findings


def camera_structure(c2w, w2c):
    require(c2w.shape == w2c.shape and c2w.ndim == 3 and c2w.shape[1:] == (4, 4), "camera shape/count")
    require(np.isfinite(c2w).all() and np.isfinite(w2c).all(), "all cameras must be finite")
    bottom = float(max(np.max(np.abs(c2w[:, 3]-[0, 0, 0, 1])), np.max(np.abs(w2c[:, 3]-[0, 0, 0, 1]))))
    orth = float(max(np.max(np.linalg.norm(p[:, :3, :3].transpose(0, 2, 1)@p[:, :3, :3]-np.eye(3), axis=(1, 2))) for p in (c2w, w2c)))
    det = float(max(np.max(np.abs(np.linalg.det(p[:, :3, :3])-1)) for p in (c2w, w2c)))
    inverse = float(max(np.max(np.abs(c2w@w2c-np.eye(4))), np.max(np.abs(w2c@c2w-np.eye(4)))))
    require(bottom == 0 and orth < 1e-6 and det < 1e-6 and inverse < 2e-6, "noncanonical or inconsistent rigid cameras")
    require(np.array_equal(c2w[0], np.eye(4)) and np.array_equal(w2c[0], np.eye(4)), "first camera identity")
    return {"bottom_row_max_error": bottom, "rotation_orthogonality_frobenius_max": orth,
            "rotation_determinant_error_max": det, "bidirectional_inverse_max_error": inverse}


def check_source_causality(source, frame_ids, rows, events, initial_count):
    require(np.asarray(source).dtype.kind in "iu", "source IDs must be integers")
    require(set(source).issubset(frame_ids), "Gaussian source ID outside stored sensor frames")
    expected = Counter({frame_ids[0]: initial_count})
    for event in events:
        index = event["current_local_index"]
        require(all(0 <= i <= index for i in event["window_local_indices"]), "future observation in mapping window")
        require(event["window_frame_ids"] == [frame_ids[i] for i in event["window_local_indices"]], "window frame labels")
        if event.get("new_points_added", 0):
            expected[frame_ids[index]] += event["new_points_added"]
            require(event["point_addition_camera_source"] == "current accepted estimated pose", "point camera must be estimated")
    require(Counter(source) == expected, "per-Gaussian source counts differ from insertion events")
    for row in rows:
        require(sum(count for frame, count in expected.items() if frame <= row["frame_id"]) == row["point_count"], "per-frame point counts contain future or missing insertions")
    return dict(sorted(expected.items()))


def check_tracking_mode(mode, rows, probe):
    if mode == "initializer-only":
        require(probe["status"] == "not_applied", "initializer-only must not run GS pose probe")
        for row in rows[1:]:
            require(not row["gs_pose_refinement_applied"] and row["gs_refinement_steps_requested"] == row["gs_refinement_steps_completed"] == 0, "initializer-only applied GS pose updates")
            require(row["best_step_selected_by_tracking_objective_only"] == 0, "initializer-only selected optimization step")
    else:
        require(mode == "optimize", "unknown tracking mode")
        for row in rows[1:]:
            enabled = row["icp"]["accepted"]
            require(row["gs_pose_refinement_applied"] is enabled, "refinement on rejected ICP or skipped accepted initializer")
            require(0 <= row["gs_refinement_steps_completed"] <= row["gs_refinement_steps_requested"], "tracking step counts")
            if not enabled:
                require(row["gs_refinement_steps_requested"] == 0, "rejected ICP requested refinement")
        require(probe["status"] in ("passed", "failed", "not_applied"), "invalid gradient probe status")
        if probe["status"] == "failed":
            require(any("pose_gradient_finite_difference_check_failed" in row["failure_flags"] for row in rows), "failed gradient probe flag was lost")
    return {"mode": mode, "gradient_probe_status": probe["status"],
            "total_pose_steps_completed": sum(row.get("gs_refinement_steps_completed", 0) for row in rows)}


def metric_differences(computed, original):
    differences = {key: abs(computed[key]-original[alias if alias in original else key]) for key, alias in METRIC_PAIRS.items()}
    require(max(value for key, value in differences.items() if key.endswith("_m")) < 1e-6, "translation metric disagreement")
    require(differences["adjacent_rpe_rotation_rmse_deg"] < 1e-5, "rotation metric disagreement")
    return differences


def structural_self_check():
    poses = np.tile(np.eye(4), (2, 1, 1)); poses[1, 0, 3] = .2
    inverses = poses.copy(); inverses[1, 0, 3] = -.2
    camera_structure(poses, inverses)
    negatives = {}
    for name, mutation in (("bottom_drift", lambda p: p.__setitem__((1, 3, 0), 1e-5)),
                           ("non_SO3", lambda p: p.__setitem__((1, 0, 0), 1.01)),
                           ("incorrect_inverse", lambda p: p.__setitem__((1, 0, 3), .3))):
        broken = poses.copy(); mutation(broken)
        try:
            camera_structure(broken, inverses)
        except AssertionError:
            negatives[name] = True
        else:
            raise AssertionError("negative camera control passed: "+name)
    rows = [{"frame_id": 30, "point_count": 2}, {"frame_id": 31, "point_count": 3}]
    events = [{"current_local_index": 1, "window_local_indices": [0, 1], "window_frame_ids": [30, 31], "new_points_added": 1, "point_addition_camera_source": "current accepted estimated pose"}]
    check_source_causality([30, 30, 31], [30, 31], rows, events, 2)
    try:
        check_source_causality([30, 30, 32], [30, 31], rows, events, 2)
    except AssertionError:
        negatives["future_source_id"] = True
    else:
        raise AssertionError("future source control passed")
    probe = {"status": "not_applied"}
    sample = {"gs_pose_refinement_applied": False, "gs_refinement_steps_requested": 0, "gs_refinement_steps_completed": 0, "best_step_selected_by_tracking_objective_only": 0}
    check_tracking_mode("initializer-only", [{}, sample], probe)
    try:
        check_tracking_mode("initializer-only", [{}, {**sample, "gs_refinement_steps_completed": 1}], probe)
    except AssertionError:
        negatives["initializer_only_pose_update"] = True
    else:
        raise AssertionError("initializer-only mutation passed")
    require(no_gt_pose_fields({"ground_truth_included": False, "configuration": {"max_gt_dt": .02}}) == [], "GT metadata whitelist")
    require(no_gt_pose_fields({"evaluation_only_gt_c2w": np.eye(4).tolist()}) != [], "GT payload detection")
    return {"status": "passed", "negative_controls_rejected": negatives, "metadata_vs_GT_payload_distinguished": True}


def audit_base_run(run, baseline, expected_source=LOCKED_SOURCE):
    directory, base = ROOT/"outputs"/run, ROOT/"outputs"/baseline
    m, bm = read_json(directory/"metrics.json"), read_json(base/"metrics.json")
    progress, evaluation = read_json(directory/"progress.json"), read_json(directory/"trajectory_evaluation.json")
    checkpoint = torch.load(directory/"online_map.pt", map_location="cpu", weights_only=True)
    last = torch.load(directory/"last_checkpoint.pt", map_location="cpu", weights_only=True)
    rows, config, count = m["frame_metrics"], m["configuration"], m["frames_processed"]
    require(m["experiment_id"] == run and m["status"].startswith("completed"), "run incomplete or ID mismatch")
    require(m["implementation_version"] == progress["implementation_version"] == last["implementation_version"] == checkpoint["implementation_version"] == 4, "implementation version")
    require(m["script_sha256"] == progress["script_sha256"] == last["script_sha256"] == expected_source, "source lock")
    require(config["pose_initializer"] == "icp-prefix" and config["depth_weight"] == 1 and config["seed"] == 0, "locked V4 control settings")
    require(config == last["configuration"] == progress["configuration"] == checkpoint["configuration"], "configuration artifacts disagree")
    require(count == config["frames"] == len(rows) == progress["completed_frame_count"] == progress["stored_camera_count"], "all requested frames retained")
    require(progress["current_local_index"] == last["current_local_index"] == count-1, "last checkpoint index")
    require(progress["frame_metrics"] == last["completed_frame_metrics"] == rows, "frame records differ")
    require(progress["mapping_events"] == last["mapping_events"] == m["mapping_events"], "mapping records differ")
    require(not no_gt_pose_fields(progress) and not no_gt_pose_fields(last), "GT payload in causal state")
    c2w = checkpoint["estimated_c2w"].numpy().astype(np.float64)
    w2c = checkpoint["estimated_w2c"].numpy().astype(np.float64)
    structure = camera_structure(c2w, w2c)
    for key, values in (("estimated_c2w", c2w), ("estimated_w2c", w2c)):
        require(np.array_equal(last[key].numpy(), values) and np.array_equal(np.asarray(progress[key]), values), key+" checkpoint/progress disagreement")
        require(np.array_equal(np.array([row[key] for row in rows]), values), key+" JSON disagreement")
    require(all(torch.equal(last["parameters"][key], value) and torch.isfinite(value).all() for key, value in checkpoint["parameters"].items()), "parameters differ/nonfinite")
    paths = {"estimate": "trajectory_estimated.txt", "absolute_gt": "trajectory_groundtruth.txt", "anchored_gt": "trajectory_groundtruth_anchored.txt", "prefix": "trajectory_icp_prefix.txt"}
    parsed = {key: parse_trajectory_text((directory/name).read_text(encoding="utf-8-sig"), name) for key, name in paths.items()}
    for item in parsed.values():
        validate_time_pairing(parsed["estimate"], item)
    pairing = validate_time_pairing(parsed["estimate"], parsed["anchored_gt"])
    require(pairing["frame_count"] == count, "trajectory drops failed frames")
    truth = checkpoint["evaluation_only_gt_c2w"].numpy()
    anchored = np.linalg.inv(truth[0])[None]@truth
    text_errors = {"estimate_matrix_max_abs": float(np.max(np.abs(parsed["estimate"]["poses"]-c2w))),
                   "absolute_gt_matrix_max_abs": float(np.max(np.abs(parsed["absolute_gt"]["poses"]-truth))),
                   "anchored_gt_matrix_max_abs": float(np.max(np.abs(parsed["anchored_gt"]["poses"]-anchored)))}
    require(max(text_errors.values()) < 1e-6, "trajectory serialization differs from checkpoint")
    computed = audit_pose_arrays(parsed["estimate"]["poses"], parsed["anchored_gt"]["poses"])
    differences = metric_differences(computed, m["trajectory"])
    require(evaluation == m["trajectory"], "evaluation JSON differs")
    frame_ids = checkpoint["frame_ids"]
    require(frame_ids == [r["frame_id"] for r in rows] == last["frame_ids_with_stored_pose"] == [r["frame_id"] for r in bm["frame_metrics"]], "standalone baseline frame IDs differ")
    require(m["association"] == bm["association"], "baseline sensor association differs")
    baseline_text = parse_trajectory_text((base/"trajectory_estimated.txt").read_text(encoding="utf-8-sig"))
    baseline_truth = parse_trajectory_text((base/"trajectory_groundtruth_anchored.txt").read_text(encoding="utf-8-sig"))
    validate_time_pairing(parsed["prefix"], baseline_text)
    validate_time_pairing(parsed["anchored_gt"], baseline_truth)
    require(np.max(np.abs(parsed["anchored_gt"]["poses"]-baseline_truth["poses"])) < 1e-7, "baseline GT gauge differs")
    prefix = checkpoint["icp_prefix_c2w_float64"].numpy()
    require(prefix.dtype == np.float64 and prefix.shape == c2w.shape, "ICP prefix dtype/shape")
    require(np.isfinite(prefix).all() and np.array_equal(prefix[0], np.eye(4)), "prefix finite/identity")
    require(np.max(np.abs(prefix[:, 3]-[0, 0, 0, 1])) == 0 and np.max(np.abs(np.linalg.det(prefix[:, :3, :3])-1)) < 1e-8, "prefix rigid structure")
    require(np.array_equal(prefix, last["icp_prefix_c2w_float64"].numpy()) and np.array_equal(prefix, np.array(progress["icp_prefix_c2w_float64"])), "prefix checkpoint/progress disagreement")
    baseline_arrays = np.array([r["estimated_c2w"] for r in bm["frame_metrics"]])
    prefix_difference = float(np.max(np.abs(prefix-baseline_arrays)))
    require(prefix_difference < 1e-12, "independent float64 ICP prefix differs from standalone baseline")
    require(np.max(np.abs(parsed["prefix"]["poses"]-prefix)) < 1e-8, "prefix text serialization")
    records = last["icp_frame_records"]
    require(records == progress["icp_frame_records"] and len(records) == count-1, "all ICP records retained")
    for index, record in enumerate(records, start=1):
        require(record == rows[index]["icp"], "frame ICP record disagreement")
        require(record["source_current_frame_id"] == frame_ids[index] and record["target_previous_frame_id"] == frame_ids[index-1], "ICP causal source/target")
        require(np.array_equal(np.array(record["icp_prefix_c2w_float64"]), prefix[index]), "record prefix differs")
        base_row = bm["frame_metrics"][index]
        require(record["accepted"] == base_row["accepted"] and record["failure_flags"] == base_row["failure_flags"], "original ICP accepted/failed flags changed")
        require(np.max(np.abs(np.array(record["registration"]["candidate_previous_from_current"])-np.array(base_row["candidate_previous_from_current"]))) < 1e-12, "raw ICP candidate differs from baseline")
        if not record["accepted"]:
            require(np.array_equal(prefix[index], prefix[index-1]) and rows[index]["pose_rejected"], "failed ICP was integrated")
            require(np.array_equal(w2c[index], w2c[index-1]) and rows[index]["new_points_added"] == 0, "failed ICP pose or points changed")
            require(all("icp_"+flag in rows[index]["failure_flags"] for flag in record["failure_flags"]), "ICP flags were dropped")
    require(m["icp_failed_frame_count"] == sum(not r["accepted"] for r in records), "ICP failure count")
    failure_count = sum(bool(r["failure_flags"]) for r in rows)
    require(failure_count == m["failure_frame_count"], "original failures selectively dropped")
    source = checkpoint["source_frame_ids_per_gaussian"]
    source_counts = check_source_causality(source, frame_ids, rows, m["mapping_events"], m["initial_point_count"])
    require(source == last["source_frame_ids_per_gaussian"] and len(source) == m["final_point_count"] == len(checkpoint["parameters"]["means"]) == progress["point_count"], "source/parameter length disagreement")
    tracking = check_tracking_mode(config["tracking_mode"], rows, m["pose_gradient_checks"])
    accepted_prefix_t = float(np.max(np.linalg.norm(c2w[:, :3, 3]-prefix[:, :3, 3], axis=1)))
    accepted_prefix_r = float(np.max(np.abs(c2w[:, :3, :3]-prefix[:, :3, :3])))
    if config["tracking_mode"] == "initializer-only":
        require(accepted_prefix_t < 2e-6 and accepted_prefix_r < 1e-6, "initializer-only pose departed from float32 canonical ICP prefix")
    prefix_metrics = audit_pose_arrays(parsed["prefix"]["poses"], parsed["anchored_gt"]["poses"])
    prefix_differences = metric_differences(prefix_metrics, m["icp_prefix_trajectory_evaluation"])
    require(m["icp_initializer"]["helper_sha256"] == bm["configuration"]["source_sha256"]["track_rgbd_icp.py"], "ICP helper source differs")
    require(len(computed["per_frame_raw_translation_error_m"]) == count and len(computed["per_interval_rpe_translation_error_m"]) == count-1, "evaluation frame retention")
    return {"status": "passed", "run": run, "baseline": baseline, "implementation_version": 4,
            "training_script_sha256": expected_source, "artifact_auditor_sha256": sha256(Path(__file__)),
            "trajectory_auditor_sha256": sha256(ROOT/"scripts/audit_tum_trajectory.py"),
            "frame_count": count, "all_requested_frames_and_original_flags_retained": True,
            "failure_frame_count": failure_count, "icp_failure_frame_count": m["icp_failed_frame_count"],
            "camera_structure": structure, "checkpoint_progress_frame_json_parameters_consistent": True,
            "GT_pose_fields_in_causal_progress_and_last_checkpoint": [],
            "GT_scan_scope": "Field payload inspection; max_gt_dt and ground_truth_included=False are metadata. Source flow reviewed separately, not formal taint proof.",
            "time_pairing": pairing, "trajectory_serialization_errors": text_errors,
            "metrics": {key: computed[key] for key in METRIC_PAIRS},
            "independent_vs_online_metric_absolute_differences": differences,
            "independent_float64_prefix_vs_standalone_array_max_abs": prefix_difference,
            "ICP_prefix_vs_online_prefix_metric_absolute_differences": prefix_differences,
            "ICP_metric_scope": "Standalone ICP and this audit share audit_pose_arrays; text/baseline artifact consistency, not a second independent metric formula.",
            "tracking_mode_checks": tracking, "accepted_vs_independent_prefix_translation_max_m": accepted_prefix_t,
            "accepted_vs_independent_prefix_rotation_matrix_max_abs": accepted_prefix_r,
            "per_gaussian_source_frame_counts": source_counts, "source_and_window_causality_passed": True,
            "input_hashes": {name: sha256(directory/name) for name in ("metrics.json", *paths.values(), "online_map.pt", "last_checkpoint.pt", "progress.json")},
            "limitations": ["CPU artifact validation does not rerender historical GPU sensor losses", "Map-only control localizes with depth ICP; Gaussian map is not its localization source", "All metrics retain rejected frames; failure flags are not evidence of tracking accuracy"]}



def numpy_prior(candidate_w2c, reference_w2c, weight, sigma_t_m, sigma_r_rad):
    """Independent NumPy float64 geometry, never imports the prior helper."""
    require(np.isfinite(weight) and weight >= 0 and sigma_t_m > 0 and sigma_r_rad > 0, "prior scales")
    if weight == 0:
        return 0.0
    candidate, reference = np.asarray(candidate_w2c, dtype=np.float64), np.asarray(reference_w2c, dtype=np.float64)
    require(candidate.shape == reference.shape == (4, 4) and np.isfinite(candidate).all() and np.isfinite(reference).all(), "prior cameras")
    rotation, ref_rotation = candidate[:3, :3].T, reference[:3, :3].T
    center, ref_center = -rotation@candidate[:3, 3], -ref_rotation@reference[:3, 3]
    return float(weight*(np.sum((center-ref_center)**2)/sigma_t_m**2
                         + np.sum((rotation-ref_rotation)**2)/(2*sigma_r_rad**2)))


def check_loss(loss, config, tracking):
    sensor = loss["sensor_objective"] if tracking else loss["objective"]
    sensor_from_channels = loss["rgb_mse"]+config["depth_weight"]*loss["depth_mae_m"]+config["alpha_weight"]*loss["alpha_loss"]
    sensor_error = abs(sensor-sensor_from_channels)
    require(sensor_error <= 2e-7+2e-6*abs(sensor), "sensor objective/channel disagreement")
    total_error = 0.
    if tracking:
        total_error = abs(loss["objective"]-sensor-loss["prior_objective"])
        require(total_error <= 2e-7+2e-6*abs(loss["objective"]), "sensor+prior != combined")
        require(loss["prior_objective"] >= 0, "negative prior")
        if config["pose_prior_weight"] == 0:
            require(loss["prior_objective"] == 0. and loss["objective"] == loss["sensor_objective"], "lambda0 is not exact off")
    else:
        require("sensor_objective" not in loss and "prior_objective" not in loss, "mapping sensor objective contains prior")
    return sensor_error, total_error


def validate_context(directory, metrics, checkpoint):
    from tum_rgbd import Frame, load_frames, scaled_intrinsics
    path, json_path = directory/"first_tracking_context.pt", directory/"first_tracking_context.json"
    context = torch.load(path, map_location="cpu", weights_only=True)
    meta = read_json(json_path)
    config, rows = metrics["configuration"], metrics["frame_metrics"]
    require(meta == metrics["first_tracking_context"] == checkpoint["first_tracking_context"], "context metadata linkage")
    require(meta["checkpoint"] == path.name and meta["checkpoint_sha256"] == sha256(path), "context PT hash")
    require(context["implementation_version"] == meta["implementation_version"] == 4, "context version")
    require(context["configuration"] == meta["configuration"] == config, "context configuration")
    require(not no_gt_pose_fields(context) and not no_gt_pose_fields(meta), "GT fields in first context")
    for field in ("ground_truth_included", "future_sensor_observations_included", "future_maps_or_trajectories_included"):
        require(context[field] is meta[field] is False, "context GT/future inclusion")
    allowed = {"status", "implementation_version", "configuration", "ground_truth_included",
               "future_sensor_observations_included", "future_maps_or_trajectories_included",
               "source", "parameters", "source_frame_ids_per_gaussian", "map_role", "image",
               "depth_m", "valid_depth_mask", "K", "initial_w2c", "prior_reference_w2c",
               "icp_prefix_c2w_float64", "sensor_metadata", "initial_tracking_loss", "tracking_objective"}
    require(set(context) == allowed, "unexpected snapshot payload fields")
    require(meta["saved_before_first_pose_gradient_probe"] is True and meta["tracking_optimizer_steps_executed_before_save"] == 0, "context not before optimizer")
    require(context["sensor_metadata"] == meta["sensor_metadata"], "context sensor linkage")
    sensor = context["sensor_metadata"]
    require(sensor["local_index"] == 1 and sensor["frame_id"] == rows[1]["frame_id"] == 201, "first tracking sensor")
    require(rows[0]["frame_id"] == 200 and set(context["source_frame_ids_per_gaussian"]) == {200}, "future/GT warmup sources")
    require(meta["map_source_frame_ids"] == [200] and meta["point_count"] == metrics["initial_point_count"] == len(context["source_frame_ids_per_gaussian"]), "warmup point count")
    params = context["parameters"]
    require(set(params) == set(checkpoint["parameters"]), "context parameter names")
    require(all(not p.requires_grad and p.device.type == "cpu" and torch.isfinite(p).all() and len(p) == meta["point_count"] for p in params.values()), "context parameters must be finite detached warmup map")
    require(context["image"].shape == (config["height"], config["width"], 3), "single current RGB shape")
    require(context["depth_m"].shape == context["valid_depth_mask"].shape == (config["height"], config["width"]), "single current depth shape")
    require(context["valid_depth_mask"].dtype == torch.bool and int(context["valid_depth_mask"].sum()) == meta["valid_depth_pixels"], "context valid pixels")
    require(list(context["image"].shape) == meta["image_shape"] and list(context["depth_m"].shape) == meta["depth_shape"], "context shape metadata")
    require(context["initial_tracking_loss"] == meta["initial_tracking_loss"] == rows[1]["initial_loss"], "first loss linkage")
    for field in ("initial_w2c", "prior_reference_w2c"):
        require(np.array_equal(context[field].numpy(), np.array(meta[field])) and np.array_equal(context[field].numpy(), np.array(rows[1][field])), "context initial/reference linkage")
    require(torch.equal(context["initial_w2c"], context["prior_reference_w2c"]), "context reference differs from initial")
    prefix = checkpoint["icp_prefix_c2w_float64"][1]
    require(context["icp_prefix_c2w_float64"].dtype == torch.float64 and torch.equal(context["icp_prefix_c2w_float64"], prefix), "snapshot prefix is not current independent prefix")
    require(np.array_equal(context["icp_prefix_c2w_float64"].numpy(), np.array(meta["icp_prefix_c2w_float64"])), "snapshot prefix metadata")
    require(np.array_equal(context["K"].numpy(), np.array(meta["K"])) and np.allclose(context["K"].numpy(), scaled_intrinsics(config["width"], config["height"]), rtol=0, atol=1e-5), "context intrinsics")
    source_paths = {"v3_script_sha256": "train_tum_online_gs.py", "script_sha256": "train_tum_online_gs_v4.py",
                    "pose_prior_helper_sha256": "geometric_pose_prior.py", "pose_helper_sha256": "track_fixed_map.py",
                    "loader_sha256": "tum_rgbd.py", "icp_helper_sha256": "track_rgbd_icp.py"}
    require(context["source"] == meta["source"], "context source linkage")
    for field, name in source_paths.items():
        require(context["source"][field] == sha256(ROOT/"scripts"/name), "context source hash "+name)
    # Reload only the current RGB-D sensor with a deliberately unknown GT.
    frame = Frame(201, -1, -1, -1, sensor["rgb_timestamp"], sensor["depth_timestamp"],
                  float("nan"), sensor["rgb_path"], sensor["depth_path"], np.full((4, 4), np.nan), float("nan"))
    image, depth, mask = load_frames(Path(config["dataset"]), [frame], config["width"], config["height"], config["min_depth"], config["max_depth"])
    require(np.array_equal(image[0], context["image"].numpy()) and np.array_equal(depth[0], context["depth_m"].numpy()) and np.array_equal(mask[0], context["valid_depth_mask"].numpy()), "snapshot differs from raw current RGB-D")
    return {"pt_sha256": sha256(path), "json_sha256": sha256(json_path), "warmup_source_frame_ids": [200],
            "warmup_point_count": meta["point_count"], "current_sensor_frame_id": 201,
            "sensor_tensor_matches_reloaded_RGB_D_with_NaN_GT": True, "frozen_parameters_CPU_finite_detached": True,
            "initial_reference_prefix_metadata_and_first_frame_record_consistent": True,
            "all_GT_future_flags_false_and_payload_schema_current_only": True,
            "saved_before_probe_optimizer_steps": 0,
            "serialization_scope": "Actual PT/JSON loaded and validated on CPU. Before-probe timing also relies on frozen source ordering and recorded metadata; no renderer rerun."}


def audit_v4_run(run, baseline, expected_source=LOCKED_SOURCE):
    result = audit_base_run(run, baseline, expected_source)
    directory = ROOT/"outputs"/run
    m = read_json(directory/"metrics.json")
    checkpoint = torch.load(directory/"online_map.pt", map_location="cpu", weights_only=True)
    config, rows, prior = m["configuration"], m["frame_metrics"], m["pose_prior"]
    require(sha256(ROOT/"scripts/train_tum_online_gs_v4.py") == expected_source == LOCKED_SOURCE, "V4 current source changed")
    require(sha256(ROOT/"scripts/train_tum_online_gs.py") == m["v3_source_sha256"] == checkpoint["v3_script_sha256"] == LOCKED_V3, "V3 source changed")
    require(sha256(ROOT/"scripts/geometric_pose_prior.py") == m["pose_prior_helper_sha256"] == checkpoint["pose_prior_helper_sha256"] == LOCKED_PRIOR, "prior source changed")
    require(config["start_frame"] == 200 and config["frames"] == 100 and config["frame_stride"] == 1 and config["tracking_mode"] == "optimize", "locked transfer configuration")
    require(config["pose_prior_weight"] in (0., .01) and config["pose_prior_translation_sigma_m"] == .01 and config["pose_prior_rotation_sigma_deg"] == 1., "locked prior parameters")
    weight, sigma_t, sigma_r = config["pose_prior_weight"], config["pose_prior_translation_sigma_m"], math.radians(config["pose_prior_rotation_sigma_deg"])
    require(prior["weight"] == weight and prior["enabled"] is (weight > 0) and prior["translation_sigma_m"] == sigma_t and prior["rotation_sigma_degrees"] == config["pose_prior_rotation_sigma_deg"] and prior["rotation_sigma_radians"] == sigma_r, "prior unit metadata")
    maxima = {"sensor_channel": 0., "combined_sum": 0., "selected_numpy_prior": 0., "raw_numpy_prior": 0., "initializer_prefix_camera": 0.}
    recomputed = []
    for row in rows[1:]:
        reference = np.array(row["prior_reference_w2c"])
        require(np.array_equal(reference, np.array(row["initial_w2c"])), "per-frame prior reference moved")
        prefix = checkpoint["icp_prefix_c2w_float64"][row["local_index"]].numpy()
        ref_c2w = np.eye(4); ref_c2w[:3, :3] = reference[:3, :3].T; ref_c2w[:3, 3] = -reference[:3, :3].T@reference[:3, 3]
        prefix_camera_error = float(np.max(np.abs(ref_c2w-prefix)))
        require(prefix_camera_error < 2e-6, "prior reference not current ICP-prefix canonical camera")
        maxima["initializer_prefix_camera"] = max(maxima["initializer_prefix_camera"], prefix_camera_error)
        losses = (("initial_loss", "initial_w2c", None), ("optimized_loss_before_mapping", "candidate_optimized_w2c", "selected_numpy_prior"), ("selected_raw_candidate_loss_before_projection", "selected_raw_w2c_before_projection", "raw_numpy_prior"))
        for field, pose_field, max_key in losses:
            loss = row[field]
            errors = check_loss(loss, config, True)
            maxima["sensor_channel"] = max(maxima["sensor_channel"], errors[0]); maxima["combined_sum"] = max(maxima["combined_sum"], errors[1])
            value = numpy_prior(row[pose_field], reference, weight, sigma_t, sigma_r)
            discrepancy = abs(value-loss["prior_objective"])
            require(discrepancy <= 1e-6+1e-5*abs(value), "saved prior differs from independent float64 geometry")
            if max_key: maxima[max_key] = max(maxima[max_key], discrepancy)
            recomputed.append({"frame_id": row["frame_id"], "loss_field": field, "numpy_float64_prior": value, "saved_float32_prior": loss["prior_objective"], "absolute_difference": discrepancy})
        require(row["initial_loss"]["prior_objective"] == 0., "nonzero prior at identical initial/reference")
        require(row["optimized_loss_before_mapping"]["objective"] <= row["initial_loss"]["objective"]+1e-8, "selected combined loss increased")
        require(0 <= row["best_step_selected_by_tracking_objective_only"] <= row["gs_refinement_steps_requested"], "invalid selected step")
        for field, sensor_field in (("initial_loss", "initial_sensor_loss"), ("optimized_loss_before_mapping", "selected_sensor_loss_before_mapping")):
            loss = row[field]; expected = {k: v for k, v in loss.items() if k not in ("sensor_objective", "prior_objective")}
            expected["objective"] = loss["sensor_objective"]
            require(expected == row[sensor_field], "sensor fields do not extract true sensor objective")
        check_loss(row["accepted_pose_loss_after_mapping"], config, False)
        icp = row["icp"]
        require(icp["gs_initial_tracking_loss"] == row["initial_loss"] and icp["gs_selected_tracking_loss_before_mapping"] == row["optimized_loss_before_mapping"], "ICP loss linkage")
        require(icp["gs_selected_prior_objective_before_mapping"] == row["optimized_loss_before_mapping"]["prior_objective"] and icp["gs_selected_combined_objective_before_mapping"] == row["optimized_loss_before_mapping"]["objective"], "ICP scalar loss linkage")
        require(icp["gs_initial_sensor_loss"] == row["initial_sensor_loss"] and icp["gs_selected_sensor_loss_before_mapping"] == row["selected_sensor_loss_before_mapping"] and icp["gs_accepted_sensor_loss_after_mapping"] == row["accepted_pose_loss_after_mapping"], "ICP sensor linkage")
        require(np.array_equal(np.array(icp["gs_selected_w2c"]), np.array(row["estimated_w2c"])), "ICP stored accepted camera linkage")
        if row["pose_rejected"]:
            index = row["local_index"]
            require(row["new_points_added"] == 0 and np.array_equal(np.array(row["estimated_w2c"]), np.array(rows[index-1]["estimated_w2c"])), "rejected pose did not freeze")
    context = validate_context(directory, m, checkpoint)
    csv_rows = list(csv.DictReader((directory/"loss.csv").open(newline="", encoding="utf-8-sig")))
    phases = Counter()
    for row in csv_rows:
        phase = row["phase"]; phases[phase] += 1
        numbers = {k: float(row[k]) for k in ("objective", "rgb_mse", "depth_mae_m", "alpha_loss")}
        if phase == "tracking":
            require(row["sensor_objective"] != "" and row["prior_objective"] != "", "CSV lost tracking columns")
            numbers.update({k: float(row[k]) for k in ("sensor_objective", "prior_objective")})
            errors = check_loss(numbers, config, True)
            maxima["combined_sum"] = max(maxima["combined_sum"], errors[1])
            require(1 <= int(row["current_local_index"]) < 100 and row["current_local_index"] == row["observation_local_index"], "CSV future tracking observation")
        else:
            require(phase in ("warmup_mapping", "online_mapping") and row["sensor_objective"] == row["prior_objective"] == "", "CSV mapping has a prior")
            check_loss(numbers, config, False)
            require(0 <= int(row["observation_local_index"]) <= int(row["current_local_index"]) < 100, "CSV future map observation")
    require(len(csv_rows) > 0 and phases["tracking"] > 0 and phases["warmup_mapping"] > 0, "CSV missing phases")
    result.update({"prior_scope": {"weight": weight, "sigma_translation_m": sigma_t, "sigma_rotation_deg": config["pose_prior_rotation_sigma_deg"], "sigma_rotation_rad": sigma_r, "engineering_regularization_not_covariance": True, "reference_per_frame_exact_initial_and_current_ICP_prefix": True, "mapping_records_sensor_only": True},
                   "first_tracking_context": context, "loss_consistency_max_absolute_errors": maxima,
                   "independent_numpy_prior_checks": recomputed, "prior_candidate_checks_count": len(recomputed),
                   "lambda0_all_saved_tracking_prior_exact_zero": weight == 0, "CSV_rows_and_phases": dict(phases),
                   "CSV_sensor_mapping_tracking_union_columns_preserved": True, "source_hashes_locked": {"v4": LOCKED_SOURCE, "v3": LOCKED_V3, "prior": LOCKED_PRIOR}})
    result["input_hashes"].update({name: sha256(directory/name) for name in ("first_tracking_context.pt", "first_tracking_context.json", "loss.csv")})
    result["limitations"].extend(["Prior recomputation uses persisted initial, raw-selected and canonical-selected w2c matrices only; individual CSV iteration cameras are not saved.",
                                 "Independent float64 NumPy prior and GPU float32 prior need a rounding tolerance; scalar exact off at lambda0 is separately strict.",
                                 "Sensor RGB/depth/alpha channel arithmetic is checked without rerendering or validating historical image derivatives.",
                                 "Fixed-reference evidence combines stored per-frame equality with frozen source flow; it is not a formal taint proof.",
                                 "A passed finite-difference probe and zero flags do not establish accurate trajectory."])
    return result


def self_check():
    controls = structural_self_check()
    camera, reference = np.eye(4), np.eye(4)
    camera[0, 3] = -.02
    value = numpy_prior(camera, reference, .01, .01, math.radians(1))
    require(abs(value-.04) < 1e-15, "translation hand prior")
    angle = .1
    camera[:3, :3] = [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    candidate = camera.copy(); reference = np.eye(4); reference[:3, 3] = [.4, -.2, .3]
    rotation = camera[:3, :3].T
    correct = numpy_prior(candidate, reference, .01, .01, math.radians(1))
    wrong_center = .01*(np.sum((candidate[:3, 3]-reference[:3, 3])**2)/.01**2 + np.sum((rotation-reference[:3, :3].T)**2)/(2*math.radians(1)**2))
    require(abs(correct-wrong_center) > .01, "incorrect w2c translation center control insensitive")
    wrong_unit = numpy_prior(camera, np.eye(4), .01, .01, 1.)
    require(abs(wrong_unit-numpy_prior(camera, np.eye(4), .01, .01, math.radians(1))) > .1, "degree/radian control insensitive")
    require(numpy_prior(candidate, np.full((4, 4), np.nan), 0., .01, math.radians(1)) == 0., "weight0 poisoned reference must be off")
    conf = {"depth_weight": 1., "alpha_weight": .01, "pose_prior_weight": .01}
    good = {"objective": .205, "sensor_objective": .2, "prior_objective": .005, "rgb_mse": .1, "depth_mae_m": .099, "alpha_loss": .1}
    check_loss(good, conf, True)
    try:
        check_loss({**good, "objective": .22}, conf, True)
    except AssertionError:
        controls["negative_controls_rejected"]["wrong_combined_loss"] = True
    else:
        raise AssertionError("wrong combined loss passed")
    controls["independent_prior_hand_translation"] = value
    controls["wrong_camera_center_and_degree_units_controls_sensitive"] = True
    controls["weight0_poisoned_reference_exact_zero"] = True
    return controls




def audit_run(run, baseline, expected_source=LOCKED_SOURCE):
    """Explicit sequence wrapper; never silently treats xyz as a desk run."""
    from tum_rgbd import associate_frames, scaled_intrinsics
    allowed_runs = {"tum-xyz-online-v4-prior0-transfer100-001": 0.,
                    "tum-xyz-online-v4-prior001-transfer100-001": .01}
    require(run in allowed_runs and baseline == "tum-xyz-icp-transfer100-001", "explicit xyz run/baseline required")
    directory = ROOT/"outputs"/run
    m, bm = read_json(directory/"metrics.json"), read_json(ROOT/"outputs"/baseline/"metrics.json")
    config = m["configuration"]
    dataset = (ROOT/"data/tum/rgbd_dataset_freiburg1_xyz").resolve()
    require(Path(config["dataset"]).resolve() == Path(bm["configuration"]["dataset"]).resolve() == dataset, "xyz actual dataset path")
    require(config["pose_prior_weight"] == allowed_runs[run], "xyz prior arm weight")
    # Same engineering controls as the already inspected desk diagnostic clip.
    desk_config = read_json(ROOT/"outputs/tum-online-v4-prior0-transfer100-001/metrics.json")["configuration"]
    differences = {key: [desk_config[key], config[key]] for key in desk_config if desk_config[key] != config[key]}
    require(set(config) == set(desk_config) and set(differences).issubset({"dataset", "run", "pose_prior_weight"}), "cross-sequence retuning detected")
    preparation_path = ROOT/"manifests/tum-fr1-xyz-preparation.json"
    association_path = ROOT/"manifests/tum-fr1-xyz-association.json"
    prep, assoc = read_json(preparation_path), read_json(association_path)
    require(prep["dataset"] == assoc["dataset"] == "rgbd_dataset_freiburg1_xyz", "wrong dataset manifests")
    require(prep["status"] == "completed_cpu_dataset_preparation", "dataset preparation incomplete")
    require(prep["dataset_validation"]["association_manifest_sha256"] == sha256(association_path), "association manifest changed")
    require(assoc["loader_sha256"] == m["loader_sha256"] == sha256(ROOT/"scripts/tum_rgbd.py") == "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11", "loader lock")
    require(prep["locked_sources_before"] == prep["locked_sources_after"], "preparation changed frozen sources")
    index_hashes = {}
    for name, info in prep["dataset_validation"]["indices"].items():
        index_hashes[name] = sha256(dataset/name)
        require(index_hashes[name] == info["sha256"], "xyz raw index file changed "+name)
    frames, statistics = associate_frames(dataset, config["max_association_dt"], config["max_gt_dt"])
    require(statistics == assoc["statistics"] == m["association"] == bm["association"], "xyz actual association statistics")
    require(len(frames) == len(assoc["frames"]) == 748, "all associated frame count")
    actual_sensor_rows = []
    for frame in frames:
        metadata = {k: v for k, v in frame.metadata().items() if k not in ("c2w", "w2c")}
        actual_sensor_rows.append(metadata)
    require(actual_sensor_rows == assoc["frames"], "xyz manifest/current loader row mismatch")
    require(m["selected_frame_ids"] == list(range(200, 300)), "xyz selected clip frames")
    require(m["selected_sensor_metadata"] == actual_sensor_rows[200:300], "xyz selected sensor metadata")
    checkpoint = torch.load(directory/"online_map.pt", map_location="cpu", weights_only=True)
    actual_gt = np.stack([frame.c2w for frame in frames[200:300]])
    require(np.array_equal(actual_gt, checkpoint["evaluation_only_gt_c2w"].numpy()), "posthoc GT checkpoint versus raw associated GT")
    truth_difference = 0.
    parsed = parse_trajectory_text((directory/"trajectory_estimated.txt").read_text())
    actual_times = [Decimal(Path(frame.rgb_path).stem) for frame in frames[200:300]]
    require(parsed["timestamps"] == actual_times, "TUM text times differ from exact raw RGB filenames")
    result = audit_v4_run(run, baseline, expected_source)
    result["explicit_sequence_validation"] = {
        "sequence": "fr1_xyz", "dataset_directory": dataset.as_posix(),
        "baseline": baseline, "associated_frames": 748, "selected_frames": [200, 299],
        "fixed_desk_configuration_differences": differences,
        "preparation_manifest_sha256": sha256(preparation_path),
        "association_manifest_sha256": sha256(association_path),
        "actual_index_hashes": index_hashes,
        "all_748_association_rows_equal_actual_frozen_loader": True,
        "selected_100_exact_Decimal_RGB_filename_times_match_text": True,
        "evaluation_GT_checkpoint_equal_raw_associated_GT": True,
        "GT_audit_scope": "Auditor reads actual GT only to validate final evaluation serialization; it runs no tracking, registration, mapping or pose selection.",
        "same_Freiburg1_intrinsics_used": scaled_intrinsics(config["width"], config["height"]).tolist(),
        "warmup_point_count": m["initial_point_count"]}
    result["input_hashes"].update({"xyz_preparation_manifest": sha256(preparation_path), "xyz_association_manifest": sha256(association_path)})
    result["limitations"].append("fr1_xyz is a second sequence from the same Freiburg1 sensor/domain, not proof of broad cross-scene generalization. Desk start200 was already inspected.")
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--icp-baseline")
    parser.add_argument("--expected-source", default=LOCKED_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = {"self_checks": self_check()} if args.self_test else {}
    if args.run:
        if not args.icp_baseline:
            parser.error("--icp-baseline is required with --run")
        require(all(Path(name).name == name and name not in ("", ".", "..") for name in (args.run, args.icp_baseline)), "single directory names required")
        result.update(audit_run(args.run, args.icp_baseline, args.expected_source))
    elif not args.self_test:
        parser.error("provide --run and --icp-baseline or --self-test")
    output = args.output or (ROOT/"outputs"/args.run/"independent_xyz_v4_artifact_audit.json" if args.run else None)
    if output:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
