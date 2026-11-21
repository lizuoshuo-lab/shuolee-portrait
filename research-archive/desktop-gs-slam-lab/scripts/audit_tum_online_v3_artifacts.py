"""CPU audit of locked V3 online artifacts, including standalone ICP parity.

This never imports the training script or uses CUDA. Online trajectory metrics
are recomputed with the separate NumPy auditor. ICP and this audit share that
metric implementation, so ICP comparison is artifact/serialization consistency,
not a second independent ATE/RPE formula.
"""
from __future__ import annotations

import argparse
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
LOCKED_SOURCE = "45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff"
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
            require(row["best_step_selected_by_sensor_loss_only"] == 0, "initializer-only selected optimization step")
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


def self_check():
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
    sample = {"gs_pose_refinement_applied": False, "gs_refinement_steps_requested": 0, "gs_refinement_steps_completed": 0, "best_step_selected_by_sensor_loss_only": 0}
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


def audit_run(run, baseline, expected_source=LOCKED_SOURCE):
    directory, base = ROOT/"outputs"/run, ROOT/"outputs"/baseline
    m, bm = read_json(directory/"metrics.json"), read_json(base/"metrics.json")
    progress, evaluation = read_json(directory/"progress.json"), read_json(directory/"trajectory_evaluation.json")
    checkpoint = torch.load(directory/"online_map.pt", map_location="cpu", weights_only=True)
    last = torch.load(directory/"last_checkpoint.pt", map_location="cpu", weights_only=True)
    rows, config, count = m["frame_metrics"], m["configuration"], m["frames_processed"]
    require(m["experiment_id"] == run and m["status"].startswith("completed"), "run incomplete or ID mismatch")
    require(m["implementation_version"] == progress["implementation_version"] == last["implementation_version"] == checkpoint["implementation_version"] == 3, "implementation version")
    require(m["script_sha256"] == progress["script_sha256"] == last["script_sha256"] == expected_source, "source lock")
    require(config["pose_initializer"] == "icp-prefix" and config["depth_weight"] == 1 and config["seed"] == 0, "locked V3 control settings")
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
    return {"status": "passed", "run": run, "baseline": baseline, "implementation_version": 3,
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
    output = args.output or (ROOT/"outputs"/args.run/"independent_v3_artifact_audit.json" if args.run else None)
    if output:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
