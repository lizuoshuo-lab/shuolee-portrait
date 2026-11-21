"""Read-only explicit recovery collector; the original failed batch stays failed.

The only accepted selection is the original first three completed arms plus the
single protocol-declared replacement of the fourth arm, restarted from frame 0.
--check-inputs never consumes trajectory GT, tensors, or writes an aggregate.
Metric formulas are reused from the exact frozen original collector. This is a
transparent completion recovery, not a new claim of independent metric code.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "scripts/summarize_v4_full_sequence_controls.py"
LEGACY_SHA = "5ce0f9dcfe8a30ec3c63560725c1783ed49f1f957df7d0483e19050b81cef5bf"
FAILED_RUN = "tum-xyz-online-v4-prior001-full748-001"
REPLACEMENT_RUN = "tum-xyz-online-v4-prior001-full748-002"
RECOVERY_PROTOCOL = "manifests/tum-online-v4-full-sequence-recovery-protocol-001.json"
RECOVERY_PROTOCOL_SHA = "3e1a127dfbffd1885862d2f1a5eb633a36228788041f5aa3d5c8de3b5c90ea3c"
RECOVERY_BATCH = "outputs/tum-online-v4-full-sequence-recovery-001"
RECOVERY_TERMINAL = "recovery_run_completed"
SELECTION_RULE = "original_first_three_completed_plus_explicit_fourth_replacement_no_metric_selection"
OUTPUT_RUN = "tum-online-v4-full-sequence-recovery-summary-001"
OLD_FAILED_STATUS = "batch_stopped_failure_no_retry"
OLD_FAILED_RECORD = "failed_process_exit_all_original_artifacts_retained"
FAILED_FINAL_ABSENT = ["outputs/" + FAILED_RUN + "/metrics.json", "outputs/" + FAILED_RUN + "/online_map.pt"]
RECOVERY_SOURCE_LOCKS = {
    "scripts/run_v4_full_recovery.py": "d9171ef2391f0af2db98bbe6da2029c97fc6b633fbca01feeb8629090bd2ece4",
    "scripts/check_tum_v4_full_recovery_artifacts.py": "5360bcd229205ad2f8a6a47967f824f3ad21cca804e0a91355b1702bfce63221",
    "scripts/summarize_tum_v4_full_recovery.py": "effee2a8e16758b2330f7f2633cbda3694d5c4f80e2f15e5e273485abd91680b",
    "scripts/check_tum_v4_recovery_cpu.py": "e895275cd2849aa309286ce77e7cecc7276381adc24d73865461321ce1e31e58",
}
RECOVERY_REVIEW_LOCKS = {
    RECOVERY_BATCH + "/root_protocol_review.json": "8180e1b97876b7f1f41d2ef5e19df5e4655b83f0e911059f3b05e1c875cbf2d1",
    RECOVERY_BATCH + "/cpu_recovery_wrapper_checks-001.json": "8cc16c3c2acb0324394c1f61e94903a0b146adea75efa04c06b46ece7f83530b",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if file_sha(LEGACY_PATH) != LEGACY_SHA:
    raise RuntimeError("Frozen original collector source changed")
_spec = importlib.util.spec_from_file_location("frozen_full_v4_collector_5ce", LEGACY_PATH)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
require, sha, read_json = legacy.require, legacy.sha, legacy.read_json
safe_path, verify_hash, stamp = legacy.safe_path, legacy.verify_hash, legacy.stamp
bounded, exclusive_json = legacy.bounded, legacy.exclusive_json
MAX_BYTES = legacy.MAX_BYTES
LOCKED = legacy.LOCKED


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def original_failure_gate(protocol, ledger):
    """Validate the real failed ledger without ever relabeling it completed."""
    require(ledger["status"] == OLD_FAILED_STATUS, "Original failed batch state must be retained")
    require(ledger["protocol_sha256"] == legacy.PROTOCOL_SHA, "Original protocol SHA")
    require(ledger["driver_sha256"] == protocol["driver_sha256"], "Original driver SHA")
    require(ledger["source_sha256"] == protocol["source_sha256"], "Original six frozen sources")
    require(ledger["deadline_beijing"] == protocol["deadline_beijing"], "Original deadline changed")
    require(ledger["GPU_runs_started"] == 4 and ledger["current_run"] == FAILED_RUN, "Original attempt accounting")
    require(ledger["all_original_artifacts_and_logs_retained"] is True, "Old failure retention flag")
    rows = ledger["run_records"]
    require(len(rows) == len(protocol["new_runs"]) == 4, "Original failure record omitted")
    deadline = datetime.fromisoformat(protocol["deadline_beijing"])
    previous_end = stamp(ledger["started_at"])
    durations = {}
    for index, (expected, actual) in enumerate(zip(protocol["new_runs"], rows)):
        require(actual["run"] == expected["run"] and actual["order"] == index + 1, "Original run order changed")
        require(actual["configuration"] == expected["configuration"], "Original configuration changed")
        require(actual["log"] == expected["log_relative_path"] and hash_string(actual["log_sha256"]), "Original log identity/hash")
        ready, guard, popped, ended = [stamp(actual[k]) for k in ("ready_at", "spawn_guard_at", "Popen_returned_at", "ended_at")]
        require(previous_end <= ready <= guard <= popped <= ended, "Original serial ordering")
        require(guard < deadline, "Original child started at/after deadline")
        if index < 3:
            require(actual["returncode"] == 0 and actual["status"] == "completed", "Original first three must be exit0")
            require(actual["frames"] == expected["configuration"]["frames"] and hash_string(actual["metrics_sha256"]), "Original completed frames/hash")
        else:
            require(actual["run"] == FAILED_RUN and actual["returncode"] == 1 and actual["status"] == OLD_FAILED_RECORD, "Fourth original failure overwritten")
            require("metrics_sha256" not in actual and "trajectory_summary" not in actual, "Failed original has fabricated final metrics")
        durations[actual["run"]] = (ended - guard).total_seconds()
        previous_end = ended
    require(stamp(ledger["failed_at"]) >= previous_end, "Original failure predates child exit")
    try:
        legacy.validate_ledger(protocol, ledger)
    except ValueError:
        pass
    else:
        raise ValueError("Original collector unexpectedly accepted failed batch")
    return durations


def replacement_configuration_gate(original, replacement):
    require(replacement["run"] == REPLACEMENT_RUN, "Only the explicit replacement is eligible; no metric-based selection")
    require(replacement["scene"] == "xyz" and replacement["full_ICP_reference"] == original["full_ICP_reference"], "Replacement scene/baseline")
    expected = copy.deepcopy(original["configuration"])
    expected["run"] = REPLACEMENT_RUN
    require(replacement["configuration"] == expected, "Replacement configuration may differ only by run name")
    require(expected["frames"] == 748 and expected["start_frame"] == 0 and expected["frame_stride"] == 1, "Full 748 frame0 restart required")
    require(replacement["output_relative_path"] == "outputs/" + REPLACEMENT_RUN, "Replacement output location")
    require(replacement["log_relative_path"] == "logs/" + REPLACEMENT_RUN + ".log", "Replacement log location")


def recovery_protocol_gate(original, recovery):
    require(recovery["original_protocol_sha256"] == legacy.PROTOCOL_SHA, "Recovery original protocol binding")
    require(recovery["original_failed_run"] == FAILED_RUN, "Recovery original failed identity")
    expected_selection = [r["run"] for r in original["new_runs"][:3]] + [REPLACEMENT_RUN]
    require(recovery["selected_final_runs"] == expected_selection, "No best-metric or alternative run selection")
    require(recovery["source_sha256"] == original["source_sha256"] and len(recovery["source_sha256"]) == 6, "Recovery six frozen sources changed")
    require(recovery["deadline_beijing"] == original["deadline_beijing"], "Recovery deadline changed")
    require(recovery["new_run_count"] == 1 and len(recovery["new_runs"]) == 1, "Exactly one replacement declaration")
    replacement_configuration_gate(original["new_runs"][3], recovery["new_runs"][0])
    require(recovery["driver_sha256"] == RECOVERY_SOURCE_LOCKS["scripts/run_v4_full_recovery.py"], "Recovery driver hash missing/mismatched")
    binding = recovery["replacement_binding"]
    require(binding["original_failed_run"] == FAILED_RUN and binding["replacement_run"] == REPLACEMENT_RUN,
        "Replacement identity binding differs")
    require(binding["restart_from_frame"] == 0 and binding["not_checkpoint_resume"] is True and
        binding["all_36_configuration_fields_only_run_differs"] is True, "Replacement frame0/configuration binding")
    require(binding["original_failed_command"] == original["new_runs"][3]["command"], "Original failed command binding")
    expected_command = copy.deepcopy(binding["original_failed_command"])
    expected_command[expected_command.index("--run") + 1] = REPLACEMENT_RUN
    require(recovery["new_runs"][0]["command"] == expected_command, "Replacement command differs outside run name")
    require(recovery["dataset_archives"] == original["dataset_archives"] and
        recovery["full_CPU_baseline_references"] == original["full_CPU_baseline_references"], "Recovery data/baseline references changed")
    required = {
        legacy.BATCH + "/failure.json", legacy.BATCH + "/progress.json",
        legacy.BATCH + "/fourth_run_failure_evidence.json",
        "outputs/" + FAILED_RUN + "/failure.json", "outputs/" + FAILED_RUN + "/progress.json",
        "outputs/" + FAILED_RUN + "/progress.json.tmp", "outputs/" + FAILED_RUN + "/last_checkpoint.pt",
        "outputs/" + FAILED_RUN + "/first_tracking_context.pt", "outputs/" + FAILED_RUN + "/first_tracking_context.json",
        "logs/" + FAILED_RUN + ".log",
    }
    evidence = recovery["original_failure_artifact_sha256"]
    require(required.issubset(evidence) and all(hash_string(v) for v in evidence.values()), "Old failure/log/checkpoint evidence omitted")
    snapshots = recovery["original_failure_snapshot"]
    require(hash_string(snapshots["manifest_sha256"]) and snapshots["files"], "Failure snapshot manifest or files missing")
    require({r["original"] for r in snapshots["files"]} >= required, "Failure snapshot coverage incomplete")
    for item in snapshots["files"]:
        require(item["sha256"] == evidence[item["original"]] and item["snapshot"] != item["original"], "Failure snapshot hash/source binding")
    return recovery["new_runs"][0]


def recovery_ledger_gate(original, failure, recovery, recovery_sha, ledger):
    original_failure_gate(original, failure)
    row = recovery_protocol_gate(original, recovery)
    require(ledger["status"] == RECOVERY_TERMINAL, "Replacement terminal success pending")
    require(ledger["protocol_sha256"] == recovery_sha, "Replacement protocol SHA mismatch")
    require(ledger["driver_sha256"] == recovery["driver_sha256"] and ledger["source_sha256"] == original["source_sha256"], "Replacement source/driver mismatch")
    require(ledger["deadline_beijing"] == recovery["deadline_beijing"], "Replacement ledger deadline")
    require(ledger["GPU_runs_started"] == 1 and ledger["current_run"] is None and len(ledger["run_records"]) == 1, "Exactly one collected replacement required")
    actual = ledger["run_records"][0]
    require(actual["run"] == row["run"] and actual["configuration"] == row["configuration"], "Replacement row/configuration mismatch")
    require(actual["status"] == "completed" and actual["returncode"] == 0 and actual["frames"] == 748, "Replacement exit0 full748 required")
    require(actual["log"] == row["log_relative_path"] and all(hash_string(actual[k]) for k in ("log_sha256", "metrics_sha256")), "Replacement metrics/log provenance")
    start, ready, guard, popped, ended = [stamp(ledger["started_at"])] + [stamp(actual[k]) for k in ("ready_at", "spawn_guard_at", "Popen_returned_at", "ended_at")]
    require(stamp(failure["failed_at"]) <= start <= ready <= guard <= popped <= ended <= stamp(ledger["ended_at"]), "Replacement overlaps failure or time order differs")
    require(guard < datetime.fromisoformat(recovery["deadline_beijing"]), "Replacement started at/after deadline")
    return actual, (ended - guard).total_seconds()


def validate_replacement_run(row, protocol, record, observed):
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
            audit["protocol_sha256"] == protocol["_exact_recovery_protocol_sha256"] and audit["GPU_calls"] == 0, "Unknown CPU full checker")
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
    legacy.check_chunks(folder, original, read_json(folder / "summary.json"), audit, observed)
    return original, audit


def validate_replacement_artifacts(row, protocol, record, observed):
    original, audit = validate_replacement_run(row, protocol, record, observed)
    summary = read_json(safe_path(ROOT, row["output_relative_path"]) / "summary.json")
    provenance = summary["summary_provenance"]
    summary_wrapper = RECOVERY_SOURCE_LOCKS["scripts/summarize_tum_v4_full_recovery.py"]
    checker_wrapper = RECOVERY_SOURCE_LOCKS["scripts/check_tum_v4_full_recovery_artifacts.py"]
    require(audit["recovery_checker_wrapper_sha256"] == checker_wrapper and
        audit["recovery_summary_wrapper_sha256"] == summary_wrapper, "Recovery artifact wrapper source mismatch")
    require(provenance["recovery_wrapper_sha256"] == summary_wrapper, "Recovery summary wrapper source mismatch")
    for part in (audit, provenance):
        require(part["recovery_protocol_sha256"] == protocol["_exact_recovery_protocol_sha256"] and
            part["original_protocol_sha256"] == legacy.PROTOCOL_SHA and part["original_failed_run"] == FAILED_RUN,
            "Recovery wrapper protocol/failure binding")
        require(part["explicit_replacement_binding"] == protocol["replacement_binding"], "Recovery wrapper replacement binding")
    return original, audit


def verify_failure_files(recovery, failure, observed):
    """Do not parse live progress: these hashes are for a collected failed child."""
    for relative, expected in recovery["original_failure_artifact_sha256"].items():
        verify_hash(ROOT, relative, expected, observed)
    for relative in FAILED_FINAL_ABSENT:
        require(not safe_path(ROOT, relative).exists(), "Failed original was overwritten by final output: " + relative)
    require(read_json(ROOT / legacy.BATCH / "progress.json") == failure, "Old failure/progress terminal records differ")
    details = read_json(ROOT / "outputs" / FAILED_RUN / "failure.json")
    require(details["status"] == "failed" and details["exception_type"] == "PermissionError", "Expected filesystem failure not preserved")
    require("progress.json" in details["exception_message"], "Failure destination differs")
    snapshot = recovery["original_failure_snapshot"]
    verify_hash(ROOT, snapshot["manifest"], snapshot["manifest_sha256"], observed)
    manifest = read_json(safe_path(ROOT, snapshot["manifest"]))
    require(manifest["files"] == snapshot["files"], "Snapshot manifest files differ from frozen recovery protocol")
    for item in snapshot["files"]:
        path = verify_hash(ROOT, item["snapshot"], item["sha256"], observed)
        if "bytes" in item:
            require(path.stat().st_size == safe_path(ROOT, item["original"]).stat().st_size == item["bytes"], "Snapshot byte count differs")
    require(sha(ROOT / "logs" / (FAILED_RUN + ".log")) == failure["run_records"][3]["log_sha256"], "Old failed log does not match original ledger")
    return {"original_batch_status": failure["status"], "original_failed_run": FAILED_RUN,
        "original_returncode": failure["run_records"][3]["returncode"],
        "original_failure_type": details["exception_type"],
        "original_failure_completed_frame_count": details.get("completed_frame_count"),
        "original_failure_current_local_index": details.get("current_local_index"),
        "original_failure_artifact_count": len(recovery["original_failure_artifact_sha256"]),
        "snapshot_manifest": snapshot["manifest"], "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "snapshot_file_count": len(snapshot["files"]), "failed_final_metrics_still_absent": True,
        "failure_not_silently_counted_as_success": True,
        "cause_scope": "The observed failure is Windows PermissionError during progress replacement; possible reader locking is disclosed, not established causally."}


def load_recovery_bundle(recovery_sha, require_terminal, observed=None):
    require(hash_string(recovery_sha), "Exact 64-character recovery protocol SHA is required")
    require(recovery_sha == RECOVERY_PROTOCOL_SHA, "Only the reviewed explicit recovery protocol SHA is allowed")
    if observed is None:
        observed = {}
    verify_hash(ROOT, "scripts/summarize_v4_full_sequence_controls.py", LEGACY_SHA, observed)
    verify_hash(ROOT, legacy.PROTOCOL, legacy.PROTOCOL_SHA, observed)
    original = read_json(ROOT / legacy.PROTOCOL)
    verify_hash(ROOT, RECOVERY_PROTOCOL, recovery_sha, observed)
    recovery = read_json(ROOT / RECOVERY_PROTOCOL)
    replacement = recovery_protocol_gate(original, recovery)
    for relative, expected in {**original["source_sha256"], **legacy.LOCKED}.items():
        verify_hash(ROOT, relative, expected, observed)
    for relative, expected in original["CPU_check_artifact_sha256"].items():
        verify_hash(ROOT, relative, expected, observed)
    for item in original["dataset_archives"].values():
        path = verify_hash(ROOT, item["relative_path"], item["sha256"], observed)
        require(path.stat().st_size == item["bytes"], "Archive byte size changed")
    for relative, expected in RECOVERY_SOURCE_LOCKS.items():
        verify_hash(ROOT, relative, expected, observed)
    for relative, expected in recovery["CPU_check_artifact_sha256"].items():
        verify_hash(ROOT, relative, expected, observed)
    for relative, expected in RECOVERY_REVIEW_LOCKS.items():
        verify_hash(ROOT, relative, expected, observed)
    approval = read_json(safe_path(ROOT, recovery["root_review_relative_path"]))
    observed[recovery["root_review_relative_path"]] = sha(safe_path(ROOT, recovery["root_review_relative_path"]))
    require(approval["status"] == "passed" and approval["protocol_sha256"] == recovery_sha, "Recovery root protocol approval missing or mismatched")
    require(approval["driver_sha256"] == recovery["driver_sha256"] and approval["frozen_source_sha256"] == original["source_sha256"] and
        approval["new_GPU_run_authorized"] == REPLACEMENT_RUN, "Root review source/run binding differs")
    failure = read_json(ROOT / legacy.BATCH / "failure.json")
    old_durations = original_failure_gate(original, failure)
    failure_summary = verify_failure_files(recovery, failure, observed)
    require(not (ROOT / legacy.BATCH / "metrics.json").exists(), "Original failed batch must not acquire completed metrics")
    records = copy.deepcopy(failure["run_records"][:3])
    selected = copy.deepcopy(original)
    selected["new_runs"] = copy.deepcopy(original["new_runs"][:3]) + [copy.deepcopy(replacement)]
    replacement_ledger, duration = None, None
    metrics_path = ROOT / RECOVERY_BATCH / "metrics.json"
    if metrics_path.exists():
        replacement_ledger = read_json(metrics_path)
        observed[RECOVERY_BATCH + "/metrics.json"] = sha(metrics_path)
        record, duration = recovery_ledger_gate(original, failure, recovery, recovery_sha, replacement_ledger)
        require(read_json(ROOT / RECOVERY_BATCH / "progress.json") == replacement_ledger, "Replacement terminal progress/metrics differ")
        observed[RECOVERY_BATCH + "/progress.json"] = sha(ROOT / RECOVERY_BATCH / "progress.json")
        records.append(record)
    elif require_terminal:
        raise ValueError("Replacement terminal metrics.json absent; no real aggregation")
    recovery = copy.deepcopy(recovery)
    recovery["_exact_recovery_protocol_sha256"] = recovery_sha
    return {"original": original, "recovery": recovery, "selected": selected,
        "original_failed_ledger": failure, "replacement_ledger": replacement_ledger,
        "selected_records": records, "observed_sha256": observed, "failure_summary": failure_summary,
        "attempt_wall_seconds": {**old_durations, **({REPLACEMENT_RUN: duration} if duration is not None else {})}}


def check_inputs(recovery_sha):
    """No tensor loads, raw GT reads, trajectory aggregation or output writes."""
    observed, issues, run_inputs, baselines = {}, [], [], {}
    try:
        bundle = load_recovery_bundle(recovery_sha, require_terminal=False, observed=observed)
        observed.update(bundle["observed_sha256"])
    except (KeyError, ValueError, OSError) as error:
        return {"status": "not_ready_recovery_protocol_or_evidence_gate", "ready": False,
            "issues": [str(error)], "observed_sha256": observed, "GPU_calls": 0,
            "trajectory_metrics_recomputed": False, "GT_poses_read": False,
            "torch_imported": "torch" in sys.modules, "output_written_by_check_inputs": False,
            "collector_sha256": sha(Path(__file__)), "legacy_collector_sha256": LEGACY_SHA}
    original, recovery = bundle["original"], bundle["recovery"]
    if bundle["replacement_ledger"] is None:
        issues.append("replacement terminal collected metrics.json pending")
    for scene, reference in original["full_CPU_baseline_references"].items():
        try:
            for relative, expected in reference["artifact_sha256"].items():
                verify_hash(ROOT, relative, expected, observed)
            folder = ROOT / "outputs" / reference["run"]
            for relative, expected in reference["validation_evidence_sha256"].items():
                verify_hash(folder, relative, expected, observed)
            for name, expected in reference["source_sha256"].items():
                verify_hash(ROOT, "scripts/" + name, expected, observed)
            require(read_json(folder / "validation.json")["status"] == "passed", "Full ICP validation not passed")
            baselines[scene] = {"status": "ready", "run": reference["run"], "frames": reference["frames"]}
        except (KeyError, ValueError, OSError) as error:
            baselines[scene] = {"status": "blocked", "issue": str(error)}
            issues.append(scene + " baseline: " + str(error))
    records = {r["run"]: r for r in bundle["selected_records"]}
    for row in bundle["selected"]["new_runs"]:
        item = {"run": row["run"], "frames_required": row["configuration"]["frames"]}
        try:
            record = records.get(row["run"])
            require(record is not None, "Replacement terminal record pending")
            if row["run"] == REPLACEMENT_RUN:
                validate_replacement_artifacts(row, recovery, record, observed)
            else:
                legacy.validate_run(row, original, record, observed)
            item["status"] = "ready_artifacts"
        except (KeyError, ValueError, OSError) as error:
            item.update(status="pending_or_invalid", issue=str(error))
            issues.append(row["run"] + ": " + str(error))
        run_inputs.append(item)
    return {"status": "ready_original_three_plus_explicit_replacement" if not issues else "not_ready_no_aggregation",
        "ready": not issues, "issues": issues, "run_inputs": run_inputs, "baselines": baselines,
        "observed_sha256": observed, "failure_retention": bundle["failure_summary"],
        "original_collector_rejects_failed_batch": True, "selection_rule": SELECTION_RULE,
        "collector_sha256": sha(Path(__file__)), "legacy_collector_sha256": LEGACY_SHA,
        "GPU_calls": 0, "trajectory_metrics_recomputed": False, "GT_poses_read": False,
        "torch_imported": "torch" in sys.modules, "output_written_by_check_inputs": False}


# Pure metric formulas are the exact, reviewed legacy functions. No loader,
# registration, audit metric or renderer is imported by these functions.
from collections import Counter
trajectory, reconstruct_truth = legacy.trajectory, legacy.reconstruct_truth
compare_reported, angle_degrees = legacy.compare_reported, legacy.angle_degrees
column_table, mean = legacy.column_table, legacy.mean


def validate_selected_run(row, protocol, record, observed):
    if row["run"] == REPLACEMENT_RUN:
        return validate_replacement_artifacts(row, protocol["_recovery"], record, observed)
    return legacy.validate_run(row, protocol, record, observed)

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
        metrics, audit = validate_selected_run(choice, protocol, record, observed)
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
    require(context["checker_sha256"] == sha(Path(__file__)), "Context checker source drift")
    require(len(context["input_tensors"]) == 7 and len(context["warmup_map_parameters"]) == 5, "Context tensor counts")
    result["actual_first_context_pair_comparison"] = context
    return result, details, plot_arrays


class SelectedContextChecker:
    """Same seven/five tensor observation, with explicit reviewed run selection."""
    INPUTS = ("image", "depth_m", "valid_depth_mask", "K", "initial_w2c", "prior_reference_w2c", "icp_prefix_c2w_float64")

    def __init__(self, protocol, observed):
        import torch
        self.torch, self.protocol, self.observed = torch, protocol, observed
        require(not torch.cuda.is_initialized(), "CPU collector entered with CUDA initialized")

    def details(self, tensor):
        require(tensor.device.type == "cpu" and not tensor.requires_grad and self.torch.isfinite(tensor).all(), "Context tensor not finite/detached/CPU")
        array = tensor.numpy()
        return {"shape": list(array.shape), "dtype": str(array.dtype),
            "raw_C_order_bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}

    def compare(self, scene):
        choices = [r for r in self.protocol["new_runs"] if r["scene"] == scene]
        require(len(choices) == 2 and [r["configuration"]["pose_prior_weight"] for r in choices] == [0., .01], "Selected context arm order")
        contexts, hashes = [], {}
        for row in choices:
            folder = safe_path(ROOT, row["output_relative_path"])
            metrics, meta = read_json(folder / "metrics.json"), read_json(folder / "first_tracking_context.json")
            require(metrics["status"] == "completed" and metrics["configuration"] == row["configuration"], "Context completed configuration")
            require(metrics["first_tracking_context"] == meta, "Context metadata linkage")
            path = verify_hash(folder, "first_tracking_context.pt", meta["checkpoint_sha256"], self.observed)
            context = self.torch.load(path, map_location="cpu", weights_only=True)
            require(context["configuration"] == row["configuration"], "Context configuration differs")
            require(all(context[k] is False for k in ("ground_truth_included", "future_sensor_observations_included", "future_maps_or_trajectories_included")), "Context includes future or GT")
            require(meta["saved_before_first_pose_gradient_probe"] and meta["tracking_optimizer_steps_executed_before_save"] == 0, "Context timing")
            contexts.append(context)
            hashes[row["run"]] = {"metrics_sha256": sha(folder / "metrics.json"), "context_PT_sha256": sha(path), "context_JSON_sha256": sha(folder / "first_tracking_context.json")}
        left, right = contexts
        require({k for k in left["configuration"] if left["configuration"][k] != right["configuration"][k]} == {"run", "pose_prior_weight"}, "Paired configs differ outside run/weight")
        inputs, maps = {}, {}
        for key in self.INPUTS:
            a, b = self.details(left[key]), self.details(right[key])
            inputs[key] = {"prior0": a, "prior001": b, "shape_dtype_and_raw_bytes_equal": a == b}
            require(a == b, "Paired sensor/initial/reference tensor differs: " + key)
        require(set(left["parameters"]) == set(right["parameters"]) == {"means", "log_scales", "quats", "color_logits", "opacity_logits"}, "Five map fields required")
        for key in left["parameters"]:
            p, q = left["parameters"][key], right["parameters"][key]
            a, b = self.details(p), self.details(q)
            require(a["shape"] == b["shape"] and a["dtype"] == b["dtype"], "Map shape/dtype differ")
            maps[key] = {"prior0": a, "prior001": b, "raw_bytes_equal": a == b,
                "max_absolute_element_difference": float(np.max(np.abs(p.numpy().astype(np.float64) - q.numpy().astype(np.float64))))}
        require(left["source_frame_ids_per_gaussian"] == right["source_frame_ids_per_gaussian"] and set(left["source_frame_ids_per_gaussian"]) == {0}, "Warmup source order differs")
        require(left["sensor_metadata"] == right["sensor_metadata"] and left["source"] == right["source"], "Context sensor/source differs")
        return {"status": "passed", "scene": scene, "runs": [r["run"] for r in choices],
            "checker_sha256": sha(Path(__file__)), "legacy_context_checker_reference_sha256": legacy.LOCKED["scripts/check_tum_v4_full_context_pairs.py"],
            "original_protocol_sha256": legacy.PROTOCOL_SHA, "recovery_protocol_sha256": self.protocol["_recovery"]["_exact_recovery_protocol_sha256"],
            "artifact_hashes": hashes, "seven_common_sensor_initial_reference_tensors_byte_identical": True,
            "input_tensors": inputs, "all_five_warmup_map_parameter_tensors_byte_identical": all(v["raw_bytes_equal"] for v in maps.values()),
            "warmup_map_parameters": maps, "warmup_point_count": len(left["source_frame_ids_per_gaussian"]),
            "first_map_source_ids_and_order_equal": True, "GT_read_or_online_pose_selection": False, "GPU_calls": 0,
            "scope": "Explicit whole-pipeline pair including the predeclared recovery, not identical-frozen-map causal isolation; measured raw tensor bytes, not assumed determinism."}


def execute(run, recovery_sha):
    require(run == OUTPUT_RUN, "Only the explicit exclusive recovery summary output is allowed")
    output = ROOT / "outputs" / run
    output.mkdir(exist_ok=False)
    try:
        preflight = check_inputs(recovery_sha)
        exclusive_json(output / "input_preflight.json", preflight)
        require(preflight["ready"], "Recovery gate not ready; no trajectory metrics aggregated")
        bundle = load_recovery_bundle(recovery_sha, require_terminal=True)
        observed = {**preflight["observed_sha256"], **bundle["observed_sha256"]}
        protocol = bundle["selected"]
        protocol["_recovery"] = bundle["recovery"]
        # This internal selected-record container is not an original batch ledger.
        selected_records = {"run_records": bundle["selected_records"]}
        checker = SelectedContextChecker(protocol, observed)
        result = {"status": "completed_read_only_original_three_plus_explicit_recovery",
            "collector_sha256": sha(Path(__file__)), "legacy_metric_collector_sha256": LEGACY_SHA,
            "original_protocol_sha256": legacy.PROTOCOL_SHA, "recovery_protocol_sha256": recovery_sha,
            "original_batch_status": OLD_FAILED_STATUS, "recovery_batch_status": RECOVERY_TERMINAL,
            "selection_rule": SELECTION_RULE, "original_failed_attempt": bundle["failure_summary"],
            "attempt_count": 5, "completed_arm_count": 4, "failed_attempt_count": 1,
            "partial_failed_original_frames_not_used_as_final_trajectory": True,
            "scope": "Two Freiburg1 complete associated sequences; one seed per completed arm with one disclosed frame0 restart. No cross-scene mean or single-seed SD.",
            "metric_implementation": "Exact frozen 5ce pure formulas: raw stored estimated_c2w and raw TUM xyzw, SciPy Rotation, analytical rigid inverse, Kabsch SVD scale1, math.fsum. No loader/solver/auditor metric imported.",
            "GT_usage": "Only completed selected trajectories posthoc; first raw GT anchor and no-scale metric alignment; no trajectory, map, threshold, weight or recovery selection modification.",
            "RPE_definition": "inv(inv(GT[i-1])@GT[i]) @ (inv(EST[i-1])@EST[i]); all delta1 adjacent frames",
            "timing_scope": "Original observed child durations include I/O and serialization. The failed original duration is retained separately, not erased from attempt accounting or represented as FPS.",
            "all_attempt_wall_seconds": bundle["attempt_wall_seconds"], "scenes": {}, "GPU_calls": 0}
        payloads, plots = {}, {}
        for scene in ("desk", "xyz"):
            stats, frames, arrays = summarize_scene(scene, protocol, selected_records, observed, checker)
            result["scenes"][scene] = stats
            payloads[scene + "_frames.json"] = bounded(frames)
            payloads[scene + "_warmup_context_pair_checks.json"] = bounded(stats["actual_first_context_pair_comparison"])
            plots[scene] = arrays
        require(not checker.torch.cuda.is_initialized(), "Read-only collector initialized CUDA")
        for relative, expected in observed.items():
            require(sha(safe_path(ROOT, relative)) == expected, "Consumed evidence changed: " + relative)
        for relative in FAILED_FINAL_ABSENT:
            require(not safe_path(ROOT, relative).exists(), "Failed original overwritten during collection")
        input_data = bounded({"status": "all_consumed_inputs_rehashed_unchanged", "input_sha256": observed})
        payloads["input_hashes.json"] = input_data
        result.update(input_hashes_file="input_hashes.json", input_hashes_file_sha256=hashlib.sha256(input_data).hexdigest(),
            input_hash_count=len(observed), all_inputs_rehashed_unchanged=True, torch_CUDA_initialized=False)
        payloads["summary.json"] = bounded(result)
        for name, data in payloads.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
        for scene, arrays in plots.items():
            legacy.make_plot(output / (scene + "_comparison.png"), scene, arrays)
        for name, source in (("collector_source.py", Path(__file__)), ("legacy_metric_collector_source.py", LEGACY_PATH)):
            with (output / name).open("xb") as stream:
                stream.write(source.read_bytes())
        exclusive_json(output / "completion.json", {"status": "completed", "source_sha256": sha(Path(__file__)),
            "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()},
            "four_selected_completed_arms_two_baselines_all_frames_retained": True,
            "original_failed_attempt_preserved": True, "GPU_calls": 0})
        print(json.dumps({"status": result["status"], "output": str(output)}))
    except BaseException as error:
        exclusive_json(output / "failure.json", {"status": "failed_partial_collector_evidence_retained",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "GPU_calls": 0, "original_inputs_modified": False, "no_frame_or_flag_truncation": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--recovery-protocol-sha", required=True)
    parser.add_argument("--run", default=OUTPUT_RUN)
    args = parser.parse_args()
    if args.check_inputs:
        print(json.dumps(check_inputs(args.recovery_protocol_sha), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        execute(args.run, args.recovery_protocol_sha)


if __name__ == "__main__":
    main()
