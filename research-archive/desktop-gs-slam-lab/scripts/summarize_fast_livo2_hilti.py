"""Summarize ONLY completed, strict paired Hilti exp14 local evaluations.

No ROS, replay, scoring, source edits, progress.json updates or website edits.
Default inputs are outputs/fast-livo2-hilti-gt-001/evaluation/{metrics,
execution,preregistration}.json. --validate-only performs all acceptance checks
without writing. Normal invocation creates two NEW files, refusing overwrite.

The root's real evaluator subprocess receipt must have:
  status="actual_evaluator_completed", child_exit_code=0,
  started_utc, finished_utc, command (list of actual argv strings),
  metrics_sha256, evaluator_source_sha256, evaluation_plan_sha256,
  preregistration_sha256,
  run_bindings={lio/livo: {run_directory, status_sha256}},
  stdout_path, stdout_sha256, stderr_path, stderr_sha256.
All receipt paths may be LAB-relative or absolute paths inside this LAB.
The command must directly name the frozen evaluator and its two --*-run args.
This script does not manufacture that receipt or its actual exit status.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


LAB_DEFAULT = Path(__file__).resolve().parents[1]
GROUP = "outputs/fast-livo2-hilti-gt-001"
COMMIT = "0d2c0346107b75b59934975adec9a6eeeb913c64"
BAG_SHA = "18eea013076b8263ababf4f36eb68d331564a0a493262b006dc0dddc2aa85b82"
GT_SHA = "dec7b20cc11002482ccb416a6e1a1c410d385b843504d3aedf7b5956746af1b6"
GT_FILE = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\fast-livo2-data\hilti2022-recovery-003-exp14\ground_truth\exp14_basement_2_imu.txt")
ID = "fast-livo2-hilti-gt-001"
STAT_KEYS = ("ate_position_se3", "first_matched_pose_gauge_position",
             "common_reference_adjacent_rpe_translation", "common_reference_adjacent_rpe_rotation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time(value):
    stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "UTC offset required in receipt timestamp")
    return stamp.astimezone(dt.timezone.utc)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def all_checks(checks, label):
    require(isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values()), label)


def flatten(parameters, prefix=""):
    result = {}
    for key, value in parameters.items():
        name = prefix + "/" + str(key).strip("/")
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def summarize(lab, execution_path, seal_path):
    # Standard library reads plus installed YAML parser; no evaluator import or run.
    import yaml

    lab = lab.resolve()
    evidence, observed = {}, {}

    def local(value):
        raw = Path(str(value).replace("\\", "/"))
        path = (raw if raw.is_absolute() else lab / raw).resolve()
        require(path.is_relative_to(lab), "Evidence path outside LAB: " + str(value))
        return path

    def track(value, expected=None):
        path = local(value)
        actual = sha(path)
        require(expected is None or actual == expected, "SHA256 mismatch: " + str(value))
        require(path not in observed or observed[path] == actual, "Input changed during summary: " + str(value))
        observed[path] = actual
        evidence[path.relative_to(lab).as_posix()] = {"sha256": actual, "bytes": path.stat().st_size}
        return path

    plan_path = track(GROUP + "/evaluation-plan.json")
    plan_sha = sha(plan_path)
    plan = read(plan_path)
    seal_path = track(seal_path)
    seal = read(seal_path)
    require(plan["status"] == "preregistered_before_actual_Hilti_replays", "Final preregistered plan required")
    require(seal["status"] == "sealed_before_actual_replays", "Real pre-replay seal required")
    require(seal["evaluation_plan_sha256"] == plan_sha, "Plan changed after preregistration")
    prepared = plan["prepared_source_SHA256"]
    require(prepared == seal["prepared_source_sha256"], "Seal and final plan source set differ")
    require(len(prepared) == 6, "Final six-source protocol binding required")
    for name, expected in prepared.items():
        track(name, expected)
    track(plan["static_runner_review"], plan["static_runner_review_sha256"])
    require(seal["static_runner_review_sha256"] == plan["static_runner_review_sha256"], "Runner review seal mismatch")
    evaluator_sha = prepared["scripts/evaluate_fast_livo2_hilti.py"]
    runner_sha = prepared["scripts/fast_livo2_linux/replay_hilti_gt.py"]
    controls = read(local(GROUP + "/evaluation-controls/metrics.json"))
    require(controls["status"] == "passed" and controls["check_count"] == 17, "17 meaningful evaluator controls required")
    all_checks(controls["checks"], "Evaluator controls failed")
    require(len(controls["checks"]) == 17 and controls["evaluator_source_sha256"] == evaluator_sha,
            "Control count or evaluator SHA mismatch")
    prepared_summary = read(local(GROUP + "/preflight-summary.json"))
    for binding in prepared_summary["configs"].values():
        track(binding["path"], binding["sha256"])
    require(plan["reference_SHA256"] == GT_SHA and plan["same_reference_grid_for_both_modes"] is True,
            "Preregistered dense paired reference required")
    amendment = plan["amendment_before_any_actual_replay"]
    require(amendment["minimum_common_reference_fraction"] == .9
            and amendment["minimum_common_reference_rows_for_689"] == 621,
            "Strict final coverage amendment required")

    metrics_path = track(GROUP + "/evaluation/metrics.json")
    metric = read(metrics_path)
    execution_path = track(execution_path)
    execution = read(execution_path)
    require(execution["status"] == "actual_evaluator_completed" and type(execution["child_exit_code"]) is int
            and execution["child_exit_code"] == 0, "Real successful evaluator child exit required")
    require(execution["child_started"] is True and execution["child_python_optimization_disabled"] is True
            and type(execution["actual_child_pid"]) is int and execution["actual_child_pid"] > 0,
            "Actual child identity and assertions-enabled execution required")
    wrapper_sha = execution["wrapper_source_sha256"]
    track("scripts/run_hilti_paired_evaluation.py", wrapper_sha)
    attempt = local(execution["actual_attempt"])
    require(attempt.parent == local(GROUP + "/evaluation/execution-attempts"), "Actual unique evaluator attempt required")
    track(attempt / "wrapper-source.py", wrapper_sha)
    track(attempt / "evaluator-source.py", evaluator_sha)
    track(attempt / "evaluation-plan.json", plan_sha)
    track(attempt / "preregistration.json", sha(seal_path))
    require(execution["input_sha256_before"] == execution["input_sha256_after"],
            "Evaluator inputs changed during actual scoring")
    for name, expected in execution["input_sha256_before"].items():
        track(name, expected)
    for key, expected in (("metrics_sha256", sha(metrics_path)), ("evaluator_source_sha256", evaluator_sha),
                          ("evaluation_plan_sha256", plan_sha), ("preregistration_sha256", sha(seal_path))):
        require(execution[key] == expected, "Evaluator execution binding failed: " + key)
    for stream in ("stdout", "stderr"):
        track(execution[stream + "_path"], execution[stream + "_sha256"])
    command = execution["command"]
    require(isinstance(command, list) and len(command) >= 6 and all(isinstance(v, str) for v in command),
            "Actual direct evaluator argv required")
    require("-O" not in command and "-OO" not in command, "Evaluator assertions must be enabled")
    require(any(Path(token).name == "evaluate_fast_livo2_hilti.py" for token in command), "Wrong evaluator command")
    evaluator_token = next(token for token in command if Path(token).name == "evaluate_fast_livo2_hilti.py")
    require(local(evaluator_token) == local("scripts/evaluate_fast_livo2_hilti.py"), "Command must use the frozen evaluator")
    require(metric["status"] == "accepted_local_two_mode_dense_reference_comparison", "Strict paired evaluation did not succeed")
    require(metric["evaluator_source_sha256"] == evaluator_sha and metric["dense_GT_sha256"] == GT_SHA,
            "Metric producer or dense GT SHA mismatch")
    require(metric["dense_reference_rows"] == 689 and metric["minimum_preregistered_common_reference_fraction"] == .9,
            "Wrong reference or minimum coverage")
    require(metric["maximum_estimate_interpolation_gap_s"] == .2
            and metric["maximum_RPE_reference_pair_interval_s"] == .2,
            "Unregistered interpolation or RPE interval")
    require(metric["actual_parameters_comparable_except_mode_and_output_identity"] is True,
            "Two-mode actual parameters were not accepted")
    indices = metric["common_reference_indices"]
    require(isinstance(indices, list) and all(type(i) is int and 0 <= i < 689 for i in indices)
            and indices == sorted(set(indices)), "Invalid shared original GT indices")
    require(metric["common_reference_rows"] == len(indices) >= 621
            and metric["common_reference_fraction"] == len(indices) / 689 >= .9,
            "Strict paired common reference coverage failed")
    require(set(metric["input_receipts"]) == {"lio", "livo"}
            and set(metric["metrics"]) == {"lio", "livo"}
            and set(execution["run_bindings"]) == {"lio", "livo"}, "Exactly one actual run per mode required")

    require(sha(GT_FILE) == GT_SHA, "Original publisher dense reference changed")
    reference_times = []
    for line in GT_FILE.read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            reference_times.append(int(Decimal(line.split()[0]) * Decimal(1_000_000_000)))
    require(len(reference_times) == 689 and all(b > a for a, b in zip(reference_times, reference_times[1:])),
            "Original GT query timestamps invalid")
    shared_times = [reference_times[i] for i in indices]
    require(metric["common_start_timestamp_ns"] == shared_times[0]
            and metric["common_end_timestamp_ns"] == shared_times[-1], "Shared GT interval mismatch")
    rpe_pairs = sum(b == a + 1 and tb - ta <= 200_000_000
                    for a, b, ta, tb in zip(indices, indices[1:], shared_times, shared_times[1:]))
    require(rpe_pairs > 0, "No unbridged original-reference RPE pairs")

    build_path = track("outputs/fast-livo2-shutdown-fix-001/build-after-verification.json")
    build = read(build_path)
    audit_relative = "offline-audit-runs/20261010T094728Z_ae2afedf/receipt.json"
    recovery = "outputs/fast-livo2-gt-input-recovery-003/"
    input_seal = read(track(recovery + "receipt.json"))
    audit_binding = next((row for key, row in input_seal["evidence"].items()
                          if key.replace("\\", "/") == audit_relative), None)
    require(audit_binding is not None, "Complete input seal lacks actual sensor audit")
    sensor_audit = read(track(recovery + audit_relative, audit_binding["sha256"]))
    sensor_bounds = sensor_audit["observed_topics"]["/hesai/pandar"]
    modes, normalized, timestamps, supported = {}, {}, {}, {}
    for mode in ("lio", "livo"):
        receipt = metric["input_receipts"][mode]
        run = local(receipt["run_directory"])
        require(run.parent == local(GROUP + "/runs"), "Only actual Hilti runs accepted, not preflight or other projects")
        bound = execution["run_bindings"][mode]
        require(local(bound["run_directory"]) == run, "Actual evaluator input run differs from metric receipt")
        require(command.count("--" + mode + "-run") == 1, "Actual argv has missing or duplicate mode run")
        arg_index = command.index("--" + mode + "-run") + 1
        require(arg_index < len(command) and local(command[arg_index]) == run, "Actual argv mode path differs")
        status_path = track(run / "status.json", receipt["actual_status_sha256"])
        require(bound["status_sha256"] == sha(status_path), "Execution and actual status SHA differ")
        state = read(status_path)
        require(state["run_name"] == run.name and state["mode"] == mode and state["source_commit"] == COMMIT,
                "Actual run identity/source pin mismatch")
        require(state["status"] == "actual_local_functional_replay_evidence_passed"
                and state["fusion_started"] is True and state["bag_playback_completed"] is True
                and type(state["fusion_child_exit_code"]) is int and state["fusion_child_exit_code"] == 0,
                "Actual fusion run was incomplete or failed")
        require(state.get("error") is None and state.get("caller_interrupted") is False, "Interrupted/error run rejected")
        all_checks(state["runtime_evidence_checks"], "Actual runtime checks failed")
        children_path = track(run / "child-exit-evidence.json")
        children = read(children_path)
        require(children == state["child_exit_evidence"] and children["all_required_children_exit_zero"] is True,
                "Actual child terminal evidence not bound")
        require(state["rate"] == .5 and state["post_vio_image_interval"] == 20,
                "Half-speed and registered image sampling required")
        require(state["bag"]["bytes"] == 6260771085 and state["bag"]["sha256_before"]
                == state["bag"]["sha256_after"] == BAG_SHA and state["bag"]["unchanged"] is True,
                "Complete official bag before/after binding required")
        prep_path = track(run / "preflight.json", receipt["preflight_sha256"])
        prep = read(prep_path)
        require(prep == state["preflight"] and prep["status"] == "read_only_preflight_passed"
                and len(prep["checks"]) == 13, "Actual embedded preflight mismatch")
        all_checks(prep["checks"], "Actual preflight checks failed")
        require(prep["config"]["sha256"] == prepared_summary["configs"]["HILTI22." + mode + ".upstream-zero.yaml"]["sha256"]
                and prep["config"]["camera_sha256"] == prepared_summary["configs"]["camera_fisheye_HILTI22.yaml"]["sha256"],
                "Actual preflight config/camera differs from preregistered byte hashes")
        require(prep["accepted_build_receipt_SHA256"] == sha(build_path)
                and len(prep["compiled_files"]) == 6
                and all(row["sha256"] == row["expected_sha256"] for row in prep["compiled_files"]),
                "Six compiled files or accepted ABI build binding failed")
        require(state["binary"]["sha256"] == receipt["compiled_binary_sha256"]
                == build["binary_after_sha256"] == prep["compiled_files"][0]["sha256"], "Compiled binary binding failed")
        libraries = {row["path"]: row["sha256"] for row in build["libraries_after"]}
        require(len(libraries) == 5 and {row["path"]: row["sha256"] for row in prep["compiled_files"][1:]} == libraries,
                "Actual five algorithm libraries differ from frozen ABI build")
        before = track(run / "source-before.json", receipt["source_before_sha256"])
        after = track(run / "source-after.json", receipt["source_after_sha256"])
        source = read(before)
        require(source == read(after) and source["selected_receipt_files_verified"] == 75,
                "75 actual source files changed during replay")
        track(run / "runner-source.py", runner_sha)
        require(receipt["runner_sha256"] == runner_sha, "Actual runner differs from final protocol")
        actual_path = track(run / "actual-parameters.yaml", receipt["actual_parameters_sha256"])
        actual = yaml.safe_load(actual_path.read_text(encoding="utf-8"))
        planned_path = track(run / "planned-parameters.yaml")
        planned = yaml.safe_load(planned_path.read_text(encoding="utf-8"))
        flat = flatten(actual)
        require(len(planned) >= 76 and all(key in flat and flat[key] == value for key, value in planned.items()),
                "Full planned camera/extrinsic/algorithm parameters not actually loaded")
        parameter_checks = read(track(run / "actual-parameter-checks.json"))
        require(parameter_checks and all(row["passed"] is True for row in parameter_checks.values()), "Runtime parameter checks failed")
        require(receipt["full_planned_parameters_bound_to_actual"] is True and actual["common"]["img_en"] == int(mode == "livo")
                and actual["image_save"]["interval"] == 20, "Wrong actual mode or sampling")
        require(all(actual["time_offset"][name] == 0 for name in ("img_time_offset", "imu_time_offset", "lidar_time_offset")),
                "Unregistered time offset")
        actual["common"].pop("img_en")
        actual["image_save"].pop("img_save_en")
        actual["evo"].pop("seq_name")
        actual.pop("run_id", None)
        actual.pop("roslaunch", None)
        normalized[mode] = actual
        trajectory = track(run / receipt["trajectory_relative_path"], receipt["trajectory_sha256"])
        require(trajectory.is_relative_to(run / "official-log-after/result"), "Only saved sensor-time primary trajectory accepted")
        require(receipt["trajectory_sha256"] == state["pre_vio_trajectory"]["sha256"]
                and receipt["original_pose_rows"] == state["pre_vio_trajectory"]["rows"], "Primary trajectory binding failed")
        stamps = []
        for line in trajectory.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            require(len(fields) == 8 and all(math.isfinite(float(v)) for v in fields), "Malformed primary trajectory")
            stamps.append(int(Decimal(fields[0]) * Decimal(1_000_000_000)))
            require(abs(sum(float(v) ** 2 for v in fields[4:]) ** .5 - 1) <= 1e-3, "Invalid primary quaternion")
        require(len(stamps) == receipt["original_pose_rows"] >= 3
                and all(b > a for a, b in zip(stamps, stamps[1:])), "Primary sensor timestamps duplicate or regress")
        require("IMU body" in receipt["phase"] and receipt["phase"] == state["pre_vio_trajectory"]["phase"], "Wrong primary output frame/phase")
        require(receipt["primary_startup_delay_from_first_lidar_header_s"] <= 5
                and -.5 <= receipt["primary_end_delay_from_last_lidar_header_s"] <= .5,
                "Primary trajectory lacks full registered sensor-end coverage")
        startup = (stamps[0] - sensor_bounds["first_header_ns"]) / 1e9
        end_delay = (stamps[-1] - sensor_bounds["last_header_ns"]) / 1e9
        require(startup == receipt["primary_startup_delay_from_first_lidar_header_s"] <= 5
                and -.5 <= end_delay == receipt["primary_end_delay_from_last_lidar_header_s"] <= .5,
                "Actual sensor timestamps do not meet or match reported coverage")
        require(receipt["input_offline_audit_sha256"] == audit_binding["sha256"], "Run used a different sensor audit")
        valid_indices, outside, excessive_gap = set(), 0, 0
        for index, query in enumerate(reference_times):
            bracket = bisect_left(stamps, query)
            if bracket < len(stamps) and stamps[bracket] == query:
                valid_indices.add(index)
            elif bracket == 0 or bracket == len(stamps):
                outside += 1
            elif stamps[bracket] - stamps[bracket - 1] > 200_000_000:
                excessive_gap += 1
            else:
                valid_indices.add(index)
        require(receipt["individually_supported_reference_rows"] == len(valid_indices)
                and receipt["rejected_reference_queries"] == {"outside_estimate_span": outside,
                                                              "estimate_bracket_gap_exceeded": excessive_gap},
                "Actual original timestamps disagree with evaluator support/rejection counts")
        supported[mode] = valid_indices
        if mode == "lio":
            markers = state["visual_log_evidence"]["markers"]
            require(markers["handle_vio_raw_features"]["lines"] == markers["retrieved_visual_candidates"]["lines"] == 0
                    and markers["jacobian_stage"]["timing_lines"] == markers["ekf_stage"]["timing_lines"] == 0,
                    "LIO mode unexpectedly executed visual stages")
        start, finish = time(state["started_utc"]), time(state["finished_utc"])
        require(time(seal["sealed_utc"]) < start <= finish <= time(execution["started_utc"]),
                "Preregistration/run/evaluation temporal order invalid")
        timestamps[mode] = (start, finish)
        modes[mode] = {"run_name": run.name, "run_directory": run.relative_to(lab).as_posix(),
                       "status": state["status"], "actual_status_sha256": sha(status_path),
                       "fusion_child_exit_code": state["fusion_child_exit_code"],
                       "trajectory_sha256": sha(trajectory), "trajectory_relative_path": trajectory.relative_to(lab).as_posix(),
                       "original_pose_rows": len(stamps), "sensor_start_timestamp_ns": stamps[0],
                       "sensor_end_timestamp_ns": stamps[-1], "started_utc": state["started_utc"],
                       "finished_utc": state["finished_utc"], "source_before_sha256": sha(before),
                       "source_after_sha256": sha(after), "actual_parameters_sha256": sha(actual_path),
                       "compiled_binary_sha256": state["binary"]["sha256"], "runner_source_sha256": runner_sha,
                       "phase": receipt["phase"], "metrics": metric["metrics"][mode]}
    require(normalized["lio"] == normalized["livo"], "Unregistered two-mode actual parameter difference")
    require(sorted(supported["lio"] & supported["livo"]) == indices, "Shared reference grid differs from actual support intersection")
    require(modes["lio"]["run_name"] != modes["livo"]["run_name"]
            and timestamps["lio"][1] <= timestamps["livo"][0], "Unique serial LIO then LIVO runs required")
    require(time(execution["started_utc"]) <= time(execution["finished_utc"]), "Evaluation time interval invalid")

    for mode in ("lio", "livo"):
        model_metrics = metric["metrics"][mode]
        require(model_metrics["scale"] == 1., "No scale fit permitted")
        for name in STAT_KEYS:
            value = model_metrics[name]
            expected_count = rpe_pairs if "rpe" in name else len(indices)
            require(value["count"] == expected_count and value["unit"] == ("deg" if name.endswith("rotation") else "m"),
                    "Metric sample count/unit mismatch")
            require(all(finite(value[key]) and value[key] >= 0 for key in ("rmse", "mean", "median", "max")),
                    "Invalid finite nonnegative metrics")
        for key in ("last_common_point_aligned_error_m", "last_common_point_first_matched_pose_gauge_error_m"):
            require(finite(model_metrics[key]) and model_metrics[key] >= 0, "Invalid final point error")
        require(model_metrics["reference_adjacent_pairs_excluded"] == len(indices) - 1 - rpe_pairs,
                "Original-reference excluded-pair count mismatch")
    csv_path = track(GROUP + "/evaluation/paired-errors.csv")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(indices), "Raw paired error count mismatch")
    require([int(row["original_GT_index"]) for row in rows] == indices
            and [int(row["timestamp_ns"]) for row in rows] == shared_times, "Raw paired original GT grid mismatch")
    for mode in ("lio", "livo"):
        for column, name in ((mode + "_aligned_position_error_m", "ate_position_se3"),
                             (mode + "_first_matched_pose_error_m", "first_matched_pose_gauge_position")):
            values = [float(row[column]) for row in rows]
            require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid raw paired errors")
            raw_stats = {"rmse": math.sqrt(statistics.fmean(v * v for v in values)),
                         "mean": statistics.fmean(values), "median": statistics.median(values), "max": max(values)}
            for key, value in raw_stats.items():
                require(math.isclose(value, metric["metrics"][mode][name][key], rel_tol=1e-8, abs_tol=1e-10),
                        "Raw CSV and metric statistic mismatch: " + mode + "/" + name + "/" + key)
        track(GROUP + "/evaluation/" + mode + "-aligned-common-position.txt")
    track(GROUP + "/evaluation/paired-reference-errors.png")
    require(set(execution["outputs"]) == {"metrics.json", "paired-errors.csv", "paired-reference-errors.png",
                                          "lio-aligned-common-position.txt", "livo-aligned-common-position.txt"},
            "Actual wrapper output set incomplete")
    for name, binding in execution["outputs"].items():
        path = track(binding["path"], binding["sha256"])
        require(path == local(GROUP + "/evaluation/" + name) and path.stat().st_size == binding["bytes"] > 0,
                "Actual evaluator raw/plot output path/byte binding failed")
    for path, expected in observed.items():
        require(sha(path) == expected, "Evidence changed before output creation: " + str(path))
    require(sha(GT_FILE) == GT_SHA, "Dense reference changed during summary")
    return metric, modes, evidence, execution, plan_sha, sha(seal_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=LAB_DEFAULT)
    parser.add_argument("--execution-receipt", type=Path, default=Path(GROUP + "/evaluation/execution.json"))
    parser.add_argument("--preregistration-receipt", type=Path, default=Path(GROUP + "/evaluation/preregistration.json"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        metric, modes, evidence, execution, plan_sha, seal_sha = summarize(args.lab, args.execution_receipt, args.preregistration_receipt)
        lab = args.lab.resolve()
        progress_path = lab / "manifests/progress.json"
        progress_before = sha(progress_path)
        progress = read(progress_path)
        require(isinstance(progress.get("results"), list) and progress["results"], "Existing progress record shape required")
        require(all({"id", "title", "status", "metrics", "artifacts", "scope"}.issubset(row)
                    for row in progress["results"]), "Progress result item schema changed")
        manifest_path = lab / "manifests/fast-livo2-hilti-gt.json"
        candidate_path = lab / GROUP / "progress-record-candidate.json"
        shared = {key: metric[key] for key in ("dense_GT_sha256", "dense_reference_rows", "common_reference_rows",
                                              "common_reference_fraction", "common_reference_indices",
                                              "common_start_timestamp_ns", "common_end_timestamp_ns")}
        summary = {"id": ID, "status": "completed_local_two_mode_dense_reference_comparison",
                   "updated_utc": execution["finished_utc"], "dataset": "Hilti 2022 exp14_basement_2",
                   "source_commit": COMMIT, "modes": modes, "common_reference": shared,
                   "evaluation_metrics_sha256": evidence[GROUP + "/evaluation/metrics.json"]["sha256"],
                   "actual_evaluator_child_exit_code": execution["child_exit_code"],
                   "evaluation_plan_sha256": plan_sha, "preregistration_sha256": seal_sha,
                   "evaluator_source_sha256": metric["evaluator_source_sha256"],
                   "summarizer_source_sha256": sha(Path(__file__)), "protocol": metric["protocol"],
                   "limitations": metric["limits"], "evidence": evidence,
                   "dense_reference_file": str(GT_FILE), "dense_reference_sha256": GT_SHA,
                   "progress_json_edited": False, "local_only": True}
        def artifact_relative(path):
            candidate = path if path.is_absolute() else lab / path
            return candidate.resolve().relative_to(lab).as_posix()
        artifacts = ["manifests/fast-livo2-hilti-gt.json", GROUP + "/evaluation/metrics.json",
                     artifact_relative(args.execution_receipt), artifact_relative(args.preregistration_receipt),
                     GROUP + "/evaluation/paired-errors.csv", GROUP + "/evaluation/paired-reference-errors.png",
                     GROUP + "/evaluation-plan.json"]
        record = {"id": ID, "title": "FAST-LIVO2 本机 Hilti exp14 双模式与共同 IMU 开发真值对照",
                  "status": summary["status"],
                  "metrics": {"actual_runs": modes, "common_reference": shared,
                              "actual_evaluator_child_exit_code": execution["child_exit_code"],
                              "evaluation_metrics_sha256": summary["evaluation_metrics_sha256"],
                              "evaluation_plan_sha256": plan_sha, "preregistration_sha256": seal_sha,
                              "evaluator_source_sha256": metric["evaluator_source_sha256"]},
                  "artifacts": artifacts,
                  "scope": "完整官方 exp14 输入，串行 0.5× LIO/LIVO 本机回放；共同原始 dense IMU 开发参考时刻，"
                           "各模式事后 SE3 尺度1对齐。主轨迹为 LiDAR 后、当前 VIO 前；LIVO 含既往视觉历史。"
                           "无杆臂或时间偏移拟合，无稀疏测量成绩，不证明物理标定、同步、实时性或跨场景优越性。"}
        if args.validate_only:
            print(json.dumps({"status": "strict_completion_evidence_verified_no_files_written",
                              "runs": [modes[key]["run_name"] for key in ("lio", "livo")],
                              "common_reference_rows": len(metric["common_reference_indices"])}))
            return 0
        # All evidence is checked before either completion file is opened.
        require(not manifest_path.exists() and not candidate_path.exists(), "Refusing to overwrite a previous completion artifact")
        require(sha(progress_path) == progress_before, "Shared progress.json changed while checking its schema")
        for path, value in ((candidate_path, record), (manifest_path, summary)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        require(sha(progress_path) == progress_before, "Shared progress.json changed during independent export")
        print(json.dumps({"status": summary["status"], "manifest": str(manifest_path),
                          "progress_record_candidate": str(candidate_path), "progress_json_edited": False}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print("Hilti completion rejected: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
