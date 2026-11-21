"""Prepare/review four frozen V4 full-sequence controls, then execute serially.

Preparation is CPU-only. Execute requires the exact reviewed protocol SHA.
No source, solver, objective, scale, weight or threshold is changed. Every
associated frame and failure is retained; no retries or extra runs are added.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/tum-online-v4-full-sequence-protocol-001.json"
OUTPUT = ROOT / "outputs/tum-online-v4-full-sequence-001"
PYTHON = ROOT / ".venv/Scripts/python.exe"
TRAINER = ROOT / "scripts/train_tum_online_gs_v4.py"
WRAPPER = ROOT / "scripts/run_local.py"
BJT = dt.timezone(dt.timedelta(hours=8))
DEADLINE = dt.datetime.fromisoformat("2026-10-09T08:00:00+08:00")
LOCKED = {
    "scripts/train_tum_online_gs_v4.py": "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f",
    "scripts/geometric_pose_prior.py": "d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b",
    "scripts/tum_rgbd.py": "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11",
    "scripts/track_fixed_map.py": "c6f8ca3e80747decc7805aa567db44179f5a6e17b8ee7c36ef8565273463fdab",
    "scripts/track_rgbd_icp.py": "21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a",
    "scripts/run_local.py": "9f62cfc9cb4370ca966b6920cd318c700c1c575620d3fda84f5be39888ae0cc5",
}
REFERENCE_SOURCES = {
    "track_rgbd_icp.py": LOCKED["scripts/track_rgbd_icp.py"],
    "tum_rgbd.py": LOCKED["scripts/tum_rgbd.py"],
    "audit_tum_trajectory.py": "e7acfaea60297dbc31eccc62b6ea9f50f577fb317209ac50e3342630424a42d7",
    "track_rgbd_icp_full.py": "3bad533c9c3f29104772ea5969bf9bf573205f3519b12c867e99ed3cdbcff5a5",
    "track_rgbd_icp_xyz_full.py": "ca3cd9b9b65c2d0c3c1c80b2055d98bccb3c8ee6ad657e52dcee6ba13be700c2",
}
SCENES = {
    "desk": {"frames": 549, "baseline": "tum-icp-full549-001",
             "archive_sha256": "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
             "previous_runs": ["tum-online-v4-prior0-transfer100-001", "tum-online-v4-prior001-transfer100-001"]},
    "xyz": {"frames": 748, "baseline": "tum-xyz-icp-full748-001",
            "archive_sha256": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
            "previous_runs": ["tum-xyz-online-v4-prior0-transfer100-001", "tum-xyz-online-v4-prior001-transfer100-001"]},
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp(value: dt.datetime | None = None) -> dict:
    value = value or utc_now()
    return {"utc": value.isoformat(), "beijing": value.astimezone(BJT).isoformat()}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_guard() -> dict:
    actual = {relative: sha(ROOT / relative) for relative in LOCKED}
    if actual != LOCKED:
        raise ValueError("Unknown or changed source; no training subprocess may start")
    return actual


def archive_records() -> dict:
    records = {}
    for scene, settings in SCENES.items():
        relative = f"data/downloads/rgbd_dataset_freiburg1_{scene}.tgz"
        path = ROOT / relative
        actual = sha(path)
        if actual != settings["archive_sha256"]:
            raise ValueError("Archive hash mismatch: "+scene)
        records[scene] = {"relative_path": relative, "sha256": actual,
                          "bytes": path.stat().st_size,
                          "hash_provenance": "local exact-file hash, not a publisher checksum"}
    return records


def command_for(configuration: dict) -> list[str]:
    command = [str(PYTHON), str(WRAPPER), str(TRAINER)]
    for key, value in configuration.items():
        if key == "check_only":
            if value is not False:
                raise ValueError("Experimental configuration must not be check-only")
            continue
        command += ["--"+key.replace("_", "-"), str(value)]
    return command


def reference_records() -> tuple[dict, dict]:
    baselines, previous = {}, {}
    for scene, settings in SCENES.items():
        folder = ROOT / "outputs" / settings["baseline"]
        metric, summary, validation, manifest = [read_json(folder/name) for name in
            ("metrics.json", "summary.json", "validation.json", "source_manifest.json")]
        if (metric["status"] != "completed" or summary["status"] != "completed" or
                validation["status"] != "passed" or (folder/"failure.json").exists() or
                summary["frames_processed"] != settings["frames"] or
                validation["frames_in_metrics_and_both_trajectories"] != settings["frames"]):
            raise ValueError("Full CPU baseline is not finalized/validated: "+scene)
        for name, expected in validation["evidence_sha256"].items():
            if sha(folder/name) != expected:
                raise ValueError("Full CPU baseline evidence hash mismatch: "+name)
        if summary["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Full CPU reference source summaries disagree")
        for name, expected in manifest["source_sha256"].items():
            if REFERENCE_SOURCES.get(name) != expected or sha(ROOT/"scripts"/name) != expected:
                raise ValueError("Unknown or changed full CPU reference source: "+name)
        names = ("metrics.json", "summary.json", "validation.json", "source_manifest.json", "selected_sensor_metadata.json")
        baselines[scene] = {"run": settings["baseline"], "frames": settings["frames"],
            "frame_ids_inclusive": [0, settings["frames"]-1], "wrapper_finalization_status": "passed",
            "source_sha256": manifest["source_sha256"],
            "artifact_sha256": {"outputs/"+settings["baseline"]+"/"+name: sha(folder/name) for name in names},
            "validation_evidence_sha256": validation["evidence_sha256"],
            "used_as_reference_only_not_rerun": True}
        previous[scene] = []
        for name, weight in zip(settings["previous_runs"], (0., .01)):
            path = ROOT/"outputs"/name/"metrics.json"
            record = read_json(path)
            config = record["configuration"]
            if (record["status"] != "completed" or record["script_sha256"] != LOCKED[str(TRAINER.relative_to(ROOT)).replace("\\", "/")] or
                    len(config) != 36 or config["seed"] != 0 or config["frames"] != 100 or
                    config["start_frame"] != 200 or config["pose_prior_weight"] != weight or
                    config["depth_weight"] != 1 or config["alpha_weight"] != .01 or
                    config["pose_prior_translation_sigma_m"] != .01 or config["pose_prior_rotation_sigma_deg"] != 1 or
                    config["dataset"] != (ROOT/f"data/tum/rgbd_dataset_freiburg1_{scene}").as_posix()):
                raise ValueError("Unexpected frozen 100-frame witness configuration: "+name)
            previous[scene].append({"run": name, "metrics_relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                    "metrics_sha256": sha(path), "configuration": config,
                                    "existing_result_not_rerun_or_overwritten": True})
        left, right = [row["configuration"] for row in previous[scene]]
        if {key for key in left if left[key] != right[key]} != {"run", "pose_prior_weight"}:
            raise ValueError("Frozen weight-pair differs in unexpected fields")
    left, right = [previous[scene][0]["configuration"] for scene in SCENES]
    if {key for key in left if left[key] != right[key]} != {"run", "dataset"}:
        raise ValueError("Frozen scene configurations differ outside run/dataset")
    return baselines, previous


def parse_checks(runs: list[dict]) -> dict:
    sys.path.insert(0, str(ROOT/"scripts"))
    spec = importlib.util.spec_from_file_location("frozen_v4_full_protocol_parser", TRAINER)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    if trainer.torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation must not initialize CUDA")
    results, previous_argv = [], sys.argv
    try:
        for row in runs:
            sys.argv = [str(TRAINER), *row["command"][3:]]
            parsed = vars(trainer.arguments())
            parsed["dataset"] = parsed["dataset"].resolve().as_posix()
            if parsed != row["configuration"] or len(parsed) != 36:
                raise ValueError("Actual frozen CLI parsing differs from protocol")
            results.append({"run": row["run"], "field_count": len(parsed), "configuration": parsed,
                            "actual_frozen_arguments_parser": True, "trainer_main_called": False})
    finally:
        sys.argv = previous_argv
    if trainer.torch.cuda.is_initialized():
        raise RuntimeError("CLI parsing unexpectedly initialized CUDA")
    return {"status": "passed", "all_four_actual_CLI_configurations_exact": True,
            "CUDA_initialized": False, "runs": results}


def sensor_preflight(runs: list[dict]) -> dict:
    result = {}
    for scene, settings in SCENES.items():
        row = next(row for row in runs if row["scene"] == scene)
        stdout_path = OUTPUT/("cpu_preflight_"+scene+".json")
        stderr_path = OUTPUT/("cpu_preflight_"+scene+".stderr.log")
        command = [str(PYTHON), str(TRAINER), *row["command"][3:], "--check-only"]
        # Frozen main returns before any CUDA availability call or GS import.
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        if completed.returncode:
            raise RuntimeError("Frozen CPU check failed; original output retained: "+scene)
        checked = read_json(stdout_path)
        expected = {**row["configuration"], "check_only": True}
        baseline_sensor = read_json(ROOT/"outputs"/settings["baseline"]/"selected_sensor_metadata.json")
        if (checked["status"] != "cpu_checks_passed" or checked["configuration"] != expected or
                checked["association"]["associated_count"] != settings["frames"] or
                checked["selected_frame_ids"] != list(range(settings["frames"])) or
                checked["selected_sensor_metadata"] != baseline_sensor["selected_frames"] or
                checked["association"] != baseline_sensor["association"] or
                checked["script_sha256"] != LOCKED["scripts/train_tum_online_gs_v4.py"] or
                checked["icp_initializer"]["helper_sha256"] != LOCKED["scripts/track_rgbd_icp.py"] or
                not checked["cpu_checks"]["unique_depth_files"] or
                not checked["cpu_checks"]["initialization_camera_identity"]):
            raise ValueError("Full sensor/association/CPU checks differ from frozen baseline: "+scene)
        result[scene] = {"status": "passed", "command": command,
            "all_sensor_metadata_rows_exact_to_full_CPU_baseline": True, "all_frames_included": settings["frames"],
            "frame_ids_inclusive": [0, settings["frames"]-1], "association": checked["association"],
            "first_frame_initialization": checked["cpu_checks"]["first_frame_initialization"],
            "first_and_last_rgb_timestamps": [checked["selected_sensor_metadata"][index]["rgb_timestamp"] for index in (0, -1)],
            "check_only_difference_is_only_check_only_true": True,
            "same_sensor_preflight_applies_to_both_weights": True,
            "stdout_relative_path": str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
            "stdout_sha256": sha(stdout_path), "stderr_sha256": sha(stderr_path),
            "GT_usage": "timestamp association only; no GT pose optimization or model selection",
            "GPU_runs_started": 0}
    return {"status": "passed", "scenes": result}


def launch_before_deadline(command: list[str], stream, *, clock=utc_now, popen=subprocess.Popen):
    # All source/archive checks, progress serialization and log-open I/O must be
    # complete BEFORE this check. Do no I/O between the clock check and Popen.
    guard = clock()
    if guard >= DEADLINE:
        return None, guard
    return popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT), guard


def deadline_checks() -> dict:
    events, calls = [], []
    def fake_popen(command, **kwargs):
        events.append("Popen")
        calls.append(command)
        return object()
    def check(at):
        events.extend(["hash_guard", "progress_write", "log_open"])
        def fake_clock():
            events.append("actual_clock_guard")
            return at
        return launch_before_deadline(["CPU-only-sentinel"], None, clock=fake_clock, popen=fake_popen)
    started, _ = check(DEADLINE-dt.timedelta(microseconds=1))
    at, _ = check(DEADLINE)
    after, _ = check(DEADLINE+dt.timedelta(microseconds=1))
    if started is None or at is not None or after is not None or len(calls) != 1:
        raise AssertionError("Deadline boundary allowed a forbidden process")
    if events[:5] != ["hash_guard", "progress_write", "log_open", "actual_clock_guard", "Popen"]:
        raise AssertionError("Deadline guard is not immediately before spawn")
    return {"status": "passed", "events": events,
            "one_microsecond_before_allowed": True, "at_and_after_deadline_blocked": True,
            "real_training_or_GPU_subprocesses_started": 0,
            "no_I_O_between_actual_clock_guard_and_Popen": True}


def prepare() -> None:
    if PROTOCOL.exists() or OUTPUT.exists():
        raise FileExistsError("New protocol/batch folder must not overwrite existing evidence")
    sources, archives = source_guard(), archive_records()
    baselines, previous = reference_records()
    runs = []
    for scene, settings in SCENES.items():
        for index, label in enumerate(("0", "001")):
            witness = previous[scene][index]
            name = f"tum-{scene}-online-v4-prior{label}-full{settings['frames']}-001"
            config = {**witness["configuration"], "run": name, "start_frame": 0, "frames": settings["frames"]}
            differences = {key for key in config if config[key] != witness["configuration"][key]}
            if differences != {"run", "start_frame", "frames"}:
                raise ValueError("Unexpected full-sequence configuration changes")
            folder, log = ROOT/"outputs"/name, ROOT/"logs"/(name+".log")
            if folder.exists() or log.exists():
                raise FileExistsError("Predeclared run/log exists: "+name)
            runs.append({"order": len(runs)+1, "run": name, "scene": scene,
                "configuration": config, "configuration_differences_from_100frame_witness": sorted(differences),
                "previous_100frame_reference": witness["run"], "full_ICP_reference": settings["baseline"],
                "output_relative_path": "outputs/"+name, "log_relative_path": "logs/"+name+".log",
                "command": command_for(config)})
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        parsing = parse_checks(runs)
        atomic_json(OUTPUT/"cpu_cli_checks.json", parsing)
        sensor = sensor_preflight(runs)
        atomic_json(OUTPUT/"cpu_sensor_checks.json", sensor)
        deadline = deadline_checks()
        atomic_json(OUTPUT/"cpu_deadline_checks.json", deadline)
        if source_guard() != sources:
            raise ValueError("Frozen source changed during CPU preparation")
        protocol = {"protocol_id": "tum-online-v4-full-sequence-protocol-001",
            "status": "prepared_awaiting_root_review_before_GPU", "prepared_at": stamp(),
            "deadline_beijing": DEADLINE.isoformat(), "new_run_count": 4, "new_runs": runs,
            "source_sha256": sources, "driver_sha256": sha(Path(__file__)), "dataset_archives": archives,
            "full_CPU_baseline_references": baselines, "previous_100frame_records": previous,
            "CPU_check_artifact_sha256": {"outputs/"+OUTPUT.name+"/"+name: sha(OUTPUT/name) for name in
                ("cpu_cli_checks.json", "cpu_sensor_checks.json", "cpu_deadline_checks.json", "cpu_preflight_desk.json", "cpu_preflight_xyz.json")},
            "controls": {"order": [row["run"] for row in runs], "seed": 0, "weights": [0., .01],
                "engineering_sigma_translation_m": .01, "engineering_sigma_rotation_degrees": 1.,
                "depth_weight": 1., "alpha_weight": .01, "all_36_configuration_fields_recorded": True,
                "only_run_start_frame_frames_changed_from_matching_100frame_reference": True,
                "every_associated_frame_including_all_failures_retained": True,
                "GT_poses_final_evaluation_only": True, "no_GT_weight_pose_or_frame_selection": True,
                "strictly_serial_one_GPU_child_at_a_time": True, "no_retries_or_extra_runs": True,
                "source_archive_reference_guards_before_every_new_run": True,
                "nonzero_exit_or_unknown_source_stops_batch_with_original_artifacts": True,
                "actual_deadline_guard_after_hash_progress_log_IO_immediately_before_Popen": True,
                "at_or_after_deadline_no_new_Popen": True, "already_started_process_collected": True},
            "limitations": ["Complete-sequence length controls on two Freiburg1 sequences, not a broad SLAM benchmark",
                "Same engineering weights/scales selected earlier on desk; no full-sequence GT tuning",
                "CUDA reductions are not guaranteed deterministic; no backend or global determinism switch changed",
                "Separate warmup maps may differ between weight arms; exact equality is measured, not presumed",
                "Trajectory reports retain raw anchored error, SE3 no-scale ATE, adjacent RPE and all failure flags",
                "Observed end-to-end wall duration includes loading and serialization; not isolated SLAM FPS"]}
        with PROTOCOL.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2, ensure_ascii=False, allow_nan=False)
        atomic_json(OUTPUT/"progress.json", {"status": "awaiting_root_protocol_review", "prepared_at": stamp(),
            "protocol_sha256": sha(PROTOCOL), "GPU_runs_started": 0, "completed_runs": [],
            "deadline_beijing": DEADLINE.isoformat()})
        print(json.dumps({"protocol": str(PROTOCOL), "protocol_sha256": sha(PROTOCOL),
            "driver_sha256": protocol["driver_sha256"], "runs": [row["run"] for row in runs],
            "CPU_checks": "passed", "GPU_runs_started": 0}, indent=2), flush=True)
    except BaseException as error:
        atomic_json(OUTPUT/"failure.json", {"phase": "CPU_preparation", "at": stamp(),
            "exception_type": type(error).__name__, "exception_message": str(error),
            "traceback": traceback.format_exc(), "GPU_runs_started": 0, "original_evidence_retained": True})
        raise


def execution_guards(protocol: dict, row: dict) -> None:
    if source_guard() != protocol["source_sha256"] or archive_records() != protocol["dataset_archives"]:
        raise ValueError("Frozen source/archive changed since review")
    if sha(Path(__file__)) != protocol["driver_sha256"]:
        raise ValueError("Driver changed since exact protocol review")
    for reference in protocol["full_CPU_baseline_references"].values():
        for relative, expected in reference["artifact_sha256"].items():
            if sha(ROOT/relative) != expected:
                raise ValueError("Full CPU reference changed: "+relative)
        for name, expected in reference["validation_evidence_sha256"].items():
            if sha(ROOT/"outputs"/reference["run"]/name) != expected:
                raise ValueError("Full CPU reference evidence changed: "+name)
        for name, expected in reference["source_sha256"].items():
            if sha(ROOT/"scripts"/name) != expected:
                raise ValueError("Full CPU reference source changed: "+name)
    for records in protocol["previous_100frame_records"].values():
        for record in records:
            if sha(ROOT/record["metrics_relative_path"]) != record["metrics_sha256"]:
                raise ValueError("Previous fixed-configuration witness changed")
    for relative, expected in protocol["CPU_check_artifact_sha256"].items():
        if sha(ROOT/relative) != expected:
            raise ValueError("Reviewed CPU preflight evidence changed")
    if command_for(row["configuration"]) != row["command"]:
        raise ValueError("Command differs from full protocol configuration")


def execute(expected_sha: str) -> None:
    if sha(PROTOCOL) != expected_sha:
        raise ValueError("Protocol SHA differs from exact root-reviewed bytes")
    protocol = read_json(PROTOCOL)
    if sha(Path(__file__)) != protocol["driver_sha256"]:
        raise ValueError("Driver SHA changed after review")
    with (OUTPUT/"execution_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"at": stamp(), "protocol_sha256": expected_sha}, stream, indent=2)
    progress = {"status": "running", "started_at": stamp(), "protocol_sha256": expected_sha,
        "driver_sha256": protocol["driver_sha256"], "source_sha256": protocol["source_sha256"],
        "deadline_beijing": DEADLINE.isoformat(), "GPU_runs_started": 0, "run_records": [], "current_run": None}
    process = None
    try:
        for row in protocol["new_runs"]:
            if utc_now() >= DEADLINE:
                progress["status"] = "deadline_reached_no_further_runs_started"
                progress["not_started_runs"] = [item["run"] for item in protocol["new_runs"][row["order"]-1:]]
                break
            execution_guards(protocol, row)
            folder, log = ROOT/row["output_relative_path"], ROOT/row["log_relative_path"]
            if folder.exists() or log.exists():
                raise FileExistsError("Never overwrite run/log: "+row["run"])
            record = {"order": row["order"], "run": row["run"], "configuration": row["configuration"],
                      "status": "ready_before_actual_deadline_guard", "log": row["log_relative_path"],
                      "ready_at": stamp()}
            progress["current_run"] = row["run"]
            progress["run_records"].append(record)
            atomic_json(OUTPUT/"progress.json", progress)
            print(json.dumps({"event": "ready", "run": row["run"], "order": row["order"]}), flush=True)
            with log.open("xb") as stream:
                # The actual clock guard occurs inside launch, after this log
                # open and every preceding guard/progress operation.
                process, guard = launch_before_deadline(row["command"], stream)
                if process is not None:
                    record.update({"status": "running", "pid": process.pid, "spawn_guard_at": stamp(guard),
                                   "Popen_returned_at": stamp()})
                    progress["GPU_runs_started"] += 1
                    atomic_json(OUTPUT/"progress.json", progress)
                    returncode = process.wait()
            if process is None:
                record.update({"status": "not_started_deadline_reached_after_log_open", "spawn_guard_at": stamp(guard),
                               "empty_reserved_log_retained": True})
                progress["status"] = "deadline_reached_no_further_runs_started"
                progress["not_started_runs"] = [item["run"] for item in protocol["new_runs"][row["order"]-1:]]
                progress["current_run"] = None
                break
            record.update({"returncode": returncode, "ended_at": stamp(), "log_sha256": sha(log)})
            process = None
            if returncode != 0:
                record["status"] = "failed_process_exit_all_original_artifacts_retained"
                raise RuntimeError("Child failed; stop without retry: "+row["run"])
            metric_path = folder/"metrics.json"
            metric = read_json(metric_path)
            if (metric["configuration"] != row["configuration"] or
                    metric["script_sha256"] != LOCKED["scripts/train_tum_online_gs_v4.py"] or
                    not metric["status"].startswith("completed") or
                    metric["selected_frame_ids"] != list(range(row["configuration"]["frames"])) or
                    len(metric["frame_metrics"]) != row["configuration"]["frames"]):
                raise RuntimeError("Exit0 artifact does not match complete protocol frames/config/source")
            record.update({"status": metric["status"], "metrics_sha256": sha(metric_path),
                "frames": len(metric["frame_metrics"]), "failure_frame_count": metric["failure_frame_count"],
                "pose_gradient_checks": metric["pose_gradient_checks"],
                "trajectory_summary": {key: metric["trajectory"][key] for key in
                    ("ate_rmse_m_se3_no_scale", "raw_anchored_translation_rmse_m", "adjacent_rpe_translation_rmse_m", "adjacent_rpe_rotation_rmse_degrees")}})
            progress["current_run"] = None
            atomic_json(OUTPUT/"progress.json", progress)
            print(json.dumps({"event": "completed", **record}, ensure_ascii=False), flush=True)
        else:
            progress["status"] = "all_four_predeclared_full_sequence_runs_completed"
        progress["ended_at"] = stamp()
        atomic_json(OUTPUT/"progress.json", progress)
        atomic_json(OUTPUT/"metrics.json", progress)
        print(json.dumps({"status": progress["status"], "GPU_runs_started": progress["GPU_runs_started"]}), flush=True)
    except BaseException as error:
        # Even an error after Popen/progress I/O cannot leave a launched child
        # competing with another GPU job. Collect it before reporting failure.
        collected = None
        if process is not None:
            collected = {"pid": process.pid, "returncode": process.wait(), "collected_at": stamp()}
        progress.update({"status": "batch_stopped_failure_no_retry", "failed_at": stamp(),
            "exception_type": type(error).__name__, "exception_message": str(error),
            "traceback": traceback.format_exc(), "already_started_child_collected": collected,
            "all_original_artifacts_and_logs_retained": True})
        atomic_json(OUTPUT/"progress.json", progress)
        atomic_json(OUTPUT/"failure.json", progress)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha", help="Exact root-reviewed SHA required for execute")
    args = parser.parse_args()
    if args.prepare:
        if args.protocol_sha is not None:
            parser.error("--protocol-sha applies only to --execute")
        prepare()
    else:
        if not args.protocol_sha or len(args.protocol_sha) != 64 or any(c not in "0123456789abcdef" for c in args.protocol_sha):
            parser.error("--execute requires the 64-character lowercase reviewed protocol SHA")
        execute(args.protocol_sha)


if __name__ == "__main__":
    main()
