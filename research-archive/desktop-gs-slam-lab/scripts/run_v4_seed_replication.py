"""Prepare/run the predeclared eight V4 seed1/seed2 controls, strictly serial.

This driver never changes algorithms, thresholds, weights, engineering scales,
or an existing experiment. It calls only the locked trainer through run_local.
Prepare first, then execute with the reviewed exact protocol SHA. No new run
starts on/after the Beijing deadline; an already-started run is collected.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/tum-online-v4-seed-replication-protocol-001.json"
OUTPUT = ROOT / "outputs/tum-online-v4-seed-replication-001"
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
}
ARCHIVES = {
    "desk": "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
    "xyz": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
}
SEED0 = {
    "desk": ("tum-online-v4-prior0-transfer100-001", "tum-online-v4-prior001-transfer100-001"),
    "xyz": ("tum-xyz-online-v4-prior0-transfer100-001", "tum-xyz-online-v4-prior001-transfer100-001"),
}


def now() -> dict:
    utc = dt.datetime.now(dt.timezone.utc)
    return {"utc": utc.isoformat(), "beijing": utc.astimezone(BJT).isoformat()}


def before_deadline(at: dt.datetime | None = None) -> bool:
    return (at or dt.datetime.now(dt.timezone.utc)) < DEADLINE


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


def locked_sources() -> dict:
    actual = {relative: sha(ROOT / relative) for relative in LOCKED}
    if actual != LOCKED:
        raise ValueError("Unknown or changed training/helper source SHA; no subprocess may start")
    return actual


def archive_records() -> dict:
    result = {}
    for scene, expected in ARCHIVES.items():
        relative = f"data/downloads/rgbd_dataset_freiburg1_{scene}.tgz"
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError("Dataset archive SHA mismatch: "+scene)
        result[scene] = {"relative_path": relative, "sha256": expected, "bytes": path.stat().st_size,
                         "hash_provenance": "locally computed exact-file hash, not a publisher checksum"}
    return result


def command_for(configuration: dict) -> list[str]:
    command = [str(PYTHON), str(WRAPPER), str(TRAINER)]
    for key, value in configuration.items():
        if key == "check_only":
            if value is not False:
                raise ValueError("This protocol's eight experimental runs must not be check-only")
            continue
        command += ["--"+key.replace("_", "-"), str(value)]
    return command


def prepare() -> None:
    if PROTOCOL.exists() or OUTPUT.exists():
        raise FileExistsError("Protocol/batch directory already exists; refusing replacement")
    sources, archives = locked_sources(), archive_records()
    seed0_records, bases = {}, {}
    for scene, names in SEED0.items():
        seed0_records[scene] = []
        for name, weight in zip(names, (0., .01)):
            path = ROOT / "outputs" / name / "metrics.json"
            metric = json.loads(path.read_text(encoding="utf-8"))
            config = metric["configuration"]
            if metric["status"] != "completed" or metric["script_sha256"] != LOCKED["scripts/train_tum_online_gs_v4.py"]:
                raise ValueError("Seed0 witness is not the existing locked completed experiment")
            if config["seed"] != 0 or config["pose_prior_weight"] != weight or config["frames"] != 100 or config["start_frame"] != 200:
                raise ValueError("Wrong seed0 witness/configuration")
            if config["dataset"] != (ROOT / f"data/tum/rgbd_dataset_freiburg1_{scene}").as_posix():
                raise ValueError("Wrong scene in seed0 witness")
            seed0_records[scene].append({"run": name, "weight": weight, "seed": 0,
                                         "metrics_sha256": sha(path), "configuration": config,
                                         "existing_result_reused_without_rerun": True})
        base, other = [record["configuration"] for record in seed0_records[scene]]
        if [key for key in base if base[key] != other[key] and key != "run"] != ["pose_prior_weight"]:
            raise ValueError("Seed0 pair differs in an unexpected configuration")
        bases[scene] = base.copy()
    desk, xyz = bases.values()
    if [key for key in desk if desk[key] != xyz[key] and key not in ("run", "dataset")]:
        raise ValueError("Scene base configurations are not identical")
    runs = []
    for seed in (1, 2):
        for scene in ("desk", "xyz"):
            for weight, label in ((0., "0"), (.01, "001")):
                name = f"tum-{scene}-online-v4-prior{label}-seed{seed}-transfer100-001"
                config = {**bases[scene], "run": name, "seed": seed, "pose_prior_weight": weight}
                if (ROOT / "outputs" / name).exists() or (ROOT / "logs" / (name+".log")).exists():
                    raise FileExistsError("Predeclared run/log already exists: "+name)
                runs.append({"order": len(runs)+1, "run": name, "scene": scene,
                             "configuration": config, "command": command_for(config),
                             "log_relative_path": "logs/"+name+".log",
                             "output_relative_path": "outputs/"+name,
                             "seed0_reference_run": SEED0[scene][0 if weight == 0 else 1]})
    time = now()
    protocol = {"protocol_id": "tum-online-v4-seed-replication-protocol-001",
                "status": "prepared_awaiting_root_review_before_GPU", "protocol_prepared_at": time,
                "execution_start_time_recorded_in_batch_progress": True,
                "deadline_beijing": DEADLINE.isoformat(), "new_runs": runs,
                "source_sha256": sources, "wrapper_sha256": sha(WRAPPER),
                "driver_sha256": sha(Path(__file__)), "dataset_archives": archives,
                "existing_seed0_records": seed0_records, "new_run_count": 8,
                "controls": {"seeds": [1, 2], "scenes": ["desk", "xyz"], "weights": [0., .01],
                             "engineering_sigma_translation_m": .01, "engineering_sigma_rotation_deg": 1.,
                             "strictly_serial_one_GPU_child_at_a_time": True, "no_seed0_rerun": True,
                             "no_GroundTruth_selection_or_tuning": True, "no_best_seed_selection": True,
                             "all_failed_frames_results_logs_retained": True, "no_retries_or_additional_runs": True,
                             "no_new_start_at_or_after_deadline": True, "already_started_run_collected": True,
                             "abort_batch_on_source_mismatch_or_nonzero_process_exit": True},
                "limits": ["Seeds change first-map point subsampling and optimization histories; all three seeds will be reported",
                           "CUDA reduction/rasterization is not guaranteed deterministic; no global deterministic switch changes the backend",
                           "Paired warmup map equality must be measured, not assumed; this is not a strictly identical-map control",
                           "Only two sequences in the same Freiburg1 sensor/domain; not a global SLAM benchmark"],
                "CPU_guard_checks": {"before_deadline_now": before_deadline(),
                                     "at_deadline_is_not_before": not before_deadline(DEADLINE),
                                     "one_microsecond_before_is_before": before_deadline(DEADLINE-dt.timedelta(microseconds=1)),
                                     "one_microsecond_after_is_not_before": not before_deadline(DEADLINE+dt.timedelta(microseconds=1))}}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with PROTOCOL.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_json(OUTPUT / "progress.json", {"status": "awaiting_root_protocol_review", "prepared_at": time,
                "protocol_relative_path": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha(PROTOCOL),
                "GPU_runs_started": 0, "completed_runs": [], "deadline_beijing": DEADLINE.isoformat()})
    print(json.dumps({"protocol": str(PROTOCOL), "protocol_sha256": sha(PROTOCOL),
                       "driver_sha256": protocol["driver_sha256"], "runs": [row["run"] for row in runs],
                       "prepared_at": time, "GPU_runs_started": 0}, indent=2), flush=True)


def execute(expected_sha: str) -> None:
    if sha(PROTOCOL) != expected_sha:
        raise ValueError("Protocol SHA differs from the root-reviewed exact bytes")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha(Path(__file__)) != protocol["driver_sha256"] or sha(WRAPPER) != protocol["wrapper_sha256"]:
        raise ValueError("Driver/wrapper source changed after protocol review")
    with (OUTPUT / "execution_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"started_at": now(), "protocol_sha256": expected_sha}, stream, indent=2)
    progress = {"status": "running", "started_at": now(), "protocol_sha256": expected_sha,
                "deadline_beijing": DEADLINE.isoformat(), "run_records": [], "GPU_runs_started": 0,
                "current_run": None}
    atomic_json(OUTPUT / "progress.json", progress)
    try:
        for row in protocol["new_runs"]:
            if not before_deadline():
                progress["status"] = "deadline_reached_no_further_runs_started"
                progress["not_started_runs"] = [item["run"] for item in protocol["new_runs"][row["order"]-1:]]
                break
            if locked_sources() != protocol["source_sha256"] or archive_records() != protocol["dataset_archives"]:
                raise ValueError("Source/dataset guard changed before new run")
            if sha(WRAPPER) != protocol["wrapper_sha256"]:
                raise ValueError("Wrapper guard changed before new run")
            if command_for(row["configuration"]) != row["command"]:
                raise ValueError("Command does not match the full protocol configuration")
            folder, log = ROOT / row["output_relative_path"], ROOT / row["log_relative_path"]
            if folder.exists() or log.exists():
                raise FileExistsError("Cannot overwrite predeclared run/log: "+row["run"])
            # Hash I/O may take time. Check the actual wall clock again directly
            # before opening the log and starting the subprocess.
            if not before_deadline():
                progress["status"] = "deadline_reached_no_further_runs_started"
                progress["not_started_runs"] = [item["run"] for item in protocol["new_runs"][row["order"]-1:]]
                break
            record = {"run": row["run"], "order": row["order"], "configuration": row["configuration"],
                      "started_at": now(), "status": "running", "log": row["log_relative_path"]}
            progress["current_run"] = row["run"]
            progress["run_records"].append(record)
            atomic_json(OUTPUT / "progress.json", progress)
            print(json.dumps({"event": "starting", "order": row["order"], "run": row["run"],
                              "started_at": record["started_at"]}), flush=True)
            with log.open("xb") as stream:
                process = subprocess.Popen(row["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                progress["GPU_runs_started"] += 1
                atomic_json(OUTPUT / "progress.json", progress)
                returncode = process.wait()
            record.update({"returncode": returncode, "ended_at": now(), "log_sha256": sha(log)})
            if returncode != 0:
                record["status"] = "failed_process_exit_all_original_artifacts_retained"
                raise RuntimeError("Training process failed without retry: "+row["run"])
            metric_path = folder / "metrics.json"
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            if metric["configuration"] != row["configuration"] or metric["script_sha256"] != LOCKED["scripts/train_tum_online_gs_v4.py"]:
                raise RuntimeError("Completed artifact config/source differs from the protocol")
            if not metric["status"].startswith("completed"):
                raise RuntimeError("Exit0 did not produce a completed metric artifact")
            record.update({"status": metric["status"], "metrics_sha256": sha(metric_path),
                           "failure_frame_count": metric["failure_frame_count"],
                           "pose_gradient_checks": metric["pose_gradient_checks"],
                           "trajectory_summary": {key: metric["trajectory"][key] for key in
                                ("ate_rmse_m_se3_no_scale", "raw_anchored_translation_rmse_m",
                                 "adjacent_rpe_translation_rmse_m", "adjacent_rpe_rotation_rmse_degrees")}})
            progress["current_run"] = None
            atomic_json(OUTPUT / "progress.json", progress)
            print(json.dumps({"event": "completed", **record}, ensure_ascii=False), flush=True)
        else:
            progress["status"] = "all_eight_predeclared_runs_completed"
        progress["ended_at"] = now()
        atomic_json(OUTPUT / "progress.json", progress)
        atomic_json(OUTPUT / "metrics.json", progress)
        print(json.dumps({"status": progress["status"], "GPU_runs_started": progress["GPU_runs_started"]}), flush=True)
    except BaseException as error:
        progress.update({"status": "batch_stopped_failure_no_retry", "failed_at": now(),
                         "exception_type": type(error).__name__, "exception_message": str(error),
                         "traceback": traceback.format_exc(), "all_started_run_artifacts_retained": True})
        atomic_json(OUTPUT / "progress.json", progress)
        atomic_json(OUTPUT / "failure.json", progress)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha", help="Exact root-reviewed protocol SHA, required to execute")
    args = parser.parse_args()
    if args.prepare:
        if args.protocol_sha is not None:
            parser.error("--protocol-sha only applies to --execute")
        prepare()
    else:
        if not args.protocol_sha or len(args.protocol_sha) != 64:
            parser.error("--execute requires exact reviewed --protocol-sha")
        execute(args.protocol_sha)


if __name__ == "__main__":
    main()
