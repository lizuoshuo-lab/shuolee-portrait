"""Execute the preregistered local SplaTAM queue serially, retaining failures.

Training completion is distinct from independent trajectory/runtime acceptance.
No dataset fetching, source edits, environment changes or cloud operations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / "outputs/splatam-ablation-repeat-001"
NATIVE_RUNS = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\splatam\outputs\runs")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan_path = OUT / "plan.json"
    plan = read(plan_path)
    expected_weights = {"rgbd": {"im": .5, "depth": 1.0}, "rgb_only": {"im": .5, "depth": 0.0}, "depth_only": {"im": 0.0, "depth": 1.0}}
    if plan["tracking_weights"] != expected_weights:
        raise SystemExit("Preregistered loss profiles do not match the reviewed protocol")
    native_launcher = LAB / "scripts/splatam_local/run_ablation_official.py"
    wrapper = LAB / "scripts/splatam_local/run_ablation_wsl.py"
    if sha(native_launcher) != plan["launcher_sha256"] or sha(wrapper) != plan["windows_wrapper_sha256"]:
        raise SystemExit("Reviewed launchers changed after plan freeze")
    jobs = plan["full_queue"]
    expected_jobs = [("freiburg1_desk", p, 0) for p in ("rgb_only", "depth_only")] + [("freiburg1_desk", p, s) for s in (1, 2) for p in ("rgbd", "rgb_only", "depth_only")] + [("freiburg1_room", "rgbd", 0)]
    if [(j["sequence"], j["tracking_loss"], j["seed"]) for j in jobs] != expected_jobs:
        raise SystemExit("Full queue differs from the reviewed order")
    baseline = LAB / "outputs/splatam-official-runs-001" / plan["baseline"]["run_name"]
    if sha(baseline / "execution.json") != plan["baseline"]["execution_sha256"] or sha(baseline / "params.npz") != plan["baseline"]["params_sha256"]:
        raise SystemExit("Frozen seed0 RGB-D baseline changed")
    if not args.execute:
        print(json.dumps({"status": "prepared_only", "jobs": len(jobs), "plan_sha256": sha(plan_path)}))
        return
    token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    execution = OUT / "executions" / token
    execution.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, execution / "plan.json")
    shutil.copy2(__file__, execution / "batch-source.py")
    receipt = {"schema_version": 1, "status": "running", "created_utc": stamp(), "execution_path": execution.relative_to(LAB).as_posix(), "plan_sha256": sha(plan_path), "batch_source_sha256": sha(Path(__file__)), "jobs": [], "scope": "Serial actual training queue. Independent evaluation is a separate required stage. Failed jobs remain recorded."}
    target = execution / "receipt.json"
    save(target, receipt)
    save(OUT / "batch-progress.json", receipt)
    print(json.dumps({"execution": str(execution), "status": "running", "jobs": len(jobs)}), flush=True)
    for index, job in enumerate(jobs):
        if sha(plan_path) != receipt["plan_sha256"] or sha(native_launcher) != plan["launcher_sha256"] or sha(wrapper) != plan["windows_wrapper_sha256"]:
            receipt.update(status="stopped_reviewed_inputs_changed", finished_utc=stamp(), stopped_before_job=index,
                           stop_reason="Plan or launcher changed after the previous child exited")
            save(target, receipt)
            save(OUT / "batch-progress.json", receipt)
            raise SystemExit("Plan or launcher changed during batch; refusing another job")
        prefix = f"{job['sequence']}_full_320x240_{job['tracking_loss']}_s{job['seed']}_"
        existing = {p.name for p in NATIVE_RUNS.glob(prefix + "*")}
        command = [sys.executable, "-X", "utf8", str(wrapper), "--frames", "-1", "--width", "320", "--sequence", job["sequence"], "--seed", str(job["seed"]), "--tracking-loss", job["tracking_loss"]]
        entry = dict(job, index=index, status="running", started_utc=stamp(), command=command)
        log_path = execution / f"job-{index:02d}.log"
        entry["log_path"] = log_path.relative_to(LAB).as_posix()
        receipt["jobs"].append(entry)
        start = time.monotonic()
        with log_path.open("wb") as log:
            child = subprocess.Popen(command, cwd=LAB, stdout=log, stderr=subprocess.STDOUT)
            entry["windows_wrapper_pid"] = child.pid
            save(target, receipt)
            save(OUT / "batch-progress.json", receipt)
            print(json.dumps({"job": index, "sequence": job["sequence"], "profile": job["tracking_loss"], "seed": job["seed"], "status": "started"}), flush=True)
            while child.poll() is None:
                matches = [p for p in NATIVE_RUNS.glob(prefix + "*") if p.name not in existing]
                if len(matches) > 1:
                    # Do not silently bind a competing unrelated training process.
                    entry["native_binding_error"] = "Multiple new native runs match this job"
                elif len(matches) == 1:
                    entry["run_name"] = matches[0].name
                    try:
                        native_receipt = read(matches[0] / "execution.json")
                        entry["native_status"] = native_receipt["status"]
                        entry["native_child_pid"] = native_receipt.get("pid")
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass
                entry["elapsed_wrapper_seconds"] = time.monotonic() - start
                save(target, receipt)
                save(OUT / "batch-progress.json", receipt)
                time.sleep(5)
            code = child.wait()
        entry.update(windows_wrapper_exit_code=code, wrapper_wall_seconds=time.monotonic() - start, finished_utc=stamp(), log_sha256=sha(log_path))
        matches = [p for p in NATIVE_RUNS.glob(prefix + "*") if p.name not in existing]
        if len(matches) == 1 and "native_binding_error" not in entry:
            run_name = matches[0].name
            entry["run_name"] = run_name
            mirror = LAB / "outputs/splatam-official-runs-001" / run_name
            try:
                actual = read(mirror / "execution.json")
                entry.update(actual_child_exit_code=actual.get("child_exit_code"), actual_status=actual["status"], execution_sha256=sha(mirror / "execution.json"), final_model_sha256=actual.get("params_sha256"))
                bound = actual.get("experiment_tracking_loss") == job["tracking_loss"] and read(mirror / "effective-config.json")["seed"] == job["seed"]
                entry["status"] = "completed_pending_independent_evaluation" if code == 0 and actual.get("child_exit_code") == 0 and actual["status"] == "completed_pending_independent_evaluation" and bound else "failed"
            except (OSError, KeyError, json.JSONDecodeError) as error:
                entry.update(status="failed", error=str(error))
        else:
            entry.update(status="failed", error=entry.get("native_binding_error", "No unique new native run"))
        native_phase = None
        if len(matches) == 1:
            try:
                native_phase = read(matches[0] / "execution.json")["status"]
            except (OSError, KeyError, json.JSONDecodeError):
                native_phase = "unreadable"
        entry["native_status_after_wrapper_exit"] = native_phase
        orphan_risk = (native_phase == "running" or len(matches) > 1
                       or (code != 0 and native_phase == "unreadable"))
        if orphan_risk:
            entry.update(status="failed_potential_native_orphan",
                         error="Wrapper exited but native training completion cannot be certified; no next GPU child will start")
            receipt.update(status="stopped_potential_native_orphan", finished_utc=stamp(), stopped_after_job=index,
                           stop_reason=entry["error"])
            save(target, receipt)
            save(OUT / "batch-progress.json", receipt)
            print(json.dumps({"job": index, "run_name": entry.get("run_name"), "status": receipt["status"], "native_phase": native_phase}), flush=True)
            raise SystemExit(1)
        save(target, receipt)
        save(OUT / "batch-progress.json", receipt)
        print(json.dumps({"job": index, "run_name": entry.get("run_name"), "status": entry["status"], "wrapper_exit": code}), flush=True)
    receipt.update(status="training_queue_completed_pending_independent_evaluation" if all(j["status"] == "completed_pending_independent_evaluation" for j in receipt["jobs"]) else "training_queue_finished_with_failures", finished_utc=stamp())
    receipt["frozen_baseline_unchanged"] = sha(baseline / "execution.json") == plan["baseline"]["execution_sha256"] and sha(baseline / "params.npz") == plan["baseline"]["params_sha256"]
    save(target, receipt)
    save(OUT / "batch-progress.json", receipt)
    print(json.dumps({"status": receipt["status"], "jobs": len(receipt["jobs"]), "execution": str(execution)}), flush=True)
    raise SystemExit(0 if receipt["status"] == "training_queue_completed_pending_independent_evaluation" and receipt["frozen_baseline_unchanged"] else 1)


if __name__ == "__main__":
    main()
