"""Capture one REAL frozen Hilti evaluator subprocess after both actual runs.

Usage: .venv/Scripts/python.exe scripts/run_hilti_paired_evaluation.py
       --lio-run <actual LIO directory> --livo-run <actual LIVO directory>

No ROS/GPU, no scores computed in this wrapper, no frozen source modification.
The evaluator's fixed output directory is used once. Existing execution or any
scoring artifacts cause rejection before a new child is started. Genuine failed
child logs and partial scoring output are retained, never replaced with success.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid


LAB_DEFAULT = Path(__file__).resolve().parents[1]
GROUP = "outputs/fast-livo2-hilti-gt-001"
EXPECTED_OUTPUTS = ("metrics.json", "paired-errors.csv", "paired-reference-errors.png",
                    "lio-aligned-common-position.txt", "livo-aligned-common-position.txt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parsed_time(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "UTC offset required")
    return result.astimezone(dt.timezone.utc)


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=LAB_DEFAULT)
    parser.add_argument("--lio-run", type=Path, required=True)
    parser.add_argument("--livo-run", type=Path, required=True)
    args = parser.parse_args()
    attempt, record, out, child = None, None, None, None
    try:
        require(os.name == "nt", "This wrapper captures the frozen Windows evaluator, whose GT path is WSL UNC")
        require(sys.flags.optimize == 0, "Wrapper and child require Python assertions enabled")
        lab = args.lab.resolve()

        def local(value):
            path = Path(str(value).replace("\\", "/"))
            result = (path if path.is_absolute() else lab / path).resolve()
            require(result.is_relative_to(lab), "Path outside LAB: " + str(value))
            return result

        out = local(GROUP + "/evaluation")
        require(not (out / "execution.json").exists(), "Previous actual execution receipt exists; will not overwrite")
        require(not any((out / name).exists() for name in EXPECTED_OUTPUTS),
                "Previous/partial scoring files exist; will not overwrite")
        script = local("scripts/evaluate_fast_livo2_hilti.py")
        plan_path = local(GROUP + "/evaluation-plan.json")
        seal_path = local(GROUP + "/evaluation/preregistration.json")
        plan, seal = read(plan_path), read(seal_path)
        plan_sha, seal_sha, evaluator_sha = sha(plan_path), sha(seal_path), sha(script)
        require(plan["status"] == "preregistered_before_actual_Hilti_replays"
                and seal["status"] == "sealed_before_actual_replays", "Final preregistered protocol seal required")
        require(plan_sha == seal["evaluation_plan_sha256"]
                and plan["prepared_source_SHA256"] == seal["prepared_source_sha256"], "Final plan changed after seal")
        inputs = {plan_path: plan_sha, seal_path: seal_sha, Path(__file__).resolve(): sha(Path(__file__))}
        require(len(plan["prepared_source_SHA256"]) == 6, "Final six-source plan required")
        for relative, expected in plan["prepared_source_SHA256"].items():
            path = local(relative)
            require(sha(path) == expected, "Prepared source SHA mismatch: " + relative)
            inputs[path] = expected
        require(evaluator_sha == plan["prepared_source_SHA256"]["scripts/evaluate_fast_livo2_hilti.py"], "Wrong frozen evaluator")
        review_path = local(plan["static_runner_review"])
        require(sha(review_path) == seal["static_runner_review_sha256"] == plan["static_runner_review_sha256"],
                "Static runner review seal mismatch")
        inputs[review_path] = sha(review_path)
        bindings, states, runs = {}, {}, {}
        for mode in ("lio", "livo"):
            run = local(getattr(args, mode + "_run"))
            require(run.parent == local(GROUP + "/runs"), "Only actual run directories accepted")
            status_path = run / "status.json"
            state = read(status_path)
            require(state["mode"] == mode and state["run_name"] == run.name
                    and state["status"] == "actual_local_functional_replay_evidence_passed"
                    and state["fusion_started"] is True and state["bag_playback_completed"] is True
                    and type(state["fusion_child_exit_code"]) is int and state["fusion_child_exit_code"] == 0,
                    "Actual successful " + mode + " run required")
            require(parsed_time(seal["sealed_utc"]) < parsed_time(state["started_utc"])
                    <= parsed_time(state["finished_utc"]), "Protocol was not sealed before actual run")
            bindings[mode] = {"run_directory": run.relative_to(lab).as_posix(), "status_sha256": sha(status_path)}
            inputs[status_path] = sha(status_path)
            runs[mode], states[mode] = run, state
        require(runs["lio"] != runs["livo"]
                and parsed_time(states["lio"]["finished_utc"]) <= parsed_time(states["livo"]["started_utc"]),
                "Unique serial LIO then LIVO replays required")
        command = [sys.executable, "-X", "utf8", str(script), "--lio-run", str(runs["lio"]), "--livo-run", str(runs["livo"])]
        name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        attempt = out / "execution-attempts" / name
        attempt.mkdir(parents=True, exist_ok=False)
        for path, filename in ((script, "evaluator-source.py"), (plan_path, "evaluation-plan.json"),
                               (seal_path, "preregistration.json"), (Path(__file__), "wrapper-source.py")):
            shutil.copy2(path, attempt / filename)
        stdout, stderr = attempt / "stdout.log", attempt / "stderr.log"
        record = {"status": "actual_evaluator_running", "actual_attempt": attempt.relative_to(lab).as_posix(),
                  "command": command, "child_exit_code": None, "child_started": False,
                  "started_utc": utc(), "evaluator_source_sha256": evaluator_sha,
                  "evaluation_plan_sha256": plan_sha, "preregistration_sha256": seal_sha,
                  "wrapper_source_sha256": sha(Path(__file__)), "run_bindings": bindings,
                  "child_python_optimization_disabled": True,
                  "stdout_path": stdout.relative_to(lab).as_posix(), "stderr_path": stderr.relative_to(lab).as_posix(),
                  "local_only": True}
        save_new(attempt / "started.json", record)
        env = os.environ.copy()
        env.pop("PYTHONOPTIMIZE", None)
        env["PYTHONUTF8"] = "1"
        with stdout.open("xb") as stdout_stream, stderr.open("xb") as stderr_stream:
            child = subprocess.Popen(command, cwd=lab, env=env, stdout=stdout_stream, stderr=stderr_stream)
            record["child_started"] = True
            record["actual_child_pid"] = child.pid
            record["child_exit_code"] = child.wait(timeout=300)
        record["finished_utc"] = utc()
        record["stdout_sha256"], record["stderr_sha256"] = sha(stdout), sha(stderr)
        record["input_sha256_before"] = {path.relative_to(lab).as_posix(): value for path, value in inputs.items()}
        record["input_sha256_after"] = {path.relative_to(lab).as_posix(): sha(path) for path in inputs}
        require(record["input_sha256_before"] == record["input_sha256_after"], "Frozen protocol/status inputs changed during evaluator")
        outputs = {}
        for filename in EXPECTED_OUTPUTS:
            path = out / filename
            if path.is_file():
                outputs[filename] = {"path": path.relative_to(lab).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
        record["outputs"] = outputs
        require(record["child_exit_code"] == 0, "Actual evaluator child failed; retained raw stdout/stderr")
        require(set(outputs) == set(EXPECTED_OUTPUTS) and all(row["bytes"] > 0 for row in outputs.values()),
                "Actual successful child did not generate every required metric/raw/plot artifact")
        metric = read(out / "metrics.json")
        require(metric["status"] == "accepted_local_two_mode_dense_reference_comparison"
                and metric["evaluator_source_sha256"] == evaluator_sha
                and metric["common_reference_rows"] >= 621 and metric["common_reference_fraction"] >= .9,
                "Child exit zero did not produce accepted strict paired metrics")
        for mode in ("lio", "livo"):
            receipt = metric["input_receipts"][mode]
            require(local(receipt["run_directory"]) == runs[mode]
                    and receipt["actual_status_sha256"] == bindings[mode]["status_sha256"],
                    "Produced metric run identity/status SHA differs from actual argv")
        record["metrics_sha256"] = outputs["metrics.json"]["sha256"]
        record["status"] = "actual_evaluator_completed"
        record["wrapper_exit_code"] = 0
        save_new(attempt / "receipt.json", record)
        save_new(out / "execution.json", record)
        print(json.dumps({"status": record["status"], "child_exit_code": record["child_exit_code"],
                          "execution": str(out / "execution.json"), "actual_attempt": str(attempt)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        if attempt is not None and record is not None:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                record["child_exit_code"] = child.poll()
            record.update(status="actual_evaluator_failed_or_incomplete", error=str(error),
                          finished_utc=utc(), wrapper_exit_code=1)
            for stream in ("stdout", "stderr"):
                path = attempt / (stream + ".log")
                if path.is_file():
                    record[stream + "_sha256"] = sha(path)
            (attempt / "failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if not (attempt / "receipt.json").exists():
                save_new(attempt / "receipt.json", record)
            if out is not None and not (out / "execution.json").exists():
                save_new(out / "execution.json", record)
        print("Hilti actual evaluator rejected/failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
