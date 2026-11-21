"""Explicitly recover a preflight-only 4x failure without changing frozen jobs.

Run on Linux with the same Python/CUDA environment as the original job. Only
the corrected first-stage preflight uses a separately frozen current package;
training, export, evaluation and the report use the ORIGINAL frozen package.
This controller refuses an existing 4x model or a prior recovery attempt.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha(folder):
    digest = hashlib.sha256()
    files = sorted(folder.glob("*.py"))
    require(bool(files), "Missing pipeline package")
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_preflight_only_change(old_package, new_package):
    before = {p.name: p for p in old_package.glob("*.py")}
    after = {p.name: p for p in new_package.glob("*.py")}
    require(set(before) == set(after), "Pipeline package file set changed")
    for name in before:
        if name != "engine.py":
            require(
                before[name].read_bytes() == after[name].read_bytes(),
                "Unexpected package change: " + name,
            )
    old = ast.parse(before["engine.py"].read_text(encoding="utf-8"))
    new = ast.parse(after["engine.py"].read_text(encoding="utf-8"))

    def assignment(tree):
        engine = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Engine"
        )
        preflight = next(
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name == "preflight"
        )
        found = [
            node
            for node in ast.walk(preflight)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "backward_level"
                for target in node.targets
            )
        ]
        require(
            len(found) == 1, "Expected exactly one preflight backward_level assignment"
        )
        return found[0]

    old_assignment, new_assignment = assignment(old), assignment(new)
    require(
        ast.dump(old_assignment.value) == ast.dump(ast.Constant(value=1)),
        "Original preflight is not the failed full-size version",
    )
    expected = ast.parse("2 if scale == 4 else 1", mode="eval").body
    require(
        ast.dump(new_assignment.value) == ast.dump(expected),
        "New preflight differs from the approved warmup correction",
    )
    new_assignment.value = copy.deepcopy(old_assignment.value)
    require(
        ast.dump(old, include_attributes=False)
        == ast.dump(new, include_attributes=False),
        "Engine changed outside the approved preflight assignment",
    )
    return dict(
        passed=True,
        engine_ast_only_change="Engine.preflight backward_level: 1 -> 2 if scale == 4 else 1",
        all_other_engine_ast_identical=True,
        other_package_files_byte_identical=True,
    )


def source_files(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pyo")
    )


def protected_inputs(run, job_file, job, plan):
    """Capture bytes, not learned-state copies; generated 4x outputs are excluded."""
    paths = {
        job_file,
        run / "fast.config",
        Path(job["swinir_weight"]),
        run / "swinir4x/superresolution.json",
        run / "swinir4x/pixels.json",
        run / "swinir4x/training-inputs.json",
    }
    for relative in (
        "prepared",
        "pipeline-source",
        "source",
        "original",
        "interpolation2x",
        "swinir4x/dataset",
        "swinir4x/rgb-cache-1",
        "swinir4x/rgb-cache-2",
    ):
        folder = run / relative
        require(folder.is_dir(), "Missing immutable input folder: " + relative)
        paths.update(source_files(folder))
    for row in plan["source_checks"]:
        require(
            sha(run / "source" / row["path"]) == row["sha256"],
            "Native source changed: " + row["path"],
        )
    require(
        sha(run / "fast.config") == plan["fast_sha256"], "Native config/fast changed"
    )
    for name in ("original", "interpolation2x"):
        export = run / "export" / name
        require(
            read(run / name / "execution.json")["state"] == "training_completed",
            name + " not complete",
        )
        require(
            read(run / name / "ply-parity.json")["passed"]
            and read(export / "native-ply-parity.json")["passed"]
            and read(export / "upright-parity.json")["passed"],
            name + " parity not complete",
        )
        paths.update(source_files(export))
    sr = read(run / "swinir4x/superresolution.json")
    pixels = read(run / "swinir4x/pixels.json")
    rows = read(run / "swinir4x/training-inputs.json")["records"]
    require(
        sr["state"] == "completed" and pixels["scale"] == 4,
        "SwinIR inputs are not complete",
    )
    require(
        len(sr["records"]) == pixels["count"] == len(rows) == plan["frame_count"],
        "4x frame counts differ",
    )
    sr_by_name = {row["name"]: row for row in sr["records"]}
    px_by_name = {row["name"]: row for row in pixels["items"]}
    require(
        len(sr_by_name) == len(px_by_name) == len(rows), "Duplicate 4x image identities"
    )
    hashes = {str(path): sha(path) for path in sorted(paths)}
    for row in rows:
        image = Path(row["image"])
        require(
            row["name"] in sr_by_name and row["name"] in px_by_name, "Missing SR frame"
        )
        wanted = sr_by_name[row["name"]]["output_sha256"]
        require(
            hashes[str(image)]
            == wanted
            == px_by_name[row["name"]]["image_sha256"]
            == row["source_sha256"],
            "4x input pixel hash differs: " + row["name"],
        )
    return dict(
        files=hashes,
        frame_count=len(rows),
        sr_frames=len(sr_by_name),
        scope="Original job, frozen pipeline/native source, prepared/source RGB and camera data, all existing 1x/2x runtime and export artifacts, all 4x RGB/caches/cameras, official SR weight",
        excludes="Mutable root state/plan, new 4x preflight/training/export outputs, generated Python bytecode",
    )


def verify_protected(receipt):
    for path, expected in receipt["files"].items():
        require(sha(path) == expected, "Protected file changed: " + path)


def run_stage(command, log, cwd, environment, lock_fd):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(
            [str(v) for v in command],
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(lock_fd,),
        )
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def record_status(record, receipt, phase, state):
    path = record / "测试状态.json"
    current = read(path) if path.exists() else {}
    if "pipeline_state" in current:
        current.setdefault(
            "initial_pipeline_state", copy.deepcopy(current["pipeline_state"])
        )
    if "pipeline_exit_code" in current:
        current.setdefault(
            "initial_pipeline_exit_code", current.pop("pipeline_exit_code")
        )
    canonical = read(Path(receipt["run_dir"]) / "state.json")
    current.update(
        state=canonical["state"],
        status_label=state,
        phase=phase,
        updated=now(),
        recovery=receipt,
        original_failure=receipt["original_failures"],
        pipeline_state=canonical,
        training_completed=canonical["training_completed"],
        completed_scales=canonical["completed_scales"],
        recovery_exit_code=0
        if receipt["state"] == "completed"
        else 1
        if receipt["state"] == "failed"
        else None,
        quality_accepted=False,
    )
    save(path, current)


def copy_record(run, record, recovery, receipt, phase, state):
    # Only pipeline-export names are updated; record-only regional analyses remain.
    shutil.copytree(run / "export", record, dirs_exist_ok=True)
    shutil.copy2(record / "comparison.html", record / "查看完整图倍率对照.html")
    for name in ("state", "plan", "job"):
        shutil.copy2(run / (name + ".json"), record / ("训练_" + name + ".json"))
    for source, target in (
        (recovery / "code-receipt.json", "恢复代码核查.json"),
        (recovery / "recovery-state.json", "恢复记录.json"),
        (recovery / "full4-capacity.json", "四倍显存容量核查.json"),
        (recovery / "archive/state.json", "初次失败_state.json"),
        (recovery / "archive/plan.json", "初次失败_plan.json"),
        (
            recovery / "archive/swinir4x-preflight.log",
            "初次失败_swinir4x-preflight.log",
        ),
    ):
        shutil.copy2(source, record / target)
    record_status(record, receipt, phase, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Original portable job.json")
    parser.add_argument(
        "--record", required=True, help="Linux-accessible /mnt/d/00/记录/... directory"
    )
    parser.add_argument(
        "--package",
        default="/mnt/d/Projects/msv2/msv2",
        help="Reviewed corrected public package",
    )
    args = parser.parse_args(argv)
    require(os.name == "posix", "This controller must run on Linux")
    job_file = Path(args.job).resolve()
    if job_file.is_dir():
        job_file /= "job.json"
    job = read(job_file)
    require(job["schema"] == "msv2-job-v1", "Expected portable job.json")
    run, package, record = (
        Path(job["run_dir"]).resolve(),
        Path(args.package).resolve(),
        Path(args.record).resolve(),
    )
    require(job_file == run / "job.json", "Use the original job path")
    require(
        record != run and run not in record.parents, "Record must be outside the run"
    )
    old_snapshot = run / "pipeline-source"
    require(
        implementation_sha(old_snapshot / "msv2") == job["implementation_sha256"],
        "Original frozen pipeline changed",
    )
    ast_receipt = verify_preflight_only_change(old_snapshot / "msv2", package)
    sys.path.insert(0, str(old_snapshot))
    from msv2.pipeline import process_lock

    require(
        Path(sys.modules["msv2.pipeline"].__file__).parent.resolve()
        == (old_snapshot / "msv2").resolve(),
        "Controller imported another pipeline",
    )

    def interrupted(number, frame):
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, interrupted)
    with process_lock(run.parent.parent / "queue/gpu.lock") as lock_fd:
        state, plan = read(run / "state.json"), read(run / "plan.json")
        require(
            state["state"] == "partial" and sorted(state["completed_scales"]) == [1, 2],
            "Expected preserved 1x/2x partial result",
        )
        require(
            set(state["failures"]) == {"4"} and not state["training_completed"],
            "Expected only a failed 4x preflight",
        )
        require(
            job["options"]["scales"] == [1, 2, 4] and plan["options"] == job["options"],
            "Unexpected frozen options",
        )
        require(
            not (run / "swinir4x/model").exists()
            and not (run / "swinir4x/execution.json").exists(),
            "4x training has already started",
        )
        require(
            not (run / "swinir4x/preflight.json").exists(),
            "4x preflight already completed",
        )
        failure_log = run / "logs/swinir4x-preflight.log"
        require(
            failure_log.is_file()
            and "OutOfMemoryError" in failure_log.read_text(encoding="utf-8"),
            "Original OOM log is missing",
        )
        recovery = run / "recovery"
        archive, new_snapshot = recovery / "archive", recovery / "pipeline-source"
        require(
            not archive.exists()
            and not new_snapshot.exists()
            and not (recovery / "code-receipt.json").exists(),
            "Recovery already prepared; never overwrite an earlier attempt",
        )
        capacity_path = recovery / "full4-capacity.json"
        capacity = read(capacity_path)
        require(
            capacity["state"] == "passed"
            and capacity["diagnostic_only"]
            and capacity["model_updates"] == capacity["optimizer_updates"] == 0
            and capacity["trained_initializer"] is False
            and capacity["model_file_unchanged"],
            "Read-only full-size capacity proxy did not pass",
        )
        require(
            len(capacity["views"]) == 3
            and capacity["points"] == read(run / "original/execution.json")["points"],
            "Capacity proxy identities differ",
        )
        require(
            sha(capacity["model_source"]) == capacity["model_sha256"]
            and Path(capacity["model_source"]).resolve()
            == run
            / f"original/model/point_cloud/iteration_{job['options']['iterations']}/point_cloud.ply",
            "Capacity proxy was not the current read-only 1x model",
        )
        protected = protected_inputs(run, job_file, job, plan)
        archive.mkdir(parents=True)
        for source, name in (
            (run / "state.json", "state.json"),
            (run / "plan.json", "plan.json"),
            (job_file, "job.json"),
            (failure_log, "swinir4x-preflight.log"),
        ):
            shutil.copy2(source, archive / name)
        error_path = run / "logs/swinir4x-error.txt"
        if error_path.exists():
            shutil.copy2(error_path, archive / error_path.name)
        shutil.copytree(
            package,
            new_snapshot / "msv2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        require(
            implementation_sha(new_snapshot / "msv2") == implementation_sha(package),
            "New package changed during freezing",
        )
        verify_preflight_only_change(old_snapshot / "msv2", new_snapshot / "msv2")
        save(recovery / "protected-inputs.json", protected)
        code = dict(
            schema="msv2-unstarted4-recovery-v1",
            created=now(),
            controller=str(Path(__file__).resolve()),
            controller_sha256=sha(__file__),
            original_job=str(job_file),
            original_job_sha256=sha(job_file),
            original_implementation_sha256=job["implementation_sha256"],
            original_pipeline_source=str(old_snapshot),
            corrected_preflight_source=str(new_snapshot),
            corrected_preflight_implementation_sha256=implementation_sha(
                new_snapshot / "msv2"
            ),
            ast_check=ast_receipt,
            public_package_source=str(package),
            protected_receipt="protected-inputs.json",
            protected_file_count=len(protected["files"]),
            full_capacity_proxy=str(capacity_path),
            full_capacity_proxy_sha256=sha(capacity_path),
            capacity_scope=capacity["scope"],
            capacity_optimizer_updates=0,
            capacity_model_is_training_initializer=False,
            unchanged_native_training_source=True,
            unchanged_input_resolution=True,
            unchanged_training_algorithm=True,
            no_swinir_rerun=True,
            no_1x_2x_retraining=True,
            old_3d_used_for_fourx_initialization=False,
            stage_sources=dict(
                preflight=str(new_snapshot),
                train=str(old_snapshot),
                export=str(old_snapshot),
                evaluate=str(old_snapshot),
                report=str(old_snapshot),
            ),
            scope="Only initial native 4x warmup preflight is corrected. Fresh 4x native training/export/evaluation remain original frozen code. Proxy capacity does not certify actual completed full-stage training.",
        )
        save(recovery / "code-receipt.json", code)
        receipt = dict(
            state="recovering",
            run_dir=str(run),
            recovery_started=now(),
            pid=os.getpid(),
            original_failures=copy.deepcopy(state["failures"]),
            original_state_archive="archive/state.json",
            original_plan_archive="archive/plan.json",
            code_receipt="code-receipt.json",
            stages=[],
            completed_scales=[1, 2],
            fresh_fourx=True,
            old_3d_initializer=False,
            quality_accepted=False,
            full_capacity_proxy_only=True,
            full_fourx_training_completed=False,
        )
        history = state.setdefault("recovered_failures", [])
        history.append(
            dict(
                recorded=now(),
                failures=copy.deepcopy(state["failures"]),
                original_finished=state.get("finished"),
                archive=str(archive),
                recovery_started=receipt["recovery_started"],
                pid=os.getpid(),
            )
        )
        state.update(
            pid=os.getpid(),
            recovery_started=receipt["recovery_started"],
            state="running",
            training_completed=False,
            quality_accepted=False,
        )
        state.pop("finished", None)
        state.pop("error", None)
        plan.update(
            state="recovering",
            recovery_started=receipt["recovery_started"],
            recovery_code_receipt=str(recovery / "code-receipt.json"),
            training_completed=False,
            quality_accepted=False,
        )
        plan.pop("finished", None)
        save(run / "plan.json", plan)

        def update(phase):
            state.update(state="running", phase=phase, updated=now(), pid=os.getpid())
            receipt.update(state="recovering", phase=phase, updated=now())
            save(run / "state.json", state)
            save(recovery / "recovery-state.json", receipt)
            record_status(record, receipt, phase, "正在恢复")
            print("MSV2_RECOVERY_STAGE", phase, flush=True)

        def stage(action, new_preflight=False):
            phase = "recovery/swinir4x/" + action
            update(phase)
            snapshot = new_snapshot if new_preflight else old_snapshot
            environment = os.environ.copy()
            environment.update(
                OMP_NUM_THREADS=str(job["options"]["cpu_threads"]),
                PYTHONUNBUFFERED="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                PYTHONPATH=str(snapshot),
            )
            if new_preflight:
                program = (
                    "from pathlib import Path; import sys; import msv2.engine as e; "
                    "assert Path(e.__file__).parent.resolve()==(Path(sys.argv[2])/'msv2').resolve(); "
                    "e.Engine(sys.argv[1]).preflight(4)"
                )
                command = [
                    sys.executable,
                    "-s",
                    "-c",
                    program,
                    str(job_file),
                    str(new_snapshot),
                ]
            elif action == "report":
                program = (
                    "import sys; from msv2.engine import Engine; from msv2.report import publish; "
                    "publish(Engine(sys.argv[1]),[1,2,4],{})"
                )
                command = [sys.executable, "-s", "-c", program, str(job_file)]
            else:
                command = [
                    sys.executable,
                    "-s",
                    "-m",
                    "msv2.worker",
                    action,
                    str(job_file),
                    "--scale",
                    "4",
                ]
            log = recovery / "logs" / (action + ".log")
            started = now()
            run_stage(command, log, snapshot, environment, lock_fd)
            receipt["stages"].append(
                dict(
                    action=action,
                    started=started,
                    finished=now(),
                    source=str(snapshot),
                    log=str(log),
                    passed=True,
                )
            )
            save(recovery / "recovery-state.json", receipt)

        try:
            stage("preflight", new_preflight=True)
            preflight = read(run / "swinir4x/preflight.json")
            require(
                preflight["passed"] and preflight["backward_level"] == 2,
                "Corrected native warmup preflight did not pass",
            )
            require(
                not (run / "swinir4x/model").exists(),
                "Model unexpectedly exists before fresh training",
            )
            state["failures"].pop("4")
            save(run / "state.json", state)
            stage("train")
            execution = read(run / "swinir4x/execution.json")
            require(
                execution["state"] == "training_completed"
                and execution["iterations"] == job["options"]["iterations"]
                and read(run / "swinir4x/ply-parity.json")["passed"],
                "Actual full 4x training did not complete",
            )
            receipt["full_fourx_training_completed"] = True
            stage("export")
            stage("evaluate")
            state["completed_scales"] = [1, 2, 4]
            stage("report")
            update("recovery/verify_protected_inputs")
            verify_protected(protected)
            require(
                implementation_sha(new_snapshot / "msv2")
                == code["corrected_preflight_implementation_sha256"],
                "Corrected preflight snapshot changed",
            )
            summary = read(run / "export/comparison.json")
            require(
                summary["training_completed"]
                and summary["completed_scales"] == [1, 2, 4]
                and not summary["failures"],
                "Final comparison incomplete",
            )
            finished = now()
            state.update(
                state="completed",
                phase="awaiting_visual_review",
                finished=finished,
                updated=finished,
                training_completed=True,
                quality_accepted=False,
                failures={},
                report=str(run / "export/comparison.html"),
            )
            plan = read(run / "plan.json")
            plan.update(
                state="completed",
                finished=finished,
                training_completed=True,
                quality_accepted=False,
            )
            receipt.update(
                state="completed",
                phase="awaiting_visual_review",
                finished=finished,
                completed_scales=[1, 2, 4],
                protected_inputs_unchanged=True,
                full_fourx_training_completed=True,
                training_completed=True,
            )
            save(run / "state.json", state)
            save(run / "plan.json", plan)
            save(recovery / "recovery-state.json", receipt)
            copy_record(
                run,
                record,
                recovery,
                receipt,
                "awaiting_visual_review",
                "训练完成，待效果对照",
            )
            print(
                "MSV2_RECOVERY_COMPLETED",
                json.dumps(
                    dict(
                        run=str(run),
                        record=str(record),
                        frames=plan["frame_count"],
                        quality_accepted=False,
                    )
                ),
                flush=True,
            )
        except BaseException as error:
            failure = dict(error=repr(error), phase=state.get("phase"), failed=now())
            (recovery / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            state.update(
                state="partial",
                finished=now(),
                updated=now(),
                training_completed=False,
                quality_accepted=False,
            )
            state["failures"]["4"] = failure
            plan = read(run / "plan.json")
            plan.update(
                state="partial",
                finished=now(),
                training_completed=False,
                quality_accepted=False,
            )
            receipt.update(
                state="failed",
                phase=state.get("phase"),
                failure=failure,
                finished=now(),
                completed_scales=state["completed_scales"],
                training_completed=False,
            )
            save(run / "state.json", state)
            save(run / "plan.json", plan)
            save(recovery / "recovery-state.json", receipt)
            try:
                copy_record(
                    run,
                    record,
                    recovery,
                    receipt,
                    "recovery_failed",
                    "恢复失败，已保留原始和两倍结果",
                )
            except Exception as record_error:
                save(
                    recovery / "record-copy-error.json", dict(error=repr(record_error))
                )
            raise


if __name__ == "__main__":
    main()
