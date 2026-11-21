"""Fresh 4x retry with an explicitly verified lossless workspace wrapper.

The original job, pipeline snapshot, native source, source images, 1x/2x
results and earlier controller remain unchanged. Failed 4x artifacts are
renamed into a new archive, never used as an initializer. No stage starts
without a passed, hash-bound GPU gradient/stress proof supplied by the caller.
"""

import argparse
import ast
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback


def helper_module():
    path = Path(__file__).with_name("recover_msv2_unstarted4.py")
    spec = importlib.util.spec_from_file_location("original_recovery_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = helper_module()
require, read, save, sha, now = h.require, h.read, h.save, h.sha, h.now


def verify_resource_change(old_package, new_package):
    before = {p.name: p for p in old_package.glob("*.py")}
    after = {p.name: p for p in new_package.glob("*.py")}
    require(set(after) == set(before) | {"memory.py"}, "Only memory.py may be added")
    for name in before:
        if name != "engine.py":
            require(
                before[name].read_bytes() == after[name].read_bytes(),
                "Unexpected package change: " + name,
            )
    old = ast.parse(before["engine.py"].read_text(encoding="utf-8"))
    new = ast.parse(after["engine.py"].read_text(encoding="utf-8"))
    engine = next(
        n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "Engine"
    )
    preflight = next(
        n
        for n in engine.body
        if isinstance(n, ast.FunctionDef) and n.name == "preflight"
    )
    assignments = [
        n
        for n in ast.walk(preflight)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "backward_level" for t in n.targets)
    ]
    require(len(assignments) == 1, "Unexpected preflight structure")
    require(
        ast.dump(assignments[0].value)
        == ast.dump(ast.parse("2 if scale == 4 else 1", mode="eval").body),
        "Expected approved native-warmup preflight correction",
    )
    assignments[0].value = ast.Constant(value=1)
    train = next(
        n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    allowed = [
        ast.parse("from .memory import wrap_training_render").body[0],
        ast.parse("m.render_imp = wrap_training_render(m.render_imp, self.repo)").body[
            0
        ],
    ]
    positions = []
    for expected in allowed:
        found = [
            i
            for i, node in enumerate(train.body)
            if ast.dump(node) == ast.dump(expected)
        ]
        require(
            len(found) == 1,
            "Expected exactly one approved train workspace-wrapper statement",
        )
        positions.append(found[0])
    require(
        positions[1] == positions[0] + 1,
        "Workspace wrapper statements must be adjacent",
    )
    for position in sorted(positions, reverse=True):
        train.body.pop(position)
    declarations = [
        n
        for n in ast.walk(train)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "dict"
    ]
    removed = 0
    for node in declarations:
        for keyword in list(node.keywords):
            if keyword.arg == "image_buffer_offload":
                expected = ast.parse("m.render_imp.memory_stats", mode="eval").body
                require(
                    ast.dump(keyword.value) == ast.dump(expected),
                    "Unexpected workspace receipt expression",
                )
                node.keywords.remove(keyword)
                removed += 1
    require(removed == 1, "Expected exactly one workspace execution-receipt field")
    require(
        ast.dump(old, include_attributes=False)
        == ast.dump(new, include_attributes=False),
        "Engine changed outside the explicit preflight/resource-wrapper/receipt allowlist",
    )
    return dict(
        passed=True,
        engine_byte_identical=False,
        engine_ast_allowlist=[
            "preflight backward_level=2 if scale==4 else1",
            "train import wrap_training_render and wrap m.render_imp with self.repo",
            "execution image_buffer_offload=m.render_imp.memory_stats",
        ],
        all_other_engine_ast_identical=True,
        other_existing_package_files_byte_identical=True,
        added_module="memory.py",
        memory_module_sha256=sha(new_package / "memory.py"),
    )


def proof_guard(path, job_file, package):
    proof = read(path)
    require(
        proof["state"] == "passed"
        and proof["model_updates"] == proof["optimizer_updates"] == 0,
        "A passed zero-update GPU proof is required",
    )
    require(
        proof["gradient_parity_passed"] and proof["stress_passed"],
        "GPU gradient parity/stress proof did not pass",
    )
    require(proof["job_sha256"] == sha(job_file), "GPU proof belongs to another job")
    require(
        proof["engine_sha256"] == sha(package / "engine.py")
        and proof["memory_sha256"] == sha(package / "memory.py"),
        "GPU proof does not bind the reviewed adapter bytes",
    )
    return proof


def relative_archive_hashes(folder):
    return {
        path.relative_to(folder).as_posix(): sha(path)
        for path in h.source_files(folder)
    }


def safe_moves(run, archive):
    moves = []
    for name in (
        "model",
        "execution.json",
        "progress.json",
        "progress.jsonl",
        "training-adapter.py",
    ):
        source = run / "swinir4x" / name
        target = archive / "swinir4x" / name
        require(
            source.exists() and not source.is_symlink(),
            "Expected failed 4x artifact: " + str(source),
        )
        require(
            source.resolve().is_relative_to(run)
            and target.resolve().is_relative_to(run),
            "Archive move escaped the exact run",
        )
        require(not target.exists(), "Never overwrite failed 4x archive")
        moves.append((source, target))
    return moves


def record_status(record, run, receipt, phase, label):
    path = record / "测试状态.json"
    current = read(path) if path.exists() else {}
    canonical = read(run / "state.json")
    if "pipeline_state" in current:
        current.setdefault(
            "initial_pipeline_state", copy.deepcopy(current["pipeline_state"])
        )
    if "pipeline_exit_code" in current:
        current.setdefault(
            "initial_pipeline_exit_code", current.pop("pipeline_exit_code")
        )
    if "pid" in current:
        current.setdefault("initial_controller_pid", current["pid"])
    if "finished" in current:
        current.setdefault("initial_finished", current.pop("finished"))
    if "recovery" in current:
        current.setdefault("previous_recovery", copy.deepcopy(current["recovery"]))
    current.update(
        state=canonical["state"],
        status_label=label,
        phase=phase,
        pid=canonical["pid"],
        recovery_controller_pid=receipt["pid"],
        updated=now(),
        pipeline_state=canonical,
        training_completed=canonical["training_completed"],
        completed_scales=canonical["completed_scales"],
        recovery=receipt,
        original_failure=receipt["original_failures"],
        quality_accepted=False,
        recovery_exit_code=0
        if receipt["state"] == "completed"
        else 1
        if receipt["state"] == "failed"
        else None,
    )
    if canonical.get("finished"):
        current["finished"] = canonical["finished"]
    save(path, current)


def copy_record(run, record, recovery, receipt, phase, label):
    shutil.copytree(run / "export", record, dirs_exist_ok=True)
    shutil.copy2(record / "comparison.html", record / "查看完整图倍率对照.html")
    for name in ("state", "plan", "job"):
        shutil.copy2(run / (name + ".json"), record / ("训练_" + name + ".json"))
    for source, target in (
        (recovery / "code-receipt.json", "显存恢复代码核查.json"),
        (recovery / "recovery-state.json", "显存恢复记录.json"),
        (Path(receipt["gpu_memory_proof"]), "显存修复GPU验证.json"),
        (recovery / "archive/state.json", "四倍训练失败_state.json"),
        (recovery / "archive/plan.json", "四倍训练失败_plan.json"),
        (recovery / "archive/failed-train.log", "四倍训练失败.log"),
    ):
        shutil.copy2(source, record / target)
    record_status(record, run, receipt, phase, label)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument(
        "--memory-proof",
        required=True,
        help="Passed zero-update GPU gradient/stress proof JSON",
    )
    parser.add_argument("--package", default="/mnt/d/Projects/msv2/msv2")
    args = parser.parse_args(argv)
    require(os.name == "posix", "Run this controller in Linux")
    job_file, package, record, proof_path = [
        Path(v).resolve()
        for v in (args.job, args.package, args.record, args.memory_proof)
    ]
    job = read(job_file)
    require(job["schema"] == "msv2-job-v1", "Expected portable job")
    run = Path(job["run_dir"]).resolve()
    require(
        job_file == run / "job.json" and record != run and run not in record.parents,
        "Use original job and an external record",
    )
    original_recovery = run / "recovery"
    original_receipt = read(original_recovery / "code-receipt.json")
    require(
        sha(h.__file__) == original_receipt["controller_sha256"],
        "Locked earlier controller helpers changed",
    )
    old_snapshot = run / "pipeline-source"
    require(
        h.implementation_sha(old_snapshot / "msv2") == job["implementation_sha256"],
        "Original frozen pipeline changed",
    )
    ast_receipt = verify_resource_change(old_snapshot / "msv2", package)
    proof_guard(proof_path, job_file, package)
    sys.path.insert(0, str(old_snapshot))
    from msv2.pipeline import process_lock

    require(
        Path(sys.modules["msv2.pipeline"].__file__).parent.resolve()
        == old_snapshot / "msv2",
        "Wrong original pipeline import",
    )

    def interrupted(number, frame):
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, interrupted)
    with process_lock(run.parent.parent / "queue/gpu.lock") as lock_fd:
        state, plan = read(run / "state.json"), read(run / "plan.json")
        require(
            state["state"] == "partial"
            and sorted(state["completed_scales"]) == [1, 2]
            and set(state["failures"]) == {"4"},
            "Expected preserved 1x/2x and failed 4x",
        )
        require(
            job["options"]["scales"] == [1, 2, 4] and plan["options"] == job["options"],
            "Frozen options differ",
        )
        old_execution = read(run / "swinir4x/execution.json")
        require(
            old_execution["state"] == "failed"
            and "OutOfMemoryError" in old_execution.get("error", ""),
            "Expected failed 4x memory state",
        )
        require(
            not (
                run
                / f"swinir4x/model/point_cloud/iteration_{job['options']['iterations']}/point_cloud.ply"
            ).exists(),
            "A completed final 4x model already exists",
        )
        preflight = read(run / "swinir4x/preflight.json")
        require(
            preflight["passed"] and preflight["backward_level"] == 2,
            "Same-input native warmup preflight is not valid",
        )
        recovery = run / "recovery-memory"
        archive, snapshot = recovery / "archive", recovery / "pipeline-source"
        require(
            not archive.exists()
            and not snapshot.exists()
            and not (recovery / "code-receipt.json").exists(),
            "Memory recovery already prepared",
        )
        moves = safe_moves(run, archive)
        protected = h.protected_inputs(run, job_file, job, plan)
        for path in h.source_files(original_recovery) + [
            run / "swinir4x/preflight.json",
            run / "swinir4x/preflight.png",
            proof_path,
            Path(h.__file__).resolve(),
        ]:
            protected["files"][str(path)] = sha(path)
        archive.mkdir(parents=True)
        for source, name in (
            (run / "state.json", "state.json"),
            (run / "plan.json", "plan.json"),
            (job_file, "job.json"),
            (original_recovery / "logs/train.log", "failed-train.log"),
            (run / "swinir4x/resolved-options.json", "failed-resolved-options.json"),
        ):
            shutil.copy2(source, archive / name)
        shutil.copytree(
            package,
            snapshot / "msv2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        require(
            h.implementation_sha(snapshot / "msv2") == h.implementation_sha(package),
            "Package changed during freezing",
        )
        verify_resource_change(old_snapshot / "msv2", snapshot / "msv2")
        proof = proof_guard(proof_path, job_file, snapshot / "msv2")
        # Validate every exact source and destination before performing any move.
        safe_moves(run, archive)
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        require(
            not (run / "swinir4x/model").exists()
            and not (run / "swinir4x/execution.json").exists(),
            "Fresh retry still has old model state",
        )
        archived_hashes = relative_archive_hashes(archive)
        save(recovery / "protected-inputs.json", protected)
        save(recovery / "archive-sha256.json", archived_hashes)
        implementation = h.implementation_sha(snapshot / "msv2")
        code = dict(
            schema="msv2-fresh4-workspace-recovery-v1",
            created=now(),
            controller=str(Path(__file__).resolve()),
            controller_sha256=sha(__file__),
            immutable_helper=str(Path(h.__file__).resolve()),
            immutable_helper_sha256=sha(h.__file__),
            original_job=str(job_file),
            original_job_sha256=sha(job_file),
            original_implementation_sha256=job["implementation_sha256"],
            original_pipeline_source=str(old_snapshot),
            train_override_pipeline_source=str(snapshot),
            train_override_implementation_sha256=implementation,
            train_override_engine_sha256=sha(snapshot / "msv2/engine.py"),
            memory_sha256=sha(snapshot / "msv2/memory.py"),
            ast_check=ast_receipt,
            gpu_memory_proof=str(proof_path),
            gpu_memory_proof_sha256=sha(proof_path),
            gpu_proof_model_updates=proof["model_updates"],
            gpu_proof_optimizer_updates=proof["optimizer_updates"],
            archive=str(archive),
            exact_renamed_artifacts=[
                str(target.relative_to(run)) for _, target in moves
            ],
            archive_receipt="archive-sha256.json",
            protected_receipt="protected-inputs.json",
            protected_file_count=len(protected["files"]),
            unchanged_native_training_source=True,
            unchanged_training_algorithm=True,
            unchanged_input_resolution=True,
            no_swinir_rerun=True,
            no_1x_2x_retraining=True,
            old_3d_used_as_initializer=False,
            checkpoint_loaded=False,
            stage_sources=dict(
                preflight="Preserved same-input passed warmup preflight",
                train=str(snapshot),
                export=str(old_snapshot),
                evaluate=str(old_snapshot),
                report=str(old_snapshot),
            ),
            scope="Fresh raw-SfM 4x initialization and optimizer. Only large-image saved imgBuffer storage changes; verified gradient-equivalent CPU temporary storage. Original worker snapshot guard is explicitly overridden only for the reviewed train adapter. Native model/loss/optimizer code remains original; full training is accepted only after actual completion.",
        )
        save(recovery / "code-receipt.json", code)
        receipt = dict(
            state="recovering",
            run_dir=str(run),
            pid=os.getpid(),
            recovery_started=now(),
            stages=[],
            original_failures=copy.deepcopy(state["failures"]),
            earlier_preflight_recovery=str(original_recovery),
            original_state_archive="archive/state.json",
            original_plan_archive="archive/plan.json",
            gpu_memory_proof=str(proof_path),
            code_receipt="code-receipt.json",
            completed_scales=[1, 2],
            fresh_fourx=True,
            old_3d_initializer=False,
            full_fourx_training_completed=False,
            training_completed=False,
            quality_accepted=False,
        )
        state.setdefault("recovered_failures", []).append(
            dict(
                recorded=now(),
                failures=copy.deepcopy(state["failures"]),
                original_finished=state.get("finished"),
                archive=str(archive),
                recovery_started=receipt["recovery_started"],
                pid=os.getpid(),
                recovery_kind="lossless_workspace_storage",
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
            memory_recovery_code_receipt=str(recovery / "code-receipt.json"),
            training_completed=False,
            quality_accepted=False,
        )
        plan.pop("finished", None)
        save(run / "plan.json", plan)

        def update(phase):
            state.update(state="running", phase=phase, pid=os.getpid(), updated=now())
            receipt.update(state="recovering", phase=phase, updated=now())
            save(run / "state.json", state)
            save(recovery / "recovery-state.json", receipt)
            record_status(record, run, receipt, phase, "正在从头训练四倍版本")
            print("MSV2_MEMORY_RECOVERY_STAGE", phase, flush=True)

        def stage(action):
            update("recovery-memory/swinir4x/" + action)
            source = snapshot if action == "train" else old_snapshot
            environment = os.environ.copy()
            environment.update(
                OMP_NUM_THREADS=str(job["options"]["cpu_threads"]),
                PYTHONUNBUFFERED="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                PYTHONPATH=str(source),
            )
            if action == "train":
                program = (
                    "from pathlib import Path; import sys; import msv2.engine as e; from msv2.pipeline import implementation_fingerprint; "
                    "assert Path(e.__file__).parent.resolve()==(Path(sys.argv[2])/'msv2').resolve(); "
                    "assert implementation_fingerprint(Path(sys.argv[2])/'msv2')==sys.argv[3]; "
                    "e.Engine(sys.argv[1]).train(4)"
                )
                command = [
                    sys.executable,
                    "-s",
                    "-c",
                    program,
                    str(job_file),
                    str(snapshot),
                    implementation,
                ]
            elif action == "report":
                command = [
                    sys.executable,
                    "-s",
                    "-c",
                    "import sys; from msv2.engine import Engine; from msv2.report import publish; publish(Engine(sys.argv[1]),[1,2,4],{})",
                    str(job_file),
                ]
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
            started = now()
            log = recovery / "logs" / (action + ".log")
            h.run_stage(command, log, source, environment, lock_fd)
            receipt["stages"].append(
                dict(
                    action=action,
                    started=started,
                    finished=now(),
                    passed=True,
                    source=str(source),
                    log=str(log),
                )
            )
            save(recovery / "recovery-state.json", receipt)

        try:
            state["failures"].pop("4")
            stage("train")
            execution = read(run / "swinir4x/execution.json")
            require(
                execution["state"] == "training_completed"
                and execution["iterations"] == job["options"]["iterations"]
                and execution["adapter_sha256"] == code["train_override_engine_sha256"]
                and read(run / "swinir4x/ply-parity.json")["passed"],
                "Actual fresh 4x training did not finish correctly",
            )
            receipt["full_fourx_training_completed"] = True
            stage("export")
            stage("evaluate")
            state["completed_scales"] = [1, 2, 4]
            stage("report")
            update("recovery-memory/verify_protected_inputs")
            h.verify_protected(protected)
            require(
                relative_archive_hashes(archive) == archived_hashes,
                "Archived failed attempt changed",
            )
            require(
                h.implementation_sha(snapshot / "msv2") == implementation,
                "Train override snapshot changed",
            )
            summary = read(run / "export/comparison.json")
            require(
                summary["training_completed"]
                and summary["completed_scales"] == [1, 2, 4]
                and not summary["failures"],
                "Final report incomplete",
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
                failed_attempt_archive_unchanged=True,
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
            print("MSV2_MEMORY_RECOVERY_COMPLETED", str(run), flush=True)
        except BaseException as error:
            failure = dict(error=repr(error), phase=state.get("phase"), failed=now())
            (recovery / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            if state.get("state") == "completed" and state.get("training_completed"):
                # An output-copy failure cannot undo real completed training.
                receipt.update(record_copy_failed=True, record_copy_error=failure)
                save(recovery / "record-copy-error.json", failure)
                save(recovery / "recovery-state.json", receipt)
                try:
                    record_status(
                        record,
                        run,
                        receipt,
                        "record_copy_failed",
                        "训练已完成，记录复制需要补全",
                    )
                except Exception:
                    pass
                raise
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
                    "memory_recovery_failed",
                    "四倍重训失败，已保留原始和两倍结果",
                )
            except Exception as copy_error:
                save(recovery / "record-copy-error.json", dict(error=repr(copy_error)))
            raise


if __name__ == "__main__":
    main()
