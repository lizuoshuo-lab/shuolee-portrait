"""Recover orchestration I/O and adopt one live Windows wrapper by real handle.

The frozen plan, model launchers and old controller remain unchanged. Default
is read-only preparation; --execute adopts a verified live process, records its
actual terminal exit code, then continues the original serial queue. No kill,
duplicate training, synthetic exit status or independent accuracy acceptance.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / "outputs/splatam-ablation-repeat-001"
NATIVE_RUNS = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\splatam\outputs\runs")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    # A unique temporary file prevents collisions; Windows readers may briefly
    # omit FILE_SHARE_DELETE, so retry replacement without truncating the target.
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    for retry in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if retry == 39:
                raise
            time.sleep(.25)


def fresh_cim(pid):
    code = "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; if(!$p){exit 3}; "
    code += "$p | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine,@{Name='CreationUTC';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", code], capture_output=True, timeout=20)
    require(result.returncode == 0, "Fresh CIM target not alive/readable: " + result.stderr.decode(errors="replace"))
    return json.loads(result.stdout.decode("utf-8-sig"))


def windows_argv(command):
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel.LocalFree.restype = wintypes.HLOCAL
    count = ctypes.c_int()
    pointers = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not pointers:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [pointers[i] for i in range(count.value)]
    finally:
        kernel.LocalFree(ctypes.cast(pointers, wintypes.HLOCAL))


class LiveHandle:
    def __init__(self, pid, entry):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            require(self.kernel.WaitForSingleObject(self.handle, 0) == 258, "Adoption requires an actually live target")
            first = fresh_cim(pid)
            require(first["ProcessId"] == pid and windows_argv(first["CommandLine"]) == entry["command"],
                    "PID command differs from exact frozen active queue entry")
            require(Path(first["ExecutablePath"]).resolve() == Path(entry["command"][0]).resolve(), "Unexpected executable for adopted wrapper")
            times = [wintypes.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(self.handle, *[ctypes.byref(value) for value in times]):
                raise ctypes.WinError(ctypes.get_last_error())
            ticks = times[0].dwLowDateTime | (times[0].dwHighDateTime << 32)
            unix_ns = (ticks - 116444736000000000) * 100
            # PowerShell emits seven fractional digits; Python 3.10 accepts
            # six. Keep the original CIM string and exact FILETIME in evidence.
            cim_parseable = re.sub(r'(\.\d{6})\d+(?=[+-]|Z|$)', r'\1', first["CreationUTC"])
            cim_time = dt.datetime.fromisoformat(cim_parseable.replace("Z", "+00:00"))
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
            delta = cim_time - epoch
            cim_ns = ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
            require(abs(unix_ns - cim_ns) <= 10_000, "CIM/handle creation FILETIME mismatch; possible PID reuse")
            second = fresh_cim(pid)
            require(second == first, "PID identity changed while opening the retained handle")
            require(self.kernel.WaitForSingleObject(self.handle, 0) == 258, "Target ended before live adoption was certified")
            self.identity = dict(first, handle_creation_FILETIME_ticks=ticks, handle_creation_unix_ns=unix_ns,
                                 verified_utc=stamp(), access="SYNCHRONIZE|PROCESS_QUERY_LIMITED_INFORMATION",
                                 observation_method="own retained Windows handle")
        except BaseException:
            self.close()
            raise

    def poll(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 258:
            return None
        require(result == 0, "WaitForSingleObject failed: " + str(result))
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(code.value)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class RecordedTerminalHandle:
    """Only root's actual retained-handle terminal receipt can replace adoption."""
    def __init__(self, execution, pid, entry):
        folder = execution / "controller-recovery"
        start_path = folder / "original-wrapper-handle-started.json"
        exit_path = folder / "original-wrapper-handle-exit.json"
        started, ended = read(start_path), read(exit_path)
        observer_sha = "e66b50f86ba19f17e76db45ae077382ad2b2d7cabb9a9a9db28e7b789e8b9380"
        require(sha(LAB / "scripts/splatam_local/hold_existing_wrapper.py") == observer_sha,
                "Root retained-handle observer source changed")
        require(started["status"] == "holding_original_wrapper_handle"
                and ended["status"] == "original_wrapper_terminal_exit_observed"
                and started["actual_exit_code"] is None and type(ended["actual_exit_code"]) is int,
                "No genuine root terminal handle receipt available")
        require(started["pid"] == ended["pid"] == pid and started["run_name"] == ended["run_name"] == entry["run_name"]
                and started["process_creation_filetime_ticks"] == ended["process_creation_filetime_ticks"] == 134361030258347675
                and started["source_sha256"] == ended["source_sha256"] == observer_sha,
                "Root terminal receipt does not bind the actual original process")
        require(windows_argv(started["cim_observation"]["CommandLine"]) == entry["command"]
                and started["cim_observation"] == ended["cim_observation"], "Root handle command identity mismatch")
        for key, value in started.items():
            if key not in ("status", "actual_exit_code"):
                require(ended[key] == value, "Root handle initial/terminal receipt changed: " + key)
        self.code = ended["actual_exit_code"]
        self.identity = {"ProcessId": pid, "handle_creation_FILETIME_ticks": ended["process_creation_filetime_ticks"],
                         "observation_method": "root actual GetExitCodeProcess from already-retained Windows handle",
                         "root_started_receipt_SHA256": sha(start_path), "root_terminal_receipt_SHA256": sha(exit_path),
                         "observer_source_SHA256": observer_sha, "terminal_observed_utc": ended["terminal_observed_utc"]}

    def poll(self):
        return self.code

    def close(self):
        pass


def verify_protocol(plan, receipt):
    require(sha(OUT / "plan.json") == receipt["plan_sha256"], "Frozen plan changed")
    require(sha(LAB / "scripts/splatam_local/run_ablation_official.py") == plan["launcher_sha256"], "Native launcher changed")
    require(sha(LAB / "scripts/splatam_local/run_ablation_wsl.py") == plan["windows_wrapper_sha256"], "Windows wrapper changed")
    baseline = LAB / "outputs/splatam-official-runs-001" / plan["baseline"]["run_name"]
    require(sha(baseline / "execution.json") == plan["baseline"]["execution_sha256"]
            and sha(baseline / "params.npz") == plan["baseline"]["params_sha256"], "Frozen baseline changed")


def no_orphans(allowed=None):
    for path in NATIVE_RUNS.glob("*/execution.json"):
        state = read(path)
        if state["status"] == "running":
            require(path.parent.name == allowed, "Uncertified live native run prevents next GPU job: " + path.parent.name)


def certify(entry, job, plan, code):
    run_name = entry["run_name"]
    native_path = NATIVE_RUNS / run_name
    mirror = LAB / "outputs/splatam-official-runs-001" / run_name
    native = read(native_path / "execution.json")
    entry["native_status_after_wrapper_exit"] = native["status"]
    require(native["status"] != "running", "Native phase remains running; potential orphan")
    actual = read(mirror / "execution.json")
    config = read(mirror / "effective-config.json")
    require(code == 0 and actual["child_exit_code"] == native["child_exit_code"] == 0
            and actual["status"] == native["status"] == "completed_pending_independent_evaluation",
            "Actual wrapper/native child did not complete successfully")
    require(actual["run_name"] == native["run_name"] == run_name and actual["experiment_tracking_loss"] == job["tracking_loss"]
            and config["seed"] == job["seed"] and config["data"]["sequence"] == "rgbd_dataset_" + job["sequence"]
            and actual["requested_frames"] == -1 and actual["resolution"] == [320, 240], "Actual model protocol mismatch")
    require(actual["launcher_sha256"] == plan["launcher_sha256"] and sha(mirror / "launcher-source.py") == plan["launcher_sha256"], "Actual native launcher not frozen source")
    require(sha(mirror / "input_config.py") == actual["config_sha256"]
            and sha(mirror / "config.py") == actual["config_sha256"]
            and sha(mirror / "params.npz") == actual["params_sha256"] == native["params_sha256"], "Actual config/final model SHA mismatch")
    require(sha(mirror / "execution.json") == sha(native_path / "execution.json"), "Native/mirror execution receipts differ")
    wrapper_receipt = read(mirror / "mirror-receipt.json")
    require(wrapper_receipt["windows_wrapper_exit"] == 0 and wrapper_receipt["native_run_name"] == run_name
            and wrapper_receipt["windows_wrapper_sha256"] == plan["windows_wrapper_sha256"]
            and sha(mirror / "windows-wrapper-source.py") == plan["windows_wrapper_sha256"], "Actual mirror producer mismatch")
    entry.update(status="completed_pending_independent_evaluation", actual_child_exit_code=actual["child_exit_code"],
                 actual_status=actual["status"], execution_sha256=sha(mirror / "execution.json"),
                 final_model_sha256=actual["params_sha256"], config_sha256=actual["config_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--adopt-wrapper-pid", type=int, required=True)
    args = parser.parse_args()
    handle, recovery, receipt, execution = None, None, None, None
    try:
        require(os.name == "nt", "Real Windows process handle required")
        execution = args.execution_dir.resolve()
        require(execution.parent == (OUT / "executions").resolve(), "Must resume the same original queue execution")
        target = execution / "receipt.json"
        receipt = read(target)
        plan = read(OUT / "plan.json")
        jobs = plan["full_queue"]
        expected = [("freiburg1_desk", p, 0) for p in ("rgb_only", "depth_only")] + [("freiburg1_desk", p, s) for s in (1, 2) for p in ("rgbd", "rgb_only", "depth_only")] + [("freiburg1_room", "rgbd", 0)]
        require([(j["sequence"], j["tracking_loss"], j["seed"]) for j in jobs] == expected, "Frozen queue order changed")
        require(plan["tracking_weights"] == {"rgbd": {"im": .5, "depth": 1.}, "rgb_only": {"im": .5, "depth": 0.}, "depth_only": {"im": 0., "depth": 1.}}, "Tracking profiles changed")
        require(receipt["execution_path"] == execution.relative_to(LAB).as_posix() and receipt["status"] == "running", "Original running receipt required")
        require(len(receipt["jobs"]) == 4 and receipt["jobs"][-1]["index"] == 3, "Expected three completed plus current job3")
        entry = receipt["jobs"][-1]
        require(entry["status"] == "running" and entry["windows_wrapper_pid"] == args.adopt_wrapper_pid
                and entry["tracking_loss"] == "rgb_only" and entry["seed"] == 1 and entry["sequence"] == "freiburg1_desk", "Wrong adopted queue job")
        expected_command = [str(LAB / ".venv/Scripts/python.exe"), "-X", "utf8", str(LAB / "scripts/splatam_local/run_ablation_wsl.py"),
                            "--frames", "-1", "--width", "320", "--sequence", "freiburg1_desk", "--seed", "1", "--tracking-loss", "rgb_only"]
        require(entry["command"] == expected_command, "Original active command differs from frozen required full desk rgb_only seed1 args")
        require(sha(execution / "batch-source.py") == receipt["batch_source_sha256"]
                == sha(LAB / "scripts/splatam_local/run_ablation_batch.py"), "Original controller source changed")
        require(sha(execution / "plan.json") == receipt["plan_sha256"], "Original execution plan snapshot changed")
        if not args.execute:
            verify_protocol(plan, receipt)
            print(json.dumps({"status": "prepared_only", "adopt_job": 3, "remaining_jobs": 5, "script_sha256": sha(Path(__file__))}))
            return 0
        # Retain the process handle BEFORE slow hashing or snapshot I/O.
        try:
            handle = LiveHandle(args.adopt_wrapper_pid, entry)
        except (OSError, RuntimeError):
            # Missing terminal receipt rejects; process disappearance is never
            # converted into success or an assumed zero status.
            handle = RecordedTerminalHandle(execution, args.adopt_wrapper_pid, entry)
        native = read(NATIVE_RUNS / entry["run_name"] / "execution.json")
        require(native["run_name"] == entry["run_name"] and native["pid"] == entry["native_child_pid"]
                and native["experiment_tracking_loss"] == "rgb_only" and native["resolution"] == [320, 240]
                and native["requested_frames"] == -1, "Adopted process not bound to current native run")
        token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        recovery = execution / "recoveries" / token
        recovery.mkdir(parents=True, exist_ok=False)
        snapshots = {}
        for source, name in ((target, "old-receipt.json"), (OUT / "batch-progress.json", "old-batch-progress.json"),
                             (execution / "batch-source.py", "original-batch-source.py"), (execution / "plan.json", "original-plan.json"),
                             (Path(__file__), "resume-source.py")):
            shutil.copy2(source, recovery / name)
            snapshots[name] = sha(recovery / name)
        for name in ("original-wrapper-handle-started.json", "original-wrapper-handle-exit.json", "controller34802-failure.json"):
            source = execution / "controller-recovery" / name
            if source.is_file():
                shutil.copy2(source, recovery / name)
                snapshots[name] = sha(recovery / name)
        failure_path = execution / "controller-recovery/controller34802-failure.json"
        require(failure_path.is_file() and sha(failure_path) == "ce0f8ce8005ca85ddeaaac20e80d0c5d75346cc7b3030f0f5638a3c7d6d77e9b",
                "Root actual original controller failure evidence missing or changed")
        amendment = {"status": "actual_recovery_adopting_live_wrapper", "created_utc": stamp(),
                     "original_execution_path": receipt["execution_path"], "original_controller_session_id": 34802,
                     "original_controller_actual_exit_code": 1, "cause": "PermissionError WinError 5 replacing batch-progress.json due to transient Windows reader sharing lock",
                     "cause_observer": "Root observed actual session exit1 and original save(temp.replace) traceback; exact failure transcript may be attached separately by root",
                     "adopted_windows_wrapper": handle.identity, "adopted_native_run": entry["run_name"],
                     "adopted_native_child_pid": entry["native_child_pid"], "snapshots_sha256": snapshots,
                     "resume_source_sha256": sha(Path(__file__)), "model_protocol_changed": False,
                     "amendment": "orchestration atomic-save retry and retained Windows handle adoption only"}
        save(recovery / "recovery.json", amendment)
        receipt["orchestration_recovery"] = {"path": recovery.relative_to(LAB).as_posix(), **amendment}
        verify_protocol(plan, receipt)
        for previous in receipt["jobs"][:-1]:
            require(previous["status"] == "completed_pending_independent_evaluation"
                    and previous["windows_wrapper_exit_code"] == 0, "Earlier queue job not actually complete")
            certify(previous, jobs[previous["index"]], plan, previous["windows_wrapper_exit_code"])
        no_orphans(entry["run_name"])

        def persist():
            save(target, receipt)
            save(OUT / "batch-progress.json", receipt)

        adopted_start = time.monotonic()
        print(json.dumps({"status": "adopted_live_wrapper", "pid": args.adopt_wrapper_pid, "run_name": entry["run_name"], "recovery": str(recovery)}), flush=True)
        while True:
            code = handle.poll()
            if code is not None:
                break
            current = read(NATIVE_RUNS / entry["run_name"] / "execution.json")
            entry.update(native_status=current["status"], native_child_pid=current.get("pid"), recovered_monitor_elapsed_s=time.monotonic() - adopted_start)
            persist()
            time.sleep(10)
        entry.update(windows_wrapper_exit_code=code, exit_code_observation=handle.identity["observation_method"],
                     adoption_wait_seconds=time.monotonic() - adopted_start, finished_utc=stamp(),
                     log_sha256=sha(LAB / entry["log_path"]))
        certify(entry, jobs[3], plan, code)
        handle.close()
        handle = None
        amendment.update(status="adopted_wrapper_actual_terminal_exit_observed", adopted_wrapper_actual_exit_code=code, adopted_wrapper_finished_utc=stamp())
        save(recovery / "recovery.json", amendment)
        receipt["orchestration_recovery"].update(amendment)
        persist()
        for index in range(4, len(jobs)):
            job = jobs[index]
            verify_protocol(plan, receipt)
            no_orphans()
            prefix = f"{job['sequence']}_full_320x240_{job['tracking_loss']}_s{job['seed']}_"
            existing = {path.name for path in NATIVE_RUNS.glob(prefix + "*")}
            wrapper = LAB / "scripts/splatam_local/run_ablation_wsl.py"
            command = [sys.executable, "-X", "utf8", str(wrapper), "--frames", "-1", "--width", "320", "--sequence", job["sequence"], "--seed", str(job["seed"]), "--tracking-loss", job["tracking_loss"]]
            entry = dict(job, index=index, status="running", started_utc=stamp(), command=command)
            log_path = execution / f"job-{index:02d}.log"
            entry["log_path"] = log_path.relative_to(LAB).as_posix()
            # Opening exclusively precedes child creation: old logs are retained.
            with log_path.open("xb") as log:
                receipt["jobs"].append(entry)
                persist()
                start = time.monotonic()
                child = subprocess.Popen(command, cwd=LAB, stdout=log, stderr=subprocess.STDOUT)
                entry["windows_wrapper_pid"] = child.pid
                persist()
                print(json.dumps({"status": "started", "job": index, "pid": child.pid}), flush=True)
                while child.poll() is None:
                    matches = [p for p in NATIVE_RUNS.glob(prefix + "*") if p.name not in existing]
                    if len(matches) > 1:
                        entry["native_binding_error"] = "Multiple new native runs match this job"
                    elif len(matches) == 1:
                        entry["run_name"] = matches[0].name
                        try:
                            native = read(matches[0] / "execution.json")
                            entry.update(native_status=native["status"], native_child_pid=native.get("pid"))
                        except (OSError, KeyError, json.JSONDecodeError):
                            pass
                    entry["elapsed_wrapper_seconds"] = time.monotonic() - start
                    persist()
                    time.sleep(10)
                code = child.wait()
            entry.update(windows_wrapper_exit_code=code, wrapper_wall_seconds=time.monotonic() - start,
                         finished_utc=stamp(), log_sha256=sha(log_path))
            matches = [p for p in NATIVE_RUNS.glob(prefix + "*") if p.name not in existing]
            require(len(matches) == 1 and "native_binding_error" not in entry, "No unique completed native job; potential orphan")
            entry["run_name"] = matches[0].name
            certify(entry, job, plan, code)
            persist()
            print(json.dumps({"job": index, "run_name": entry["run_name"], "status": entry["status"], "wrapper_exit": code}), flush=True)
        verify_protocol(plan, receipt)
        require(len(receipt["jobs"]) == 9 and all(j["status"] == "completed_pending_independent_evaluation" for j in receipt["jobs"]), "All nine actual jobs required")
        no_orphans()
        receipt.update(status="training_queue_completed_pending_independent_evaluation", finished_utc=stamp(), frozen_baseline_unchanged=True)
        persist()
        amendment.update(status="recovered_training_queue_completed_pending_independent_evaluation", finished_utc=stamp())
        save(recovery / "recovery.json", amendment)
        print(json.dumps({"status": receipt["status"], "jobs": 9, "execution": str(execution)}), flush=True)
        return 0
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as error:
        # Do not terminate an adopted/new wrapper or launch another GPU child.
        if receipt is not None and recovery is not None:
            receipt.update(status="stopped_recovery_failure_potential_native_orphan", finished_utc=stamp(), stop_reason=str(error))
            try:
                save(execution / "receipt.json", receipt)
                save(OUT / "batch-progress.json", receipt)
                save(recovery / "failure.json", {"status": receipt["status"], "error": str(error), "utc": stamp(), "no_process_killed": True})
            except OSError as save_error:
                print("Recovery receipt save also failed: " + str(save_error), file=sys.stderr)
        print("Recovery stopped, no next GPU job: " + str(error), file=sys.stderr)
        return 1
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    sys.exit(main())
