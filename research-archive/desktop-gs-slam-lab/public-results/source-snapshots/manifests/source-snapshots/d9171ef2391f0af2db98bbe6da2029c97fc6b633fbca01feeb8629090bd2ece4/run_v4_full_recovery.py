"""CPU prepare, then exact-reviewed single V4 restart after the recorded I/O failure.

The frozen trainer is unchanged. This is an explicitly disclosed frame0 restart,
not a resumed checkpoint, silent retry, parameter search, or metric selection.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"manifests/tum-online-v4-full-sequence-recovery-protocol-001.json"
OUTPUT = ROOT/"outputs/tum-online-v4-full-sequence-recovery-001"
ORIGINAL_PROTOCOL = ROOT/"manifests/tum-online-v4-full-sequence-protocol-001.json"
ORIGINAL_SHA = "ee635a6d65cc02dd26ff16e3f83721afa745a56b9f7a7464453833651a4963af"
OLD_DRIVER = ROOT/"scripts/run_v4_full_sequence.py"
OLD_DRIVER_SHA = "bfda7d3388ec7683a6e64f17415f11302beac9f64fdfb375f2278094d27493d0"
OLD_BATCH = ROOT/"outputs/tum-online-v4-full-sequence-001"
FAILED = "tum-xyz-online-v4-prior001-full748-001"
REPLACEMENT = "tum-xyz-online-v4-prior001-full748-002"
DEADLINE = dt.datetime.fromisoformat("2026-10-09T08:00:00+08:00")
BJT = dt.timezone(dt.timedelta(hours=8))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def relative(path):
    return Path(path).relative_to(ROOT).as_posix()

def now():
    return dt.datetime.now(dt.timezone.utc)

def stamp(value=None):
    value = value or now()
    return {"utc": value.isoformat(), "beijing": value.astimezone(BJT).isoformat()}

def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)

def atomic(path, value):
    temporary = Path(path).with_name(Path(path).name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def frozen_driver():
    if sha(ORIGINAL_PROTOCOL) != ORIGINAL_SHA or sha(OLD_DRIVER) != OLD_DRIVER_SHA:
        raise ValueError("Original reviewed protocol/driver changed")
    spec = importlib.util.spec_from_file_location("frozen_full_recovery_reference_driver", OLD_DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def replacement_row(original):
    old = original["new_runs"][3]
    if old["run"] != FAILED:
        raise ValueError("Original fourth run is not the declared failed run")
    row = copy.deepcopy(old)
    row.update(order=1, run=REPLACEMENT, output_relative_path="outputs/"+REPLACEMENT,
               log_relative_path="logs/"+REPLACEMENT+".log")
    row["configuration"]["run"] = REPLACEMENT
    index = row["command"].index("--run")+1
    row["command"][index] = REPLACEMENT
    if {key for key in old["configuration"] if old["configuration"][key] != row["configuration"][key]} != {"run"}:
        raise ValueError("Only the output run name may change")
    if [i for i,(a,b) in enumerate(zip(old["command"], row["command"])) if a != b] != [index]:
        raise ValueError("Command changed outside --run value")
    return row

def original_failure_gate(original):
    failure = read(OLD_BATCH/"failure.json")
    if (failure["status"] != "batch_stopped_failure_no_retry" or failure["protocol_sha256"] != ORIGINAL_SHA or
            failure["driver_sha256"] != OLD_DRIVER_SHA or len(failure["run_records"]) != 4 or
            failure["GPU_runs_started"] != 4 or failure != read(OLD_BATCH/"progress.json")):
        raise ValueError("Original batch did not stop with the retained four-run failure")
    for declared, record in zip(original["new_runs"], failure["run_records"]):
        if record["run"] != declared["run"] or record["configuration"] != declared["configuration"]:
            raise ValueError("Original terminal ledger command/configuration mismatch")
        if sha(ROOT/record["log"]) != record["log_sha256"]:
            raise ValueError("Original log changed")
        if record["run"] != FAILED:
            if record["returncode"] != 0 or not record["status"].startswith("completed") or sha(ROOT/declared["output_relative_path"]/"metrics.json") != record["metrics_sha256"]:
                raise ValueError("One of the original first three results is not completed and unchanged")
        elif record["returncode"] != 1 or record["status"] != "failed_process_exit_all_original_artifacts_retained":
            raise ValueError("Original fourth-run failure is not preserved")
    folder = ROOT/"outputs"/FAILED
    trainer_failure = read(folder/"failure.json")
    if (trainer_failure["exception_type"] != "PermissionError" or trainer_failure["current_local_index"] != 520 or
            (folder/"metrics.json").exists() or (folder/"online_map.pt").exists()):
        raise ValueError("Failed partial521 run must not be relabeled as completed748")
    return failure

def snapshot(original):
    observation = OUTPUT/"root_observation.json"
    write_new(observation, {"recorded_at": stamp(), "source": "root agent explicit message",
        "observation": "Root reports PowerShell Get-Content streamed the live failed progress file around 04:48 Beijing, then its own PS session49245 was stopped with CtrlC.",
        "interpretation": "Possible file-lock contributor; historical owner and unique causation are not established. Reported coarse time is not independently verified overlap.",
        "isolated_control": "controls002 reproduces open-reader replace PermissionError and close-reader success; both tested sharing variants fail while open, so delete-sharing alone is not asserted to fix it.",
        "recovery_monitoring": "Do not open the trainer live progress.json; use log tail/process inventory until child exits. Trainer/source remain unchanged."})
    paths = list((ROOT/"outputs"/FAILED).rglob("*"))
    paths = [path for path in paths if path.is_file()]
    paths += [ORIGINAL_PROTOCOL, OLD_DRIVER, OLD_BATCH/"failure.json", OLD_BATCH/"progress.json",
              OLD_BATCH/"fourth_run_failure_evidence.json", ROOT/"logs"/(FAILED+".log"), observation,
              ROOT/"outputs/windows-progress-file-sharing-controls-002/checks.json",
              ROOT/"scripts/check_windows_progress_file_sharing.py",
              ROOT/"outputs/windows-progress-file-sharing-controls-001/failure.json",
              ROOT/"outputs/windows-progress-file-sharing-controls-001/source_script.py"]
    paths += [ROOT/name for name in original["source_sha256"]]
    records = []
    destination = OUTPUT/"original_failure_snapshot"
    destination.mkdir(exist_ok=False)
    for source in sorted(set(paths)):
        target = destination/relative(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if sha(target) != expected or sha(source) != expected:
            raise RuntimeError("Failure source/copy changed during snapshot")
        records.append({"original": relative(source), "snapshot": relative(target), "sha256": expected, "bytes": target.stat().st_size})
    manifest = destination/"manifest.json"
    write_new(manifest, {"status": "copied_and_hash_verified", "at": stamp(), "failed_run": FAILED,
        "historical_cause_not_uniquely_established": True, "files": records})
    return {"manifest": relative(manifest), "manifest_sha256": sha(manifest), "files": records}

def check_hashes(records):
    for path, expected in records.items():
        if sha(ROOT/path) != expected:
            raise ValueError("Reviewed evidence/source changed: "+path)

def launch(command, stream, *, clock=now, popen=subprocess.Popen):
    guard = clock()
    if guard >= DEADLINE:
        return None, guard
    return popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT), guard

def deadline_checks():
    calls=[]
    def fake_popen(command, **kwargs):
        calls.append(command); return object()
    results=[]
    for offset in (-1,0,1):
        process, guard=launch(["CPU-control-no-real-process"], None,
            clock=lambda: DEADLINE+dt.timedelta(microseconds=offset), popen=fake_popen)
        results.append(process is not None)
    if results != [True,False,False] or len(calls)!=1:
        raise AssertionError("Deadline boundary control failed")
    return {"status":"passed","one_microsecond_before_allowed":True,"at_and_after_blocked":True,
        "Popen_immediately_after_actual_clock_no_IO_between":True,"real_subprocesses_or_GPU_started":0}

def prepare():
    if OUTPUT.exists() or PROTOCOL.exists():
        raise FileExistsError("Recovery preparation must be unique")
    old=frozen_driver(); original=read(ORIGINAL_PROTOCOL)
    original_failure_gate(original)
    row=replacement_row(original)
    if (ROOT/row["output_relative_path"]).exists() or (ROOT/row["log_relative_path"]).exists():
        raise FileExistsError("Recovery run/log already exists")
    if old.source_guard()!=original["source_sha256"] or old.archive_records()!=original["dataset_archives"]:
        raise ValueError("Original frozen source/archive no longer matches")
    OUTPUT.mkdir(exist_ok=False)
    try:
        snap=snapshot(original)
        parsing=old.parse_checks([row]); parsing.pop("all_four_actual_CLI_configurations_exact")
        parsing["single_actual_CLI_configuration_exact"]=True
        parsing["only_difference_from_failed_command_and_configuration"]="run name"
        write_new(OUTPUT/"cpu_cli_checks.json",parsing)
        stdout,stderr=OUTPUT/"cpu_preflight_xyz.json",OUTPUT/"cpu_preflight_xyz.stderr.log"
        command=[str(old.PYTHON),str(old.TRAINER),*row["command"][3:],"--check-only"]
        with stdout.open("xb") as out,stderr.open("xb") as err:
            completed=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err)
        if completed.returncode:
            raise RuntimeError("Frozen CPU check-only failed")
        checked=read(stdout)
        baseline=read(ROOT/"outputs/tum-xyz-icp-full748-001/selected_sensor_metadata.json")
        if (checked["status"]!="cpu_checks_passed" or checked["configuration"]!={**row["configuration"],"check_only":True} or
            checked["selected_frame_ids"]!=list(range(748)) or checked["selected_sensor_metadata"]!=baseline["selected_frames"] or
            checked["association"]!=baseline["association"] or checked["script_sha256"]!=original["source_sha256"]["scripts/train_tum_online_gs_v4.py"]):
            raise ValueError("Recovery CPU sensor/source/full748 configuration differs")
        write_new(OUTPUT/"cpu_sensor_checks.json",{"status":"passed","frames":748,"selected_frame_ids_exact":True,
            "all_sensor_and_association_rows_exact_to_frozen_full748_baseline":True,"configuration_fields":36,
            "command":command,"GPU_runs_started":0,"source":checked["script_sha256"]})
        write_new(OUTPUT/"cpu_deadline_checks.json",deadline_checks())
        evidence={record["original"]:record["sha256"] for record in snap["files"]}
        cpu_names=("cpu_cli_checks.json","cpu_sensor_checks.json","cpu_preflight_xyz.json","cpu_preflight_xyz.stderr.log","cpu_deadline_checks.json")
        protocol={"protocol_id":"tum-online-v4-full-sequence-recovery-protocol-001","status":"prepared_awaiting_root_review_before_GPU",
            "prepared_at":stamp(),"deadline_beijing":DEADLINE.isoformat(),"new_run_count":1,"new_runs":[row],
            "original_protocol_relative_path":relative(ORIGINAL_PROTOCOL),"original_protocol_sha256":ORIGINAL_SHA,
            "original_driver_sha256":OLD_DRIVER_SHA,"original_failed_run":FAILED,
            "original_failure_artifact_sha256":evidence,"original_failure_snapshot":snap,
            "replacement_binding":{"original_failed_run":FAILED,"replacement_run":REPLACEMENT,"restart_from_frame":0,
                "not_checkpoint_resume":True,"all_36_configuration_fields_only_run_differs":True,"original_failed_command":original["new_runs"][3]["command"]},
            "source_sha256":original["source_sha256"],"driver_sha256":sha(Path(__file__)),
            "dataset_archives":original["dataset_archives"],"full_CPU_baseline_references":original["full_CPU_baseline_references"],
            "CPU_check_artifact_sha256":{relative(OUTPUT/name):sha(OUTPUT/name) for name in cpu_names},
            "selected_final_runs":[r["run"] for r in original["new_runs"][:3]]+[REPLACEMENT],
            "root_review_relative_path":relative(OUTPUT/"root_protocol_review.json"),
            "artifact_check_plan":{"original_three":"unchanged original full-protocol CPU checks",
                "replacement":"same frozen full artifact mathematics in a separate recovery-bound wrapper with explicit source/protocol provenance"},
            "controls":{"only_one_new_run":True,"no_retries_or_parameter_changes":True,"all_failed_artifacts_snapshot_retained":True,
                "single_GPU_child_wait_and_collect":True,"actual_deadline_guard_after_all_IO_immediately_before_Popen":True,
                "already_started_child_collected_even_after_deadline_or_driver_error":True,"no_live_trainer_progress_reads":True,
                "GT_final_evaluation_only_no_best_metric_selection":True},
            "limitations":["Original failed run retained partial521 cameras; no completed748 metrics/finalmap exists for it",
                "External open-reader fixture establishes a possible Windows locking mechanism, not the unique historical lock owner",
                "Same seed does not guarantee identical CUDA warmup maps; recovery is explicitly disclosed"]}
        check_hashes(evidence); old.source_guard()
        write_new(PROTOCOL,protocol)
        atomic(OUTPUT/"progress.json",{"status":"awaiting_root_protocol_review","protocol_sha256":sha(PROTOCOL),"GPU_runs_started":0,"prepared_at":stamp()})
        print(json.dumps({"protocol":relative(PROTOCOL),"protocol_sha256":sha(PROTOCOL),"driver_sha256":protocol["driver_sha256"],
            "run":REPLACEMENT,"CPU_checks":"passed","snapshot_files":len(snap["files"]),"GPU_runs_started":0},indent=2),flush=True)
    except BaseException as error:
        atomic(OUTPUT/"failure.json",{"phase":"CPU_preparation","at":stamp(),"exception":type(error).__name__,
            "message":str(error),"traceback":traceback.format_exc(),"GPU_runs_started":0})
        raise

def reviewed_protocol(expected_sha):
    if sha(PROTOCOL)!=expected_sha:
        raise ValueError("Exact reviewed protocol SHA required")
    protocol=read(PROTOCOL); old=frozen_driver(); original=read(ORIGINAL_PROTOCOL)
    if sha(Path(__file__))!=protocol["driver_sha256"] or protocol["new_runs"]!=[replacement_row(original)]:
        raise ValueError("Reviewed driver or sole unchanged replacement command mismatch")
    original_failure_gate(original)
    if old.source_guard()!=protocol["source_sha256"] or old.archive_records()!=protocol["dataset_archives"]:
        raise ValueError("Frozen source/archive guard failed")
    check_hashes(protocol["original_failure_artifact_sha256"])
    check_hashes(protocol["CPU_check_artifact_sha256"])
    snap=protocol["original_failure_snapshot"]
    if sha(ROOT/snap["manifest"])!=snap["manifest_sha256"] or read(ROOT/snap["manifest"])["files"]!=snap["files"]:
        raise ValueError("Failure snapshot manifest changed")
    check_hashes({record["snapshot"]:record["sha256"] for record in snap["files"]})
    for reference in protocol["full_CPU_baseline_references"].values():
        check_hashes(reference["artifact_sha256"])
        check_hashes({"outputs/"+reference["run"]+"/"+name:expected for name,expected in reference["validation_evidence_sha256"].items()})
    return protocol

def execute(expected_sha):
    protocol=reviewed_protocol(expected_sha); row=protocol["new_runs"][0]
    with (OUTPUT/"execution_started.json").open("x",encoding="utf-8") as stream:
        json.dump({"at":stamp(),"protocol_sha256":expected_sha},stream,indent=2)
    progress={"status":"running","started_at":stamp(),"protocol_sha256":expected_sha,"driver_sha256":protocol["driver_sha256"],
        "source_sha256":protocol["source_sha256"],"deadline_beijing":DEADLINE.isoformat(),"GPU_runs_started":0,"run_records":[],
        "original_failed_run":FAILED,"original_protocol_sha256":ORIGINAL_SHA,"current_run":REPLACEMENT}
    process=None
    try:
        folder,log=ROOT/row["output_relative_path"],ROOT/row["log_relative_path"]
        if folder.exists() or log.exists():
            raise FileExistsError("Never overwrite recovery run/log")
        record={"order":1,"run":REPLACEMENT,"configuration":row["configuration"],"status":"ready_before_actual_deadline_guard", "log":relative(log),"ready_at":stamp()}
        progress["run_records"].append(record); atomic(OUTPUT/"progress.json",progress)
        with log.open("xb") as stream:
            process,guard=launch(row["command"],stream)
            record["spawn_guard_at"]=stamp(guard)
            if process is not None:
                record.update(status="running",pid=process.pid,Popen_returned_at=stamp()); progress["GPU_runs_started"]=1
                atomic(OUTPUT/"progress.json",progress); returncode=process.wait()
        if process is None:
            record["status"]="not_started_deadline_reached_after_log_open"
            progress["status"]="deadline_reached_no_recovery_started"
        else:
            record.update(returncode=returncode,ended_at=stamp(),log_sha256=sha(log)); process=None
            if returncode!=0:
                record["status"]="failed_process_exit_all_original_artifacts_retained"
                raise RuntimeError("Recovery child failed; no further retry")
            metric_path=folder/"metrics.json"; metric=read(metric_path)
            if (metric["configuration"]!=row["configuration"] or metric["script_sha256"]!=protocol["source_sha256"]["scripts/train_tum_online_gs_v4.py"] or
                not metric["status"].startswith("completed") or metric["frames_processed"]!=748 or metric["selected_frame_ids"]!=list(range(748)) or len(metric["frame_metrics"])!=748):
                raise ValueError("Exit0 did not yield exact complete recovery748 artifacts")
            record.update(status=metric["status"],metrics_sha256=sha(metric_path),frames=748,failure_frame_count=metric["failure_frame_count"],pose_gradient_checks=metric["pose_gradient_checks"],
                trajectory_summary={key:metric["trajectory"][key] for key in ("ate_rmse_m_se3_no_scale","raw_anchored_translation_rmse_m","adjacent_rpe_translation_rmse_m","adjacent_rpe_rotation_rmse_degrees")})
            check_hashes(protocol["original_failure_artifact_sha256"])
            progress["status"]="recovery_run_completed"
        progress.update(current_run=None,ended_at=stamp()); atomic(OUTPUT/"progress.json",progress); write_new(OUTPUT/"metrics.json",progress)
        print(json.dumps(progress,indent=2),flush=True)
    except BaseException as error:
        collected=None
        if process is not None:
            collected={"pid":process.pid,"returncode":process.wait(),"at":stamp()}
        progress.update(status="recovery_stopped_failure_no_retry",failed_at=stamp(),exception_type=type(error).__name__,
            exception_message=str(error),traceback=traceback.format_exc(),already_started_child_collected=collected)
        atomic(OUTPUT/"progress.json",progress); write_new(OUTPUT/"failure.json",progress)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true"); group.add_argument("--execute",action="store_true")
    parser.add_argument("--protocol-sha"); args=parser.parse_args()
    if args.prepare:
        if args.protocol_sha: parser.error("protocol SHA is only for execution")
        prepare()
    else:
        if not args.protocol_sha or len(args.protocol_sha)!=64 or any(c not in "0123456789abcdef" for c in args.protocol_sha):
            parser.error("execution requires exact lowercase64 SHA")
        execute(args.protocol_sha)

if __name__=="__main__": main()
