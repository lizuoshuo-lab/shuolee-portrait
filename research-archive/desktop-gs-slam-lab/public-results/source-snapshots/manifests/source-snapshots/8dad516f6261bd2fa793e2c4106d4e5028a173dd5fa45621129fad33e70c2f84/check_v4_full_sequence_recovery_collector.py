"""CPU-only synthetic recovery gates and actual non-aggregating readiness.

No original result is edited, no recovery is selected using a metric, and no
registration, renderer or training entrypoint is called. Old metric controls
are reused transparently; these tests do not independently prove solver quality.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import timedelta
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile

import summarize_v4_full_sequence_recovery_controls as c

CHECKER_REFERENCE = "scripts/check_v4_full_sequence_collector.py"
CHECKER_REFERENCE_SHA = "249fe4f6b2999b7fb46eb0bc41a88395f2f6eb3a0edf02f30df936f1e39429b9"


def rejected(call):
    try:
        call()
    except (ValueError, KeyError, OSError) as error:
        return {"rejected": True, "reason": str(error)}
    raise AssertionError("Invalid recovery fixture unexpectedly passed")


def fixture(original, controls):
    failed = controls.synthetic_ledger(original)
    failed["status"] = c.OLD_FAILED_STATUS
    failed["current_run"] = c.FAILED_RUN
    failed["all_original_artifacts_and_logs_retained"] = True
    failed["failed_at"] = failed.pop("ended_at")
    last = failed["run_records"][-1]
    last["returncode"], last["status"] = 1, c.OLD_FAILED_RECORD
    last.pop("metrics_sha256")
    replacement = copy.deepcopy(original["new_runs"][-1])
    replacement.update(run=c.REPLACEMENT_RUN, output_relative_path="outputs/"+c.REPLACEMENT_RUN,
        log_relative_path="logs/"+c.REPLACEMENT_RUN+".log")
    replacement["configuration"]["run"] = c.REPLACEMENT_RUN
    replacement["command"][replacement["command"].index("--run")+1] = c.REPLACEMENT_RUN
    files = [c.legacy.BATCH+"/"+name for name in ("failure.json", "progress.json", "fourth_run_failure_evidence.json")]
    files += ["outputs/"+c.FAILED_RUN+"/"+name for name in
        ("failure.json", "progress.json", "progress.json.tmp", "last_checkpoint.pt", "first_tracking_context.pt", "first_tracking_context.json")]
    files += ["logs/"+c.FAILED_RUN+".log"]
    recovery = {"original_protocol_sha256": c.legacy.PROTOCOL_SHA,
        "original_failed_run": c.FAILED_RUN,
        "selected_final_runs": [r["run"] for r in original["new_runs"][:3]] + [c.REPLACEMENT_RUN],
        "source_sha256": copy.deepcopy(original["source_sha256"]), "deadline_beijing": original["deadline_beijing"],
        "new_run_count": 1, "new_runs": [replacement], "driver_sha256": c.RECOVERY_SOURCE_LOCKS["scripts/run_v4_full_recovery.py"],
        "original_failure_artifact_sha256": {p:"e"*64 for p in files},
        "replacement_binding":{"original_failed_run":c.FAILED_RUN,"replacement_run":c.REPLACEMENT_RUN,
            "restart_from_frame":0,"not_checkpoint_resume":True,"all_36_configuration_fields_only_run_differs":True,
            "original_failed_command":copy.deepcopy(original["new_runs"][3]["command"])},
        "dataset_archives":copy.deepcopy(original["dataset_archives"]),
        "full_CPU_baseline_references":copy.deepcopy(original["full_CPU_baseline_references"]),
        "original_failure_snapshot": {"manifest":"outputs/synthetic-recovery/snapshot.json", "manifest_sha256":"f"*64,
            "files":[{"original": p, "snapshot": "outputs/synthetic-recovery/snapshot/"+str(i), "sha256":"e"*64} for i,p in enumerate(files)]}}
    record = copy.deepcopy(failed["run_records"][0])
    record.update(run=c.REPLACEMENT_RUN, configuration=copy.deepcopy(replacement["configuration"]),
        frames=748, log=replacement["log_relative_path"], order=1)
    start = c.stamp(failed["failed_at"]) + timedelta(minutes=1)
    for field, seconds in (("ready_at", 1), ("spawn_guard_at", 2), ("Popen_returned_at", 3), ("ended_at", 62)):
        record[field] = controls.time_record(start + timedelta(seconds=seconds))
    ledger = {"status": c.RECOVERY_TERMINAL, "protocol_sha256":"a"*64,
        "driver_sha256":recovery["driver_sha256"], "source_sha256":copy.deepcopy(original["source_sha256"]),
        "deadline_beijing":original["deadline_beijing"], "GPU_runs_started":1, "current_run":None,
        "run_records":[record], "started_at":controls.time_record(start),
        "ended_at":controls.time_record(start+timedelta(seconds=63))}
    return failed, recovery, ledger


def gate_checks(original, controls):
    failed, recovery, ledger = fixture(original, controls)
    durations = c.original_failure_gate(original, failed)
    assert len(durations) == 4
    record, duration = c.recovery_ledger_gate(original, failed, recovery, "a"*64, ledger)
    assert record["run"] == c.REPLACEMENT_RUN and duration == 60
    results = {"valid_original_three_plus_one_failed_and_one_explicit_recovery_fixture":"passed",
        "legacy_rejects_failed_original":rejected(lambda:c.legacy.validate_ledger(original,failed))}
    old_cases = {}
    x=copy.deepcopy(failed);x["run_records"].pop();old_cases["original_failure_record_omitted"]=x
    x=copy.deepcopy(failed);x["status"]=c.legacy.TERMINAL;old_cases["failed_batch_relabeled_success"]=x
    x=copy.deepcopy(failed);x["run_records"][-1]["returncode"]=0;old_cases["failed_child_exit_overwritten"]=x
    x=copy.deepcopy(failed);x["run_records"][-1]["metrics_sha256"]="a"*64;old_cases["failed_child_fabricated_metrics"]=x
    x=copy.deepcopy(failed);x["run_records"][0]["returncode"]=1;old_cases["original_first_three_nonzero"]=x
    for name, obj in old_cases.items():
        results[name]=rejected(lambda obj=obj:c.original_failure_gate(original,obj))
    protocol_cases={}
    x=copy.deepcopy(recovery);x["new_runs"]=[];protocol_cases["replacement_missing"]=x
    x=copy.deepcopy(recovery);x["new_runs"][0]["configuration"]["pose_prior_weight"]=.02;protocol_cases["replacement_weight_changed"]=x
    x=copy.deepcopy(recovery);x["new_runs"][0]["configuration"]["start_frame"]=521;protocol_cases["checkpoint_resume_instead_of_frame0"]=x
    x=copy.deepcopy(recovery);x["source_sha256"]["scripts/tum_rgbd.py"]="0"*64;protocol_cases["source_changed"]=x
    x=copy.deepcopy(recovery);x["selected_final_runs"]="best_ATE";protocol_cases["selection_by_best_metric"]=x
    x=copy.deepcopy(recovery);x["new_runs"][0]["run"]="tum-xyz-online-v4-prior001-full748-003";protocol_cases["alternative_run_selected"]=x
    x=copy.deepcopy(recovery);x["original_failure_artifact_sha256"].pop("logs/"+c.FAILED_RUN+".log");protocol_cases["failed_log_omitted"]=x
    x=copy.deepcopy(recovery);x["original_failure_snapshot"]["files"].pop();protocol_cases["snapshot_file_omitted"]=x
    x=copy.deepcopy(recovery);x["original_failure_snapshot"]["files"][0]["sha256"]="0"*64;protocol_cases["snapshot_hash_mismatch"]=x
    x=copy.deepcopy(recovery);x["replacement_binding"]["not_checkpoint_resume"]=False;protocol_cases["recovery_resume_binding"]=x
    for name,obj in protocol_cases.items():
        results[name]=rejected(lambda obj=obj:c.recovery_protocol_gate(original,obj))
    ledger_cases={}
    x=copy.deepcopy(ledger);x["status"]="running";ledger_cases["replacement_not_terminal"]=x
    x=copy.deepcopy(ledger);x["run_records"]=[];ledger_cases["replacement_record_missing"]=x
    x=copy.deepcopy(ledger);x["run_records"][0]["returncode"]=1;ledger_cases["replacement_nonzero"]=x
    x=copy.deepcopy(ledger);x["run_records"][0]["frames"]=747;ledger_cases["replacement_incomplete_frames"]=x
    x=copy.deepcopy(ledger);x["source_sha256"]["scripts/train_tum_online_gs_v4.py"]="0"*64;ledger_cases["replacement_ledger_source_mismatch"]=x
    x=copy.deepcopy(ledger);x["protocol_sha256"]="b"*64;ledger_cases["replacement_protocol_hash_mismatch"]=x
    x=copy.deepcopy(ledger);x["run_records"][0]["configuration"]["tracking_steps"]=99;ledger_cases["replacement_ledger_config_mismatch"]=x
    x=copy.deepcopy(ledger);x["GPU_runs_started"]=2;ledger_cases["extra_recovery_attempt"]=x
    x=copy.deepcopy(ledger);x["run_records"][0]["log_sha256"]="unknown";ledger_cases["replacement_log_hash_missing"]=x
    x=copy.deepcopy(ledger);x["started_at"]=failed["started_at"];x["run_records"][0]["ready_at"]=failed["started_at"];ledger_cases["recovery_overlaps_failed_original"]=x
    for name,obj in ledger_cases.items():
        results[name]=rejected(lambda obj=obj:c.recovery_ledger_gate(original,failed,recovery,"a"*64,obj))
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary); path=folder/"old-failure.json";path.write_bytes(b"original failure")
        original_sha=c.sha(path);path.write_bytes(b"overwritten")
        results["old_failure_file_overwritten"]=rejected(lambda:c.verify_hash(folder,path.name,original_sha,{}))
        results["snapshot_path_escape"]=rejected(lambda:c.safe_path(folder,"../escape.json"))
    return results


def inherited_gate_body_checks():
    original = inspect.getsource(c.legacy.validate_run)
    expected = original.replace("def validate_run(", "def validate_replacement_run(")
    expected = expected.replace('audit["protocol_sha256"] == PROTOCOL_SHA',
        'audit["protocol_sha256"] == protocol["_exact_recovery_protocol_sha256"]')
    expected = expected.replace('    check_chunks(folder, original, read_json(folder / "summary.json"), audit, observed)',
        '    legacy.check_chunks(folder, original, read_json(folder / "summary.json"), audit, observed)')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(inspect.getsource(c.validate_replacement_run)))
    for name in ("trajectory", "reconstruct_truth", "compare_reported", "angle_degrees", "column_table", "mean"):
        assert getattr(c, name) is getattr(c.legacy, name)
    return {"replacement_gate_body_same_except_explicit_protocol_checker_identity_and_frozen_chunk_call": True,
        "metric_function_objects_are_the_exact_locked_legacy_functions": True,
        "not_claimed_as_independent_second_metric_implementation": True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="v4-full-sequence-recovery-collector-cpu-check-001")
    parser.add_argument("--recovery-protocol-sha",required=True)
    args=parser.parse_args()
    if not args.run.startswith("v4-full-sequence-recovery-collector-cpu-check-") or any(x in args.run for x in ("/","\\")):
        parser.error("Exclusive CPU-check name required")
    output=c.ROOT/"outputs"/args.run;output.mkdir(exist_ok=False)
    try:
        source=c.ROOT/CHECKER_REFERENCE
        assert c.sha(source)==CHECKER_REFERENCE_SHA, "Original CPU controls source changed"
        spec=importlib.util.spec_from_file_location("frozen_collector_cpu_controls",source)
        controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
        original=c.read_json(c.ROOT/c.legacy.PROTOCOL)
        gates=gate_checks(original,controls)
        original_controls=controls.gate_checks(original)
        formulas=controls.formula_checks()
        inherited=inherited_gate_body_checks()
        actual_failure=c.read_json(c.ROOT/c.legacy.BATCH/"failure.json")
        c.original_failure_gate(original,actual_failure)
        legacy_rejection=rejected(lambda:c.legacy.validate_ledger(original,actual_failure))
        preflight=c.check_inputs(args.recovery_protocol_sha)
        assert "torch" not in sys.modules and "gsplat" not in sys.modules
        result={"status":"passed_CPU_recovery_gate_controls","collector_sha256":c.sha(Path(c.__file__)),
            "checker_sha256":c.sha(Path(__file__)),"legacy_collector_sha256":c.LEGACY_SHA,
            "legacy_CPU_controls_sha256":c.sha(source),"synthetic_recovery_gates":gates,
            "unchanged_original_gate_controls":original_controls,"reused_closed_form_metric_controls":formulas,
            "inherited_source_body_checks":inherited,
            "actual_original_failed_ledger_sha256":c.sha(c.ROOT/c.legacy.BATCH/"failure.json"),
            "actual_legacy_rejects_failed_batch":legacy_rejection,"actual_input_preflight":preflight,
            "real_trajectory_aggregation_runs":0,"GPU_calls":0,"torch_and_gsplat_not_imported":True,
            "scope":"Synthetic gate controls plus collected-artifact readiness. Not a training rerun, independent metric implementation, or claim of solver accuracy."}
        c.exclusive_json(output/"checks.json",result)
        for p in (Path(c.__file__),Path(__file__)):
            with (output/p.name).open("xb") as stream:stream.write(p.read_bytes())
        print(json.dumps({"status":result["status"],"actual_ready":preflight["ready"],
            "gate_count":len(gates),"output":str(output),"collector_sha256":result["collector_sha256"]},indent=2))
    except BaseException as error:
        c.exclusive_json(output/"failure.json",{"status":"failed_CPU_recovery_check","error":str(error),"GPU_calls":0})
        raise


if __name__=="__main__":
    main()
