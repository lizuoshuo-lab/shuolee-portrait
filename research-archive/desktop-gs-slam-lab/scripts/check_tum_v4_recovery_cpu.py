"""CPU preparation controls for recovery export/check wrappers; no recovery metrics are fabricated."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import torch
import run_v4_full_recovery as recovery
import summarize_tum_v4_full_recovery as wrapper

def main():
    protocol_sha=recovery.sha(recovery.PROTOCOL)
    protocol=recovery.reviewed_protocol(protocol_sha)
    negative={}
    try: recovery.reviewed_protocol("0"*64)
    except ValueError: negative["unknown_protocol_rejected_before_execution"]=True
    else: raise AssertionError("unknown protocol accepted")
    try: wrapper.completed_gate(protocol_sha)
    except (FileNotFoundError,ValueError): negative["unstarted_recovery_has_no_complete_export"]=True
    else: raise AssertionError("unstarted recovery was called completed")
    with tempfile.TemporaryDirectory(prefix="recovery_marker_cpu_") as directory:
        marker=Path(directory)/"execution_started.json"
        recovery.write_new(marker,{"CPU_fixture_only":True})
        try: recovery.write_new(marker,{"CPU_fixture_only":True})
        except FileExistsError: negative["exclusive_marker_no_overwrite"]=True
        else: raise AssertionError("exclusive marker overwritten")
    module=wrapper.engine()
    source_run="tum-xyz-online-v4-prior0-full748-001"
    source=recovery.ROOT/"outputs"/source_run/"metrics.json"
    raw=source.read_bytes(); original=json.loads(raw)
    row=recovery.read(recovery.ORIGINAL_PROTOCOL)["new_runs"][2]
    module.validate_original(original,row,protocol["source_sha256"])
    result,data,chunks=module.build_payloads(source_run,original,raw)
    incomplete={**original,"frames_processed":747}
    try: module.validate_original(incomplete,row,protocol["source_sha256"])
    except ValueError: negative["incomplete_count_rejected"]=True
    else: raise AssertionError("incomplete count accepted")
    try: module.build_payloads(source_run,{**original,"status":"running"},raw)
    except ValueError: negative["running_metrics_rejected"]=True
    else: raise AssertionError("running metrics accepted")
    engine_path=recovery.ROOT/"scripts/check_tum_v4_full_artifacts.py"
    expected="68f2e360558f81f4cf3dc040be99c94d0390fea590bb4ebc2d446712834405d3"
    if recovery.sha(engine_path)!=expected: raise AssertionError("frozen checker source changed")
    spec=importlib.util.spec_from_file_location("recovery_cpu_shared_math_checks",engine_path)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    math_checks=checker.shared.self_check()
    if torch.cuda.is_initialized(): raise AssertionError("CPU fixture initialized CUDA")
    output=recovery.OUTPUT/"cpu_recovery_wrapper_checks-001.json"
    scripts=("run_v4_full_recovery.py","summarize_tum_v4_full_recovery.py","check_tum_v4_full_recovery_artifacts.py","check_tum_v4_recovery_cpu.py")
    recovery.write_new(output,{"status":"passed","at":recovery.stamp(),"protocol_sha256":protocol_sha,
        "source_sha256":{name:recovery.sha(recovery.ROOT/"scripts"/name) for name in scripts},
        "negative_controls":negative,"deadline_controls":recovery.deadline_checks(),
        "real_completed_source_chunk_fixture":{"run":source_run,"metrics_sha256":hashlib.sha256(raw).hexdigest(),
            "metric_bytes":len(raw),"frames":748,"summary_bytes":len(data),"chunk_count":len(chunks),
            "max_chunk_bytes":max(map(len,chunks.values())),"no_original_files_written":True,
            "summary_engine_sha256":wrapper.ENGINE_SHA},
        "independent_shared_math_self_checks":math_checks,
        "not_a_completed_recovery_artifact_audit":True,"GPU_calls":0,"CUDA_initialized":False})
    print(json.dumps({"status":"passed","output":recovery.relative(output),"GPU_calls":0,
        "source_sha256":{name:recovery.sha(recovery.ROOT/"scripts"/name) for name in scripts}},indent=2))

if __name__=="__main__":main()
