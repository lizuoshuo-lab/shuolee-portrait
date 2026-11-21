"""CPU chunk export for the explicitly bound replacement, retaining old engine provenance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import run_v4_full_recovery as recovery

ROOT=recovery.ROOT
ENGINE=ROOT/"scripts/summarize_tum_online_v4_full.py"
ENGINE_SHA="4203a4aadb6547157e27ba160a9ae36e4c028f68d55670834411259c44cb4eac"

def engine():
    if recovery.sha(ENGINE)!=ENGINE_SHA:
        raise ValueError("Frozen full summary engine changed")
    spec=importlib.util.spec_from_file_location("frozen_full_recovery_summary_engine",ENGINE)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def completed_gate(protocol_sha):
    protocol=recovery.reviewed_protocol(protocol_sha)
    row=protocol["new_runs"][0]
    terminal=recovery.read(recovery.OUTPUT/"metrics.json")
    if (terminal["status"]!="recovery_run_completed" or terminal["protocol_sha256"]!=protocol_sha or
        terminal["GPU_runs_started"]!=1 or len(terminal["run_records"])!=1):
        raise ValueError("Recovery child must terminate completely before CPU export")
    record=terminal["run_records"][0]
    if (record["run"]!=row["run"] or record["returncode"]!=0 or not record["status"].startswith("completed") or
        record["configuration"]!=row["configuration"] or record["frames"]!=748 or
        recovery.sha(ROOT/row["log_relative_path"])!=record["log_sha256"] or
        recovery.sha(ROOT/row["output_relative_path"]/"metrics.json")!=record["metrics_sha256"] or
        (recovery.OUTPUT/"failure.json").exists()):
        raise ValueError("Recovery terminal/configuration/artifact hash mismatch")
    return protocol,row

def summarize(protocol_sha):
    protocol,row=completed_gate(protocol_sha); module=engine()
    folder=ROOT/row["output_relative_path"]; raw=(folder/"metrics.json").read_bytes(); original=json.loads(raw)
    module.validate_original(original,row,protocol["source_sha256"])
    result,unused,chunks=module.build_payloads(row["run"],original,raw)
    result["summary_provenance"].update({
        "recovery_wrapper_sha256":recovery.sha(Path(__file__)),"recovery_protocol_sha256":protocol_sha,
        "original_protocol_sha256":recovery.ORIGINAL_SHA,"original_failed_run":recovery.FAILED,
        "explicit_replacement_binding":protocol["replacement_binding"],
        "summarizer_sha256_role":"unchanged original chunk/field-selection engine; recovery wrapper separately identified"})
    data=module.encode(result)
    if len(data)>module.MAX_BYTES:
        raise ValueError("Recovery summary exceeds byte ceiling; no evidence dropped")
    destination=folder/"summary_chunks"
    if (folder/"summary.json").exists() or destination.exists():
        raise FileExistsError("Recovery summary/chunks must not be overwritten")
    destination.mkdir(exist_ok=False)
    for name,payload in chunks.items():
        with (destination/name).open("xb") as stream: stream.write(payload)
    with (folder/"summary.json").open("xb") as stream: stream.write(data)
    if recovery.sha(folder/"metrics.json")!=hashlib.sha256(raw).hexdigest():
        raise RuntimeError("Original completed recovery metrics changed during export")
    return {"run":row["run"],"status":original["status"],"frames":original["frames_processed"],
        "failure_frame_count":original["failure_frame_count"],"summary_sha256":recovery.sha(folder/"summary.json"),
        "summary_bytes":len(data),"chunk_count":len(chunks),"max_chunk_bytes":max(map(len,chunks.values())),
        "summarizer_sha256":ENGINE_SHA,"recovery_wrapper_sha256":recovery.sha(Path(__file__)),"recovery_protocol_sha256":protocol_sha}

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol-sha",required=True)
    args=parser.parse_args(); print(json.dumps(summarize(args.protocol_sha),indent=2,allow_nan=False))

if __name__=="__main__":main()
