"""CPU recovery-bound wrapper; frozen full artifact mathematics are reused unchanged."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import run_v4_full_recovery as recovery
import summarize_tum_v4_full_recovery as summary_wrapper

ROOT=recovery.ROOT
ENGINE=ROOT/"scripts/check_tum_v4_full_artifacts.py"
ENGINE_SHA="68f2e360558f81f4cf3dc040be99c94d0390fea590bb4ebc2d446712834405d3"

def check(protocol_sha):
    protocol,row=summary_wrapper.completed_gate(protocol_sha)
    if recovery.sha(ENGINE)!=ENGINE_SHA:
        raise ValueError("Frozen full artifact check engine changed")
    spec=importlib.util.spec_from_file_location("frozen_full_recovery_artifact_engine",ENGINE)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The source bytes remain untouched. Only this private module's protocol
    # binding is changed so its same generic count/configuration checks apply
    # to the one disclosed replacement; provenance retains both sources.
    module.PROTOCOL=recovery.PROTOCOL; module.PROTOCOL_SHA=protocol_sha
    result=module.check_run(row["run"])
    summary=recovery.read(ROOT/row["output_relative_path"]/"summary.json")
    provenance=summary["summary_provenance"]
    if (provenance["recovery_wrapper_sha256"]!=recovery.sha(Path(summary_wrapper.__file__)) or
        provenance["recovery_protocol_sha256"]!=protocol_sha or provenance["original_protocol_sha256"]!=recovery.ORIGINAL_SHA or
        provenance["explicit_replacement_binding"]!=protocol["replacement_binding"]):
        raise ValueError("Recovery summary wrapper/protocol/binding provenance mismatch")
    result.update({"recovery_checker_wrapper_sha256":recovery.sha(Path(__file__)),
        "recovery_summary_wrapper_sha256":recovery.sha(Path(summary_wrapper.__file__)),
        "recovery_protocol_sha256":protocol_sha,"original_protocol_sha256":recovery.ORIGINAL_SHA,
        "original_failed_run":recovery.FAILED,"explicit_replacement_binding":protocol["replacement_binding"],
        "full_checker_sha256_role":"unchanged artifact mathematics; this separately identified wrapper binds recovery protocol"})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol-sha",required=True)
    args=parser.parse_args(); result=check(args.protocol_sha)
    output=ROOT/"outputs"/recovery.REPLACEMENT/"cpu_full_v4_artifact_checks.json"
    recovery.write_new(output,result)
    print(json.dumps({key:result[key] for key in ("status","run","frame_count","failure_frame_count","full_checker_sha256","recovery_checker_wrapper_sha256","recovery_protocol_sha256","GPU_calls")},indent=2))

if __name__=="__main__":main()
