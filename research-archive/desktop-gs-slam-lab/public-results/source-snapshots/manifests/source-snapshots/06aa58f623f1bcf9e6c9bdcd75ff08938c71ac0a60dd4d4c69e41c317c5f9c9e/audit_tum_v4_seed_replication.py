"""Protocol-bound CPU artifact auditing for eight fixed V4 seed controls.

Reuses the frozen V4 artifact check source in an isolated namespace, with only
two declared validation adaptations: seed1/2 and sequence-dependent point count.
No trainer/main/GPU/registration/renderer is executed.
"""
from __future__ import annotations
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"manifests/tum-online-v4-seed-replication-protocol-001.json"
PROTOCOL_SHA = "b86496b7731352457e5d86becdd4d06e4709efee794c1ae3d81546cdccc75eb9"
CORE = ROOT/"scripts/audit_tum_online_v4_artifacts.py"
CORE_SHA = "5496175b5e09bff227e92a4fd4ffec066980b312eaf74e7893ebb065da31670f"
BASELINES = {"desk": "tum-icp-transfer100-001", "xyz": "tum-xyz-icp-transfer100-001"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def core_functions():
    require(sha(CORE) == CORE_SHA, "frozen artifact check source changed")
    source = CORE.read_text()
    old_seed = 'config["seed"] == 0'
    old_count = '== len(context["source_frame_ids_per_gaussian"]) == 12108'
    require(source.count(old_seed) == source.count(old_count) == 1, "validation adaptations ambiguous")
    source = source.replace(old_seed, 'config["seed"] in (1, 2)').replace(old_count, '== len(context["source_frame_ids_per_gaussian"])')
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "_frozen_v4_seed_check_core"}
    exec(compile(source, str(CORE), "exec"), namespace)
    return namespace


def protocol():
    require(sha(PROTOCOL) == PROTOCOL_SHA, "protocol changed")
    p = read(PROTOCOL)
    require(len(p["new_runs"]) == p["new_run_count"] == 8, "grid count")
    require({(r["scene"], r["configuration"]["seed"], r["configuration"]["pose_prior_weight"]) for r in p["new_runs"]} == {(c,s,w) for c in ("desk","xyz") for s in (1,2) for w in (0.,.01)}, "grid changed")
    for name,digest in p["source_sha256"].items():
        require(sha(ROOT/name) == digest, "frozen trainer/helper changed "+name)
    require(sha(ROOT/"scripts/run_v4_seed_replication.py") == p["driver_sha256"] and sha(ROOT/"scripts/run_local.py") == p["wrapper_sha256"], "driver/wrapper changed")
    return p


def find_record(p, run):
    rows = [r for r in p["new_runs"] if r["run"] == run]
    require(len(rows) == 1, "not uniquely predeclared")
    return rows[0]


def summary_audit(run):
    directory = ROOT/"outputs"/run
    if not (directory/"summary.json").exists():
        return {"status":"not_available","run":run,"statistics_not_imputed":True}
    original,summary = read(directory/"metrics.json"),read(directory/"summary.json")
    provenance = summary["summary_provenance"]
    require(provenance["full_metrics_sha256"] == sha(directory/"metrics.json") and provenance["full_metrics_bytes"] == (directory/"metrics.json").stat().st_size, "summary original metric hash")
    require(provenance["summarizer_sha256"] == sha(ROOT/"scripts/summarize_tum_online_v4.py"), "summary source")
    leaves = 0
    def walk(value, base, path=""):
        nonlocal leaves
        require(type(value) is type(base), "summary type "+path)
        if isinstance(value,dict):
            for key,item in value.items():
                require(key in base,"invented summary field")
                walk(item,base[key],path+"/"+key)
        elif isinstance(value,list):
            require(len(value) == len(base),"summary list/failed frame drop")
            for item,source in zip(value,base):walk(item,source,path)
        else:
            require(value == base,"summary scalar "+path)
            leaves += 1
    walk({k:v for k,v in summary.items() if k != "summary_provenance"},original)
    for name in ("configuration","selected_frame_ids","mapping_events","pose_gradient_checks","pose_prior","first_tracking_context"):
        require(summary[name] == original[name],"mandatory summary field "+name)
    require(len(summary["frame_metrics"]) == original["frames_processed"] == 100,"summary count")
    require([r["failure_flags"] for r in summary["frame_metrics"]] == [r["failure_flags"] for r in original["frame_metrics"]],"summary flags")
    require([r["icp"].get("failure_flags") for r in summary["frame_metrics"]] == [r["icp"].get("failure_flags") for r in original["frame_metrics"]],"summary ICP flags")
    required = {"initial_loss","optimized_loss_before_mapping","initial_sensor_loss","selected_sensor_loss_before_mapping","tracking_objective_scope","best_step_selected_by_tracking_objective_only"}
    require(all(required.issubset(r) for r in summary["frame_metrics"][1:]),"missing public prior/sensor data")
    return {"status":"passed","run":run,"scalar_leaves_type_and_value_exact":leaves,"summary_bytes":(directory/"summary.json").stat().st_size,"summary_sha256":sha(directory/"summary.json"),"metrics_sha256":sha(directory/"metrics.json"),"all_frames_flags_probe_context_retained":True}


def audit_run(run):
    from tum_rgbd import associate_frames
    p = protocol();record = find_record(p,run)
    directory = ROOT/"outputs"/run;m = read(directory/"metrics.json");config = m["configuration"]
    require(config == record["configuration"],"full protocol configuration")
    scene = record["scene"]
    require(Path(config["dataset"]).resolve() == (ROOT/f"data/tum/rgbd_dataset_freiburg1_{scene}").resolve(),"actual scene")
    reference = next(r for r in p["existing_seed0_records"][scene] if r["weight"] == config["pose_prior_weight"])
    require(sha(ROOT/"outputs"/reference["run"]/"metrics.json") == reference["metrics_sha256"],"seed0 witness changed")
    differences = sorted(k for k in config if config[k] != reference["configuration"][k])
    require(differences == ["run","seed"],"unexpected seed tuning")
    core = core_functions()
    result = core["audit_run"](run,BASELINES[scene])
    frames,association = associate_frames(Path(config["dataset"]),config["max_association_dt"],config["max_gt_dt"])
    require(association == m["association"],"actual dataset association")
    selected = frames[200:300]
    metadata = [{k:v for k,v in f.metadata().items() if k not in ("c2w","w2c")} for f in selected]
    require(metadata == m["selected_sensor_metadata"] and [f.frame_id for f in selected] == m["selected_frame_ids"],"actual selected rows")
    checkpoint = torch.load(directory/"online_map.pt",map_location="cpu",weights_only=True)
    require(np.array_equal(np.stack([f.c2w for f in selected]),checkpoint["evaluation_only_gt_c2w"].numpy()),"posthoc GT checkpoint")
    text = core["parse_trajectory_text"]((directory/"trajectory_estimated.txt").read_text())
    serialized = [Decimal(format(f.rgb_timestamp,".9f")) for f in selected]
    require(text["timestamps"] == serialized,"epoch float9 serialization")
    errors = [abs(a-Decimal(Path(f.rgb_path).stem)) for a,f in zip(serialized,selected)]
    bounds = [Decimal(str(float(np.spacing(f.rgb_timestamp))/2))+Decimal(".0000000005") for f in selected]
    require(all(e <= b for e,b in zip(errors,bounds)),"raw source Decimal timestamp bound")
    result["protocol_checks"] = {"protocol_sha256":PROTOCOL_SHA,"scene":scene,"seed":config["seed"],"weight":config["pose_prior_weight"],"configuration_fields_exact":len(config),"seed0_reference":reference["run"],"seed0_changed_fields":differences}
    result["reuse_scope"] = {"frozen_artifact_check_source_sha256":CORE_SHA,"isolated_namespace":True,"validation_adaptations":["seed1/2 rather than seed0","actual initial-point count rather than desk12108"]}
    result["actual_dataset_checks"] = {"all_100_selected_sensor_rows_equal":True,"evaluation_GT_checkpoint_equal_posthoc_raw_GT":True,"strict_Decimal_text_pairing":True,"maximum_float9_text_vs_raw_decimal_seconds":str(max(errors)),"maximum_half_epoch_ULP_plus_half_ns_bound_seconds":str(max(bounds)),"GT_scope":"Auditor reads GT only to validate final evaluation; no tracking/mapping/registration or pose selection."}
    result["public_summary"] = summary_audit(run)
    require(not torch.cuda.is_initialized(),"CPU initialized CUDA")
    return result


def batch_status():
    p = protocol();states=[]
    for r in p["new_runs"]:
        directory = ROOT/"outputs"/r["run"]
        state = {"run":r["run"],"scene":r["scene"],"seed":r["configuration"]["seed"],"weight":r["configuration"]["pose_prior_weight"]}
        if (directory/"metrics.json").exists():
            metric = read(directory/"metrics.json");state.update({"status":metric["status"],"failure_frame_count":metric.get("failure_frame_count"),"probe":metric.get("pose_gradient_checks"),"metrics_sha256":sha(directory/"metrics.json")})
        elif (directory/"failure.json").exists():
            state.update({"status":"failed_original_artifacts_retained","failure_sha256":sha(directory/"failure.json")})
        else:
            state["status"] = "started_not_completed" if directory.exists() else "not_started"
        if (directory/"independent_seed_v4_artifact_audit.json").exists():
            audit = read(directory/"independent_seed_v4_artifact_audit.json")
            require(audit["artifact_auditor_sha256"] == sha(Path(__file__)),"stale audit source")
            state.update({"artifact_audit":audit["status"],"artifact_audit_sha256":sha(directory/"independent_seed_v4_artifact_audit.json"),"metrics":audit["metrics"]})
        states.append(state)
    pairs=[]
    for scene in ("desk","xyz"):
        for seed in (1,2):
            members=[s for s in states if s["scene"] == scene and s["seed"] == seed]
            paths=[ROOT/"outputs"/s["run"]/"first_tracking_context.pt" for s in members]
            row={"scene":scene,"seed":seed,"runs":[s["run"] for s in members]}
            if all(path.exists() for path in paths):
                contexts=[torch.load(path,map_location="cpu",weights_only=True) for path in paths]
                row["warmup_parameters"]={key:{"byte_values_equal":torch.equal(contexts[0]["parameters"][key],contexts[1]["parameters"][key]),"same_seed_order_elementwise_max_abs_difference":float((contexts[0]["parameters"][key]-contexts[1]["parameters"][key]).abs().max())} for key in contexts[0]["parameters"]}
                row["warmup_byte_values_identical"]=all(v["byte_values_equal"] for v in row["warmup_parameters"].values())
                row["same_current_RGB_D_K_reference_prefix"]=all(torch.equal(contexts[0][k],contexts[1][k]) for k in ("image","depth_m","valid_depth_mask","K","initial_w2c","prior_reference_w2c","icp_prefix_c2w_float64"))
                row["comparison_scope"]="Within the same seed pair only. Cross-seed ordered arrays can contain reordered/differently sampled physical points."
            else:
                row["comparison_status"]="not_available"
            pairs.append(row)
    complete=all(s.get("artifact_audit") == "passed" for s in states)
    return {"status":"all_eight_artifacts_audited" if complete else "partial_snapshot_no_statistics","protocol_sha256":PROTOCOL_SHA,"artifact_auditor_sha256":sha(Path(__file__)),"runs":states,"paired_context_checks":pairs,"all_predeclared_runs_and_original_failures_retained":True,"group_means":None,"statistics_scope":"No early/imputed means or best-seed selection; report all fixed runs. Root handles subsequent statistics.","limits":["Trajectory math reuses audit_pose_arrays also used by ICP; not a second independent ICP formula.","Actual same-seed warmup equality is measured rather than assumed.","Only two Freiburg1 sequences/sensor domain.","CPU only; no registration or sensor rerender."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--run");modes.add_argument("--batch-status",action="store_true")
    parser.add_argument("--summary-only",action="store_true")
    parser.add_argument("--self-test",action="store_true");parser.add_argument("--output",type=Path)
    args=parser.parse_args();result={}
    if args.self_test:result["self_checks"]=core_functions()["self_check"]()
    if args.run:
        find_record(protocol(),args.run)
        result.update(summary_audit(args.run) if args.summary_only else audit_run(args.run))
    elif args.batch_status:result.update(batch_status())
    elif not args.self_test:parser.error("provide run/batch-status/self-test")
    output=args.output
    if output is None and args.run:output=ROOT/"outputs"/args.run/("independent_seed_public_summary_audit.json" if args.summary_only else "independent_seed_v4_artifact_audit.json")
    if output:
        with output.open("x",encoding="utf-8")as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
