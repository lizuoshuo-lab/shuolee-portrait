"""Pure-JSON fixed summary of all twenty seen-view renders, gated by a real independent audit.

Prepare performs CPU structure/hash controls only. Summarize requires the exact
reviewed summary protocol and a completed independent raw-prediction audit.
No numerical metric implementation, map/pose update, renderer, or GPU import.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"outputs/tum-online-v4-full-seen-map-evaluation-004"
OUTPUT=ROOT/"outputs/tum-v4-full-seen-map-summary-001"
PROTOCOL=OUTPUT/"protocol.json"
EVALUATION_PROTOCOL=ROOT/"manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json"
METRIC_SHA="f2b0c23f57df18a968901f77dcba881425cc7062cbf194027d444e3808da7159"
EVALUATION_PROTOCOL_SHA="7f07d29ebbc96b9f96d48d84e97bc2f9c9c042186a70f1f8a5fa87e23c8f6e40"
SOURCES={
    "scripts/evaluate_tum_v4_full_seen_maps.py":"41ee8343536096b38e6effbaa415a2fbcadc43ea7ffe7dcb5d73cdc9e51c41d2",
    "scripts/audit_v4_full_seen_map_quality.py":"020451555333f7137aa568cd8792db9e28818aba6e612d15bb87458b91271abe",
    "scripts/audit_v4_final_map_quality.py":"aabd7c5197711567e679457ce6d203939252395945f558bc7e2b120f83b8f695",
}
AUDIT=INPUT/"independent_full_seen_map_audit.json"
MAX_BYTES=1024*1024

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def relative(path):return Path(path).relative_to(ROOT).as_posix()

def encode(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")

def write_new(path,value):
    data=encode(value);require(len(data)<=MAX_BYTES,"One-MiB ceiling exceeded; do not omit requested flags/views")
    with Path(path).open("xb") as stream:stream.write(data)

def hash_guard(records):
    for name,expected in records.items():require(sha(ROOT/name)==expected,"Changed locked source/artifact: "+name)

def validate_structure(metric,evaluation):
    require(metric["status"] in ("completed","completed_with_render_failures") and metric["render_attempts"]==20 and
        metric["optimization_steps"]==0 and metric["evaluation_role"]=="seen_training_view_diagnostic","Incorrect completed render scope/count")
    require(len(metric["maps"])==4 and len(metric["pairs"])==2,"All four maps and two common-C pairs required")
    require(evaluation["requested_renders"]==20 and evaluation["evaluation_role"]=="seen_training_view_diagnostic" and
        evaluation["render_and_metric_settings"]["alignment_fit"]=="none","Unexpected seen-view protocol/GT fit")
    expected_views=evaluation["fixed_views"]
    failed=0
    for actual,expected in zip(metric["maps"],evaluation["maps"]):
        require(actual["run"]==expected["run"] and actual["scene"]==expected["scene"] and actual["prior_weight"]==expected["prior_weight"],"Map ordering/selection changed")
        require([row["frame_id"] for row in actual["views"]]==expected_views[actual["scene"]] and len(actual["views"])==5,"Missing/reselected requested view")
        require(actual["parameters_before"]==actual["parameters_after"]==expected["parameter_tensors"] and
            actual["parameter_bytes_unchanged"] is True,"Frozen map parameter evidence inconsistent")
        require(actual["history"]==expected["history"] and actual["point_count"]==expected["point_count"],"Model/history count changed")
        require(actual["aggregate"]["requested_views"]==5 and actual["training_failure_frame_count"]==expected["training_failure_frame_count"],"View denominator/training flags changed")
        failed+=sum(row["status"]!="completed" for row in actual["views"])
    require(metric["failed_render_count"]==failed and metric["successful_render_count"]==20-failed,"Failed/successful view count mismatch")
    for pair in metric["pairs"]:
        require([row["frame_id"] for row in pair["views"]]==expected_views[pair["scene"]] and
            pair["pooled_common_C"]["requested_paired_views"]==5,"Missing common-C requested rows")
    require(metric["input_tensor_bytes_before_after_unchanged"] is True and metric["original_map_and_failure_artifacts_unchanged"] is True,"Input/original failure retention gate")

def frozen_inputs():
    require(sha(EVALUATION_PROTOCOL)==EVALUATION_PROTOCOL_SHA and sha(INPUT/"metrics.json")==METRIC_SHA,"Completed eval bytes changed")
    hash_guard(SOURCES)
    metric,evaluation=read(INPUT/"metrics.json"),read(EVALUATION_PROTOCOL)
    validate_structure(metric,evaluation)
    records={relative(EVALUATION_PROTOCOL):EVALUATION_PROTOCOL_SHA,relative(INPUT/"metrics.json"):METRIC_SHA,**SOURCES}
    for mapping in metric["maps"]:
        records[mapping["history"]["path"]]=mapping["history"]["sha256"]
        records[mapping["fixed_diagnostic_figure"]["path"]]=mapping["fixed_diagnostic_figure"]["sha256"]
        for view in mapping["views"]:
            if "prediction" in view:records[view["prediction"]["path"]]=view["prediction"]["sha256"]
    for record in evaluation["maps"]:records.update(record["artifact_sha256"])
    records.update(evaluation["lineage"]["evidence_sha256"])
    hash_guard(records)
    return metric,evaluation,records

def audit_gate():
    audit=read(AUDIT)
    expected={"status":"passed_independent_full_seen_map_audit",
        "audit_source_sha256":SOURCES["scripts/audit_v4_full_seen_map_quality.py"],
        "independent_math_helper_sha256":SOURCES["scripts/audit_v4_final_map_quality.py"],
        "evaluation_script_sha256":SOURCES["scripts/evaluate_tum_v4_full_seen_maps.py"],
        "protocol_sha256":EVALUATION_PROTOCOL_SHA,"metrics_sha256":METRIC_SHA,
        "map_count":4,"requested_view_count":20,"actual_NPZ_count":20,
        "actual_prediction_payload_array_count":60,"all_training_rows_including_flags":2594,"requested_commonC_views":10,"GPU_calls":0}
    for key,value in expected.items():require(audit.get(key)==value,"Actual independent audit gate: "+key)
    return {"path":relative(AUDIT),"sha256":sha(AUDIT),"required_passed_fields":expected}

def tables(metric,evaluation):
    rows=[]
    for actual,model in zip(metric["maps"],evaluation["maps"]):
        pooled=actual["aggregate"].get("pooled")
        item={"run":actual["run"],"scene":actual["scene"],"prior_weight":actual["prior_weight"],
            "completed_views":actual["aggregate"]["completed_views"],"failed_views":actual["aggregate"]["failed_views"],
            "point_count":model["point_count"],"point_cap":model["point_cap"],"cap_reached":model["cap_reached"],
            "first_cap_frame":model["first_cap_frame"],"source_frame_histogram":model["source_frame_histogram"],
            "first_frame_source_gaussians":model["source_frame_histogram"].get("0",0),"history":model["history"],
            "all_training_failure_flags_retained_in_history":True}
        if pooled:
            item.update(rgb_psnr_db=pooled["rgb"]["psnr_db"],rgb_mse=pooled["rgb"]["mse"],
                rgb_psnr_positive_infinity=pooled["rgb"]["psnr_positive_infinity"],
                sensor_valid_pixels=pooled["valid_depth_pixels"],covered_valid_pixels=pooled["covered_valid_pixels"],
                sensor_valid_coverage=pooled["sensor_valid_coverage"],covered_depth=pooled["covered_depth"],
                all_valid_holes_as_zero_depth=pooled["all_valid_holes_as_zero_depth"])
        rows.append(item)
    return rows

def build(metric,evaluation,audit_record):
    # Retain every original value and requested row. No recomputation, filtering,
    # cross-scene mean, per-view PSNR averaging, or failed-row filling.
    result=copy.deepcopy(metric)
    result["summary_table"]=tables(metric,evaluation)
    result["summary_provenance"]={"format":"complete_20_seen_view_summary_v1","summary_source_sha256":sha(Path(__file__)),
        "evaluation_protocol_sha256":EVALUATION_PROTOCOL_SHA,"complete_metrics_sha256":METRIC_SHA,
        "independent_audit":audit_record,"metric_calculation_by_summary":"none; all scalar values copied from audited complete original metrics",
        "all_original_render_view_pair_parameter_and_failure_fields_retained":True,
        "training_flags_and_source_additions": "All2594 frame rows remain in four hash-indexed history files, each below one MiB",
        "pooling":"Each map sums five views' actual SSE and counts before RMSE/PSNR; no cross-scene mean or single-seed SD",
        "selection":"Original completed three plus the predeclared replacement002, with original failed001 lineage in frozen protocol",
        "numerical_summary_does_not_establish_strict_same_map_causal_effect":True}
    require(len(encode(result))<=MAX_BYTES,"Complete summary exceeds byte cap")
    return result

def prepare():
    require(not OUTPUT.exists(),"Do not overwrite summary preparation")
    metric,evaluation,records=frozen_inputs()
    OUTPUT.mkdir(exist_ok=False)
    negative={}
    for label,change in (("incomplete_view",lambda value:value["maps"][0]["views"].pop()),
        ("fake_optimizer",lambda value:value.update(optimization_steps=1)),
        ("changed_order",lambda value:value["maps"].reverse())):
        altered=copy.deepcopy(metric);change(altered)
        try:validate_structure(altered,evaluation)
        except ValueError:negative[label]=True
        else:raise AssertionError("Control not rejected: "+label)
    fixture=build(metric,evaluation,{"status":"CPU_size_fixture_only_not_an_actual_passed_audit"})
    cpu={"status":"passed_structure_and_byte_size_only","source_sha256":sha(Path(__file__)),"negative_controls":negative,
        "all_view_count":20,"unchanged_original_metric_fields":True,"summary_fixture_bytes":len(encode(fixture)),
        "actual_independent_audit_already_passed":False,"audit_required_before_actual_summary":True,"GPU_calls":0}
    write_new(OUTPUT/"cpu_preflight.json",cpu)
    protocol={"protocol_id":"tum-v4-full-seen-map-summary-001","status":"prepared_pending_root_source_review_and_actual_independent_audit",
        "source_sha256":sha(Path(__file__)),"input_sha256":records,
        "cpu_preflight":{"path":relative(OUTPUT/"cpu_preflight.json"),"sha256":sha(OUTPUT/"cpu_preflight.json")},
        "audit_expected_path":relative(AUDIT),"audit_gate":"Require actual passed status, frozen sources/protocol/metrics and all4/20/60/2594/10 count fields",
        "outputs":["summary.json","validation.json"],"byte_limit":MAX_BYTES,"GPU_calls":0}
    write_new(PROTOCOL,protocol)
    print(json.dumps({"status":"prepared","protocol":relative(PROTOCOL),"protocol_sha256":sha(PROTOCOL),
        "source_sha256":protocol["source_sha256"],"fixture_bytes":cpu["summary_fixture_bytes"],"GPU_calls":0},indent=2))

def summarize(protocol_sha):
    require(sha(PROTOCOL)==protocol_sha,"Exact root-reviewed summary protocol SHA required")
    protocol=read(PROTOCOL)
    require(protocol["source_sha256"]==sha(Path(__file__)),"Summary source changed after review")
    hash_guard(protocol["input_sha256"])
    hash_guard({protocol["cpu_preflight"]["path"]:protocol["cpu_preflight"]["sha256"]})
    metric,evaluation,records=frozen_inputs();require(records==protocol["input_sha256"],"Summary input set changed")
    audit=audit_gate();result=build(metric,evaluation,audit)
    write_new(OUTPUT/"summary.json",result)
    actual=read(OUTPUT/"summary.json")
    require({key:actual[key] for key in metric}==metric,"Original metric fields lost or altered")
    hash_guard(records);require(sha(AUDIT)==audit["sha256"],"Independent audit changed during export")
    validation={"status":"passed_exact_original_field_and_flag_retention","summary_sha256":sha(OUTPUT/"summary.json"),
        "summary_bytes":(OUTPUT/"summary.json").stat().st_size,"summary_source_sha256":sha(Path(__file__)),
        "summary_protocol_sha256":protocol_sha,"independent_audit":audit,"input_file_count":len(records),
        "all_inputs_rehashed_unchanged":True,"all_original_twenty_views_two_pairs_and_flags_exact":True,"GPU_calls":0}
    write_new(OUTPUT/"validation.json",validation)
    print(json.dumps({"status":validation["status"],"summary_sha256":validation["summary_sha256"],
        "summary_bytes":validation["summary_bytes"],"GPU_calls":0},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true");group.add_argument("--summarize",action="store_true");parser.add_argument("--protocol-sha")
    args=parser.parse_args()
    if args.prepare:
        if args.protocol_sha:parser.error("Protocol SHA applies only to actual summary")
        prepare()
    else:
        if not args.protocol_sha:parser.error("Exact reviewed summary protocol SHA required")
        summarize(args.protocol_sha)

if __name__=="__main__":main()
