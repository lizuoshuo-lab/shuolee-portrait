"""Read saved old8D/hybrid prefixes; never import/call a registration solver.

Compare all fields of the frozen 15c regular return, all nested iterations,
initialization, serialized sensor identities and boundary final decisions.
Preserve schema differences separately; no original row/file byte equality.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOLVER="scripts/rgbd_affine_photometric_odometry.py"
SOLVER_SHA="15c7f160aa0f2134ea2a40d0fc0817c5642ca9e71f053e4cb0896235ac402349"
SCENES=[
    ("desk","tum-desk-lastaccepted-8d-full549-001","52bf33985013dda1f654348aae2814aa1d5c797c2e0c2abf3ba782dac31a4262",
     "tum-desk-affine-icp-fallback-full549-001","f4b6cef8240f96054163dc3ac4f354cc00bdf98a5f484569254b3b6b5bb35b3d"),
    ("xyz","tum-xyz-lastaccepted-8d-full748-001","312259b7fd1545b2ff30ba05146a8a92456cf0b8470616a70bea1924f9539f9f",
     "tum-xyz-affine-icp-fallback-full748-001","c60f3b5827bccbee82996cc54fcafa09be6c8c91107c8af133ebe44ec933569c"),
]
SENSOR_KEYS=["frame_id","rgb_index","depth_index","rgb_timestamp","depth_timestamp","rgb_depth_abs_dt_s","rgb_path","depth_path"]
INIT_KEYS=["pair_initial_transform","pair_initial_log_relative_gain","pair_initial_bias_intensity","brightness_accumulated_across_pairs","pair_initialized_from_GT"]
POST_KEYS=["frame_id","local_index","current_frame_id","accepted","failure_flags","reference_frame_id","reference_gap_frames","last_accepted_frame_id_before","last_accepted_frame_id_after","estimated_c2w"]


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def differences(a,b,path=""):
    """Every typed leaf and shape difference; no tolerances, no skipped keys."""
    result=[]
    numeric=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool)
    if type(a) is not type(b):
        result.append({"path":path,"kind":"numeric_type" if numeric(a) and numeric(b) else "value_type",
            "old_type":type(a).__name__,"new_type":type(b).__name__,"old":a,"new":b})
    elif isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            p=path+"/"+str(k)
            if k not in a:result.append({"path":p,"kind":"schema_new_only","new":b[k]})
            elif k not in b:result.append({"path":p,"kind":"schema_old_only","old":a[k]})
            else:result.extend(differences(a[k],b[k],p))
    elif isinstance(a,list):
        if len(a)!=len(b):result.append({"path":path,"kind":"list_length","old_length":len(a),"new_length":len(b)})
        for i in range(max(len(a),len(b))):
            p=path+"/"+str(i)
            if i>=len(a):result.append({"path":p,"kind":"list_new_only","new":b[i]})
            elif i>=len(b):result.append({"path":p,"kind":"list_old_only","old":a[i]})
            else:result.extend(differences(a[i],b[i],p))
    elif a!=b:
        record={"path":path,"kind":"numeric_value" if numeric(a) else "scalar_value","old":a,"new":b}
        if numeric(a):
            require(math.isfinite(a) and math.isfinite(b),"Saved numbers must be finite")
            record["absolute_difference"]=abs(b-a)
        result.append(record)
    return result


def fixture_checks():
    require(not differences({"a":[1,{"x":2.}]},{"a":[1,{"x":2.}]}),"Exact nested fixture")
    require(differences({"a":1.},{"a":2.})[0]["kind"]=="numeric_value","Numeric change fixture")
    require(differences({"a":1},{"a":1.})[0]["kind"]=="numeric_type","Numeric type must not be silently coerced")
    require(differences({"a":1},{"a":1,"b":None})[0]["kind"]=="schema_new_only","Additional field must be visible")
    require(any(x["kind"]=="list_old_only" for x in differences([1,2],[1])),"Missing list item retained")
    return {"status":"passed_five_readback_comparator_fixtures","solver_calls":0}


def return_fields():
    require(sha(ROOT/SOLVER)==SOLVER_SHA,"Frozen 15c solver source changed")
    tree=ast.parse((ROOT/SOLVER).read_text(encoding="utf-8"))
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="register_pair_affine")
    returned=next(x for x in reversed(function.body) if isinstance(x,ast.Return))
    require(isinstance(returned.value,ast.Dict) and all(isinstance(k,ast.Constant) and isinstance(k.value,str) for k in returned.value.keys),"Regular return field list from AST only")
    fields=[k.value for k in returned.value.keys]
    require(len(fields)==len(set(fields))==19,"Full frozen regular 19-field return")
    return fields


def initialization(row,primary):
    result={k:row[k] for k in INIT_KEYS}
    history=primary.get("iteration_history",[])
    result["first_iteration_initialization_observed"]=bool(history)
    if history:
        result["first_iteration_transform_before"]=history[0]["transform_before"]
        result["first_iteration_log_gain_before"]=history[0]["log_gain_before"]
        result["first_iteration_bias_before"]=history[0]["bias_before"]
    return result


def compare_scene(scene,old_name,new_name,fields,append):
    old=json.loads((ROOT/"outputs"/old_name/"metrics.json").read_bytes())
    new=json.loads((ROOT/"outputs"/new_name/"metrics.json").read_bytes())
    require(old["status"]=="completed_with_pair_rejections" and new["status"]=="completed","Both original terminal results required")
    require(old["source_sha256"][Path(SOLVER).name]==new["source_sha256"][Path(SOLVER).name]==SOLVER_SHA,"Same frozen solver SHA in actual saved results")
    ar,br=old["frame_metrics"],new["frame_metrics"]
    require(len(ar)==len(br)==old["frames_processed"]==new["frames_processed"],"No saved frames dropped")
    first=next(r["frame_id"] for r in br if r["fallback_selected"])
    require(first+1<len(br),"Next frame exists for causal boundary comparison")
    by_pair=[]
    config_diff=differences(old["configuration"],new["configuration"],"/pipeline_configuration")
    for d in config_diff:append(scene,None,"pipeline_configuration",d)
    for i in range(1,first+2):
        a,b=ar[i],br[i];p=b["primary"]
        require(a["frame_id"]==b["frame_id"]==i,"Exact chronology")
        # Old wrapper flattens regular solver return. It cannot be compared as
        # one whole row to new nested primary+fallback/post-integration state.
        ap={k:a[k] for k in fields if k in a}
        solver_diff=differences(ap,p,"/solver_return")
        missing_old=[k for k in fields if k not in a]
        missing_new=[k for k in fields if k not in p]
        extras_new=[k for k in p if k not in fields]
        for d in solver_diff:append(scene,i,"all_solver_return_fields",d)
        old_current,old_ref=a["source_sensor_metadata"],a["reference_sensor_metadata"]
        new_current,new_ref=b["primary_current_sensor"],b["primary_reference_sensor"]
        common_current=differences({k:old_current[k] for k in SENSOR_KEYS},{k:new_current[k] for k in SENSOR_KEYS},"/shared_current_sensor_identity")
        common_ref=differences({k:old_ref[k] for k in SENSOR_KEYS},{k:new_ref[k] for k in SENSOR_KEYS},"/shared_reference_sensor_identity")
        full_current=differences(old_current,new_current,"/full_current_sensor_metadata")
        full_ref=differences(old_ref,new_ref,"/full_reference_sensor_metadata")
        for section,ds in (("common_current_sensor",common_current),("common_reference_sensor",common_ref),
                           ("full_sensor_metadata_schema_and_values",full_current+full_ref)):
            for d in ds:append(scene,i,section,d)
        init_old,init_new=initialization(a,ap),initialization(b,p)
        init_diff=differences(init_old,init_new,"/initialization")
        for d in init_diff:append(scene,i,"initialization",d)
        post_old,post_new={k:a[k] for k in POST_KEYS},{k:b[k] for k in POST_KEYS}
        post_diff=differences(post_old,post_new,"/post_integration")
        for d in post_diff:append(scene,i,"post_integration",d)
        row={"frame_id":i,"scope":"before_first_fallback" if i<first else "first_fallback" if i==first else "immediate_next_frame",
            "old_reference_frame_id":a["reference_frame_id"],"new_reference_frame_id":b["reference_frame_id"],
            "full_regular_solver_field_count":len(fields),"old_missing_return_fields":missing_old,"new_missing_return_fields":missing_new,
            "new_unexpected_solver_fields_retained":extras_new,"all_nested_solver_return_difference_count":len(solver_diff),
            "different_solver_fields":[k for k in fields if differences(ap.get(k),p.get(k))],
            "first_solver_difference":solver_diff[0] if solver_diff else None,
            "old_regular_return_canonical_JSON_sha256":canonical(ap),"new_complete_primary_canonical_JSON_sha256":canonical(p),
            "canonical_full_return_objects_equal":not solver_diff,"not_original_row_or_file_byte_equality":True,
            "shared_current_sensor_identity_difference_count":len(common_current),"shared_reference_sensor_identity_difference_count":len(common_ref),
            "full_sensor_metadata_schema_difference_count":sum(x["kind"].startswith("schema_") for x in full_current+full_ref),
            "old_sensor_provenance_fields_not_in_new_primary_sensor":sorted(set(old_current)-set(new_current)),
            "initialization_saved_metadata_and_first_iteration":init_old,"initialization_difference_count":len(init_diff),
            "post_integration_difference_count":len(post_diff),"old_final_accept":a["accepted"],"new_primary_accept":p["accepted"],
            "new_final_accept":b["accepted"],"new_selection":b["selection"],
            "old_last_accepted_after":a["last_accepted_frame_id_after"],"new_last_final_accepted_after":b["last_accepted_frame_id_after"]}
        by_pair.append(row)
    common_config=set(old["configuration"])&set(new["configuration"])
    common_config_diff={k:differences(old["configuration"][k],new["configuration"][k]) for k in sorted(common_config)}
    first_diff=next((r for r in by_pair if r["all_nested_solver_return_difference_count"]),None)
    boundary=[]
    for i in (first,first+1):
        a,b=ar[i],br[i]
        boundary.append({"frame_id":i,"old_reference_sensor":a["reference_sensor_metadata"],"new_primary_reference_sensor":b["primary_reference_sensor"],
            "old_current_sensor":a["source_sensor_metadata"],"new_current_sensor":b["primary_current_sensor"],
            "old_primary_candidate":a["candidate_reference_from_current"],"new_primary_candidate":b["primary"]["candidate_reference_from_current"],
            "new_fallback_previous_sensor_frame_id":b["fallback_reference_frame_id"],"new_fallback_original_accepted":b["fallback"]["original_accepted"],
            "new_fallback_relative_candidate":b["fallback"]["candidate_previous_from_current"],
            "old_estimated_c2w":a["estimated_c2w"],"new_estimated_c2w":b["estimated_c2w"],
            "old_primary_flags":a["failure_flags"],"new_primary_flags":b["primary"]["failure_flags"],"new_final_flags":b["failure_flags"],
            "old_final_accept":a["accepted"],"new_final_accept":b["accepted"],"new_selection":b["selection"],
            "old_reference_after":a["last_accepted_frame_id_after"],"new_reference_after":b["last_accepted_frame_id_after"]})
    return {"scene":scene,"old_run":old_name,"hybrid_run":new_name,"all_saved_frame_count_each":len(ar),
        "first_fallback_frame_id":first,"strictly_pre_fallback_nonanchor_pairs":first-1,
        "compared_pair_range_including_boundary_next":[1,first+1],"all_return_fields_derived_from_frozen_source_AST":fields,
        "old_outer_row_keys":sorted(ar[1]),"hybrid_outer_row_keys":sorted(br[1]),"hybrid_primary_keys":sorted(br[1]["primary"]),
        "wrapper_schemas_differ_from_frame1":set(ar[1])!=set(br[1]),
        "all_saved_full_row_or_file_byte_equality_claimed":False,
        "pipeline_configuration_common_field_count":len(common_config),"pipeline_configuration_common_fields_with_differences":[k for k,v in common_config_diff.items() if v],
        "pipeline_configuration_old_only_fields":sorted(set(old["configuration"])-set(new["configuration"])),
        "pipeline_configuration_new_only_fields":sorted(set(new["configuration"])-set(old["configuration"])),
        "first_solver_return_difference_frame":first_diff["frame_id"] if first_diff else None,
        "first_solver_return_difference":first_diff["first_solver_difference"] if first_diff else None,
        "per_pair":by_pair,"boundary_final_reference_and_candidates":boundary,
        "raw_sensor_pyramid_arrays_recomputed":False,"GT_metrics_recomputed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default="affine-hybrid-primary-prefix-comparison-001")
    args=parser.parse_args();require(args.run and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.run),"Safe unique run")
    output=ROOT/"outputs"/args.run;output.mkdir(exist_ok=False)
    with (output/"source_script.py").open("xb") as stream:stream.write(Path(__file__).read_bytes())
    inputs={SOLVER:SOLVER_SHA,"scripts/compare_affine_hybrid_primary_prefix.py":sha(__file__),
        "scripts/track_rgbd_affine_full.py":"879eafcceaa2ed03150755bae3bfc9f0063fded86280e27ad2f273087885086c",
        "scripts/track_rgbd_affine_icp_fallback.py":"fb25204d6c9780378cf2f58eec6bc4a27d6c32297ea2c7d283e22f4129cb6922"}
    for _,old,old_sha,new,new_sha in SCENES:
        inputs[f"outputs/{old}/metrics.json"]=old_sha;inputs[f"outputs/{new}/metrics.json"]=new_sha
    try:
        require(all(sha(ROOT/name)==expected for name,expected in inputs.items()),"All actual source/terminal metric bytes before read")
        controls=fixture_checks();fields=return_fields();counter=Counter();row_count=0
        with (output/"all_corresponding_field_differences.jsonl").open("x",encoding="utf-8") as stream:
            def append(scene,frame,section,diff):
                nonlocal row_count
                stream.write(json.dumps({"scene":scene,"frame_id":frame,"section":section,**diff},ensure_ascii=False,allow_nan=False)+"\n")
                counter[f"{scene}/{section}/{diff['kind']}"]+=1;row_count+=1
            scenes=[compare_scene(scene,old,new,fields,append) for scene,old,_,new,_ in SCENES]
        require(all(sha(ROOT/name)==expected for name,expected in inputs.items()),"Original result/source bytes unchanged after read")
        result={"status":"completed_read_only_actual_primary_prefix_field_comparison","source_sha256":sha(__file__),
            "actual_consumed_file_sha256":inputs,"all_inputs_rehashed_unchanged":True,"comparator_fixtures":controls,
            "scenes":scenes,"all_corresponding_difference_record_count":row_count,"difference_kind_counts":dict(counter),
            "all_difference_records_path":"all_corresponding_field_differences.jsonl","all_difference_records_sha256":sha(output/"all_corresponding_field_differences.jsonl"),
            "whole_metric_JSON_including_GT_derived_evaluation_fields_read":True,
            "GT_scope":"Entire original metric JSONs are parsed, including posthoc trajectory/GT association metadata. No GT values select pairs or candidates, and no trajectory/GT metric is evaluated. First boundary is selected by original hybrid fallback_selected flag.",
            "limits":["All 19 frozen regular solver return fields and every nested iteration/trial are compared with types/shapes/scalars retained; no tolerances or internal filtering.",
                "Old flattened solver return and new nested primary have different wrapper schemas. Canonical selected return-object equality is not original row/file byte equality.",
                "Eight shared RGB/depth sensor identity fields are compared; old-only GT association provenance fields are separately retained as schema differences, not silently erased.",
                "Only each pre-fallback prefix, first fallback and its immediate next frame are compared. No all-future equivalence claim.",
                "Serialized identity/config/initialization and first recorded iteration are observed; sensor pyramids are not regenerated and no registrations, GT metrics or speed comparisons run.",
                "Reference selection changes after fallback, so subsequent primary differences cannot be called a brightness-only causal effect."],
            "registration_calls":0,"optimizer_steps":0,"renderer_calls":0,"GPU_calls":0}
        write(output/"metrics.json",result)
        saved={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.iterdir()) if p.is_file()}
        write(output/"completion.json",{"status":"completed","all_saved_files":saved,"source_sha256":sha(__file__),"registration_calls":0,"GPU_calls":0})
    except BaseException as error:
        write(output/"failure.json",{"status":"failed_read_only_comparison_preserved","exception_type":type(error).__name__,"exception_message":str(error),"traceback":traceback.format_exc(),"registration_calls":0,"GPU_calls":0})
        raise
    print(json.dumps({"status":result["status"],"scene_count":len(scenes),"all_difference_record_count":row_count,"registration_calls":0,"GPU_calls":0}))


if __name__=="__main__":main()
