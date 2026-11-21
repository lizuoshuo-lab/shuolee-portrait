"""Read-only6D per-row proposal-field addendum to the frozen actual full audit.

Prepare locks source/evidence and tests synthetic guards only. Actual audit,
after root review, explicitly reads each accepted iteration's own proposed
six-vector and checks its limited accepted vector. No inherited local from a
preceding iteration, trajectory calculation, registration, GPU or new score.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/rgbd-affine-full-6d-proposal-addendum-cpu-preparation-001"
PROTOCOL = PREP / "protocol.json"
OUTPUT = ROOT / "outputs/rgbd-affine-full-6d-proposal-addendum-001"
BATCH = ROOT / "outputs/rgbd-affine-full-validation-001"
AUDIT_SHA = "a9480a1a6a7134b8811dc8b500045419f1fbb1dbfaf4dbb41caae6f050e4afd5"
AUDITOR_SHA = "155bfe57fbf873d7717eac40d8ac9a18134ee7f7824eed92efab800d5577c690"
SUMMARY_SHA = "d6a5f9e304579e98b0b97912762eb6c7618273c89398bf3c95e09809e76635d3"
PIPELINE_PROTOCOL_SHA = "ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a"
LOCKS = {
    "scripts/audit_rgbd_affine_full_results.py":AUDITOR_SHA,
    "scripts/summarize_rgbd_affine_full_results.py":"13e05091ee1c4f8be87c4f11fa88f7efda96f73e0780039c62d54b0965f72309",
    "scripts/audit_rgbd_photogeometric_results.py":"e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53",
    "scripts/audit_rgbd_affine_real_results.py":"b549a0af2642b2e203e8f3b5f0deae46dcaf2458f608d7e27435c87b33f8c6d5",
    "outputs/rgbd-affine-full-validation-001/metrics.json":"d2f93140c9119467b831a592efb8f10c46ad99e2196dc78ac508ee1ef36a987c",
    "outputs/rgbd-affine-full-validation-001/independent_full_affine_result_audit.json":AUDIT_SHA,
    "outputs/rgbd-affine-full-comparison-001/summary.json":SUMMARY_SHA,
    "outputs/rgbd-affine-full-cpu-preparation-001/protocol.json":PIPELINE_PROTOCOL_SHA,
    "outputs/rgbd-affine-full-cpu-preparation-001/root_protocol_review.json":"977fae21421e9c71d51c29756b8999d4b075b5b8ed53a9c16d46176c88ecd44f",
    "outputs/rgbd-affine-full-cpu-preparation-001/root-independent-full-preflight-review.json":"d815c85825decc9d7330594033416fbfdf1dfc09d472c081176a4329265dfdeb",
    "outputs/tum-desk-lastaccepted-6d-full549-001/metrics.json":"317cc482a07d462c89283cd50531552ac34edc326443d6e1074786512fa0cd71",
    "outputs/tum-xyz-lastaccepted-6d-full748-001/metrics.json":"d34b698dbac30400856f88c9f49f6d7c81ec4e632e463f1758fa1b43ca301fb6",
}
RUNS = ["tum-desk-lastaccepted-6d-full549-001","tum-xyz-lastaccepted-6d-full748-001"]
MAX_BYTES = 1024 * 1024
ABS_TOL = 3e-12


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(MAX_BYTES),b""):digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_new(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode("utf-8")
    require(len(raw)<=MAX_BYTES,"Addendum exceeds one MiB; never drop failures or frame denominators")
    with Path(path).open("xb") as stream:stream.write(raw)


def hash_guard(records):
    for name,digest in records.items():require(sha(ROOT/name)==digest,"Frozen binding changed: "+name)


def six_vector(value,name):
    require(isinstance(value,list) and len(value)==6,"Exactly six components for "+name)
    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in value),
            "Finite physical six-vector for "+name)
    return value


def accepted_row_guard(row):
    require(row["accepted"] is True,"This guard applies to accepted ITERATION rows only")
    # These are explicit reads of THIS row. No state from a preceding call or
    # iteration is available or substituted for the required exported field.
    proposed=six_vector(row["proposed_delta_twist_m_rad"],"current-row proposed twist")
    accepted=six_vector(row["accepted_delta_twist_m_rad"],"current-row accepted twist")
    factor=row["step_limit_factor"]; scale=row["line_search_scale"]
    require(isinstance(factor,(int,float)) and not isinstance(factor,bool) and math.isfinite(factor)
            and 0<factor<=1,"Finite physical step-limit factor")
    require(isinstance(scale,(int,float)) and not isinstance(scale,bool) and math.isfinite(scale)
            and 0<scale<=1,"Finite accepted line-search scale")
    limited=[component*factor*scale for component in proposed]
    translation=max(abs(a-b) for a,b in zip(limited[:3],accepted[:3]))
    rotation=max(abs(a-b) for a,b in zip(limited[3:],accepted[3:]))
    require(max(translation,rotation)<ABS_TOL,"Accepted delta equals THIS row proposed*factor*scale")
    return {"translation_component_max_abs_difference_m":translation,
            "rotation_component_max_abs_difference_rad":rotation}


def lineage_metadata():
    hash_guard(LOCKS)
    batch=read(BATCH/"metrics.json");audit=read(BATCH/"independent_full_affine_result_audit.json")
    summary=read(ROOT/"outputs/rgbd-affine-full-comparison-001/summary.json")
    require(batch["status"]=="all_four_predeclared_CPU_full_sequence_rounds_completed"
            and batch["runs_started"]==4 and batch["current_run"] is None and len(batch["run_records"])==4
            and all(row["returncode"]==0 for row in batch["run_records"]),"Four original terminal exit0 rounds")
    require(audit["status"]=="passed_independent_full_input_reference_step_and_trajectory_audit"
            and audit["audit_source_sha256"]==AUDITOR_SHA and audit["protocol_sha256"]==PIPELINE_PROTOCOL_SHA
            and audit["complete_frame_rows"]==2594 and audit["tracked_nonanchor_pairs"]==2590,
            "Frozen original actual audit scope/source")
    require(summary["status"]=="completed_after_independent_full_result_audit"
            and summary["audit_sha256"]==AUDIT_SHA and summary["audit_source_sha256"]==AUDITOR_SHA
            and summary["complete_new_frame_rows"]==2594 and summary["tracked_new_pairs"]==2590,
            "Original public numerical summary binding")
    records=[row for row in batch["run_records"] if row["model_dimension"]==6]
    require([row["run"] for row in records]==RUNS,"Both original6D runs in declared order")
    expected=[]
    for record in records:
        old=next(row for row in audit["runs"] if row["run"]==record["run"])
        expected.append({"run":record["run"],"frames":record["frames_processed"],
            "metrics_path":relative(ROOT/"outputs"/record["run"]/"metrics.json"),"metrics_sha256":record["metrics_sha256"],
            "old_audited_steps":old["all_pair_step_checks"]["steps"],
            "old_audited_accepted_updates":old["all_pair_step_checks"]["accepted_updates"],
            "old_failure_frame_count":old["failure_frame_count"]})
    return expected


def cpu_controls():
    row={"accepted":True,"proposed_delta_twist_m_rad":[.03,-.02,.01,.04,.02,-.03],
         "step_limit_factor":.8,"line_search_scale":.5,
         "accepted_delta_twist_m_rad":[.03*.8*.5,-.02*.8*.5,.01*.8*.5,.04*.8*.5,.02*.8*.5,-.03*.8*.5]}
    positive=accepted_row_guard(row)
    mutations={"missing_current_proposed":lambda r:r.pop("proposed_delta_twist_m_rad"),
        "malformed8D_proposed":lambda r:r["proposed_delta_twist_m_rad"].extend([0.,0.]),
        "malformed8D_accepted":lambda r:r["accepted_delta_twist_m_rad"].extend([0.,0.]),
        "NaN_proposed":lambda r:r["proposed_delta_twist_m_rad"].__setitem__(0,float('nan')),
        "NaN_accepted":lambda r:r["accepted_delta_twist_m_rad"].__setitem__(0,float('nan')),
        "wrong_limited_delta":lambda r:r["accepted_delta_twist_m_rad"].__setitem__(0,.9),
        "missing_current_accepted":lambda r:r.pop("accepted_delta_twist_m_rad"),
        "nonfinite_factor":lambda r:r.update(step_limit_factor=float('inf'))}
    results={}
    for label,mutate in mutations.items():
        # Call the positive row immediately before EACH malformed row. A
        # missing current field cannot inherit even after an accepted call.
        accepted_row_guard(row);value=copy.deepcopy(row);mutate(value)
        try:accepted_row_guard(value)
        except (ValueError,KeyError) as error:results[label]={"rejected":True,"exception":type(error).__name__,"message":str(error)}
        else:raise AssertionError("Synthetic guard did not reject "+label)
    require(not any(name in sys.modules for name in ("numpy","scipy","torch","gsplat","tum_rgbd")),
            "Field addendum imported a numerical/optimizer/loader/GPU stack")
    return {"status":"passed_synthetic_current_row_guards_only","positive":positive,"negative_controls":results,
            "actual_metrics_iteration_rows_checked":0,"actual_registration_calls":0,"trajectory_recalculations":0,"GPU_calls":0}


def prepare():
    require(not PREP.exists(),"Never overwrite addendum CPU preparation")
    PREP.mkdir(exist_ok=False)
    try:
        expected=lineage_metadata();controls=cpu_controls()
        evidence=ROOT/"outputs/rgbd-affine-full-auditor-independent-source-review-002/additional-missing-current-proposal-control.json"
        records={**LOCKS,relative(evidence):sha(evidence)}
        write_new(PREP/"cpu_preflight.json",controls)
        with (PREP/"source_script.py").open("xb") as stream:stream.write(Path(__file__).read_bytes())
        protocol={"protocol_id":"rgbd-affine-full-6d-proposal-field-addendum-001",
            "status":"CPU_prepared_awaiting_root_source_review_before_actual_addendum",
            "prepared_utc":stamp(),"source_sha256":sha(Path(__file__)),"input_sha256":records,"runs":expected,
            "root_review_relative_path":relative(PREP/"root_source_review.json"),
            "cpu_preflight":{"path":relative(PREP/"cpu_preflight.json"),"sha256":sha(PREP/"cpu_preflight.json")},
            "scope":"Accepted iteration rows in both original6D runs, including steps inside finally rejected pairs",
            "explicit_current_row_reads":["row['proposed_delta_twist_m_rad']","row['accepted_delta_twist_m_rad']"],
            "finite_vector_shape":[6],"formula":"accepted_delta = this_row_proposed * this_row_step_limit_factor * this_row_line_search_scale",
            "absolute_scalar_component_tolerance":ABS_TOL,"units_components":["m","m","m","rad","rad","rad"],
            "original_frozen_auditor_source_sha256":AUDITOR_SHA,"original_actual_audit_sha256":AUDIT_SHA,"original_summary_sha256":SUMMARY_SHA,
            "actual_output":relative(OUTPUT/"metrics.json"),"new_trajectory_metrics":0,"registration_calls":0,"GPU_calls":0,
            "limits":["A supplemental exported-field guard, not a replacement or rerun of the frozen actual full audit.",
                "No trajectory, GT camera, correspondence, rank, cost or optimization is recomputed.",
                "All source/actual result/old audit/summary bytes stay unchanged; the new addendum is bound separately.",
                "CPU preparation hashes actual files but does not read or check actual iteration rows."]}
        write_new(PROTOCOL,protocol)
        print(json.dumps({"status":"CPU_prepared_only","source_sha256":protocol["source_sha256"],"protocol_sha256":sha(PROTOCOL),
            "negative_controls":len(controls["negative_controls"]),"actual_iterations_checked":0,"GPU_calls":0},indent=2))
    except BaseException as error:
        write_new(PREP/"failure.json",{"phase":"CPU_prepare","exception_type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(),"GPU_calls":0})
        raise


def audit(protocol_sha):
    require(not OUTPUT.exists(),"Never overwrite actual addendum output")
    OUTPUT.mkdir(exist_ok=False)
    progress={"phase":"actual_current_row_field_addendum","started_utc":stamp(),"GPU_calls":0}
    try:
        require(sha(PROTOCOL)==protocol_sha,"Exact root-reviewed protocol SHA required")
        protocol=read(PROTOCOL);source_before=sha(Path(__file__))
        require(source_before==protocol["source_sha256"],"Addendum source changed")
        review_path=ROOT/protocol["root_review_relative_path"];review=read(review_path)
        require(review["status"]=="passed" and review["protocol_sha256"]==protocol_sha
                and review["source_sha256"]==source_before,"Exact actual root addendum source/protocol review required")
        hash_guard(protocol["input_sha256"])
        hash_guard({protocol["cpu_preflight"]["path"]:protocol["cpu_preflight"]["sha256"]})
        review_before=sha(review_path)
        require(lineage_metadata()==protocol["runs"],"Original actual audit/6D results/summary metadata changed")
        rows=[];checked=0;all_steps=0;progress["completed_runs"]=[]
        for binding in protocol["runs"]:
            data=read(ROOT/binding["metrics_path"]);frames=data["frame_metrics"]
            require(data["frames_processed"]==len(frames)==binding["frames"]
                    and [f["frame_id"] for f in frames]==list(range(binding["frames"])),"Complete original6D frames")
            accepted_total=0;steps=0;accepted_in_rejected_pairs=0;max_t=0.;max_r=0.;per_frame=[]
            for frame in frames[1:]:
                frame_accepted=0
                for iteration in frame["iteration_history"]:
                    steps+=1
                    if iteration["accepted"] is True:
                        progress.update(current_run=binding["run"],current_frame_id=frame["frame_id"],
                                        current_level=iteration["level"],current_iteration=iteration["iteration"])
                        result=accepted_row_guard(iteration)
                        frame_accepted+=1;accepted_total+=1
                        accepted_in_rejected_pairs+=int(frame["accepted"] is False)
                        max_t=max(max_t,result["translation_component_max_abs_difference_m"])
                        max_r=max(max_r,result["rotation_component_max_abs_difference_rad"])
                per_frame.append({"frame_id":frame["frame_id"],"pair_accepted":frame["accepted"],
                    "retained_pair_failure_flags":frame["failure_flags"],"iteration_rows":len(frame["iteration_history"]),
                    "accepted_iteration_rows_checked":frame_accepted})
            require(steps==binding["old_audited_steps"] and accepted_total==binding["old_audited_accepted_updates"],
                    "Every old actual step and accepted update remains in the addendum denominator")
            require(sum(not f["pair_accepted"] for f in per_frame)==binding["old_failure_frame_count"],"Finally rejected pair rows retained")
            row={**binding,"all_iteration_rows_seen":steps,"accepted_iteration_rows_checked":accepted_total,
                 "accepted_iteration_rows_in_finally_rejected_pairs_checked":accepted_in_rejected_pairs,
                 "all_current_proposed_and_accepted_fields_present_finite6":True,
                 "maximum_translation_delta_component_difference_m":max_t,"maximum_rotation_delta_component_difference_rad":max_r,
                 "all_nonanchor_frame_counts_including_rejected_pairs":per_frame}
            rows.append(row);progress["completed_runs"].append(binding["run"]);checked+=accepted_total;all_steps+=steps
            del data,frames
        hash_guard(protocol["input_sha256"])
        require(sha(Path(__file__))==source_before and sha(review_path)==review_before,"Source/root review changed during actual addendum")
        result={"status":"passed_current_row_6d_proposal_field_addendum","source_sha256":source_before,"protocol_sha256":protocol_sha,
            "root_review_sha256":review_before,"original_frozen_auditor_source_sha256":AUDITOR_SHA,"original_actual_audit_sha256":AUDIT_SHA,
            "original_summary_sha256":SUMMARY_SHA,"actual_6D_run_count":2,"complete_frame_rows":sum(r["frames"] for r in rows),
            "all_nonanchor_pairs":sum(r["frames"]-1 for r in rows),"all_iteration_rows_seen":all_steps,"accepted_iteration_rows_checked":checked,
            "runs":rows,"all_input_bytes_before_after_unchanged":True,"input_file_count":len(protocol["input_sha256"]),
            "original_audit_and_summary_unchanged":True,"new_trajectory_metrics":0,"registration_calls":0,"GPU_calls":0,
            "limits":protocol["limits"],"ended_utc":stamp()}
        write_new(OUTPUT/"metrics.json",result)
        print(json.dumps({"status":result["status"],"accepted_iteration_rows_checked":checked,"all_steps":all_steps,
                          "artifact_sha256":sha(OUTPUT/"metrics.json"),"GPU_calls":0},indent=2))
    except BaseException as error:
        write_new(OUTPUT/"failure.json",{**progress,"exception_type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true");group.add_argument("--audit",action="store_true");parser.add_argument("--protocol-sha")
    args=parser.parse_args()
    if args.prepare:
        if args.protocol_sha:parser.error("Protocol SHA only applies to actual field addendum")
        prepare()
    else:
        if not args.protocol_sha:parser.error("Exact reviewed protocol SHA required")
        audit(args.protocol_sha)


if __name__ == "__main__":main()
