"""Read-only final-map diagnosis at five fixed seen views per full sequence.

CPU preparation locks the three original completed maps and the explicitly
disclosed fourth replacement. A reviewed SHA permits one job of 20 renders.
No map/pose optimization, fitted alignment, frame selection, or retries.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import run_v4_full_recovery as recovery

ROOT=recovery.ROOT
PROTOCOL=ROOT/"manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json"
OUTPUT=ROOT/"outputs/tum-online-v4-full-seen-map-evaluation-004"
HELPER=ROOT/"scripts/evaluate_tum_v4_final_maps.py"
HELPER_SHA="0e6d8d976c924bf9ce492d2a405735d7cb0f4288b7da68a6518bf6589f970896"
RECOVERY_PROTOCOL_SHA="3e1a127dfbffd1885862d2f1a5eb633a36228788041f5aa3d5c8de3b5c90ea3c"
WIDTH,HEIGHT=160,120
VIEWS={"desk":[0,137,274,411,548],"xyz":[0,187,374,561,747]}
SOURCE_LOCKS={
    "scripts/evaluate_tum_v4_final_maps.py":HELPER_SHA,
    "scripts/run_v4_full_recovery.py":"d9171ef2391f0af2db98bbe6da2029c97fc6b633fbca01feeb8629090bd2ece4",
    "scripts/summarize_tum_v4_full_recovery.py":"effee2a8e16758b2330f7f2633cbda3694d5c4f80e2f15e5e273485abd91680b",
    "scripts/check_tum_v4_full_recovery_artifacts.py":"5360bcd229205ad2f8a6a47967f824f3ad21cca804e0a91355b1702bfce63221",
    "scripts/check_tum_v4_full_artifacts.py":"68f2e360558f81f4cf3dc040be99c94d0390fea590bb4ebc2d446712834405d3",
    "scripts/summarize_tum_online_v4_full.py":"4203a4aadb6547157e27ba160a9ae36e4c028f68d55670834411259c44cb4eac",
    "scripts/summarize_v4_full_sequence_recovery_controls.py":"911d8198c67c29bbf9cf8d06786560305c7478602209cff2fca2361cfc9cc76a",
    "scripts/audit_v4_full_recovery_summary.py":"f5d53a0ce23a82ed860bf66993cd0b80e6bebb283fa5384a8dce7989e70cb9bc",
}
COMPLETED_SUMMARY_EVIDENCE={
    "outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json":"97b65b413abd5b18eb8677a20ff7d1eef422205289d929d373f4ea5edd10c81d",
    "outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json":"70f65dfced84eabf1ec313e93a908062d565a19401b1ae4c49b0d4d71801417e",
    "outputs/tum-online-v4-full-sequence-recovery-summary-001/completion.json":"0ab455b0eb5fcbecf9e050aea9815bd093aacf1d14014a3c61b3808635f5dc58",
}
LIMITATIONS=[
    "Five fixed seen-training views per map, not heldout reconstruction or an untouched benchmark.",
    "GT cameras use inverse(GT frame0 c2w) @ GT view c2w; no SE3/Sim3 fit and no map transformation.",
    "The fourth original run failed during progress-file replacement after partial521 observations; a separate frame0 restart002 is explicitly selected, not hidden or chosen by metrics.",
    "CUDA warmup maps and rendering may be numerically nondeterministic. Whole-pipeline weight pairs do not share byte-identical maps and are observational controls.",
    "GT association is nearest at RGB time, not interpolated; RGB-depth offsets, distortion, and registration assumptions remain.",
    "RGB+ED uses conditional expected depth from the original V4 renderer. Own/common coverage and holes-as-zero have distinct denominators.",
    "Every requested view and training flag is retained; failed renders are not replaced by zero or retried.",
]

def require(value,message):
    if not value: raise ValueError(message)

def helper():
    require(recovery.sha(HELPER)==HELPER_SHA,"Frozen metric/render helper changed")
    spec=importlib.util.spec_from_file_location("frozen_full_seen_metric_render_helper",HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def write_small(path,value):
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
    require(len(data)<=1024*1024,"JSON exceeds one-MiB ceiling; do not drop flags")
    with Path(path).open("xb") as stream:stream.write(data)

def lineage():
    recovery.check_hashes(SOURCE_LOCKS)
    recovery.check_hashes(COMPLETED_SUMMARY_EVIDENCE)
    summary=recovery.read(ROOT/"outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json")
    audit=recovery.read(ROOT/"outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json")
    require(summary["status"]=="completed_read_only_original_three_plus_explicit_recovery" and
        summary["collector_sha256"]==SOURCE_LOCKS["scripts/summarize_v4_full_sequence_recovery_controls.py"] and
        summary["completed_arm_count"]==4 and summary["failed_attempt_count"]==1 and
        summary["selection_rule"]=="original_first_three_completed_plus_explicit_fourth_replacement_no_metric_selection",
        "Completed independent trajectory summary/selection lineage mismatch")
    require(audit["status"]=="passed_independent_lineage_full_trajectory_chunk_and_tensor_audit" and
        audit["collector_sha256"]==summary["collector_sha256"] and
        audit["auditor_sha256"]==SOURCE_LOCKS["scripts/audit_v4_full_recovery_summary.py"] and
        audit["summary_sha256"]==COMPLETED_SUMMARY_EVIDENCE["outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json"],
        "Independent completed trajectory summary audit/source mismatch")
    recovered=recovery.reviewed_protocol(RECOVERY_PROTOCOL_SHA)
    original=recovery.read(recovery.ORIGINAL_PROTOCOL)
    terminal=recovery.read(recovery.OUTPUT/"metrics.json")
    require(terminal["status"]=="recovery_run_completed" and terminal["protocol_sha256"]==RECOVERY_PROTOCOL_SHA and
        terminal["GPU_runs_started"]==1 and len(terminal["run_records"])==1,"Recovery is not terminal-completed")
    record=terminal["run_records"][0]
    require(record["run"]==recovery.REPLACEMENT and record["returncode"]==0 and record["frames"]==748 and
        record["metrics_sha256"]==recovery.sha(ROOT/"outputs"/recovery.REPLACEMENT/"metrics.json"),"Recovery terminal metric mismatch")
    review=ROOT/recovered["root_review_relative_path"]
    require(recovery.read(review)["status"]=="passed","Explicit recovery root review missing")
    rows=original["new_runs"][:3]+recovered["new_runs"]
    require([row["run"] for row in rows]==recovered["selected_final_runs"],"Wrong selection; no metric-based replacement")
    source={**original["source_sha256"],**SOURCE_LOCKS}
    recovery.check_hashes(source)
    evidence={recovery.relative(recovery.ORIGINAL_PROTOCOL):recovery.ORIGINAL_SHA,
        recovery.relative(recovery.PROTOCOL):RECOVERY_PROTOCOL_SHA,
        recovery.relative(recovery.OUTPUT/"metrics.json"):recovery.sha(recovery.OUTPUT/"metrics.json"),
        recovery.relative(review):recovery.sha(review),
        recovery.relative(recovery.OLD_BATCH/"failure.json"):recovery.sha(recovery.OLD_BATCH/"failure.json"),
        recovered["original_failure_snapshot"]["manifest"]:recovered["original_failure_snapshot"]["manifest_sha256"],
        **COMPLETED_SUMMARY_EVIDENCE}
    return {"selected_run_order":[row["run"] for row in rows],"original_protocol_sha256":recovery.ORIGINAL_SHA,
        "recovery_protocol_sha256":RECOVERY_PROTOCOL_SHA,"original_failed_run":recovery.FAILED,
        "explicit_replacement_binding":recovered["replacement_binding"],"evidence_sha256":evidence,
        "original_failure_snapshot_file_count":len(recovered["original_failure_snapshot"]["files"]),
        "source_sha256":source,"completed_full_trajectory_summary_and_independent_audit_locked":True,
        "all_selected_runs_are_predeclared_not_metric_selected":True},rows

def rigid_inverse(matrix,module):
    value=np.asarray(matrix,dtype=np.float64)
    module.rigid_check(value,1e-10)
    result=np.eye(4);result[:3,:3]=value[:3,:3].T;result[:3,3]=-value[:3,:3].T@value[:3,3]
    return result

def prepare_dataset(scene,module):
    from tum_rgbd import associate_frames,load_frames,safe_data_path,scaled_intrinsics
    dataset=ROOT/f"data/tum/rgbd_dataset_freiburg1_{scene}"
    frames,association=associate_frames(dataset,.02,.02,32.)
    count=549 if scene=="desk" else 748
    require(len(frames)==count and [frame.frame_id for frame in frames]==list(range(count)),"Full association count/chronology changed")
    selected=[frames[index] for index in VIEWS[scene]]
    images,depths,masks=load_frames(dataset,selected,WIDTH,HEIGHT,.2,5.)
    K=scaled_intrinsics(WIDTH,HEIGHT)
    anchor=frames[0].c2w
    absolute=np.stack([frame.c2w for frame in selected])
    relative=rigid_inverse(anchor,module)[None]@absolute
    # Exact affine bottom, then rigid inverse. No estimated trajectory or fit.
    relative[:,3,:]=[0,0,0,1]
    w2c=np.stack([rigid_inverse(value,module) for value in relative])
    require(np.allclose(anchor[None]@relative,absolute,rtol=0,atol=1e-12),"Wrong GT gauge/transform direction")
    require(np.allclose(relative[0],np.eye(4),rtol=0,atol=1e-12),"First GT-relative camera must be identity")
    checks={"GT_anchor":module.rigid_check(anchor,1e-10),"GT_relative_c2w":module.rigid_check(relative,1e-10),
        "GT_relative_w2c":module.rigid_check(w2c,1e-10),"render_w2c_float32":module.rigid_check(w2c.astype(np.float32))}
    inputs={"rgb":images,"depth":depths,"valid_depth_mask":masks,"K_render_float32":K.astype(np.float32)[None],
        "GT_anchor_c2w":anchor,"GT_absolute_c2w":absolute,"GT_relative_c2w":relative,
        "GT_relative_w2c":w2c,"render_w2c_float32":w2c.astype(np.float32)}
    file_records={}
    for frame in selected:
        for name in ("rgb_path","depth_path"):
            path=safe_data_path(dataset,getattr(frame,name));file_records[recovery.relative(path)]=recovery.sha(path)
    for name in ("rgb.txt","depth.txt","groundtruth.txt"):
        file_records[recovery.relative(dataset/name)]=recovery.sha(dataset/name)
    sensors=[module.sensor_metadata(frame) for frame in frames]
    record={"scene":scene,"dataset":recovery.relative(dataset),"association":association,"all_training_frame_count":count,
        "evaluation_role":"seen_training_view_diagnostic","evaluation_frame_ids":VIEWS[scene],"seen_training_overlap_count":5,
        "evaluation_sensor_metadata":[module.sensor_metadata(frame) for frame in selected],"input_file_sha256":file_records,
        "input_tensors":{name:module.array_record(value) for name,value in inputs.items()},"K":K.tolist(),
        "anchor_GT_c2w":anchor.tolist(),"relative_GT_c2w":relative.tolist(),"relative_GT_w2c":w2c.tolist(),
        "camera_checks":checks,"valid_depth_pixels_per_view":[int(mask.sum()) for mask in masks]}
    require(all(value>0 for value in record["valid_depth_pixels_per_view"]),"Empty sensor mask")
    return record,inputs,sensors

def inspect_map(row,dataset,sensors,module,*,write_history):
    folder=ROOT/row["output_relative_path"]
    metric=recovery.read(folder/"metrics.json");audit=recovery.read(folder/"cpu_full_v4_artifact_checks.json")
    checkpoint_path=folder/"last_checkpoint.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    config=row["configuration"];count=config["frames"]
    require(metric["status"] in module.COMPLETE and checkpoint["status"] in module.COMPLETE and audit["status"]=="passed" and
        not (folder/"failure.json").exists(),"Selected map is not complete and CPU-checked")
    require(metric["configuration"]==checkpoint["configuration"]==config and len(config)==36,"Full map configuration changed")
    require(metric["frames_processed"]==count and metric["selected_frame_ids"]==checkpoint["frame_ids_with_stored_pose"]==list(range(count)) and
        checkpoint["current_local_index"]==count-1 and len(checkpoint["completed_frame_metrics"])==count,"Missing full final cameras/frames")
    require(metric["selected_sensor_metadata"]==sensors and metric["association"]==dataset["association"],"Full sensor/source metadata changed")
    require(metric["script_sha256"]==checkpoint["script_sha256"]==module.SOURCE_HASHES["train_tum_online_gs_v4.py"] and
        metric["implementation_version"]==checkpoint["implementation_version"]==4,"Map trainer version/hash changed")
    require(audit["full_checker_sha256"]==SOURCE_LOCKS["scripts/check_tum_v4_full_artifacts.py"] and
        audit["protocol_sha256"]==(RECOVERY_PROTOCOL_SHA if row["run"]==recovery.REPLACEMENT else recovery.ORIGINAL_SHA),"Wrong artifact audit protocol/source")
    if row["run"]==recovery.REPLACEMENT:
        require(audit["recovery_checker_wrapper_sha256"]==SOURCE_LOCKS["scripts/check_tum_v4_full_recovery_artifacts.py"] and
            audit["recovery_summary_wrapper_sha256"]==SOURCE_LOCKS["scripts/summarize_tum_v4_full_recovery.py"],"Recovery wrapper provenance changed")
    require(checkpoint["ground_truth_included"] is False and not any("evaluation_only_gt" in key for key in checkpoint),"Render checkpoint is not causal/GT-free")
    parameters=checkpoint["parameters"];n=len(parameters["means"])
    require(set(parameters)==set(module.PARAMETER_SHAPES) and n==metric["final_point_count"] and 0<n<=config["max_points"],"Wrong final map count/fields")
    for key,tail in module.PARAMETER_SHAPES.items():
        tensor=parameters[key]
        require(tensor.dtype==torch.float32 and tensor.device.type=="cpu" and not tensor.requires_grad and
            tuple(tensor.shape)==(n,*tail) and torch.isfinite(tensor).all().item(),"Bad frozen parameter: "+key)
    require((torch.linalg.vector_norm(parameters["quats"],dim=-1)>1e-12).all().item(),"Zero quaternion")
    require(torch.isfinite(parameters["log_scales"].exp()).all().item() and (parameters["log_scales"].exp()>0).all().item(),"Invalid activated scales")
    K=np.asarray(dataset["K"],dtype=np.float32)[None]
    require(np.array_equal(checkpoint["K"].numpy(),K),"Actual trained K differs from rendering K")
    require(np.array_equal(checkpoint["estimated_c2w"][0].numpy(),np.eye(4)),"Map world gauge is not first identity")
    for key in ("estimated_c2w","estimated_w2c"):
        require(checkpoint[key].shape==(count,4,4),"Incomplete stored camera count")
        module.rigid_check(checkpoint[key].numpy())
    ids=np.asarray(checkpoint["source_frame_ids_per_gaussian"],dtype=np.int64)
    require(ids.shape==(n,) and set(ids).issubset(range(count)),"Invalid map source IDs")
    histogram={str(key):int(value) for key,value in sorted(Counter(ids.tolist()).items())}
    history=[];cumulative=metric["initial_point_count"]
    expected_histogram={"0":cumulative}
    for index,frame in enumerate(metric["frame_metrics"]):
        require(frame["frame_id"]==frame["local_index"]==index,"Training history skipped a frame")
        added=int(frame["new_points_added"])
        if index: cumulative+=added
        else: require(added==0,"Initial frame should not be a later addition")
        require(frame["point_count"]==cumulative<=config["max_points"],"Additions/count budget inconsistent")
        if index and added:expected_histogram[str(index)]=added
        history.append({"frame_id":index,"rgb_timestamp":frame["rgb_timestamp"],"new_points_added":added,
            "point_count":frame["point_count"],"failure_flags":frame["failure_flags"],
            "map_coverage_warning_flags":frame.get("map_coverage_warning_flags"),"pose_rejected":frame.get("pose_rejected"),
            "optional_tracking_field_presence":{"map_coverage_warning_flags":"map_coverage_warning_flags" in frame,
                "pose_rejected":"pose_rejected" in frame}})
    require(cumulative==n and expected_histogram==histogram,"Source history and final Gaussians differ")
    caps=[frame for frame in history if frame["point_count"]==config["max_points"]]
    history_value={"run":row["run"],"training_frame_count":count,"point_cap":config["max_points"],"initial_point_count":metric["initial_point_count"],
        "final_point_count":n,"cap_reached":bool(caps),"first_cap_frame":caps[0] if caps else None,
        "source_frame_histogram":histogram,"all_frames_including_all_flags":history,"mapping_event_count":len(metric["mapping_events"]),
        "training_failure_frame_count":metric["failure_frame_count"],"training_icp_failure_count":metric["icp_failed_frame_count"]}
    history_path=OUTPUT/"map_histories"/(row["run"]+".json")
    if write_history:write_small(history_path,history_value)
    else:require(recovery.read(history_path)==history_value,"Saved budget/flag history changed")
    artifacts={recovery.relative(folder/name):recovery.sha(folder/name) for name in
        ("metrics.json","last_checkpoint.pt","online_map.pt","configuration.json","summary.json","cpu_full_v4_artifact_checks.json")}
    record={"run":row["run"],"scene":row["scene"],"prior_weight":config["pose_prior_weight"],"configuration":config,
        "checkpoint":recovery.relative(checkpoint_path),"checkpoint_sha256":recovery.sha(checkpoint_path),
        "checkpoint_role":"completed causal final map; final online_map consistency was checked separately by the artifact auditor",
        "ground_truth_in_render_checkpoint":False,"artifact_sha256":artifacts,"point_count":n,"point_cap":config["max_points"],
        "cap_reached":bool(caps),"first_cap_frame":caps[0]["frame_id"] if caps else None,
        "parameter_tensors":{key:module.array_record(value) for key,value in parameters.items()},"K_tensor":module.array_record(checkpoint["K"]),
        "source_frame_ids_tensor":module.array_record(ids),"source_frame_histogram":histogram,
        "training_failure_frame_count":metric["failure_frame_count"],"all_training_flags_retained":True,
        "history":{"path":recovery.relative(history_path),"sha256":recovery.sha(history_path),"bytes":history_path.stat().st_size}}
    return record,{"parameters":parameters,"K":checkpoint["K"]}

def cpu_controls(module,render_node):
    binding=module.renderer_binding_cpu_control(render_node)
    target=np.full((2,2),2.,dtype=np.float32);valid=np.array([[True,True],[True,False]])
    rgb=np.zeros((2,2,3),dtype=np.float32)
    alpha=np.array([[.8,.2],[.8,.8]],dtype=np.float32)
    value=module.view_metrics(rgb,target,alpha,rgb,target,valid)
    require(value["covered_valid_pixels"]==2 and value["covered_depth"]["rmse_m"]==0 and
        np.isclose(value["all_valid_holes_as_zero_depth"]["rmse_m"],math.sqrt(4/3)) and
        value["rgb"]["psnr_db"] is None and value["rgb"]["psnr_positive_infinity"],"Coverage/holes/zero-MSE semantics changed")
    empty=module.view_metrics(rgb,target,np.zeros((2,2)),rgb,target,valid)
    require(empty["covered_depth"]["pixels"]==0 and empty["covered_depth"]["rmse_m"] is None and
        empty["all_valid_holes_as_zero_depth"]["rmse_m"]==2,"Empty coverage must retain denominator/null")
    other=np.array([[.1,.9],[.9,.9]],dtype=np.float32)
    pair=module.paired_metrics(target,alpha,target,other,target,valid)
    require(pair["intersection_pixels"]==1 and pair["intersection_sensor_valid_coverage"]==1/3,"Common-C denominator changed")
    a=np.eye(4);a[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];a[:3,3]=[1,2,3]
    b=np.eye(4);b[:3,3]=[4,1,2]
    rel=rigid_inverse(a,module)@b
    require(np.allclose(a@rel,b,atol=1e-14,rtol=0) and not np.allclose(a@(b@rigid_inverse(a,module)),b),"GT gauge direction negative control ineffective")
    import_calls=[]
    def fake_import():import_calls.append(True);return "CPU-backend-import-control"
    for offset in (-1,0,1):
        try:
            load_renderer_backend(clock=lambda:recovery.DEADLINE+dt.timedelta(microseconds=offset),loader=fake_import)
            require(offset==-1,"Backend import allowed at/after deadline")
        except ValueError:
            require(offset>=0,"Backend import blocked before deadline")
    require(len(import_calls)==1,"Deadline backend-import negative control failed")
    require(not torch.cuda.is_initialized(),"CPU controls initialized CUDA")
    return {"status":"passed","renderer_binding":binding,"hand_computed_coverage_holes_common_empty_and_RGB_infinity":True,
        "GT_gauge_composition_and_wrong_order_negative_control":True,"backend_import_blocked_at_and_after_deadline":True,"GPU_calls":0}

def prepare():
    require(not OUTPUT.exists() and not PROTOCOL.exists(),"Never overwrite prepared map-evaluation evidence")
    OUTPUT.mkdir(exist_ok=False);(OUTPUT/"map_histories").mkdir(exist_ok=False)
    try:
        module=helper();sources,render_node=module.source_checks();chain,rows=lineage()
        datasets,inputs,sensors={},{},{}
        for scene in VIEWS:datasets[scene],inputs[scene],sensors[scene]=prepare_dataset(scene,module)
        maps=[]
        for row in rows:maps.append(inspect_map(row,datasets[row["scene"]],sensors[row["scene"]],module,write_history=True)[0])
        controls=cpu_controls(module,render_node)
        write_small(OUTPUT/"cpu_preflight.json",{"status":"passed","at":recovery.stamp(),"maps":4,"requested_renders":20,
            "full_frame_counts":[549,549,748,748],"all_four_final_maps_and_CPU_audits_complete":True,
            "views":VIEWS,"controls":controls,"all_source_and_input_guards_passed":True,"GPU_calls":0})
        historical_cpu={}
        for name in ("001/failure.json","001/source_script.py","002/cpu_preflight.json","002/source_script.py","003/cpu_preflight.json","003/source_script.py"):
            path=ROOT/("outputs/tum-online-v4-full-seen-map-evaluation-"+name)
            historical_cpu[recovery.relative(path)]=recovery.sha(path)
        for revision in ("002","003"):
            path=ROOT/f"manifests/tum-online-v4-full-seen-map-evaluation-protocol-{revision}.json"
            historical_cpu[recovery.relative(path)]=recovery.sha(path)
        protocol={"protocol_id":"tum-online-v4-full-seen-map-evaluation-protocol-004","status":"prepared_awaiting_root_review_before_GPU",
            "prepared_at":recovery.stamp(),"deadline_beijing":recovery.DEADLINE.isoformat(),"evaluation_script_sha256":recovery.sha(Path(__file__)),
            "evaluation_role":"seen_training_view_diagnostic","requested_renders":20,"views_per_map":5,"fixed_views":VIEWS,
            "source":sources,"source_record_role":"Unchanged original render/metric helper source; actual evaluation script SHA is separately recorded",
            "source_sha256":chain["source_sha256"],"lineage":chain,"datasets":datasets,"maps":maps,
            "historical_CPU_evidence_sha256":historical_cpu,
            "cpu_preflight":{"path":recovery.relative(OUTPUT/"cpu_preflight.json"),"sha256":recovery.sha(OUTPUT/"cpu_preflight.json")},
            "render_and_metric_settings":{"width":WIDTH,"height":HEIGHT,"render_mode":"RGB+ED","alpha_threshold_strict":.5,
                "valid_depth_range_m_inclusive":[.2,5.],"RGB_data_range":1,"RGB_metric_clipping":False,
                "GT_world_gauge":"inverse(GT_frame0_c2w) @ GT_view_c2w","alignment_fit":"none",
                "own_C":"V & alpha>.5 & finite(ED) & ED>0","common_C":"own_C_prior0 & own_C_prior001",
                "holes_as_zero":"predicted depth set to0 outside own_C, error on all V","empty_C_depth_metrics":None,
                "PSNR_zero_MSE":"null dB with positive_infinity flag","optimization_steps":0,"retries":0},
            "previous_CPU_preflight_failure":{"failure_path":"outputs/tum-online-v4-full-seen-map-evaluation-001/failure.json",
                "failure_sha256":recovery.sha(ROOT/"outputs/tum-online-v4-full-seen-map-evaluation-001/failure.json"),
                "preserved_source":"outputs/tum-online-v4-full-seen-map-evaluation-001/source_script.py",
                "preserved_source_sha256":recovery.sha(ROOT/"outputs/tum-online-v4-full-seen-map-evaluation-001/source_script.py"),
                "reason":"Initialization frame has no tracking-only warning/rejection fields; optional field absence is now explicitly retained, not filled with successful values.","GPU_calls":0},
            "previous_passed_CPU_preflight":{"protocol":"manifests/tum-online-v4-full-seen-map-evaluation-protocol-003.json",
                "protocol_sha256":recovery.sha(ROOT/"manifests/tum-online-v4-full-seen-map-evaluation-protocol-003.json"),
                "preserved_source":"outputs/tum-online-v4-full-seen-map-evaluation-003/source_script.py",
                "preserved_source_sha256":recovery.sha(ROOT/"outputs/tum-online-v4-full-seen-map-evaluation-003/source_script.py"),
                "revision_reason":"Bind root-requested actual completed recovery trajectory summary, independent audit, completion and both source hashes; no renderer/metric changes.","GPU_calls":0},
            "outputs":{"raw_predictions":"raw_predictions/<run>/view_<frameid>.npz",
                "fixed_first_middle_last_figures":"public_diagnostics/<run>.png","map_history":"map_histories/<run>.json"},
            "limitations":LIMITATIONS}
        require(not torch.cuda.is_initialized(),"CPU preparation initialized CUDA")
        write_small(PROTOCOL,protocol)
        recovery.atomic(OUTPUT/"progress.json",{"status":"awaiting_root_protocol_review","protocol_sha256":recovery.sha(PROTOCOL),"GPU_calls":0})
        print(json.dumps({"protocol":recovery.relative(PROTOCOL),"protocol_sha256":recovery.sha(PROTOCOL),
            "script_sha256":protocol["evaluation_script_sha256"],"maps":4,"requested_renders":20,"CPU":"passed","GPU_calls":0},indent=2))
    except BaseException as error:
        recovery.write_new(OUTPUT/"failure.json",{"phase":"CPU_prepare","at":recovery.stamp(),"exception_type":type(error).__name__,
            "message":str(error),"traceback":traceback.format_exc(),"GPU_calls":0})
        raise

def aggregate(rows,module):
    good=[row["metrics"] for row in rows if row["status"]=="completed"]
    result={"requested_views":5,"completed_views":len(good),"failed_views":5-len(good),"scope":"successful views only; all requested rows retained"}
    if good:
        rgb_sse=sum(row["rgb"]["squared_error_sum"] for row in good);rgb_count=sum(row["rgb"]["channel_values"] for row in good)
        valid=sum(row["valid_depth_pixels"] for row in good);covered=sum(row["covered_valid_pixels"] for row in good)
        result["pooled"]={"rgb":{"squared_error_sum":rgb_sse,"channel_values":rgb_count,**module.psnr(rgb_sse/rgb_count)},"valid_depth_pixels":valid,"covered_valid_pixels":covered,
            "sensor_valid_coverage":covered/valid,"covered_depth":module.sum_depth([row["covered_depth"] for row in good]),
            "all_valid_holes_as_zero_depth":module.sum_depth([row["all_valid_holes_as_zero_depth"] for row in good])}
    return result

def export_figure(run,scene,inputs,rows,predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=[0,2,4];fig,axes=plt.subplots(3,5,figsize=(11,6.5),layout="constrained")
    names=("observed RGB","render RGB","observed depth m","ED m (0..5)","alpha")
    for k in range(5):axes[0,k].set_title(names[k],fontsize=10)
    for r,index in enumerate(selected):
        frame=VIEWS[scene][index]
        axes[r,0].imshow(inputs["rgb"][index]);axes[r,2].imshow(inputs["depth"][index],vmin=0,vmax=5,cmap="viridis")
        prediction=predictions.get(frame)
        if prediction is not None:
            axes[r,1].imshow(np.clip(prediction[0],0,1));axes[r,3].imshow(prediction[1],vmin=0,vmax=5,cmap="viridis");axes[r,4].imshow(prediction[2],vmin=0,vmax=1,cmap="gray")
        else:
            for k in (1,3,4):axes[r,k].text(.5,.5,"FAILED",ha="center",va="center",transform=axes[r,k].transAxes)
        for axis in axes[r]:axis.set_xticks([]);axis.set_yticks([])
        axes[r,0].set_ylabel("frame "+str(frame))
    fig.suptitle(run+"\nSeen training views; fixed GT frame0 gauge; RGB clipping for display only",fontsize=10)
    path=OUTPUT/"public_diagnostics"/(run+".png");fig.savefig(path,dpi=140);plt.close(fig)
    return {"path":recovery.relative(path),"sha256":recovery.sha(path),"fixed_frame_ids":[VIEWS[scene][index] for index in selected],"RGB_clipping_display_only":True}

def load_renderer_backend(*,clock=recovery.now,loader=None):
    require(clock()<recovery.DEADLINE,"Deadline reached before GPU backend import; do not start")
    if loader is not None:return loader()
    from gsplat import rasterization
    return rasterization

def execute(expected_sha):
    module=helper()
    protocol,_=module.decode_reviewed_protocol(PROTOCOL.read_bytes(),expected_sha)
    require(protocol["evaluation_script_sha256"]==recovery.sha(Path(__file__)),"Evaluation source changed since review")
    recovery.write_new(OUTPUT/"execution_started.json",{"at":recovery.stamp(),"protocol_sha256":expected_sha})
    progress={"status":"running_CPU_guards","at":recovery.stamp(),"protocol_sha256":expected_sha,"GPU_started":False,"render_attempts":0,"maps":[]}
    try:
        sources,render_node=module.source_checks();chain,rows=lineage()
        require(sources==protocol["source"] and chain==protocol["lineage"],"Frozen source/terminal lineage changed")
        recovery.check_hashes(protocol["historical_CPU_evidence_sha256"])
        recovery.check_hashes({protocol["cpu_preflight"]["path"]:protocol["cpu_preflight"]["sha256"]})
        datasets,inputs,sensors={},{},{}
        for scene in VIEWS:
            datasets[scene],inputs[scene],sensors[scene]=prepare_dataset(scene,module)
            require(datasets[scene]==protocol["datasets"][scene],"Evaluation raw sensor/K/GT input bytes changed")
        checkpoints=[]
        for row,expected in zip(rows,protocol["maps"]):
            actual,checkpoint=inspect_map(row,datasets[row["scene"]],sensors[row["scene"]],module,write_history=False)
            require(actual==expected,"Actual final map tensors/artifacts/history changed")
            checkpoints.append(checkpoint)
        for name in ("raw_predictions","public_diagnostics"):(OUTPUT/name).mkdir(exist_ok=False)
        recovery.atomic(OUTPUT/"progress.json",progress)
        rasterization=load_renderer_backend()
        all_predictions={};job_start=time.perf_counter()
        for record,checkpoint in zip(protocol["maps"],checkpoints):
            scene,run=record["scene"],record["run"]
            parameters_cpu=checkpoint["parameters"]
            if not progress["GPU_started"]:
                # Every guard/hash/input/read and progress write above finishes
                # before this actual deadline check immediately before CUDA copy.
                gpu_guard=recovery.now()
                require(gpu_guard<recovery.DEADLINE,"Deadline reached; no GPU job started")
                progress.update(GPU_started=True,GPU_started_at=recovery.stamp(gpu_guard),status="rendering")
            parameters={key:value.to("cuda").detach() for key,value in parameters_cpu.items()}
            require(all(not value.requires_grad for value in parameters.values()),"Parameters not frozen")
            K=torch.from_numpy(inputs[scene]["K_render_float32"]).to("cuda")
            render=module.bind_original_render(render_node,parameters,K,rasterization)
            raw_folder=OUTPUT/"raw_predictions"/run;raw_folder.mkdir(exist_ok=False)
            result={"run":run,"scene":scene,"prior_weight":record["prior_weight"],"point_count":record["point_count"],
                "history":record["history"],"training_failure_frame_count":record["training_failure_frame_count"],
                "views":[],"parameters_before":record["parameter_tensors"]}
            predictions={}
            for index,frame in enumerate(VIEWS[scene]):
                view={"frame_id":frame,"rgb_timestamp":datasets[scene]["evaluation_sensor_metadata"][index]["rgb_timestamp"],"role":"seen_training_view_diagnostic"}
                progress["render_attempts"]+=1
                try:
                    w2c=torch.from_numpy(inputs[scene]["render_w2c_float32"][index]).to("cuda")
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.inference_mode():rgb,depth,alpha=render(w2c)
                    torch.cuda.synchronize();view["render_wall_seconds_including_sync"]=time.perf_counter()-start
                    arrays=(rgb.cpu().numpy(),depth.cpu().numpy(),alpha.cpu().numpy())
                    prediction_path=raw_folder/f"view_{frame:04d}.npz"
                    with prediction_path.open("xb") as stream:
                        np.savez_compressed(stream,rgb=arrays[0],depth_ED_m=arrays[1],alpha=arrays[2])
                    view["prediction"]={"path":recovery.relative(prediction_path),"sha256":recovery.sha(prediction_path)}
                    view["metrics"]=module.view_metrics(*arrays,inputs[scene]["rgb"][index],inputs[scene]["depth"][index],inputs[scene]["valid_depth_mask"][index])
                    view["status"]="completed";predictions[frame]=arrays
                except Exception as error:
                    view.update(status="failed",exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                result["views"].append(view)
                recovery.atomic(OUTPUT/"progress.json",{**progress,"current_map":run,"completed_current_map_views":result["views"]})
            result["parameters_after"]={key:module.array_record(value) for key,value in parameters.items()}
            require(result["parameters_after"]==result["parameters_before"],"Render changed map parameter bytes")
            require(recovery.sha(ROOT/record["checkpoint"])==record["checkpoint_sha256"],"Original final map file changed")
            result["parameter_bytes_unchanged"]=True;result["aggregate"]=aggregate(result["views"],module)
            result["fixed_diagnostic_figure"]=export_figure(run,scene,inputs[scene],result["views"],predictions)
            progress["maps"].append(result);all_predictions[run]=predictions
            del render,parameters,K
        pairs=[]
        for scene in VIEWS:
            pair_maps=[record for record in protocol["maps"] if record["scene"]==scene]
            off,on=[all_predictions[record["run"]] for record in pair_maps]
            paired=[]
            for index,frame in enumerate(VIEWS[scene]):
                if frame in off and frame in on:
                    item={"frame_id":frame,"status":"completed","metrics":module.paired_metrics(off[frame][1],off[frame][2],on[frame][1],on[frame][2],inputs[scene]["depth"][index],inputs[scene]["valid_depth_mask"][index])}
                else:item={"frame_id":frame,"status":"unavailable_due_to_retained_render_failure"}
                paired.append(item)
            good=[row["metrics"] for row in paired if row["status"]=="completed"]
            pooled={"completed_paired_views":len(good),"requested_paired_views":5}
            if good:
                valid=sum(row["sensor_valid_pixels"] for row in good);common=sum(row["intersection_pixels"] for row in good)
                pooled.update(sensor_valid_pixels=valid,intersection_pixels=common,intersection_sensor_valid_coverage=common/valid,
                    prior0_depth=module.sum_depth([row["prior0_depth"] for row in good]),prior001_depth=module.sum_depth([row["prior001_depth"] for row in good]))
            pairs.append({"scene":scene,"prior0_run":pair_maps[0]["run"],"prior001_run":pair_maps[1]["run"],"views":paired,"pooled_common_C":pooled})
        require(progress["render_attempts"]==20,"Missing requested render attempts")
        require(lineage()[0]==protocol["lineage"],"Final source/original failure lineage changed")
        for scene in VIEWS:
            require({name:module.array_record(value) for name,value in inputs[scene].items()}==protocol["datasets"][scene]["input_tensors"],"Evaluation inputs changed")
            recovery.check_hashes(protocol["datasets"][scene]["input_file_sha256"])
        failed=sum(view["status"]!="completed" for result in progress["maps"] for view in result["views"])
        progress.update(status="completed" if not failed else "completed_with_render_failures",ended_at=recovery.stamp(),
            failed_render_count=failed,successful_render_count=20-failed,pairs=pairs,optimization_steps=0,
            total_job_seconds_includes_render_metric_NPZ_and_figure_export=time.perf_counter()-job_start,
            input_tensor_bytes_before_after_unchanged=True,original_map_and_failure_artifacts_unchanged=True,
            evaluation_role="seen_training_view_diagnostic",limitations=LIMITATIONS)
        write_small(OUTPUT/"metrics.json",progress);recovery.atomic(OUTPUT/"progress.json",progress)
        print(json.dumps({"status":progress["status"],"renders":20,"failed_renders":failed,"optimization_steps":0,
            "metrics_sha256":recovery.sha(OUTPUT/"metrics.json")},indent=2))
    except BaseException as error:
        progress.update(status="failed",failed_at=recovery.stamp(),exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        recovery.atomic(OUTPUT/"progress.json",progress);recovery.write_new(OUTPUT/"failure.json",progress)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true");group.add_argument("--execute",action="store_true");parser.add_argument("--protocol-sha")
    args=parser.parse_args()
    if args.prepare:
        if args.protocol_sha:parser.error("Protocol SHA only applies to execute")
        prepare()
    else:
        if not args.protocol_sha:parser.error("Exact reviewed protocol SHA required")
        execute(args.protocol_sha)

if __name__=="__main__":main()
