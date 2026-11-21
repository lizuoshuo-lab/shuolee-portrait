"""Pure-JSON, audit-gated summary of frozen-map GT versus saved-camera controls.

Prepare only validates structure, locks file bytes and runs negative fixtures.
Actual export requires root source/protocol review and an actual independent
raw-array audit bound by its source and artifact SHA. No metric implementation,
renderer, tensor library, camera fit, optimization or GPU import is used here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/tum-online-v4-estimated-seen-map-evaluation-002"
OUTPUT = ROOT / "outputs/tum-v4-estimated-camera-summary-001"
PROTOCOL = OUTPUT / "protocol.json"
EVALUATION = ROOT / "manifests/tum-online-v4-estimated-seen-map-evaluation-protocol-002.json"
EVALUATION_SHA = "124571c2835445a08e696e9a08f4f64495a7fb88aa35072f260404cefa8b2094"
METRIC_SHA = "daec22d74d01c8f208907817b1067cbe61f8a6eb9ff9c23db53eca773d11b14f"
GT_EVALUATION = ROOT / "manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json"
GT_EVALUATION_SHA = "7f07d29ebbc96b9f96d48d84e97bc2f9c9c042186a70f1f8a5fa87e23c8f6e40"
GT_METRIC = ROOT / "outputs/tum-online-v4-full-seen-map-evaluation-004/metrics.json"
GT_METRIC_SHA = "f2b0c23f57df18a968901f77dcba881425cc7062cbf194027d444e3808da7159"
GT_AUDIT_SHA = "834ab16cea0efb298c44c512d3a7e0c5497c43c4075ad8caf44941a2a183ae65"
EVALUATOR_SHA = "7dfe57310c334d5e917dee5e12e0a68f9d83381b1096aeb00d0fc29bfa1bab74"
MATH_SHA = "aabd7c5197711567e679457ce6d203939252395945f558bc7e2b120f83b8f695"
AUDIT = INPUT / "independent_estimated_seen_map_audit.json"
MAX_BYTES = 1024 * 1024
RUNS = ["tum-desk-online-v4-prior0-full549-001", "tum-desk-online-v4-prior001-full549-001",
        "tum-xyz-online-v4-prior0-full748-001", "tum-xyz-online-v4-prior001-full748-002"]
VIEWS = {"desk": [0,137,274,411,548], "xyz": [0,187,374,561,747]}
FAILED_RUN = "tum-xyz-online-v4-prior001-full748-001"
AUDIT_EXPECTED = {"status": "passed_independent_estimated_seen_map_audit",
        "independent_math_helper_sha256": MATH_SHA, "evaluation_script_sha256": EVALUATOR_SHA,
        "protocol_sha256": EVALUATION_SHA, "metrics_sha256": METRIC_SHA,
        "existing_GT_metrics_sha256": GT_METRIC_SHA, "existing_GT_independent_audit_sha256": GT_AUDIT_SHA,
        "map_count": 4, "requested_view_count": 20, "actual_estimated_NPZ_count": 20,
        "existing_GT_NPZ_count": 20, "actual_prediction_payload_array_count": 120,
        "requested_same_map_commonC_views": 20, "all_training_rows_including_flags": 2594, "GPU_calls": 0}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(MAX_BYTES), b""): digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")


def write_new(path, value):
    data = encode(value)
    require(len(data) <= MAX_BYTES, "One-MiB ceiling exceeded; do not omit views, camera pairs or flags")
    with Path(path).open("xb") as stream: stream.write(data)


def hash_guard(records):
    for name, value in records.items():
        require(sha(ROOT/name) == value, "Changed locked source/input/prediction/failure evidence: "+name)


def validate_structure(metric, evaluation, gt_metric, gt_evaluation, prediction_locks):
    require(metric["status"] in ("completed", "completed_with_render_failures") and metric["render_attempts"] == 20
            and metric["new_GT_camera_renders"] == metric["optimization_steps"] == 0,
            "Incorrect completed camera-choice render count/scope")
    require(metric["evaluation_role"] == evaluation["evaluation_role"] == "same_frozen_map_camera_choice_seen_training_view_diagnostic"
            and evaluation["requested_new_renders"] == 20 and evaluation["fixed_views"] == gt_evaluation["fixed_views"] == VIEWS,
            "Wrong view role or reselected fixed views")
    settings = evaluation["render_and_metric_settings"]
    require(settings["alignment_fit"] == settings["map_transform"] == "none"
            and settings["new_GT_camera_renders"] == settings["optimization_steps"] == settings["retries"] == 0
            and settings["new_estimated_camera_renders"] == 20
            and settings["camera_choice"] == "saved accepted estimated_w2c exact float32 bytes from GT-free causal final checkpoint",
            "Fit/optimization/camera source scope changed")
    require(evaluation["lineage"] == gt_evaluation["lineage"]
            and evaluation["lineage"]["original_failed_run"] == FAILED_RUN
            and evaluation["lineage"]["selected_run_order"] == RUNS, "Original failure/replacement lineage changed")
    require([m["run"] for m in metric["maps"]] == [m["run"] for m in evaluation["maps"]] == RUNS
            and len(metric["same_map_camera_pairs"]) == 4, "Missing/reordered maps or same-map camera pairs")
    failed = 0
    for actual, expected, gt, original, pair in zip(metric["maps"], evaluation["maps"], gt_metric["maps"],
                                                   gt_evaluation["maps"], metric["same_map_camera_pairs"]):
        scene = actual["scene"]; frames = VIEWS[scene]
        require(actual["run"] == expected["run"] == gt["run"] == original["run"] == pair["run"]
                and actual["scene"] == expected["scene"] == pair["scene"]
                and actual["prior_weight"] == expected["prior_weight"], "Camera pair mixed maps/scenes/weights")
        require(actual["camera_choice"] == "saved accepted estimated camera"
                and pair["same_map_parameter_bytes"] is True
                and actual["parameters_before"] == actual["parameters_after"] == expected["parameter_tensors"]
                    == gt["parameters_before"] == gt["parameters_after"]
                and actual["parameter_bytes_unchanged"] is True, "Not the same frozen map bytes across camera choices")
        require(actual["saved_camera_tensors_before"] == expected["camera_tensors"]
                and actual["selected_camera_bytes_unchanged"] is True
                and expected["ground_truth_in_render_checkpoint"] is False
                and expected["actual_GT_camera_substitution_rejected"] is True,
                "Saved accepted camera/gauge/GT-free byte guards failed")
        require(actual["history"] == expected["history"] == original["history"]
                and actual["point_count"] == expected["point_count"] == original["point_count"]
                and actual["training_failure_frame_count"] == expected["training_failure_frame_count"],
                "Map history/source counts or training failure rows changed")
        require(actual["existing_GT_camera_aggregate"] == gt["aggregate"], "Existing GT aggregate relabeled/recomputed")
        require(len(actual["views"]) == len(pair["views"]) == 5
                and [v["frame_id"] for v in actual["views"]] == [v["frame_id"] for v in pair["views"]] == frames,
                "Missing/reselected camera-choice view or same-map pair row")
        require(actual["aggregate"]["requested_views"] == pair["pooled_same_map_common_C"]["requested_same_map_paired_views"] == 5,
                "Requested per-map denominators changed")
        for index, (view, gt_view, common) in enumerate(zip(actual["views"], gt["views"], pair["views"])):
            require(view["camera_choice"] == "saved accepted estimated w2c"
                    and view["saved_accepted_c2w"] == expected["selected_accepted_c2w"][index]
                    and view["render_saved_accepted_w2c_float32"] == expected["render_saved_accepted_w2c_float32"][index]
                    and view["same_map_GT_prediction"] == gt_view["prediction"]
                    and view["existing_GT_camera_metrics"] == gt_view["metrics"], "Mixed GT/estimated camera labels or matrices")
            require(view["rgb_timestamp"] == gt_view["rgb_timestamp"], "RGB time pairing changed")
            if "prediction" in view:
                predicted = view["prediction"]
                expected_path = relative(INPUT/"raw_predictions"/actual["run"]/f"view_{frames[index]:04d}.npz")
                require(predicted["path"] == expected_path and prediction_locks.get(expected_path) == predicted["sha256"],
                        "Wrong NPZ run/frame/path/SHA binding")
            if view["status"] == "completed":
                require("prediction" in view and common["status"] == "completed", "Missing completed prediction/common-C")
                own = view["metrics"]; old = gt_view["metrics"]; shared = common["metrics"]
                valid = evaluation["datasets"][scene]["valid_depth_pixels_per_view"][index]
                count = shared["intersection_pixels"]
                require(own["valid_depth_pixels"] == old["valid_depth_pixels"] == shared["sensor_valid_pixels"] == valid
                        and own["covered_depth"]["pixels"] == own["covered_valid_pixels"]
                        and own["all_valid_holes_as_zero_depth"]["pixels"] == valid
                        and shared["GT_camera_depth"]["pixels"] == shared["saved_estimated_camera_depth"]["pixels"] == count
                        and count <= min(own["covered_valid_pixels"], old["covered_valid_pixels"]),
                        "Own/common/holes denominators or camera labels changed")
            else:
                failed += 1
                require(common["status"] == "unavailable_due_to_retained_estimated_camera_render_failure",
                        "Failed view/common-C must be retained, not fabricated")
    require(metric["failed_render_count"] == failed and metric["successful_render_count"] == 20-failed
            and metric["observation_map_camera_and_existing_GT_baseline_bytes_unchanged"] is True,
            "Failure counts or final map/camera/observation/GT byte guards changed")


def frozen_inputs():
    require(sha(EVALUATION) == EVALUATION_SHA and sha(INPUT/"metrics.json") == METRIC_SHA
            and sha(GT_EVALUATION) == GT_EVALUATION_SHA and sha(GT_METRIC) == GT_METRIC_SHA,
            "Actual completed protocol/metric bytes changed")
    metric, evaluation, gt_metric, gt_evaluation = read(INPUT/"metrics.json"), read(EVALUATION), read(GT_METRIC), read(GT_EVALUATION)
    predictions = {v["prediction"]["path"]:v["prediction"]["sha256"] for m in metric["maps"] for v in m["views"] if "prediction" in v}
    validate_structure(metric, evaluation, gt_metric, gt_evaluation, predictions)
    records = {relative(EVALUATION):EVALUATION_SHA, relative(INPUT/"metrics.json"):METRIC_SHA,
               relative(GT_EVALUATION):GT_EVALUATION_SHA, relative(GT_METRIC):GT_METRIC_SHA,
               "scripts/evaluate_tum_v4_estimated_seen_maps.py":EVALUATOR_SHA,
               "scripts/audit_v4_final_map_quality.py":MATH_SHA, **predictions,
               **evaluation["source_sha256"], **evaluation["GT_control_evidence_sha256"],
               **evaluation["historical_CPU_evidence_sha256"], **evaluation["lineage"]["evidence_sha256"]}
    records[evaluation["cpu_preflight"]["path"]] = evaluation["cpu_preflight"]["sha256"]
    for name in ("execution_started.json", "console.log", "root_protocol_review.json"):
        records[relative(INPUT/name)] = sha(INPUT/name)
    for dataset in evaluation["datasets"].values(): records.update(dataset["input_file_sha256"])
    hash_guard(records)
    require(not any(name in sys.modules for name in ("torch", "gsplat", "numpy")), "Pure summary imported a tensor/GPU/metric library")
    return metric, evaluation, gt_metric, gt_evaluation, predictions, records


def table(metric, evaluation):
    rows = []
    for actual, expected, pair in zip(metric["maps"], evaluation["maps"], metric["same_map_camera_pairs"]):
        rows.append({"run":actual["run"], "scene":actual["scene"], "prior_weight":actual["prior_weight"],
                     "point_count":actual["point_count"], "point_cap":expected["point_cap"], "cap_reached":expected["cap_reached"],
                     "requested_views":5, "completed_views":actual["aggregate"]["completed_views"],
                     "failed_views":actual["aggregate"]["failed_views"], "history":expected["history"],
                     "GT_camera_aggregate":actual["existing_GT_camera_aggregate"],
                     "saved_estimated_camera_aggregate":actual["aggregate"],
                     "same_map_common_C":pair["pooled_same_map_common_C"],
                     "same_map_parameter_bytes":True, "saved_accepted_camera_bytes_unchanged":True})
    return rows


def build(metric, evaluation, audit_record, review_record):
    result = copy.deepcopy(metric)
    result["summary_table"] = table(metric, evaluation)
    result["summary_provenance"] = {
        "format":"complete_same_map_camera_choice_summary_v1", "summary_source_sha256":sha(Path(__file__)),
        "evaluation_protocol_sha256":EVALUATION_SHA, "complete_metrics_sha256":METRIC_SHA,
        "existing_GT_protocol_sha256":GT_EVALUATION_SHA, "existing_GT_metrics_sha256":GT_METRIC_SHA,
        "actual_independent_audit":audit_record, "root_source_review":review_record,
        "metric_calculation_by_summary":"none; original per-view and pooled values copied unchanged",
        "all_original_twenty_view_four_same_map_camera_pair_camera_gauge_byte_guard_and_failure_fields_retained":True,
        "pairing_unit":"same map tensor bytes, same sensor view; camera choice only",
        "cross_map_same_warmup_map_claim":False,
        "methods":{"camera_choice":"exact saved accepted w2c, no fit/transform/optimization",
                   "mask_scope":"ownC and holes per camera; commonC only between two camera choices of each same map",
                   "pooling":"five-view sums of original SSE/counts, no meanPSNR, no cross-scene aggregate, no single-seed SD/CI",
                   "role":"seen-training views; render consistency, not localization or world geometry accuracy",
                   "CUDA_control_boundary":"GT and estimated renders at different times; numerical reduction nondeterminism not isolated"},
        "training_flags":"All2594 original training rows remain in the four SHA-indexed history files",
        "failure_lineage":"original completed3 plus explicitly reviewed replacement002; original failed001 remains retained"}
    require(len(encode(result)) <= MAX_BYTES, "Full-retention summary exceeds size cap")
    return result


def prepare():
    require(not OUTPUT.exists(), "Never overwrite prepared summary evidence")
    OUTPUT.mkdir(exist_ok=False)
    try:
        metric, evaluation, gt_metric, gt_evaluation, predictions, records = frozen_inputs()
        controls = {}
        cases = {
            "mixed_camera_label":lambda m,e:m["maps"][0]["views"][1].update(camera_choice="GT camera"),
            "missing_requested_view":lambda m,e:m["maps"][0]["views"].pop(),
            "wrong_NPZ_SHA_binding":lambda m,e:m["maps"][0]["views"][0]["prediction"].update(sha256="0"*64),
            "wrong_failure_lineage":lambda m,e:e["lineage"].update(original_failed_run="hidden_or_replaced_failure"),
            "mixed_same_map_pair":lambda m,e:m["same_map_camera_pairs"][0].update(run=RUNS[1]),
            "fake_optimizer":lambda m,e:m.update(optimization_steps=1),
            "changed_saved_camera":lambda m,e:m["maps"][0]["views"][1]["render_saved_accepted_w2c_float32"][0].__setitem__(3,99.),
        }
        for label, mutate in cases.items():
            m, e = copy.deepcopy(metric), copy.deepcopy(evaluation); mutate(m,e)
            try: validate_structure(m,e,gt_metric,gt_evaluation,predictions)
            except ValueError: controls[label] = True
            else: raise AssertionError("Negative fixture was not rejected: "+label)
        fixture = build(metric,evaluation,{"status":"CPU_size_fixture_not_actual_audit"},{"status":"not_reviewed"})
        require({k:fixture[k] for k in metric} == metric, "Size fixture lost original fields")
        cpu = {"status":"passed_structure_byte_size_and_negative_fixtures_only", "source_sha256":sha(Path(__file__)),
               "all_maps":4, "all_requested_views":20, "all_same_map_camera_pairs":4,
               "negative_controls":controls, "summary_fixture_bytes":len(encode(fixture)),
               "input_file_count":len(records), "actual_independent_audit_already_passed":False,
               "actual_summary_written":False, "tensor_or_GPU_stack_imported":False, "GPU_calls":0}
        write_new(OUTPUT/"cpu_preflight.json",cpu)
        protocol = {"protocol_id":"tum-v4-estimated-camera-summary-001", "status":"CPU_prepared_root_review_and_actual_independent_audit_required",
                    "source_sha256":sha(Path(__file__)), "input_sha256":records,
                    "cpu_preflight":{"path":relative(OUTPUT/"cpu_preflight.json"),"sha256":sha(OUTPUT/"cpu_preflight.json")},
                    "root_review_relative_path":relative(OUTPUT/"root_source_review.json"),
                    "root_review_binding_required":{"status":"passed", "summary_protocol_sha256":"exact protocol SHA supplied to CLI",
                        "summary_source_sha256":"exact source SHA in this protocol",
                        "independent_audit_binding":{"path":relative(AUDIT), "sha256":"actual passed artifact SHA",
                                                     "source_path":"actual independent auditor source path", "source_sha256":"actual reviewed auditor source SHA"}},
                    "actual_audit_required_fields":AUDIT_EXPECTED,
                    "audit_control_boundary":"planned schema is not passed evidence; root review must bind final real source and artifact hashes before export",
                    "outputs":["summary.json","validation.json"], "byte_limit":MAX_BYTES, "GPU_calls":0}
        write_new(PROTOCOL,protocol)
        print(json.dumps({"status":"CPU_prepared_only", "protocol_sha256":sha(PROTOCOL), "source_sha256":protocol["source_sha256"],
                          "fixture_bytes":cpu["summary_fixture_bytes"], "inputs":len(records), "GPU_calls":0},indent=2))
    except BaseException as error:
        write_new(OUTPUT/"failure.json", {"phase":"CPU_prepare", "exception_type":type(error).__name__,
                  "message":str(error), "traceback":traceback.format_exc(), "GPU_calls":0})
        raise


def actual_audit_gate(protocol, protocol_sha):
    review_path = ROOT/protocol["root_review_relative_path"]; review = read(review_path)
    require(review["status"] == "passed" and review["summary_protocol_sha256"] == protocol_sha
            and review["summary_source_sha256"] == sha(Path(__file__)), "Exact actual root source/protocol review required")
    binding = review["independent_audit_binding"]
    require(binding["path"] == relative(AUDIT) and binding["source_path"].startswith("scripts/")
            and sha(AUDIT) == binding["sha256"] and sha(ROOT/binding["source_path"]) == binding["source_sha256"],
            "Actual independent audit source/artifact hashes differ from root binding")
    audit = read(AUDIT)
    for key,value in protocol["actual_audit_required_fields"].items():
        require(audit.get(key) == value, "Actual passed audit gate: "+key)
    require(audit["audit_source_sha256"] == binding["source_sha256"], "Auditor actual source SHA mismatch")
    return {**binding,"required_passed_fields":{**AUDIT_EXPECTED,"audit_source_sha256":binding["source_sha256"]}}, \
           {"path":relative(review_path),"sha256":sha(review_path)}


def summarize(protocol_sha):
    progress = {"phase":"actual_summary_export", "GPU_calls":0}
    try:
        require(sha(PROTOCOL) == protocol_sha, "Exact reviewed summary protocol SHA required")
        protocol = read(PROTOCOL)
        require(protocol["source_sha256"] == sha(Path(__file__)), "Summary source changed after review")
        hash_guard(protocol["input_sha256"])
        hash_guard({protocol["cpu_preflight"]["path"]:protocol["cpu_preflight"]["sha256"]})
        metric,evaluation,gt_metric,gt_evaluation,predictions,records = frozen_inputs()
        require(records == protocol["input_sha256"], "Frozen summary input set changed")
        audit,review = actual_audit_gate(protocol,protocol_sha)
        result = build(metric,evaluation,audit,review); write_new(OUTPUT/"summary.json",result)
        actual = read(OUTPUT/"summary.json")
        require({k:actual[k] for k in metric} == metric, "Original completed values/flags/camera fields changed")
        hash_guard(records); hash_guard({audit["path"]:audit["sha256"],audit["source_path"]:audit["source_sha256"],review["path"]:review["sha256"]})
        validation = {"status":"passed_exact_original_camera_choice_field_and_flag_retention", "summary_sha256":sha(OUTPUT/"summary.json"),
              "summary_bytes":(OUTPUT/"summary.json").stat().st_size,"summary_source_sha256":sha(Path(__file__)),
              "summary_protocol_sha256":protocol_sha,"actual_independent_audit":audit,"root_source_review":review,
              "all_original_twenty_views_four_same_map_pairs_camera_fields_and_flags_exact":True,
              "all_inputs_rehashed_unchanged":True,"input_file_count":len(records),"GPU_calls":0}
        write_new(OUTPUT/"validation.json",validation)
        print(json.dumps({"status":validation["status"],"summary_sha256":validation["summary_sha256"],
                          "summary_bytes":validation["summary_bytes"],"GPU_calls":0},indent=2))
    except BaseException as error:
        write_new(OUTPUT/"failure.json",{**progress,"exception_type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true"); group.add_argument("--summarize",action="store_true"); parser.add_argument("--protocol-sha")
    args=parser.parse_args()
    if args.prepare:
        if args.protocol_sha:parser.error("Protocol SHA applies only to actual export")
        prepare()
    else:
        if not args.protocol_sha:parser.error("Exact reviewed protocol SHA required")
        summarize(args.protocol_sha)


if __name__ == "__main__": main()
