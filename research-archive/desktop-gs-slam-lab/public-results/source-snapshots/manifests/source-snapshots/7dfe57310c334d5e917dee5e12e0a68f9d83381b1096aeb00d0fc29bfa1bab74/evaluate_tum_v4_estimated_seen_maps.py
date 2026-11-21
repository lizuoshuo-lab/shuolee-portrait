"""Frozen-map camera-choice control: saved accepted cameras versus existing GT renders.

Prepare and review first. Execute exactly 20 new renders of the predeclared
seen-training views. Map bytes, observations, K, renderer and metrics are held
fixed within each map. No fit, pose update, map update, view selection or retry.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import run_v4_full_recovery as recovery

ROOT = recovery.ROOT
BASE = ROOT / "scripts/evaluate_tum_v4_full_seen_maps.py"
BASE_SHA = "41ee8343536096b38e6effbaa415a2fbcadc43ea7ffe7dcb5d73cdc9e51c41d2"
GT_PROTOCOL = ROOT / "manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json"
GT_PROTOCOL_SHA = "7f07d29ebbc96b9f96d48d84e97bc2f9c9c042186a70f1f8a5fa87e23c8f6e40"
GT_FOLDER = ROOT / "outputs/tum-online-v4-full-seen-map-evaluation-004"
PROTOCOL = ROOT / "manifests/tum-online-v4-estimated-seen-map-evaluation-protocol-002.json"
OUTPUT = ROOT / "outputs/tum-online-v4-estimated-seen-map-evaluation-002"
EVIDENCE = {
    "scripts/evaluate_tum_v4_full_seen_maps.py": BASE_SHA,
    "manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json": GT_PROTOCOL_SHA,
    "outputs/tum-online-v4-full-seen-map-evaluation-004/metrics.json": "f2b0c23f57df18a968901f77dcba881425cc7062cbf194027d444e3808da7159",
    "outputs/tum-online-v4-full-seen-map-evaluation-004/independent_full_seen_map_audit.json": "834ab16cea0efb298c44c512d3a7e0c5497c43c4075ad8caf44941a2a183ae65",
    "outputs/tum-v4-full-seen-map-summary-001/summary.json": "822767cc1a03954c37dbd5e4292bb00f16216385b49ac8de3f8a7f42c8d5af29",
    "outputs/tum-v4-full-seen-map-summary-001/validation.json": "5d76d1705f518c2c9ed2f99f3f56a0626c93985ff60ce8d5b84b5abfbc2327f2",
    "scripts/summarize_tum_v4_full_seen_maps.py": "251c8b24fb3c16ab47df6fadb351f1b97a410298da4e322a760fc66022536d60",
    "scripts/audit_v4_full_seen_map_quality.py": "020451555333f7137aa568cd8792db9e28818aba6e612d15bb87458b91271abe",
    "scripts/audit_v4_final_map_quality.py": "aabd7c5197711567e679457ce6d203939252395945f558bc7e2b120f83b8f695",
}
LIMITATIONS = [
    "Twenty fixed seen-training views of four completed maps, not heldout or localization evaluation.",
    "The only new render input changed within each frozen map is its camera: saved accepted estimated w2c instead of the existing anchored GT camera.",
    "Estimated cameras may fit the observations used to build the map; lower render error does not establish accurate world geometry or localization.",
    "No SE3/Sim3 fit, GT-dependent camera correction, map transform, optimization, best-view selection or retries.",
    "Weight-paired maps have different warmup bytes. Only the two camera choices within the SAME map share exact map bytes.",
    "GT baseline and new estimated-camera renders are executed at different times; CUDA rendering reduction may be numerically nondeterministic.",
    "RGB-depth offsets, nearest GT association and registered Freiburg1 calibration assumptions are unchanged.",
    "Own coverage, same-map GT/estimated common coverage and holes-as-zero use different denominators; all requested failures remain visible.",
    "The original xyz prior001 full748-001 failed with partial521 evidence; the reviewed frame0 restart002 is explicitly selected and the failure lineage retained.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc)


def stamp(value=None):
    value = now() if value is None else value
    return {"utc": value.astimezone(dt.timezone.utc).isoformat(),
            "beijing": value.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()}


def load_base():
    recovery.check_hashes(EVIDENCE)
    spec = importlib.util.spec_from_file_location("frozen_GT_seen_camera_control", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_small(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    require(len(data) <= 1024 * 1024, "JSON exceeds one MiB; do not discard failures")
    with Path(path).open("xb") as stream:
        stream.write(data)


def baseline_guards(base, module):
    protocol = recovery.read(GT_PROTOCOL)
    metric = recovery.read(GT_FOLDER / "metrics.json")
    audit = recovery.read(GT_FOLDER / "independent_full_seen_map_audit.json")
    validation = recovery.read(ROOT / "outputs/tum-v4-full-seen-map-summary-001/validation.json")
    require(metric["status"] == "completed" and metric["render_attempts"] == metric["successful_render_count"] == 20
            and metric["failed_render_count"] == metric["optimization_steps"] == 0,
            "Existing GT control must contain all twenty successful frozen renders")
    require(audit["status"] == "passed_independent_full_seen_map_audit"
            and audit["evaluation_script_sha256"] == BASE_SHA and audit["protocol_sha256"] == GT_PROTOCOL_SHA
            and audit["metrics_sha256"] == EVIDENCE[recovery.relative(GT_FOLDER / "metrics.json")]
            and audit["map_count"] == 4 and audit["actual_NPZ_count"] == 20
            and audit["actual_prediction_payload_array_count"] == 60 and audit["GPU_calls"] == 0,
            "Existing independent GT audit is not the reviewed actual passed artifact")
    require(validation["status"] == "passed_exact_original_field_and_flag_retention"
            and validation["summary_sha256"] == EVIDENCE["outputs/tum-v4-full-seen-map-summary-001/summary.json"]
            and validation["all_inputs_rehashed_unchanged"] and validation["GPU_calls"] == 0,
            "Existing GT summary must retain original fields exactly")
    require(protocol["evaluation_script_sha256"] == BASE_SHA and protocol["fixed_views"] == base.VIEWS
            and protocol["requested_renders"] == 20, "GT baseline protocol/view IDs changed")
    sources, render_node = module.source_checks()
    chain, rows = base.lineage()
    require(sources == protocol["source"] and chain == protocol["lineage"], "Completed maps/source/failure lineage changed")
    recovery.check_hashes(protocol["historical_CPU_evidence_sha256"])
    recovery.check_hashes({protocol["cpu_preflight"]["path"]: protocol["cpu_preflight"]["sha256"]})
    files = dict(EVIDENCE)
    for record, gt_map in zip(protocol["maps"], metric["maps"]):
        require(record["run"] == gt_map["run"] and record["parameter_tensors"] == gt_map["parameters_before"]
                == gt_map["parameters_after"] and gt_map["parameter_bytes_unchanged"],
                "GT baseline did not render the same frozen map bytes")
        require([v["frame_id"] for v in gt_map["views"]] == base.VIEWS[record["scene"]], "GT render view order changed")
        for view in gt_map["views"]:
            require(view["status"] == "completed", "Missing actual GT prediction")
            path = ROOT / view["prediction"]["path"]
            require(recovery.sha(path) == view["prediction"]["sha256"], "GT prediction NPZ changed")
            files[view["prediction"]["path"]] = view["prediction"]["sha256"]
        files.update(record["artifact_sha256"])
        files[record["history"]["path"]] = record["history"]["sha256"]
    require(len(protocol["maps"]) == len(metric["maps"]) == len(rows) == 4, "Missing selected map")
    return protocol, metric, files, sources, render_node, chain, rows


def assert_camera_record(actual, expected, module):
    require(module.array_record(actual) == expected, "Render camera bytes differ from saved accepted estimated camera")


def accepted_camera_inputs(row, record, module, dataset):
    checkpoint = torch.load(ROOT / record["checkpoint"], map_location="cpu", weights_only=True)
    count = row["configuration"]["frames"]
    require(checkpoint["ground_truth_included"] is False
            and not any("evaluation_only_gt" in key for key in checkpoint), "Camera source is not causal GT-free checkpoint")
    require(checkpoint["frame_ids_with_stored_pose"] == list(range(count)), "Accepted camera/frame chronology incomplete")
    c2w = checkpoint["estimated_c2w"].numpy()
    w2c = checkpoint["estimated_w2c"].numpy()
    require(c2w.shape == w2c.shape == (count, 4, 4) and c2w.dtype == w2c.dtype == np.float32,
            "Saved accepted poses must be the original full float32 camera arrays")
    require(not checkpoint["estimated_c2w"].requires_grad and not checkpoint["estimated_w2c"].requires_grad,
            "Accepted camera arrays must be frozen")
    checks = {"all_saved_c2w": module.rigid_check(c2w), "all_saved_w2c": module.rigid_check(w2c)}
    identity = np.eye(4)
    inverse_error = float(np.max(np.abs(c2w.astype(np.float64) @ w2c.astype(np.float64) - identity)))
    require(inverse_error < 2e-6, "Saved c2w/w2c inverse direction or precision invalid")
    require(np.array_equal(c2w[0], identity) and np.array_equal(w2c[0], identity), "Saved estimated gauge is not initial identity")
    frame_ids = dataset["evaluation_frame_ids"]
    selected_c2w = c2w[frame_ids].copy()
    # Render EXACT saved accepted w2c bytes. No recomposition, SVD, fit or GT use.
    selected_w2c = w2c[frame_ids].copy()
    checks.update(selected_c2w=module.rigid_check(selected_c2w), selected_w2c=module.rigid_check(selected_w2c),
                  maximum_saved_inverse_product_error=inverse_error)
    gt_w2c = np.asarray(dataset["relative_GT_w2c"], dtype=np.float32)
    differences = [float(np.max(np.abs(est.astype(np.float64) - gt.astype(np.float64))))
                   for est, gt in zip(selected_w2c, gt_w2c)]
    require(differences[0] < 1e-12 and max(differences[1:]) > 1e-5,
            "Actual estimated-camera control must agree at anchor and differ from GT at later fixed views")
    cameras = {"saved_all_c2w": module.array_record(c2w), "saved_all_w2c": module.array_record(w2c),
               "selected_accepted_c2w": module.array_record(selected_c2w),
               "render_saved_accepted_w2c_float32": module.array_record(selected_w2c)}
    # Explicit actual-data negative control: substituting GT must fail the
    # selected accepted camera byte guard. Anchor alone is expected to agree.
    try:
        assert_camera_record(gt_w2c, cameras["render_saved_accepted_w2c_float32"], module)
    except ValueError:
        gt_substitution_rejected = True
    else:
        raise ValueError("Actual GT-versus-estimated camera negative control failed")
    result = {**record, "camera_source": "completed GT-free last_checkpoint saved accepted estimated_c2w and estimated_w2c",
              "camera_tensors": cameras, "selected_accepted_c2w": selected_c2w.tolist(),
              "render_saved_accepted_w2c_float32": selected_w2c.tolist(), "camera_checks": checks,
              "GT_vs_saved_estimated_w2c_max_absolute_difference_per_fixed_view": differences,
              "actual_GT_camera_substitution_rejected": gt_substitution_rejected,
              "same_map_parameter_bytes_as_GT_baseline": True}
    return result, {"parameters": checkpoint["parameters"], "K": checkpoint["K"],
                    "selected_accepted_c2w": selected_c2w, "render_saved_accepted_w2c_float32": selected_w2c}


def collect_inputs(base, module, gt_protocol, rows):
    datasets, inputs, sensors = {}, {}, {}
    for scene in base.VIEWS:
        datasets[scene], inputs[scene], sensors[scene] = base.prepare_dataset(scene, module)
        require(datasets[scene] == gt_protocol["datasets"][scene], "Raw RGB/depth/valid/K/GT reference bytes differ from GT20")
    maps, checkpoints = [], []
    for row, expected in zip(rows, gt_protocol["maps"]):
        actual, _ = base.inspect_map(row, datasets[row["scene"]], sensors[row["scene"]], module, write_history=False)
        require(actual == expected, "Final map/history/artifact bytes differ from frozen GT20 protocol")
        record, checkpoint = accepted_camera_inputs(row, actual, module, datasets[row["scene"]])
        maps.append(record)
        checkpoints.append(checkpoint)
    return datasets, inputs, maps, checkpoints


def camera_pair_metrics(module, gt_depth, gt_alpha, est_depth, est_alpha, target, valid):
    original = module.paired_metrics(gt_depth, gt_alpha, est_depth, est_alpha, target, valid)
    # Same independent masks/formulas, explicitly camera labels instead of
    # the helper's historical prior0/prior001 labels. Never compare maps here.
    return {"sensor_valid_pixels": original["sensor_valid_pixels"],
            "intersection_pixels": original["intersection_pixels"],
            "intersection_sensor_valid_coverage": original["intersection_sensor_valid_coverage"],
            "GT_camera_depth": original["prior0_depth"],
            "saved_estimated_camera_depth": original["prior001_depth"],
            "mae_difference_estimated_minus_GT_m": original["mae_difference_prior001_minus_prior0_m"],
            "rmse_difference_estimated_minus_GT_m": original["rmse_difference_prior001_minus_prior0_m"]}


def load_backend(*, clock=now, loader=None):
    require(clock() < recovery.DEADLINE, "Deadline reached before backend import; no new experiment")
    if loader is not None:
        return loader()
    from gsplat import rasterization
    return rasterization


def cpu_controls(base, module, render_node, maps):
    original = base.cpu_controls(module, render_node)
    a = np.eye(4); a[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; a[:3, 3] = [1, 2, 3]
    world_point = np.array([2., 5., 7., 1.])
    local_point = np.linalg.inv(a) @ world_point
    require(np.allclose(a @ local_point, world_point, atol=1e-14, rtol=0)
            and not np.allclose(a @ world_point, local_point), "OpenCV c2w-to-w2c direction negative control ineffective")
    target = np.full((2, 2), 2., dtype=np.float32); valid = np.ones((2, 2), dtype=bool)
    pair = camera_pair_metrics(module, target, np.array([[1., 1.], [0., 0.]]),
                               target + 1., np.array([[0., 1.], [1., 0.]]), target, valid)
    require(pair["intersection_pixels"] == 1 and pair["intersection_sensor_valid_coverage"] == .25
            and pair["GT_camera_depth"]["rmse_m"] == 0 and pair["saved_estimated_camera_depth"]["rmse_m"] == 1,
            "Same-map camera common-C metric labels or denominator changed")
    calls = []
    def fake_loader():
        calls.append(True)
        return "CPU_only"
    for offset in (-1, 0, 1):
        try:
            load_backend(clock=lambda: recovery.DEADLINE + dt.timedelta(microseconds=offset), loader=fake_loader)
            require(offset == -1, "Backend was allowed at/after deadline")
        except ValueError:
            require(offset >= 0, "Backend was blocked before deadline")
    require(len(calls) == 1 and not torch.cuda.is_initialized(), "CPU preflight initialized CUDA or cutoff negative control failed")
    return {"status": "passed", "frozen_GT_helper_coverage_and_renderer_controls": original,
            "OpenCV_world_to_camera_and_wrong_direction_negative_control": True,
            "actual_GT_camera_substitution_rejected_per_map": [r["actual_GT_camera_substitution_rejected"] for r in maps],
            "all_four_first_anchors_identity_and_later_estimated_cameras_differ": True,
            "same_map_camera_pair_common_C_hand_check": True,
            "backend_import_before_at_after_cutoff_controls": True, "GPU_calls": 0}


def prepare():
    require(not OUTPUT.exists() and not PROTOCOL.exists(), "No overwrite of prepared camera-choice evidence")
    OUTPUT.mkdir(exist_ok=False)
    try:
        base = load_base(); module = base.helper()
        gt_protocol, gt_metric, files, sources, render_node, chain, rows = baseline_guards(base, module)
        datasets, inputs, maps, _ = collect_inputs(base, module, gt_protocol, rows)
        controls = cpu_controls(base, module, render_node, maps)
        write_small(OUTPUT / "cpu_preflight.json", {"status": "passed", "at": stamp(), "GPU_calls": 0,
                    "map_count": 4, "requested_new_renders": 20, "controls": controls,
                    "same_map_and_observation_bytes_as_completed_GT20": True,
                    "selected_run_order": [r["run"] for r in maps]})
        historical = {name: recovery.sha(ROOT / name) for name in (
                "outputs/tum-online-v4-estimated-seen-map-evaluation-001/failure.json",
                "outputs/tum-online-v4-estimated-seen-map-evaluation-001/source_script.py")}
        protocol = {"protocol_id": "tum-online-v4-estimated-seen-map-evaluation-protocol-002",
                    "status": "prepared_awaiting_root_review_before_GPU", "prepared_at": stamp(),
                    "deadline_beijing": recovery.DEADLINE.isoformat(), "evaluation_script_sha256": recovery.sha(Path(__file__)),
                    "root_review_relative_path": recovery.relative(OUTPUT / "root_protocol_review.json"),
                    "evaluation_role": "same_frozen_map_camera_choice_seen_training_view_diagnostic",
                    "requested_new_renders": 20, "views_per_map": 5, "fixed_views": base.VIEWS,
                    "source": sources, "source_sha256": {**chain["source_sha256"], **{k:v for k,v in EVIDENCE.items() if k.startswith("scripts/")}},
                    "lineage": chain, "GT_control_evidence_sha256": files,
                    "historical_CPU_evidence_sha256": historical,
                    "previous_CPU_failure": {"reason": "Exact estimated identity is retained; the numerical inverse(GT0)@GT0 reference has tiny floating-point residual, checked at 1e-12 instead of bit-exact zero. Proper-rigid camera tolerance remains unchanged at 2e-6.", "GPU_calls": 0},
                    "GT_control": {"protocol": recovery.relative(GT_PROTOCOL), "protocol_sha256": GT_PROTOCOL_SHA,
                                   "metrics": recovery.relative(GT_FOLDER / "metrics.json"),
                                   "metrics_sha256": EVIDENCE[recovery.relative(GT_FOLDER / "metrics.json")],
                                   "independent_audit": recovery.relative(GT_FOLDER / "independent_full_seen_map_audit.json"),
                                   "independent_audit_sha256": EVIDENCE[recovery.relative(GT_FOLDER / "independent_full_seen_map_audit.json")],
                                   "summary_sha256": EVIDENCE["outputs/tum-v4-full-seen-map-summary-001/summary.json"],
                                   "render_count": 20, "baseline_role": "already completed read-only GT20 NPZ, no rerender"},
                    "datasets": datasets, "maps": maps,
                    "cpu_preflight": {"path": recovery.relative(OUTPUT / "cpu_preflight.json"), "sha256": recovery.sha(OUTPUT / "cpu_preflight.json")},
                    "render_and_metric_settings": {**gt_protocol["render_and_metric_settings"],
                            "camera_choice": "saved accepted estimated_w2c exact float32 bytes from GT-free causal final checkpoint",
                            "camera_c2w_reference": "saved accepted estimated_c2w in original first-camera identity gauge",
                            "GT_world_gauge": "read-only existing GT20 reference: inverse(GT_frame0_c2w) @ GT_view_c2w",
                            "common_C": "within each same map: own_C_existing_GT_camera & own_C_saved_estimated_camera",
                            "new_GT_camera_renders": 0, "new_estimated_camera_renders": 20,
                            "alignment_fit": "none", "map_transform": "none", "optimization_steps": 0, "retries": 0},
                    "comparison_plan": {"units_depth": "meters", "RGB_data_range": 1,
                            "per_view_and_pooled_own_C_and_holes": "retain each camera's distinct denominator and SSE",
                            "common_C": "five same-map paired views per map; exact same sensor V per pair",
                            "pooled_RGB": "sum all requested successful-view RGB SSE and channel count, then compute PSNR; do not average PSNR",
                            "between_map_causal_claim": False, "single_seed_standard_deviation": "not reported",
                            "estimated_camera_lower_loss_interpretation": "in-sample map/camera consistency only; not world geometry or localization accuracy"},
                    "outputs": {"raw_predictions": "raw_predictions/<run>/view_<frameid>.npz",
                            "metrics": "metrics.json with all twenty views and four same-map paired-camera blocks",
                            "failures": "failed view rows retained, no zero fill, no retry"},
                    "limitations": LIMITATIONS}
        require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
        write_small(PROTOCOL, protocol)
        recovery.atomic(OUTPUT / "progress.json", {"status": "awaiting_root_protocol_review", "protocol_sha256": recovery.sha(PROTOCOL), "GPU_calls": 0})
        print(json.dumps({"status": "CPU_prepared_only", "protocol": recovery.relative(PROTOCOL), "protocol_sha256": recovery.sha(PROTOCOL),
                          "script_sha256": protocol["evaluation_script_sha256"], "map_count": 4, "requested_new_renders": 20, "GPU_calls": 0}, indent=2))
    except BaseException as error:
        recovery.write_new(OUTPUT / "failure.json", {"phase": "CPU_prepare", "at": stamp(), "exception_type": type(error).__name__,
                           "message": str(error), "traceback": traceback.format_exc(), "GPU_calls": 0})
        raise


def execute(expected_sha):
    base = load_base(); module = base.helper()
    protocol, _ = module.decode_reviewed_protocol(PROTOCOL.read_bytes(), expected_sha)
    require(protocol["evaluation_script_sha256"] == recovery.sha(Path(__file__)), "Evaluation source changed since review")
    review = recovery.read(ROOT / protocol["root_review_relative_path"])
    require(review["status"] == "passed" and review["protocol_sha256"] == expected_sha
            and review["evaluation_script_sha256"] == protocol["evaluation_script_sha256"], "Exact actual root source/protocol review required")
    recovery.write_new(OUTPUT / "execution_started.json", {"at": stamp(), "protocol_sha256": expected_sha,
                       "root_review_sha256": recovery.sha(ROOT / protocol["root_review_relative_path"])})
    progress = {"status": "running_CPU_guards", "at": stamp(), "protocol_sha256": expected_sha,
                "evaluation_script_sha256": protocol["evaluation_script_sha256"], "GPU_started": False,
                "render_attempts": 0, "maps": [], "same_map_camera_pairs": []}
    try:
        gt_protocol, gt_metric, files, sources, render_node, chain, rows = baseline_guards(base, module)
        require(files == protocol["GT_control_evidence_sha256"] and sources == protocol["source"]
                and chain == protocol["lineage"], "Frozen GT20 source/NPZ/artifact/failure lineage changed")
        recovery.check_hashes(protocol["source_sha256"])
        recovery.check_hashes(protocol["historical_CPU_evidence_sha256"])
        recovery.check_hashes({protocol["cpu_preflight"]["path"]: protocol["cpu_preflight"]["sha256"]})
        datasets, inputs, maps, checkpoints = collect_inputs(base, module, gt_protocol, rows)
        require(datasets == protocol["datasets"] and maps == protocol["maps"], "Actual frozen input/map/accepted camera bytes changed")
        (OUTPUT / "raw_predictions").mkdir(exist_ok=False)
        recovery.atomic(OUTPUT / "progress.json", progress)
        rasterization = load_backend()
        job_start = time.perf_counter()
        for record, checkpoint, gt_map in zip(maps, checkpoints, gt_metric["maps"]):
            scene, run = record["scene"], record["run"]
            if not progress["GPU_started"]:
                # All hash/PNG/checkpoint reads, output creation and progress
                # writes complete before this immediate first-CUDA-copy guard.
                gpu_guard = now()
                require(gpu_guard < recovery.DEADLINE, "Deadline reached before first CUDA copy; do not start")
                progress.update(GPU_started=True, GPU_started_at=stamp(gpu_guard), status="rendering_saved_estimated_cameras")
            parameters = {key:value.to("cuda").detach() for key,value in checkpoint["parameters"].items()}
            require(all(not value.requires_grad for value in parameters.values()), "Map parameters are not frozen")
            K = checkpoint["K"].to("cuda").detach()
            render = module.bind_original_render(render_node, parameters, K, rasterization)
            raw_folder = OUTPUT / "raw_predictions" / run; raw_folder.mkdir(exist_ok=False)
            result = {"run": run, "scene": scene, "prior_weight": record["prior_weight"], "point_count": record["point_count"],
                      "camera_choice": "saved accepted estimated camera", "history": record["history"],
                      "training_failure_frame_count": record["training_failure_frame_count"], "views": [],
                      "parameters_before": record["parameter_tensors"], "saved_camera_tensors_before": record["camera_tensors"],
                      "existing_GT_camera_aggregate": gt_map["aggregate"]}
            paired = []
            for index, frame in enumerate(base.VIEWS[scene]):
                view = {"frame_id": frame, "rgb_timestamp": datasets[scene]["evaluation_sensor_metadata"][index]["rgb_timestamp"],
                        "role": "seen_training_view_diagnostic", "camera_choice": "saved accepted estimated w2c",
                        "saved_accepted_c2w": record["selected_accepted_c2w"][index],
                        "render_saved_accepted_w2c_float32": record["render_saved_accepted_w2c_float32"][index],
                        "same_map_GT_prediction": gt_map["views"][index]["prediction"],
                        "existing_GT_camera_metrics": gt_map["views"][index]["metrics"]}
                progress["render_attempts"] += 1
                try:
                    w2c_array = checkpoint["render_saved_accepted_w2c_float32"][index]
                    w2c = torch.from_numpy(w2c_array).to("cuda")
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.inference_mode():
                        rgb, depth, alpha = render(w2c)
                    torch.cuda.synchronize(); view["render_wall_seconds_including_sync"] = time.perf_counter() - start
                    arrays = (rgb.cpu().numpy(), depth.cpu().numpy(), alpha.cpu().numpy())
                    prediction_path = raw_folder / f"view_{frame:04d}.npz"
                    with prediction_path.open("xb") as stream:
                        np.savez_compressed(stream, rgb=arrays[0], depth_ED_m=arrays[1], alpha=arrays[2])
                    view["prediction"] = {"path": recovery.relative(prediction_path), "sha256": recovery.sha(prediction_path)}
                    view["metrics"] = module.view_metrics(*arrays, inputs[scene]["rgb"][index],
                                        inputs[scene]["depth"][index], inputs[scene]["valid_depth_mask"][index])
                    gt_path = ROOT / gt_map["views"][index]["prediction"]["path"]
                    require(recovery.sha(gt_path) == gt_map["views"][index]["prediction"]["sha256"], "Existing GT NPZ changed")
                    with np.load(gt_path, allow_pickle=False) as gt:
                        pair_value = camera_pair_metrics(module, gt["depth_ED_m"], gt["alpha"], arrays[1], arrays[2],
                                                        inputs[scene]["depth"][index], inputs[scene]["valid_depth_mask"][index])
                    paired.append({"frame_id": frame, "status": "completed", "metrics": pair_value})
                    view["status"] = "completed"
                except Exception as error:
                    view.update(status="failed", exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                    paired.append({"frame_id": frame, "status": "unavailable_due_to_retained_estimated_camera_render_failure"})
                result["views"].append(view)
                recovery.atomic(OUTPUT / "progress.json", {**progress, "current_map": run, "completed_current_map_views": result["views"]})
            result["parameters_after"] = {key:module.array_record(value) for key,value in parameters.items()}
            require(result["parameters_after"] == result["parameters_before"], "Render changed map parameter bytes")
            require(recovery.sha(ROOT / record["checkpoint"]) == record["checkpoint_sha256"], "Original checkpoint changed")
            for name in ("selected_accepted_c2w", "render_saved_accepted_w2c_float32"):
                assert_camera_record(checkpoint[name], record["camera_tensors"][name], module)
            result["parameter_bytes_unchanged"] = True
            result["selected_camera_bytes_unchanged"] = True
            result["aggregate"] = base.aggregate(result["views"], module)
            good = [row["metrics"] for row in paired if row["status"] == "completed"]
            pooled = {"requested_same_map_paired_views": 5, "completed_same_map_paired_views": len(good),
                      "scope": "successful pairs only; every requested failed row retained"}
            if good:
                valid = sum(row["sensor_valid_pixels"] for row in good); common = sum(row["intersection_pixels"] for row in good)
                pooled.update(sensor_valid_pixels=valid, intersection_pixels=common, intersection_sensor_valid_coverage=common/valid,
                              GT_camera_depth=module.sum_depth([row["GT_camera_depth"] for row in good]),
                              saved_estimated_camera_depth=module.sum_depth([row["saved_estimated_camera_depth"] for row in good]))
            progress["maps"].append(result)
            progress["same_map_camera_pairs"].append({"run": run, "scene": scene, "same_map_parameter_bytes": True,
                            "views": paired, "pooled_same_map_common_C": pooled})
            del render, parameters, K
        require(progress["render_attempts"] == 20, "Missing predeclared new render attempts")
        require(base.lineage()[0] == protocol["lineage"], "Final source or original failure lineage changed")
        recovery.check_hashes(protocol["GT_control_evidence_sha256"])
        for scene in base.VIEWS:
            require({k:module.array_record(v) for k,v in inputs[scene].items()} == protocol["datasets"][scene]["input_tensors"],
                    "Observation/reference input bytes changed")
            recovery.check_hashes(protocol["datasets"][scene]["input_file_sha256"])
        failed = sum(v["status"] != "completed" for r in progress["maps"] for v in r["views"])
        progress.update(status="completed" if not failed else "completed_with_render_failures", ended_at=stamp(),
                    failed_render_count=failed, successful_render_count=20-failed, optimization_steps=0, new_GT_camera_renders=0,
                    total_job_seconds_includes_render_metric_and_NPZ_export=time.perf_counter()-job_start,
                    observation_map_camera_and_existing_GT_baseline_bytes_unchanged=True,
                    evaluation_role=protocol["evaluation_role"], limitations=LIMITATIONS)
        write_small(OUTPUT / "metrics.json", progress); recovery.atomic(OUTPUT / "progress.json", progress)
        print(json.dumps({"status": progress["status"], "new_estimated_camera_renders": 20, "failed_renders": failed,
                          "new_GT_camera_renders": 0, "optimization_steps": 0, "metrics_sha256": recovery.sha(OUTPUT / "metrics.json")}, indent=2))
    except BaseException as error:
        progress.update(status="failed", failed_at=stamp(), exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        recovery.atomic(OUTPUT / "progress.json", progress); recovery.write_new(OUTPUT / "failure.json", progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true"); group.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha")
    args = parser.parse_args()
    if args.prepare:
        if args.protocol_sha: parser.error("Protocol SHA only applies to execute")
        prepare()
    else:
        if not args.protocol_sha: parser.error("Exact reviewed protocol SHA required")
        execute(args.protocol_sha)


if __name__ == "__main__":
    main()
