"""Frozen V4 final maps at ten future known cameras; explicit CPU/GPU stages.

This is map-quality evaluation, not tracking or an untouched final test. The
check-only path never imports gsplat or initializes CUDA. It can record pending
training runs; execution requires a complete, hash-frozen CPU protocol.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 160, 120
MIN_DEPTH, MAX_DEPTH, ALPHA_THRESHOLD = .2, 5., .5
TRAIN_IDS = list(range(200, 300))
EVAL_IDS = list(range(300, 310))
DEADLINE = datetime.datetime.fromisoformat("2026-10-09T08:00:00+08:00")
SOURCE_HASHES = {
    "train_tum_online_gs_v4.py": "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f",
    "tum_rgbd.py": "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11",
    "track_rgbd_icp.py": "21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a",
    "track_fixed_map.py": "c6f8ca3e80747decc7805aa567db44179f5a6e17b8ee7c36ef8565273463fdab",
    "geometric_pose_prior.py": "d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b",
}
EXPECTED_VERSIONS = {"torch": "2.7.1+cu128", "gsplat": "1.5.3"}
PARAMETER_SHAPES = {"means": (3,), "log_scales": (3,), "quats": (4,),
                    "color_logits": (3,), "opacity_logits": ()}
COMPLETE = {"completed", "completed_with_tracking_or_mapping_failures"}
LIMITS = [
    "Known-camera final-map evaluation only; no tracking, map optimization, ICP, or hyperparameter selection.",
    "Future associated frames 300..309 are disjoint from training 200..299 but share each sequence/domain; not an untouched final test or cross-scene generalization.",
    "Cameras use inverse(GT frame 200 c2w) @ GT eval c2w. No SE3/Sim3 alignment is fitted and no map is transformed.",
    "TUM GT is nearest at RGB time, not interpolated; RGB/depth time offsets are not motion compensated.",
    "Registered RGB-D and Freiburg1 intrinsics are assumed; no distortion correction or physical calibration validation.",
    "RGB+ED is the frozen gsplat V4 renderer; ED is conditional expected depth, not accumulated depth.",
    "Covered depth is conditional on alpha > 0.5 and positive finite ED. Holes-as-zero and common-coverage metrics expose different denominators.",
    "Rendering can retain GPU floating-point nondeterminism; unchanged model bytes do not guarantee bitwise render repeatability.",
    "All twelve specified maps and failed views remain in outputs. Partial summaries identify their successful-view denominator.",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8"))


def array_record(value) -> dict:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.nbytes, "sha256": sha(value.tobytes())}


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def decode_reviewed_protocol(data: bytes, expected_sha: str) -> tuple[dict, str]:
    actual_sha = sha(data)
    require(re.fullmatch(r"[a-f0-9]{64}", expected_sha) is not None and actual_sha == expected_sha,
            "Protocol differs from the exact reviewed SHA256")
    return json.loads(data.decode("utf-8-sig")), actual_sha


def require_before_deadline(now=None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    require(now.tzinfo is not None and now < DEADLINE, "No new GPU experiment may start at or after the fixed deadline")
    return now.isoformat()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def source_checks() -> tuple[dict, ast.FunctionDef]:
    actual = {name: file_sha(ROOT / "scripts" / name) for name in SOURCE_HASHES}
    require(actual == SOURCE_HASHES, "Frozen source hash mismatch")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    require(versions == EXPECTED_VERSIONS, "Renderer dependency versions changed")
    source_path = ROOT / "scripts/train_tum_online_gs_v4.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    render_nodes = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "render"]
    require(len(render_nodes) == 1, "Expected exactly one original render function")
    render = render_nodes[0]
    return {"source_sha256": actual, "versions": versions,
            "render_ast_sha256": sha(ast.dump(render, include_attributes=False).encode()),
            "render_source_lines": [render.lineno, render.end_lineno],
            "render_ast_is_unmodified_nested_v4_render": True,
            "evaluation_script_sha256": file_sha(Path(__file__))}, render


def run_specs() -> list[dict]:
    specs = []
    for scene in ("desk", "xyz"):
        for seed in (0, 1, 2):
            for weight, token in ((0., "0"), (.01, "001")):
                if seed == 0:
                    prefix = "tum-online" if scene == "desk" else "tum-xyz-online"
                    name = f"{prefix}-v4-prior{token}-transfer100-001"
                else:
                    name = f"tum-{scene}-online-v4-prior{token}-seed{seed}-transfer100-001"
                specs.append({"run": name, "scene": scene, "seed": seed, "prior_weight": weight})
    return specs


def rigid_check(poses: np.ndarray, tolerance=2e-6) -> dict:
    poses = np.asarray(poses, dtype=np.float64)
    require(poses.shape[-2:] == (4, 4) and np.isfinite(poses).all(), "Invalid finite camera shape")
    rotations = poses[..., :3, :3]
    ortho = float(np.max(np.abs(np.swapaxes(rotations, -2, -1) @ rotations - np.eye(3))))
    determinant = float(np.max(np.abs(np.linalg.det(rotations) - 1.)))
    bottom = float(np.max(np.abs(poses[..., 3, :] - [0, 0, 0, 1])))
    require(max(ortho, determinant, bottom) < tolerance, "Camera is not proper rigid")
    return {"maximum_orthogonality_element_error": ortho,
            "maximum_determinant_error": determinant, "maximum_bottom_row_error": bottom,
            "absolute_tolerance": tolerance}


def sensor_metadata(frame) -> dict:
    return {key: value for key, value in frame.metadata().items() if key not in ("c2w", "w2c")}


def prepare_dataset(scene: str) -> tuple[dict, dict]:
    # Import only after source_checks has validated this exact frozen loader.
    from tum_rgbd import associate_frames, load_frames, safe_data_path, scaled_intrinsics
    dataset = ROOT / f"data/tum/rgbd_dataset_freiburg1_{scene}"
    frames, association = associate_frames(dataset, .02, .02, 32.)
    require(len(frames) > EVAL_IDS[-1], "Insufficient associated frames")
    train = [frames[i] for i in TRAIN_IDS]
    evaluation = [frames[i] for i in EVAL_IDS]
    isolation = {}
    for field in ("frame_id", "rgb_index", "depth_index", "rgb_path", "depth_path"):
        a, b = {getattr(f, field) for f in train}, {getattr(f, field) for f in evaluation}
        require(not a & b and len(a) == 100 and len(b) == 10, f"Training/evaluation reuse: {field}")
        isolation[field] = {"training_unique": len(a), "evaluation_unique": len(b), "overlap": 0}
    images, depths, masks = load_frames(dataset, evaluation, WIDTH, HEIGHT, MIN_DEPTH, MAX_DEPTH)
    K = scaled_intrinsics(WIDTH, HEIGHT)
    anchor = train[0].c2w
    absolute = np.stack([frame.c2w for frame in evaluation])
    c2w = np.linalg.inv(anchor)[None] @ absolute
    w2c = np.linalg.inv(c2w)
    camera_checks = {"anchor": rigid_check(anchor, 1e-10),
                     "relative_c2w_float64": rigid_check(c2w, 1e-10),
                     "relative_w2c_float64": rigid_check(w2c, 1e-10),
                     "render_w2c_float32": rigid_check(w2c.astype(np.float32))}
    require(np.allclose(anchor[None] @ c2w, absolute, atol=1e-12, rtol=0), "GT relative gauge mismatch")
    require(images.shape == (10, HEIGHT, WIDTH, 3) and depths.shape == masks.shape == (10, HEIGHT, WIDTH),
            "Evaluation sensor shape mismatch")
    file_records = []
    for frame in train + evaluation:
        row = {"frame_id": frame.frame_id}
        for field in ("rgb_path", "depth_path"):
            filename = safe_data_path(dataset, getattr(frame, field))
            row[field] = {"path": getattr(frame, field), "sha256": file_sha(filename), "bytes": filename.stat().st_size}
        file_records.append(row)
    tensors = {"rgb": images, "depth": depths, "valid_depth_mask": masks,
               "K_float64": K, "K_render_float32": K.astype(np.float32)[None],
               "anchor_gt_c2w_float64": anchor, "evaluation_gt_c2w_absolute_float64": absolute,
               "evaluation_c2w_relative_float64": c2w, "evaluation_w2c_float64": w2c,
               "evaluation_w2c_render_float32": w2c.astype(np.float32)}
    record = {"scene": scene, "dataset": relative(dataset), "association": association,
              "training_frame_ids": TRAIN_IDS, "evaluation_frame_ids": EVAL_IDS,
              "training_sensor_metadata": [sensor_metadata(frame) for frame in train],
              "evaluation_sensor_metadata": [sensor_metadata(frame) for frame in evaluation],
              "train_eval_isolation": isolation, "raw_sensor_files": file_records,
              "index_file_sha256": {name: file_sha(dataset / name) for name in ("rgb.txt", "depth.txt", "groundtruth.txt")},
              "tensors": {name: array_record(value) for name, value in tensors.items()},
              "K": K.tolist(), "anchor_gt_c2w": anchor.tolist(),
              "evaluation_relative_c2w": c2w.tolist(), "evaluation_relative_w2c": w2c.tolist(),
              "camera_checks": camera_checks,
              "valid_depth_pixels_per_view": [int(mask.sum()) for mask in masks],
              "rgb_depth_abs_dt_s": [abs(f.rgb_timestamp-f.depth_timestamp) for f in evaluation],
              "rgb_gt_abs_dt_s": [abs(f.rgb_timestamp-f.gt_timestamp) for f in evaluation]}
    return record, tensors


def expected_config(spec: dict) -> dict:
    return {"dataset": (ROOT / f"data/tum/rgbd_dataset_freiburg1_{spec['scene']}").as_posix(),
            "run": spec["run"], "frames": 100, "start_frame": 200, "frame_stride": 1,
            "width": WIDTH, "height": HEIGHT, "max_association_dt": .02, "max_gt_dt": .02,
            "min_depth": MIN_DEPTH, "max_depth": MAX_DEPTH, "initial_points": 15000,
            "max_points": 60000, "initial_pixel_stride": 1, "new_point_stride": 2,
            "max_new_points": 3000, "warmup_steps": 150, "tracking_steps": 100,
            "mapping_steps": 80, "keyframe_interval": 4, "keyframe_window": 4,
            "pose_initializer": "icp-prefix", "tracking_mode": "optimize",
            "pose_prior_weight": spec["prior_weight"], "pose_prior_translation_sigma_m": .01,
            "pose_prior_rotation_sigma_deg": 1., "pose_lr_translation": .003, "pose_lr_rotation": .001,
            "depth_weight": 1., "alpha_weight": .01, "alpha_threshold": ALPHA_THRESHOLD,
            "failure_min_coverage": .3, "failure_max_step_m": .2, "failure_max_step_deg": 20.,
            "seed": spec["seed"], "check_only": False}


def completed_driver_witness() -> dict:
    """Require all eight child processes to have exited, not just written metrics."""
    ledger_path = ROOT / "outputs/tum-online-v4-seed-replication-001/metrics.json"
    driver_protocol_path = ROOT / "manifests/tum-online-v4-seed-replication-protocol-001.json"
    if not ledger_path.exists():
        return {"status": "pending", "reason": "Final eight-run driver ledger is not present"}
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    if ledger.get("status") != "all_eight_predeclared_runs_completed":
        return {"status": "pending", "ledger_status": ledger.get("status"), "ledger_sha256": sha(ledger_bytes)}
    driver_protocol_bytes = driver_protocol_path.read_bytes()
    driver_protocol = json.loads(driver_protocol_bytes)
    require(ledger["protocol_sha256"] == sha(driver_protocol_bytes), "Driver protocol witness changed")
    require(file_sha(ROOT / "scripts/run_v4_seed_replication.py") == driver_protocol["driver_sha256"], "Driver source changed")
    require(file_sha(ROOT / "scripts/run_local.py") == driver_protocol["wrapper_sha256"], "Training launch wrapper changed")
    specs = {s["run"]: s for s in run_specs() if s["seed"] != 0}
    records = ledger["run_records"]
    require(len(records) == 8 and len({r["run"] for r in records}) == 8 and
            {r["run"] for r in records} == set(specs), "Driver omitted or replaced a predefined run")
    require(ledger["GPU_runs_started"] == 8 and ledger.get("current_run") is None and ledger.get("ended_at"),
            "Driver ledger is not terminal")
    checked = []
    for record in records:
        require(record["returncode"] == 0 and record["status"] in COMPLETE and record.get("ended_at"),
                "A training child did not exit0 with completed status")
        require(record["configuration"] == expected_config(specs[record["run"]]), "Driver configuration differs")
        metrics_sha = file_sha(ROOT / "outputs" / record["run"] / "metrics.json")
        require(metrics_sha == record["metrics_sha256"], "Training metrics differ from terminal driver witness")
        checked.append({"run": record["run"], "returncode": 0, "status": record["status"],
                        "ended_at": record["ended_at"], "metrics_sha256": metrics_sha,
                        "configuration_sha256": canonical_sha(record["configuration"])})
    require(file_sha(ledger_path) == sha(ledger_bytes), "Driver ledger changed during read")
    return {"status": "ready", "ledger": relative(ledger_path), "ledger_sha256": sha(ledger_bytes),
            "driver_protocol": relative(driver_protocol_path), "driver_protocol_sha256": sha(driver_protocol_bytes),
            "driver_script_sha256": driver_protocol["driver_sha256"],
            "launch_wrapper_sha256": driver_protocol["wrapper_sha256"],
            "all_eight_child_processes_exit0": True, "seed0_existing_runs_do_not_require_this_driver": True,
            "records": checked}


def inspect_completed_run(spec: dict, dataset_record: dict) -> dict:
    directory = ROOT / "outputs" / spec["run"]
    row = {**spec, "status": "pending", "checkpoint": relative(directory / "last_checkpoint.pt"),
           "configuration_expected": expected_config(spec), "checkpoint_sha256": None}
    metrics_path, checkpoint_path = directory / "metrics.json", directory / "last_checkpoint.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        row["reason"] = "Completed metrics and last_checkpoint are not both present"
        row["failure_record_present"] = (directory / "failure.json").exists()
        return row
    metrics = read_json(metrics_path)
    row["training_status"] = metrics.get("status")
    if metrics.get("status") not in COMPLETE:
        row["reason"] = "Training metrics are not completed"
        return row
    checkpoint_sha = file_sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("status") not in COMPLETE:
        row["reason"] = "Final completed checkpoint has not been persisted yet"
        return row
    require(file_sha(checkpoint_path) == checkpoint_sha, "Checkpoint changed during CPU read")
    require(checkpoint.get("ground_truth_included") is False and
            not any("evaluation_only_gt" in key for key in checkpoint), "Checkpoint contains evaluation GT")
    for record in (metrics, checkpoint):
        require(record["configuration"] == expected_config(spec), f"Configuration differs: {spec['run']}")
        require(record["script_sha256"] == SOURCE_HASHES["train_tum_online_gs_v4.py"], "Training source mismatch")
        require(record["implementation_version"] == 4, "Unexpected implementation version")
    require(metrics["loader_sha256"] == SOURCE_HASHES["tum_rgbd.py"], "Training loader mismatch")
    require(metrics["pose_helper_sha256"] == SOURCE_HASHES["track_fixed_map.py"], "Pose helper mismatch")
    require(metrics["pose_prior_helper_sha256"] == SOURCE_HASHES["geometric_pose_prior.py"], "Prior helper mismatch")
    require(metrics["icp_initializer"]["helper_sha256"] == SOURCE_HASHES["track_rgbd_icp.py"], "ICP helper mismatch")
    require(metrics["selected_frame_ids"] == checkpoint["frame_ids_with_stored_pose"] == TRAIN_IDS,
            "Training frames differ from fixed 200..299")
    require(metrics["selected_sensor_metadata"] == dataset_record["training_sensor_metadata"],
            "Training association, timestamps, or sensor paths changed")
    require(metrics["frames_processed"] == 100 and checkpoint["current_local_index"] == 99,
            "Checkpoint is not the completed 100-frame state")
    require([r["frame_id"] for r in checkpoint["completed_frame_metrics"]] == TRAIN_IDS,
            "Missing or reordered stored failed/accepted frames")
    parameters = checkpoint["parameters"]
    require(set(parameters) == set(PARAMETER_SHAPES), "Unexpected Gaussian parameter fields")
    n = len(parameters["means"])
    require(n == metrics["final_point_count"] and 0 < n <= 60000, "Invalid final point count")
    for key, tail in PARAMETER_SHAPES.items():
        tensor = parameters[key]
        require(tensor.device.type == "cpu" and tensor.dtype == torch.float32 and
                tuple(tensor.shape) == (n, *tail) and torch.isfinite(tensor).all().item(),
                f"Invalid Gaussian tensor: {key}")
    require(torch.all(torch.linalg.vector_norm(parameters["quats"], dim=-1) > 1e-12).item(), "Zero quaternion")
    scales = parameters["log_scales"].exp()
    require(torch.isfinite(scales).all().item() and (scales > 0).all().item(), "Invalid activated scale")
    source_ids = np.asarray(checkpoint["source_frame_ids_per_gaussian"], dtype=np.int64)
    require(source_ids.shape == (n,) and set(source_ids).issubset(TRAIN_IDS) and not set(source_ids) & set(EVAL_IDS),
            "Gaussian source pixels include evaluation or nontraining frames")
    K = np.asarray(dataset_record["K"], dtype=np.float32)[None]
    require(np.array_equal(checkpoint["K"].numpy(), K), "Actual training and evaluation render K differ")
    cameras = {key: rigid_check(checkpoint[key].numpy()) for key in ("estimated_c2w", "estimated_w2c")}
    require(all(checkpoint[key].shape == (100, 4, 4) for key in cameras), "Stored camera count differs")
    require(np.array_equal(checkpoint["estimated_c2w"][0].numpy(), np.eye(4)), "First camera is not identity")
    unique, counts = np.unique(source_ids, return_counts=True)
    row.update(status="ready", checkpoint_sha256=checkpoint_sha, checkpoint_bytes=checkpoint_path.stat().st_size,
               metrics_sha256=file_sha(metrics_path), configuration_sha256=file_sha(directory / "configuration.json"),
               ground_truth_in_checkpoint=False, point_count=n,
               parameter_tensors={key: array_record(value) for key, value in parameters.items()},
               K_tensor=array_record(checkpoint["K"]), source_frame_ids_tensor=array_record(source_ids),
               source_frame_histogram={str(k): int(v) for k, v in zip(unique, counts)},
               source_gaussian_evaluation_frame_overlap=0, camera_checks=cameras,
               training_failure_frame_count=metrics["failure_frame_count"],
               training_icp_failure_count=metrics["icp_failed_frame_count"],
               all_training_failure_frames_retained=True)
    return row


def depth_sums(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    error = pred[mask].astype(np.float64) - target[mask].astype(np.float64)
    return {"pixels": int(mask.sum()), "absolute_error_sum_m": float(np.abs(error).sum()),
            "squared_error_sum_m2": float(np.dot(error, error))}


def depth_values(sums: dict) -> dict:
    n = sums["pixels"]
    return {**sums, "mae_m": sums["absolute_error_sum_m"] / n if n else None,
            "rmse_m": math.sqrt(sums["squared_error_sum_m2"] / n) if n else None}


def psnr(mse: float) -> dict:
    return {"mse": mse, "psnr_db": -10. * math.log10(mse) if mse > 0 else None,
            "psnr_positive_infinity": mse == 0}


def coverage_mask(depth: np.ndarray, alpha: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return valid & (alpha > ALPHA_THRESHOLD) & np.isfinite(depth) & (depth > 0)


def view_metrics(rgb, depth, alpha, observed_rgb, observed_depth, valid) -> dict:
    require(rgb.shape == observed_rgb.shape and depth.shape == alpha.shape == observed_depth.shape == valid.shape,
            "Predicted and observed shapes differ")
    require(all(np.isfinite(x).all() for x in (rgb, depth, alpha, observed_rgb, observed_depth)),
            "Nonfinite predicted or observed values")
    require(int(valid.sum()) > 0, "No valid depth observations")
    require(float(alpha.min()) >= -1e-6 and float(alpha.max()) <= 1. + 1e-6, "Alpha out of range")
    covered = coverage_mask(depth, alpha, valid)
    rgb_error = rgb.astype(np.float64) - observed_rgb.astype(np.float64)
    rgb_sse = float(np.square(rgb_error).sum())
    holes_as_zero = np.where(covered, depth, 0.)
    return {"rgb": {"squared_error_sum": rgb_sse, "channel_values": rgb_error.size,
                    **psnr(rgb_sse / rgb_error.size)},
            "valid_depth_pixels": int(valid.sum()), "covered_valid_pixels": int(covered.sum()),
            "sensor_valid_coverage": float(covered.sum() / valid.sum()),
            "covered_depth": depth_values(depth_sums(depth, observed_depth, covered)),
            "all_valid_holes_as_zero_depth": depth_values(depth_sums(holes_as_zero, observed_depth, valid))}


def sum_depth(records: list[dict]) -> dict:
    return depth_values({key: sum(r[key] for r in records)
                         for key in ("pixels", "absolute_error_sum_m", "squared_error_sum_m2")})


def mean_defined(values: list) -> dict:
    defined = [value for value in values if value is not None]
    return {"value": float(np.mean(defined)) if defined else None,
            "defined_view_count": len(defined), "undefined_view_count": len(values)-len(defined)}


def aggregate_views(rows: list[dict]) -> dict:
    good = [row["metrics"] for row in rows if row.get("status") == "completed"]
    result = {"requested_views": len(EVAL_IDS), "completed_views": len(good),
              "failed_views": sum(row.get("status") == "failed" for row in rows),
              "not_completed_views": len(EVAL_IDS)-len(good),
              "summary_scope": "successful views only; missing/failed views are not filled with zero"}
    if not good:
        return result
    rgb_sse = sum(v["rgb"]["squared_error_sum"] for v in good)
    rgb_count = sum(v["rgb"]["channel_values"] for v in good)
    valid = sum(v["valid_depth_pixels"] for v in good)
    covered = sum(v["covered_valid_pixels"] for v in good)
    result["pooled"] = {"rgb": {"squared_error_sum": rgb_sse, "channel_values": rgb_count, **psnr(rgb_sse/rgb_count)},
                        "sensor_valid_pixels": valid, "covered_valid_pixels": covered,
                        "sensor_valid_coverage": covered/valid,
                        "covered_depth": sum_depth([v["covered_depth"] for v in good]),
                        "all_valid_holes_as_zero_depth": sum_depth([v["all_valid_holes_as_zero_depth"] for v in good])}
    psnr_infinity = any(v["rgb"]["psnr_positive_infinity"] for v in good)
    result["mean_per_view"] = {
        "rgb_mse": float(np.mean([v["rgb"]["mse"] for v in good])),
        "rgb_psnr_db": None if psnr_infinity else float(np.mean([v["rgb"]["psnr_db"] for v in good])),
        "rgb_psnr_positive_infinity": psnr_infinity,
        "sensor_valid_coverage": float(np.mean([v["sensor_valid_coverage"] for v in good])),
        "covered_depth_mae_m": mean_defined([v["covered_depth"]["mae_m"] for v in good]),
        "covered_depth_rmse_m": mean_defined([v["covered_depth"]["rmse_m"] for v in good]),
        "all_valid_holes_as_zero_rmse_m": float(np.mean([v["all_valid_holes_as_zero_depth"]["rmse_m"] for v in good]))}
    return result


def paired_metrics(depth_a, alpha_a, depth_b, alpha_b, target, valid) -> dict:
    common = coverage_mask(depth_a, alpha_a, valid) & coverage_mask(depth_b, alpha_b, valid)
    a, b = depth_values(depth_sums(depth_a, target, common)), depth_values(depth_sums(depth_b, target, common))
    return {"sensor_valid_pixels": int(valid.sum()), "intersection_pixels": int(common.sum()),
            "intersection_sensor_valid_coverage": float(common.sum()/valid.sum()),
            "prior0_depth": a, "prior001_depth": b,
            "mae_difference_prior001_minus_prior0_m": b["mae_m"]-a["mae_m"] if a["pixels"] else None,
            "rmse_difference_prior001_minus_prior0_m": b["rmse_m"]-a["rmse_m"] if a["pixels"] else None}


def cpu_controls(render_node) -> dict:
    # Unequal coverage counts make per-view and pooled depth RMSE differ.
    target = np.array([[1., 2.], [3., 4.]])
    rgb_target = np.zeros((2, 2, 3))
    valid = np.ones((2, 2), dtype=bool)
    depth_a = np.array([[1.1, 9.], [2.8, 4.]])
    alpha_a = np.array([[1., .5], [.8, .2]])
    depth_b = np.array([[1.2, 2.3], [3.4, 4.5]])
    alpha_b = np.ones((2, 2))
    a = view_metrics(np.full((2, 2, 3), .1), depth_a, alpha_a, rgb_target, target, valid)
    b = view_metrics(np.full((2, 2, 3), .2), depth_b, alpha_b, rgb_target, target, valid)
    require(a["covered_valid_pixels"] == 2, "Alpha threshold must be strict > .5")
    require(abs(a["covered_depth"]["squared_error_sum_m2"]-.05) < 1e-12, "Covered SSE control failed")
    require(abs(a["all_valid_holes_as_zero_depth"]["squared_error_sum_m2"]-20.05) < 1e-12,
            "Holes-as-zero control failed")
    aggregate = aggregate_views([{"status": "completed", "metrics": a}, {"status": "completed", "metrics": b}])
    require(abs(aggregate["pooled"]["rgb"]["mse"]-.025) < 1e-12, "RGB pooled control failed")
    require(abs(aggregate["pooled"]["covered_depth"]["rmse_m"]-math.sqrt(.59/6)) < 1e-12,
            "Covered pooled denominator control failed")
    require(abs(aggregate["pooled"]["rgb"]["psnr_db"]-aggregate["mean_per_view"]["rgb_psnr_db"]) > .9,
            "PSNR mean-vs-pool negative control failed")
    joint = paired_metrics(depth_a, alpha_a, depth_b, alpha_b, target, valid)
    require(joint["intersection_pixels"] == 2 and abs(joint["prior001_depth"]["squared_error_sum_m2"]-.2) < 1e-12,
            "Paired common coverage control failed")
    empty = view_metrics(rgb_target, depth_a, np.zeros((2, 2)), rgb_target, target, valid)
    require(empty["covered_depth"]["rmse_m"] is None and empty["rgb"]["psnr_positive_infinity"],
            "Empty coverage / exact RGB control failed")
    require(empty["all_valid_holes_as_zero_depth"]["squared_error_sum_m2"] == 30., "Empty holes control failed")
    # A noncommuting anchor checks the direction of the known-camera gauge.
    from scipy.spatial.transform import Rotation
    anchor, pose = np.eye(4), np.eye(4)
    anchor[:3, :3] = Rotation.from_euler("xyz", [20, -10, 5], degrees=True).as_matrix()
    anchor[:3, 3] = [.4, -.2, .7]
    pose[:3, :3] = Rotation.from_euler("xyz", [-6, 12, 30], degrees=True).as_matrix()
    pose[:3, 3] = [-.1, .6, .3]
    relative_pose = np.linalg.inv(anchor) @ pose
    correct_w2c = np.linalg.inv(pose) @ anchor
    require(np.max(np.abs(np.linalg.inv(relative_pose)-correct_w2c)) < 1e-12, "Gauge direction control failed")
    wrong_order_error = float(np.max(np.abs(pose @ np.linalg.inv(anchor)-relative_pose)))
    require(wrong_order_error > .1, "Gauge direction negative control was not discriminating")
    binding = renderer_binding_cpu_control(render_node)
    require_before_deadline(DEADLINE-datetime.timedelta(microseconds=1))
    for clock in (DEADLINE, DEADLINE+datetime.timedelta(microseconds=1)):
        try:
            require_before_deadline(clock)
        except ValueError:
            pass
        else:
            raise ValueError("Deadline equality/after negative control failed")
    sample_protocol = b'{"status":"cpu-control"}\n'
    parsed, bound_sha = decode_reviewed_protocol(sample_protocol, sha(sample_protocol))
    require(parsed["status"] == "cpu-control" and bound_sha == sha(sample_protocol), "Protocol SHA positive control failed")
    try:
        decode_reviewed_protocol(sample_protocol+b" ", sha(sample_protocol))
    except ValueError:
        pass
    else:
        raise ValueError("Protocol byte-change negative control failed")
    return {"status": "passed", "scope": "CPU aggregation, mask denominators, null handling, rigid camera algebra and mocked renderer binding; no rasterization",
            "alpha_equal_threshold_is_uncovered": True, "known_covered_sse_m2": .05,
            "known_holes_as_zero_sse_m2": 20.05, "pooled_rgb_mse": aggregate["pooled"]["rgb"]["mse"],
            "pooled_rgb_psnr_db": aggregate["pooled"]["rgb"]["psnr_db"],
            "mean_per_view_rgb_psnr_db": aggregate["mean_per_view"]["rgb_psnr_db"],
            "unequal_coverage_pooled_depth_rmse_m": aggregate["pooled"]["covered_depth"]["rmse_m"],
            "common_coverage_pixels": joint["intersection_pixels"], "empty_coverage_is_null_not_zero": True,
            "wrong_gauge_order_max_error": wrong_order_error, "renderer_binding": binding,
            "deadline_before_equal_after_controls_passed": True,
            "protocol_exact_bytes_and_changed_bytes_controls_passed": True}


def protocol_settings() -> dict:
    return {"resolution": [WIDTH, HEIGHT], "sensor_valid_depth_m": [MIN_DEPTH, MAX_DEPTH],
            "sensor_depth_endpoints_inclusive": True, "alpha_threshold_strictly_greater": ALPHA_THRESHOLD,
            "render_mode": "RGB+ED", "near_plane_m": .1, "far_plane_m": 6., "background_rgb": [0, 0, 0],
            "training_frame_ids": TRAIN_IDS, "evaluation_frame_ids": EVAL_IDS,
            "anchor_frame_id": 200, "camera_gauge_formula": "inverse(GT_c2w_200) @ GT_c2w_eval",
            "camera_alignment_fitted": False, "rgb_clipped_before_metrics": False,
            "metric_accumulation_dtype": "float64 on CPU", "psnr_data_range": 1.,
            "holes_as_zero": "For each sensor-valid pixel outside covered mask, replace ED by zero before error",
            "psnr_zero_mse_encoding": "psnr_db null with psnr_positive_infinity true",
            "empty_coverage_encoding": "MAE/RMSE null, pixel count 0; never substitute zero error",
            "no_new_gpu_experiment_at_or_after": DEADLINE.isoformat(),
            "deadline_scope": "One experiment with all twelve maps/120 renders; if already begun before deadline, collect it without starting another experiment"}


def prepare_protocol(output: Path) -> dict:
    require(not torch.cuda.is_initialized(), "CPU preparation started after CUDA initialization")
    sources, render_node = source_checks()
    controls = cpu_controls(render_node)
    datasets = {scene: prepare_dataset(scene)[0] for scene in ("desk", "xyz")}
    driver = completed_driver_witness()
    rows = []
    for spec in run_specs():
        try:
            rows.append(inspect_completed_run(spec, datasets[spec["scene"]]))
        except Exception as error:
            rows.append({**spec, "status": "invalid", "reason": str(error), "exception_type": type(error).__name__})
    ready = sum(row["status"] == "ready" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    status = "ready_for_explicit_gpu_execution" if ready == 12 and driver["status"] == "ready" else "invalid_inputs" if invalid else "pending_training_or_driver_completion"
    result = {"schema_version": 1, "status": status, "purpose": "final-map future-known-camera quality evaluation",
              "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sources": sources,
              "settings": protocol_settings(), "cpu_controls": controls, "datasets": datasets, "runs": rows,
              "completed_training_driver": driver,
              "ready_runs": ready, "pending_runs": 12-ready-invalid, "invalid_runs": invalid,
              "expected_map_count": 12, "expected_render_count": 120, "cuda_initialized": torch.cuda.is_initialized(),
              "no_gpu_render_executed": True, "limitations": LIMITS}
    require(not result["cuda_initialized"], "CPU path initialized CUDA")
    atomic_json(output / "protocol.json", result)
    return result


def bind_original_render(render_node, parameters, K_gpu, rasterization):
    namespace = {"torch": torch, "rasterization": rasterization,
                 "map_parameters": parameters, "K_gpu": K_gpu,
                 "args": SimpleNamespace(width=WIDTH, height=HEIGHT, max_depth=MAX_DEPTH)}
    # No edits to the extracted function body, arguments, or call keywords.
    module = ast.Module(body=[copy.deepcopy(render_node)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<frozen-v4-render-ast>", "exec"), namespace)
    return namespace["render"]


def renderer_binding_cpu_control(render_node) -> dict:
    """Run original AST against CPU stand-ins, checking its API, not CUDA math."""
    parameters = {"means": torch.tensor([[0., 0., 1.], [.1, .2, 2.]]),
                  "quats": torch.tensor([[2., 0., 0., 0.], [0., 0., 3., 0.]]),
                  "log_scales": torch.log(torch.full((2, 3), .1)),
                  "opacity_logits": torch.zeros(2), "color_logits": torch.zeros((2, 3))}
    K = torch.tensor([[[129.325, 0., 79.275], [0., 129.125, 63.45], [0., 0., 1.]]])
    calls = []
    def cpu_zeros(*args, **kwargs):
        require(kwargs == {"device": "cuda"}, "Original renderer background allocation changed")
        return torch.zeros(*args, device="cpu")
    def rasterization_stub(*args, **kwargs):
        require(len(args) == 9 and args[-2:] == (WIDTH, HEIGHT), "Renderer positional API changed")
        require(kwargs["render_mode"] == "RGB+ED" and kwargs["packed"] is False and
                kwargs["near_plane"] == .1 and kwargs["far_plane"] == 6., "Renderer keyword API changed")
        require(tuple(args[5].shape) == (1, 4, 4) and args[5].is_contiguous(), "Bound camera shape wrong")
        require(torch.equal(args[6], K), "Bound camera intrinsics changed")
        require(torch.allclose(torch.linalg.vector_norm(args[1], dim=-1), torch.ones(2)), "Quaternion not normalized")
        require(torch.allclose(args[2], torch.full((2, 3), .1)) and
                torch.equal(args[3], torch.full((2,), .5)) and torch.equal(args[4], torch.full((2, 3), .5)),
                "Parameter activation binding wrong")
        require(tuple(kwargs["backgrounds"].shape) == (1, 3) and kwargs["backgrounds"].count_nonzero() == 0,
                "Background binding wrong")
        calls.append(True)
        rgb_ed = torch.zeros((1, HEIGHT, WIDTH, 4))
        rgb_ed[..., 0], rgb_ed[..., 1], rgb_ed[..., 2], rgb_ed[..., 3] = .1, .2, .3, 2.
        return rgb_ed, torch.full((1, HEIGHT, WIDTH, 1), .75), {}
    render = bind_original_render(render_node, parameters, K, rasterization_stub)
    render.__globals__["torch"] = SimpleNamespace(Tensor=torch.Tensor, nn=torch.nn, zeros=cpu_zeros)
    before = {key: array_record(value) for key, value in parameters.items()}
    with torch.no_grad():
        rgb, depth, alpha = render(torch.eye(4))
    require(len(calls) == 1 and tuple(rgb.shape) == (HEIGHT, WIDTH, 3) and
            torch.all(depth == 2.) and torch.all(alpha == .75), "Renderer output slicing wrong")
    require(before == {key: array_record(value) for key, value in parameters.items()}, "CPU stand-in mutated parameters")
    require(not torch.cuda.is_initialized(), "CPU binding check initialized CUDA")
    return {"status": "passed", "rasterizer": "CPU stand-in, not gsplat", "render_ast_unmodified": True,
            "cuda_background_allocation_intercepted_for_cpu_control_only": True,
            "positional_and_keyword_api_activations_and_output_slicing_checked": True,
            "parameter_bytes_unchanged": True, "cuda_initialized": False}


def build_pairs(run_results: list[dict], inputs: dict, output: Path) -> list[dict]:
    by_run = {row["run"]: row for row in run_results}
    pairs = []
    for scene in ("desk", "xyz"):
        for seed in (0, 1, 2):
            names = [spec["run"] for spec in run_specs() if spec["scene"] == scene and spec["seed"] == seed]
            a, b = [by_run[name] for name in names]
            views = []
            for i, frame_id in enumerate(EVAL_IDS):
                ra = next((v for v in a["views"] if v["frame_id"] == frame_id), None)
                rb = next((v for v in b["views"] if v["frame_id"] == frame_id), None)
                if not a.get("parameter_bytes_unchanged") or not b.get("parameter_bytes_unchanged") or not ra or not rb or ra["status"] != "completed" or rb["status"] != "completed":
                    views.append({"frame_id": frame_id, "status": "unavailable", "reason": "Both map renders did not complete"})
                    continue
                try:
                    for row in (ra, rb):
                        require(file_sha(output / row["prediction"]) == row["prediction_sha256"], "Saved prediction changed")
                    with np.load(output / ra["prediction"], allow_pickle=False) as pa, np.load(output / rb["prediction"], allow_pickle=False) as pb:
                        values = paired_metrics(pa["depth"], pa["alpha"], pb["depth"], pb["alpha"],
                                                inputs[scene]["depth"][i], inputs[scene]["valid_depth_mask"][i])
                    views.append({"frame_id": frame_id, "status": "completed", **values})
                except Exception as error:
                    views.append({"frame_id": frame_id, "status": "failed", "exception_type": type(error).__name__, "reason": str(error)})
            good = [v for v in views if v["status"] == "completed"]
            result = {"scene": scene, "seed": seed, "prior0_run": names[0], "prior001_run": names[1],
                      "requested_views": 10, "completed_views": len(good), "views": views,
                      "scope": "Depth on identical sensor-valid intersection pixels within each same-seed pair"}
            if good:
                result["pooled"] = {"intersection_pixels": sum(v["intersection_pixels"] for v in good),
                                    "sensor_valid_pixels": sum(v["sensor_valid_pixels"] for v in good),
                                    "prior0_depth": sum_depth([v["prior0_depth"] for v in good]),
                                    "prior001_depth": sum_depth([v["prior001_depth"] for v in good])}
                p = result["pooled"]
                p["intersection_sensor_valid_coverage"] = p["intersection_pixels"]/p["sensor_valid_pixels"]
                result["mean_per_view"] = {"intersection_sensor_valid_coverage": float(np.mean([v["intersection_sensor_valid_coverage"] for v in good])),
                                           **{f"{name}_{metric}": mean_defined([v[name][metric] for v in good])
                                              for name in ("prior0_depth", "prior001_depth") for metric in ("mae_m", "rmse_m")}}
            pairs.append(result)
    return pairs


def execute_protocol(protocol_path: Path, expected_protocol_sha: str, output: Path) -> dict:
    require(not torch.cuda.is_initialized(), "Execution preflight must begin on CPU")
    protocol_bytes = protocol_path.read_bytes()
    protocol, protocol_sha = decode_reviewed_protocol(protocol_bytes, expected_protocol_sha)
    require(protocol["status"] == "ready_for_explicit_gpu_execution", "Protocol is pending or invalid; cannot execute")
    sources, render_node = source_checks()
    require(sources == protocol["sources"] and protocol["settings"] == protocol_settings(), "Protocol source/settings changed")
    require(completed_driver_witness() == protocol["completed_training_driver"] and
            protocol["completed_training_driver"]["status"] == "ready", "Terminal training driver witness changed")
    require(len(protocol["runs"]) == 12 and [{k: row[k] for k in ("run", "scene", "seed", "prior_weight")} for row in protocol["runs"]] == run_specs(),
            "Protocol omitted, reordered, or replaced a map")
    inputs = {}
    for scene in ("desk", "xyz"):
        record, inputs[scene] = prepare_dataset(scene)
        require(record == protocol["datasets"][scene], "GT, input bytes, cameras, or association changed since freeze")
    for spec, frozen in zip(run_specs(), protocol["runs"]):
        current = inspect_completed_run(spec, protocol["datasets"][spec["scene"]])
        require(current == frozen and current["status"] == "ready", f"Map changed since freeze: {spec['run']}")
    require(not torch.cuda.is_initialized(), "Preflight unexpectedly initialized CUDA")
    result = {"schema_version": 1, "status": "running", "purpose": protocol["purpose"],
              "protocol": relative(protocol_path), "protocol_sha256": protocol_sha,
              "protocol_read_once_and_exact_bytes_preserved": True,
              "sources": sources, "settings": protocol["settings"], "limitations": LIMITS,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "all_twelve_source_maps_checked_before_cuda": True, "optimization_steps": 0,
              "known_cameras_use_gt": True, "gt_used_for_fitting_or_map_updates": False,
              "runs": [{**{k: row[k] for k in ("run", "scene", "seed", "prior_weight", "checkpoint_sha256")},
                        "status": "pending", "views": []} for row in protocol["runs"]], "pairs": []}
    with (output / "protocol_used.json").open("xb") as stream:
        stream.write(protocol_bytes)
    atomic_json(output / "progress.json", result)
    started = time.perf_counter()
    try:
        result["pre_gpu_import_deadline_check_utc"] = require_before_deadline()
        from gsplat import rasterization
        for frozen, run_result in zip(protocol["runs"], result["runs"]):
            run_result["status"] = "running"
            atomic_json(output / "progress.json", result)
            checkpoint_path = ROOT / frozen["checkpoint"]
            parameters = render = None
            try:
                require(file_sha(checkpoint_path) == frozen["checkpoint_sha256"], "Checkpoint changed before render")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                tensors = inputs[frozen["scene"]]
                prediction_directory = output / "predictions" / frozen["run"]
                prediction_directory.mkdir(parents=True, exist_ok=False)
                with torch.no_grad():
                    if "first_gpu_start_utc" not in result:
                        result["first_gpu_start_utc"] = require_before_deadline()
                    parameters = {key: value.detach().clone().to("cuda") for key, value in checkpoint["parameters"].items()}
                    K_gpu = torch.from_numpy(tensors["K_render_float32"]).to("cuda").contiguous()
                    require(all(not value.requires_grad for value in parameters.values()) and not torch.is_grad_enabled(), "Render is not no_grad")
                    before = {key: array_record(value) for key, value in parameters.items()}
                    require(before == frozen["parameter_tensors"], "Device upload changed Gaussian bytes")
                    render = bind_original_render(render_node, parameters, K_gpu, rasterization)
                    for i, frame_id in enumerate(EVAL_IDS):
                        view = {"frame_id": frame_id, "status": "running"}
                        run_result["views"].append(view)
                        try:
                            w2c = torch.from_numpy(tensors["evaluation_w2c_render_float32"][i]).to("cuda").contiguous()
                            rendered = render(w2c)
                            require(all(not value.requires_grad for value in rendered), "Render output unexpectedly has gradients")
                            rgb, depth, alpha = [value.detach().cpu().numpy().copy() for value in rendered]
                            values = view_metrics(rgb, depth, alpha, tensors["rgb"][i], tensors["depth"][i], tensors["valid_depth_mask"][i])
                            filename = prediction_directory / f"frame-{frame_id}.npz"
                            np.savez_compressed(filename, rgb=rgb, depth=depth, alpha=alpha)
                            view.update(status="completed", metrics=values, prediction=filename.relative_to(output).as_posix(),
                                        prediction_sha256=file_sha(filename), render_w2c=array_record(w2c),
                                        arrays={key: array_record(value) for key, value in zip(("rgb", "depth", "alpha"), (rgb, depth, alpha))})
                        except Exception as error:
                            view.update(status="failed", exception_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
                        atomic_json(output / "progress.json", result)
                    after = {key: array_record(value) for key, value in parameters.items()}
                require(before == after, "Renderer changed Gaussian parameter bytes")
                require(file_sha(checkpoint_path) == frozen["checkpoint_sha256"], "Original checkpoint changed during rendering")
                run_result.update(status="completed" if all(v["status"] == "completed" for v in run_result["views"]) else "completed_with_failed_views",
                                  parameter_bytes_unchanged=True, original_checkpoint_bytes_unchanged=True)
            except Exception as error:
                run_result.update(status="failed", exception_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
            finally:
                run_result["aggregate"] = aggregate_views(run_result["views"])
                parameters = render = None
                atomic_json(output / "progress.json", result)
        result["pairs"] = build_pairs(result["runs"], inputs, output)
        result["status"] = "completed" if all(r["status"] == "completed" for r in result["runs"]) and all(p["completed_views"] == 10 for p in result["pairs"]) else "completed_with_failures"
        result["completed_map_count"] = sum(row["status"] == "completed" for row in result["runs"])
        result["completed_render_count"] = sum(v["status"] == "completed" for row in result["runs"] for v in row["views"])
        result["elapsed_seconds_including_render_and_export"] = time.perf_counter()-started
        result["cuda_initialized"] = torch.cuda.is_initialized()
        if result["cuda_initialized"]:
            result["gpu"] = torch.cuda.get_device_name(0)
        atomic_json(output / "metrics.json", result)
        atomic_json(output / "progress.json", result)
        return result
    except BaseException as error:
        result.update(status="interrupted_or_failed", exception_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
        atomic_json(output / "failure.json", result)
        atomic_json(output / "progress.json", result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha", help="Exact SHA256 of the reviewed protocol bytes; required with --execute")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", args.run):
        parser.error("run must be a simple unique directory name")
    if args.execute != (args.protocol is not None) or args.execute != (args.protocol_sha is not None):
        parser.error("--execute requires --protocol and --protocol-sha; --check-only accepts neither")
    if args.protocol_sha is not None and not re.fullmatch(r"[a-f0-9]{64}", args.protocol_sha):
        parser.error("--protocol-sha must be the exact lowercase 64-character SHA256")
    output = ROOT / "outputs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = prepare_protocol(output) if args.check_only else execute_protocol(args.protocol.resolve(), args.protocol_sha, output)
        print(json.dumps({"output": relative(output), "status": result["status"],
                          "ready_runs": result.get("ready_runs"), "pending_runs": result.get("pending_runs"),
                          "invalid_runs": result.get("invalid_runs"), "cuda_initialized": torch.cuda.is_initialized()}, indent=2))
    except BaseException as error:
        if not (output / "failure.json").exists():
            atomic_json(output / "failure.json", {"status": "failed", "exception_type": type(error).__name__,
                        "reason": str(error), "traceback": traceback.format_exc(), "cuda_initialized": torch.cuda.is_initialized()})
        raise


if __name__ == "__main__":
    main()
