"""Small sequential RGB-D Gaussian tracking/mapping teaching prototype.

Only the first RGB-D frame initializes the map. Camera 0 is identity. Later
poses start from previous or constant-velocity estimates; GT is read only for
timestamp association and final evaluation, never for optimization or new points.
This short experiment is not a benchmark SLAM implementation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
import traceback

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from track_fixed_map import cpu_math_check, pose_gradient_probe, se3_exp
from tum_rgbd import (associate_frames, initialize_from_training_depth, load_frames,
                      scaled_intrinsics)

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_VERSION = 2
V1_SOURCE_SHA256 = "b8f3b69c42f8921b07a6b996f45f0c0d2100a183918e23120d773655339f5281"
_FAILURE_WRITER = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--run", default="tum-online-v2-001")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--max-association-dt", type=float, default=.02)
    parser.add_argument("--max-gt-dt", type=float, default=.02)
    parser.add_argument("--min-depth", type=float, default=.2)
    parser.add_argument("--max-depth", type=float, default=5.)
    parser.add_argument("--initial-points", type=int, default=15000)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--initial-pixel-stride", type=int, default=1)
    parser.add_argument("--new-point-stride", type=int, default=2)
    parser.add_argument("--max-new-points", type=int, default=3000)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--tracking-steps", type=int, default=100)
    parser.add_argument("--mapping-steps", type=int, default=80)
    parser.add_argument("--keyframe-interval", type=int, default=4)
    parser.add_argument("--keyframe-window", type=int, default=4)
    parser.add_argument("--pose-initializer", choices=["previous", "constant-velocity"],
                        default="constant-velocity")
    parser.add_argument("--pose-lr-translation", type=float, default=.003)
    parser.add_argument("--pose-lr-rotation", type=float, default=.001)
    parser.add_argument("--depth-weight", type=float, default=.1)
    parser.add_argument("--alpha-weight", type=float, default=.01)
    parser.add_argument("--alpha-threshold", type=float, default=.5)
    parser.add_argument("--failure-min-coverage", type=float, default=.3)
    parser.add_argument("--failure-max-step-m", type=float, default=.2)
    parser.add_argument("--failure-max-step-deg", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Run name must contain ASCII letters, numbers, '-' or '_' only")
    positive = (args.frames, args.frame_stride, args.initial_points, args.max_points,
                args.initial_pixel_stride, args.new_point_stride, args.max_new_points,
                args.warmup_steps, args.tracking_steps, args.mapping_steps,
                args.keyframe_interval, args.pose_lr_translation, args.pose_lr_rotation)
    if min(positive) <= 0 or args.frames < 3 or args.keyframe_window < 2 or args.start_frame < 0:
        parser.error("Positive counts/rates, frames>=3 and keyframe-window>=2 required")
    if args.initial_points > args.max_points or min(args.depth_weight, args.alpha_weight) < 0:
        parser.error("Initial point cap exceeds total cap or negative loss weight")
    if not 0 < args.alpha_threshold < 1 or not 0 < args.failure_min_coverage <= 1:
        parser.error("Invalid coverage threshold")
    return args


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


@torch.no_grad()
def canonicalize_rigid_camera(camera: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Project detached R to SO(3), preserve t, and set an exact affine bottom row.

    This is applied between optimization stages. Pose gradients still pass
    through the original SE3 exponential; no SVD is in its autograd graph.
    """
    if camera.shape != (4, 4) or not torch.isfinite(camera).all():
        raise ValueError("Cannot canonicalize a nonfinite/non-4x4 camera")
    source = camera.detach().double()
    u, singular, vh = torch.linalg.svd(source[:3, :3])
    sign = torch.ones(3, dtype=source.dtype, device=source.device)
    if torch.linalg.det(u @ vh) < 0:
        sign[-1] = -1
    projected = u @ torch.diag(sign) @ vh
    result = torch.eye(4, dtype=camera.dtype, device=camera.device)
    result[:3, :3] = projected.to(camera.dtype)
    result[:3, 3] = camera[:3, 3].detach()
    identity = torch.eye(3, dtype=source.dtype, device=source.device)
    result_double = result.double()
    correction = {
        "rotation_projection_frobenius": float(torch.linalg.matrix_norm(result_double[:3, :3]-source[:3, :3])),
        "rotation_orthogonality_frobenius_before": float(torch.linalg.matrix_norm(source[:3, :3].T @ source[:3, :3]-identity)),
        "rotation_orthogonality_frobenius_after": float(torch.linalg.matrix_norm(result_double[:3, :3].T @ result_double[:3, :3]-identity)),
        "bottom_row_max_abs_error_before": float((source[3]-source.new_tensor([0., 0., 0., 1.])).abs().max()),
        "bottom_row_max_abs_error_after": float((result_double[3]-source.new_tensor([0., 0., 0., 1.])).abs().max()),
        "translation_change_l2_m": float(torch.linalg.vector_norm(result[:3, 3]-camera[:3, 3])),
        "determinant_before": float(torch.linalg.det(source[:3, :3])),
        "determinant_after": float(torch.linalg.det(result_double[:3, :3])),
        "rotation_singular_values_before": singular.cpu().tolist(),
    }
    if correction["translation_change_l2_m"] != 0 or correction["bottom_row_max_abs_error_after"] != 0:
        raise RuntimeError("Rigid canonicalization must preserve t and set exact bottom row")
    if correction["rotation_orthogonality_frobenius_after"] > 2e-6 or correction["determinant_after"] <= 0:
        raise RuntimeError("Projected camera is not a proper rigid transform within float32 precision")
    return result, correction


@torch.no_grad()
def rigid_camera_inverse(camera: torch.Tensor) -> torch.Tensor:
    """Analytic rigid inverse preserves exact bottom row after canonicalization."""
    if camera.shape != (4, 4) or not torch.isfinite(camera).all():
        raise ValueError("Rigid inverse requires a finite 4x4 camera")
    if not torch.equal(camera[3], camera.new_tensor([0., 0., 0., 1.])):
        raise ValueError("Rigid inverse requires exact [0,0,0,1] bottom row")
    r = camera[:3, :3].double()
    if torch.linalg.matrix_norm(r.T @ r-torch.eye(3, dtype=r.dtype, device=r.device)) > 2e-6:
        raise ValueError("Rigid inverse requires canonical SO(3) rotation")
    if torch.linalg.det(r) <= 0:
        raise ValueError("Rigid inverse requires positive rotation determinant")
    inverse = torch.eye(4, dtype=camera.dtype, device=camera.device)
    inverse[:3, :3] = camera[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ camera[:3, 3]
    return inverse


def rigid_math_check() -> dict:
    original = torch.eye(4, dtype=torch.float64)
    original[:3, :3] = torch.from_numpy(Rotation.from_euler("xyz", [7, -3, 2], degrees=True).as_matrix())
    original[:3, 3] = torch.tensor([.1, -.2, .3], dtype=torch.float64)
    original[:3, :3] = original[:3, :3] @ torch.diag(torch.tensor([1.00002, .99998, 1.00001], dtype=torch.float64))
    original[3, 3] = 1.0000001192092896
    normalized, correction = canonicalize_rigid_camera(original)
    inverse = rigid_camera_inverse(normalized)
    point = torch.tensor([.2, .1, 1., 1.], dtype=torch.float64)
    roundtrip = inverse @ normalized @ point
    checks = {"canonicalization": correction,
              "rigid_inverse_roundtrip_max_abs_error": float((roundtrip-point).abs().max()),
              "output_requires_grad": normalized.requires_grad,
              "proper_rotation_determinant": float(torch.linalg.det(normalized[:3, :3]))}
    if checks["rigid_inverse_roundtrip_max_abs_error"] > 1e-12 or normalized.requires_grad:
        raise RuntimeError("Rigid camera normalization/inverse CPU check failed")
    return checks


def rotation_error_degrees(rotation: np.ndarray) -> float:
    skew = np.array([rotation[2, 1]-rotation[1, 2],
                     rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    return math.degrees(abs(math.atan2(np.linalg.norm(skew)/2,
                                      (np.trace(rotation)-1)/2)))


def pose_delta(a_c2w: np.ndarray, b_c2w: np.ndarray) -> dict:
    return {"translation_m": float(np.linalg.norm(a_c2w[:3, 3]-b_c2w[:3, 3])),
            "rotation_degrees": rotation_error_degrees(a_c2w[:3, :3] @ b_c2w[:3, :3].T)}


def evaluate_trajectory(estimated_c2w: np.ndarray, truth_c2w: np.ndarray) -> dict:
    """Rigid-aligned ATE with no scale estimate, anchored errors and adjacent RPE."""
    if estimated_c2w.shape != truth_c2w.shape or estimated_c2w.ndim != 3:
        raise ValueError("Trajectory shape mismatch")
    if not np.isfinite(estimated_c2w).all() or not np.isfinite(truth_c2w).all():
        raise ValueError("Nonfinite trajectory cannot be evaluated")
    anchored_truth = np.linalg.inv(truth_c2w[0])[None] @ truth_c2w
    estimate_positions, truth_positions = estimated_c2w[:, :3, 3], anchored_truth[:, :3, 3]
    source = estimate_positions-estimate_positions.mean(0)
    target = truth_positions-truth_positions.mean(0)
    u, singular, vt = np.linalg.svd(source.T @ target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = truth_positions.mean(0)-rotation @ estimate_positions.mean(0)
    alignment = np.eye(4)
    alignment[:3, :3], alignment[:3, 3] = rotation, translation
    aligned = alignment[None] @ estimated_c2w
    per_view = []
    for i, (estimate, truth, aligned_pose) in enumerate(zip(estimated_c2w, anchored_truth, aligned)):
        raw = pose_delta(estimate, truth)
        after_alignment = pose_delta(aligned_pose, truth)
        per_view.append({"local_index": i,
                         "raw_anchored_translation_error_m": raw["translation_m"],
                         "raw_anchored_rotation_error_degrees": raw["rotation_degrees"],
                         "se3_aligned_translation_error_m": after_alignment["translation_m"],
                         "se3_aligned_rotation_error_degrees": after_alignment["rotation_degrees"]})
    adjacent_rpe = []
    for i in range(len(estimated_c2w)-1):
        estimated_relative = np.linalg.inv(estimated_c2w[i]) @ estimated_c2w[i+1]
        truth_relative = np.linalg.inv(anchored_truth[i]) @ anchored_truth[i+1]
        error = np.linalg.inv(truth_relative) @ estimated_relative
        adjacent_rpe.append({"from_local_index": i, "to_local_index": i+1,
                             "translation_error_m": float(np.linalg.norm(error[:3, 3])),
                             "rotation_error_degrees": rotation_error_degrees(error[:3, :3])})
    return {"units": {"translation": "meters", "rotation": "degrees"},
            "ate_alignment": "one global SE3 rigid alignment of estimated c2w positions to first-GT-camera anchored truth; no scale estimation",
            "raw_anchored_reference": "ground truth transformed by inverse(first GT c2w); estimated first camera identity; no alignment fitted",
            "rpe_delta_frames": 1,
            "ate_rmse_m_se3_no_scale": float(np.sqrt(np.mean([v["se3_aligned_translation_error_m"]**2 for v in per_view]))),
            "raw_anchored_translation_rmse_m": float(np.sqrt(np.mean([v["raw_anchored_translation_error_m"]**2 for v in per_view]))),
            "raw_anchored_rotation_rmse_degrees": float(np.sqrt(np.mean([v["raw_anchored_rotation_error_degrees"]**2 for v in per_view]))),
            "adjacent_rpe_translation_rmse_m": float(np.sqrt(np.mean([v["translation_error_m"]**2 for v in adjacent_rpe]))),
            "adjacent_rpe_rotation_rmse_degrees": float(np.sqrt(np.mean([v["rotation_error_degrees"]**2 for v in adjacent_rpe]))),
            "alignment_matrix": alignment.tolist(), "alignment_position_covariance_singular_values": singular.tolist(),
            "alignment_position_covariance_rank": int(np.linalg.matrix_rank(source.T @ target)),
            "per_view": per_view, "adjacent_rpe": adjacent_rpe,
            "anchored_truth_c2w": anchored_truth.tolist(), "aligned_estimated_c2w": aligned.tolist()}


def evaluation_math_check() -> dict:
    """Independent noncollinear known path: exact rigid gauge and known scale error."""
    truth = np.repeat(np.eye(4)[None], 4, axis=0)
    truth[:, :3, 3] = [[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [13, -7, 5], degrees=True).as_matrix()
    gauge[:3, 3] = [.3, -.2, .1]
    rigid_result = evaluate_trajectory(gauge[None] @ truth, truth)
    if rigid_result["ate_rmse_m_se3_no_scale"] > 1e-10 or rigid_result["adjacent_rpe_translation_rmse_m"] > 1e-10:
        raise ValueError("Trajectory evaluator failed known exact rigid-gauge case")
    scaled = truth.copy()
    scaled[:, :3, 3] *= 2
    scale_result = evaluate_trajectory(scaled, truth)
    expected_ate = .075  # RMS distance of a tetrahedron vertex from its centroid.
    expected_rpe = math.sqrt((.1**2+2*(math.sqrt(2)*.1)**2)/3)
    if abs(scale_result["ate_rmse_m_se3_no_scale"]-expected_ate) > 1e-10:
        raise ValueError("ATE evaluator incorrectly removed scale drift")
    if abs(scale_result["adjacent_rpe_translation_rmse_m"]-expected_rpe) > 1e-10:
        raise ValueError("Adjacent RPE does not match known doubled-step trajectory")
    return {"known_rigid_gauge_ate_rmse_m": rigid_result["ate_rmse_m_se3_no_scale"],
            "known_2x_scale_ate_rmse_m": scale_result["ate_rmse_m_se3_no_scale"],
            "known_2x_scale_expected_ate_rmse_m": expected_ate,
            "known_2x_scale_adjacent_rpe_rmse_m": scale_result["adjacent_rpe_translation_rmse_m"],
            "known_2x_scale_expected_rpe_rmse_m": expected_rpe}


def trajectory_tum(path: Path, timestamps: list[float], poses: np.ndarray) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw; c2w; translation meters\n")
        for timestamp, pose in zip(timestamps, poses):
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            values = [timestamp, *pose[:3, 3], *quaternion]
            stream.write(" ".join(f"{value:.9f}" for value in values)+"\n")


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().numpy()
    Image.fromarray(np.round(np.clip(image, 0, 1)*255).astype(np.uint8)).save(path)


def main() -> None:
    global _FAILURE_WRITER
    args = arguments()
    start = time.perf_counter()
    output = ROOT / "outputs" / args.run
    if output.exists() and not args.check_only:
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    frames, associations = associate_frames(args.dataset, args.max_association_dt, args.max_gt_dt)
    selected_ids = list(range(args.start_frame, args.start_frame+args.frames*args.frame_stride,
                              args.frame_stride))
    if selected_ids[-1] >= len(frames):
        raise ValueError("Requested clip exceeds associated dataset length")
    selected = [frames[index] for index in selected_ids]
    K = scaled_intrinsics(args.width, args.height)
    images, depths, masks = load_frames(args.dataset, selected, args.width, args.height,
                                        args.min_depth, args.max_depth)
    # A deliberate GT barrier: optimizer receives copies with identity camera,
    # not the TUM c2w matrices. New point copies later receive estimated c2w.
    sensor_frames = [replace(frame, c2w=np.eye(4, dtype=np.float64)) for frame in selected]
    initialized, init_metadata = initialize_from_training_depth(
        sensor_frames, images, depths, masks, [0], K,
        args.initial_points, args.initial_pixel_stride, args.seed)
    if set(initialized["source_frame_ids"]) != {selected[0].frame_id}:
        raise ValueError("First map must contain only frame zero sensor observations")
    checks = {"se3": cpu_math_check(), "rigid_camera": rigid_math_check(),
              "trajectory_evaluation": evaluation_math_check(),
              "first_frame_initialization": init_metadata,
              "initialization_camera_identity": True,
              "unique_depth_files": len({frame.depth_path for frame in selected}) == len(selected)}
    if not checks["unique_depth_files"]:
        raise ValueError("Online selected frames share a depth image")
    config = {**vars(args), "dataset": args.dataset.resolve().as_posix()}
    metadata = {"implementation_version": IMPLEMENTATION_VERSION,
                "v1_source_sha256": V1_SOURCE_SHA256,
                "configuration": config, "association": associations,
                "selected_frame_ids": [frame.frame_id for frame in selected],
                "selected_sensor_metadata": [{key: value for key, value in frame.metadata().items()
                                               if key not in ("c2w", "w2c")}
                                              for frame in sensor_frames],
                "scope": "small sequential Gaussian RGB-D tracking/mapping teaching prototype",
                "ground_truth_usage": "timestamps for association; all GT poses exclusively for final trajectory evaluation, never optimization, pose initialization or point addition",
                "world_gauge": "first estimated camera c2w=identity; first frame RGB-D supplies metric depth",
                "camera_canonicalization": "between stages: detached float64 SVD R->SO3, preserve translation, exact [0,0,0,1] bottom row; analytic rigid inverse; canonical candidate loss re-evaluated before best-pose selection",
                "camera": {"K": K.tolist(), "resolution": [args.width, args.height],
                           "convention": "OpenCV column-vector c2w/w2c; Gaussian quaternions wxyz; TUM trajectory quaternions xyzw",
                           "translation_unit": "meters", "depth_raw_units_per_meter": 5000.,
                           "K_resize": "PIL integer-center affine u_new=(u_old+.5)*sx-.5",
                           "distortion_corrected": False},
                "cpu_checks": checks,
                "initialization_seconds": time.perf_counter()-start,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "loader_sha256": hashlib.sha256(Path(__file__).with_name("tum_rgbd.py").read_bytes()).hexdigest(),
                "pose_helper_sha256": hashlib.sha256(Path(__file__).with_name("track_fixed_map.py").read_bytes()).hexdigest(),
                "dataset_attribution": "TUM RGB-D Benchmark; Sturm et al., IROS 2012; CC BY 4.0"}
    if args.check_only:
        print(json.dumps({"status": "cpu_checks_passed", **metadata}, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --check-only for CPU preparation")
    from gsplat import rasterization
    output.mkdir(parents=True, exist_ok=False)
    preview_directory = output / "views"
    preview_directory.mkdir()
    write_json(output / "configuration.json", metadata)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    images_gpu = torch.from_numpy(images).cuda()
    depths_gpu = torch.from_numpy(depths).cuda()
    masks_gpu = torch.from_numpy(masks).cuda()
    K_gpu = torch.from_numpy(K).float().cuda()[None].contiguous()
    map_parameters = {"means": torch.from_numpy(initialized["means"]).cuda(),
                      "log_scales": torch.from_numpy(initialized["scales"]).cuda().log(),
                      "quats": torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(init_metadata["point_count"], 1),
                      "color_logits": torch.logit(torch.from_numpy(initialized["colors"]).cuda().clamp(.001, .999)),
                      "opacity_logits": torch.full((init_metadata["point_count"],), math.log(.8/.2), device="cuda")}
    estimated_w2c = [torch.eye(4, device="cuda")]
    keyframes = [0]
    source_frame_ids = [int(frame_id) for frame_id in initialized["source_frame_ids"]]
    loss_history, frame_metrics, mapping_events = [], [], []
    pose_gradient_checks = None
    any_failure = False
    current_view = 0

    def persist_progress(status: str, exception: BaseException | None = None) -> None:
        """No GT access: save enough causal state to diagnose interrupted runs."""
        checkpoint = {
            "status": status, "implementation_version": IMPLEMENTATION_VERSION,
            "script_sha256": metadata["script_sha256"], "configuration": config,
            "parameters": {key: value.detach().cpu() for key, value in map_parameters.items()},
            "K": K_gpu.detach().cpu(), "estimated_w2c": torch.stack(estimated_w2c).detach().cpu(),
            "estimated_c2w": torch.stack([rigid_camera_inverse(camera) for camera in estimated_w2c]).cpu(),
            "source_frame_ids_per_gaussian": source_frame_ids,
            "keyframe_local_indices": keyframes,
            "frame_ids_with_stored_pose": [selected[i].frame_id for i in range(len(estimated_w2c))],
            "completed_frame_metrics": frame_metrics, "mapping_events": mapping_events,
            "current_local_index": current_view,
            "ground_truth_included": False,
        }
        temporary = output / "last_checkpoint.pt.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, output / "last_checkpoint.pt")
        progress = {"status": status, "implementation_version": IMPLEMENTATION_VERSION,
                    "configuration": config, "script_sha256": metadata["script_sha256"],
                    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "current_local_index": current_view,
                    "current_frame_id": selected[current_view].frame_id,
                    "completed_frame_count": len(frame_metrics),
                    "stored_camera_count": len(estimated_w2c),
                    "frame_metrics": frame_metrics, "mapping_events": mapping_events,
                    "loss_history": loss_history,
                    "estimated_w2c": checkpoint["estimated_w2c"].tolist(),
                    "estimated_c2w": checkpoint["estimated_c2w"].tolist(),
                    "point_count": len(map_parameters["means"]),
                    "last_checkpoint": "last_checkpoint.pt", "ground_truth_included": False}
        if exception is not None:
            failure = {"status": "failed", "implementation_version": IMPLEMENTATION_VERSION,
                       "script_sha256": metadata["script_sha256"], "configuration": config,
                       "exception_type": type(exception).__name__, "exception_message": str(exception),
                       "traceback": traceback.format_exc(),
                       "completed_frame_count": len(frame_metrics),
                       "current_local_index": current_view,
                       "current_frame_id": selected[current_view].frame_id,
                       "stored_camera_count": len(estimated_w2c),
                       "partial_state_is_a_completed_trajectory": False,
                       "diagnostic_checkpoint": "last_checkpoint.pt"}
            progress["failure"] = failure
            atomic_json(output / "failure.json", failure)
        atomic_json(output / "progress.json", progress)

    _FAILURE_WRITER = persist_progress

    def render(w2c: torch.Tensor):
        colors_depth, alpha, _ = rasterization(
            map_parameters["means"], torch.nn.functional.normalize(map_parameters["quats"], dim=-1),
            map_parameters["log_scales"].exp(), map_parameters["opacity_logits"].sigmoid(),
            map_parameters["color_logits"].sigmoid(), w2c[None].contiguous(), K_gpu,
            args.width, args.height, packed=False, render_mode="RGB+ED", near_plane=.1,
            far_plane=args.max_depth+1., backgrounds=torch.zeros(1, 3, device="cuda"))
        return colors_depth[0, ..., :3], colors_depth[0, ..., 3], alpha[0, ..., 0]

    def data_loss(w2c: torch.Tensor, view: int) -> dict:
        rgb, depth, alpha = render(w2c)
        valid = masks_gpu[view]
        covered = valid & (alpha.detach() > args.alpha_threshold) & torch.isfinite(depth) & (depth.detach() > 0)
        count = int(covered.sum())
        rgb_loss = ((rgb-images_gpu[view])**2).mean()
        depth_loss = (depth[covered]-depths_gpu[view][covered]).abs().mean() if count else depth.sum()*0.
        alpha_loss = ((1-alpha[valid])**2).mean()
        objective = rgb_loss+args.depth_weight*depth_loss+args.alpha_weight*alpha_loss
        return {"objective": objective, "rgb_mse": rgb_loss, "depth_mae_m": depth_loss,
                "alpha_loss": alpha_loss, "covered_valid_pixels": count,
                "valid_depth_pixels": int(valid.sum()), "coverage": count/int(valid.sum())}

    def loss_numbers(value: dict) -> dict:
        return {key: float(item.detach()) if isinstance(item, torch.Tensor) else item
                for key, item in value.items()}

    def optimize_map(window: list[int], steps: int, current_view: int, phase: str) -> dict:
        nonlocal map_parameters
        if max(window) > current_view or 0 not in window:
            raise RuntimeError("Map window must use past/current observations and retain first-frame anchor")
        map_parameters = {key: torch.nn.Parameter(value.detach()) for key, value in map_parameters.items()}
        lr = {"means": .001, "log_scales": .003, "quats": .001,
              "color_logits": .01, "opacity_logits": .01}
        optimizer = torch.optim.Adam([{"params": [value], "lr": lr[key]}
                                      for key, value in map_parameters.items()], eps=1e-15)
        failures = []
        start_map = time.perf_counter()
        for step in range(steps):
            view = window[step % len(window)]  # Every old/window view receives a constraint.
            losses = data_loss(estimated_w2c[view].detach(), view)
            objective = losses["objective"]
            if not torch.isfinite(objective):
                failures.append("mapping_nonfinite_objective")
                break
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            if any(value.grad is None or not torch.isfinite(value.grad).all()
                   for value in map_parameters.values()):
                failures.append("mapping_missing_or_nonfinite_gradient")
                break
            torch.nn.utils.clip_grad_norm_(list(map_parameters.values()), max_norm=1.)
            optimizer.step()
            with torch.no_grad():
                map_parameters["log_scales"].clamp_(math.log(.001), math.log(.15))
                map_parameters["opacity_logits"].clamp_(-10, 10)
            if step % 10 == 0 or step == steps-1:
                loss_history.append({"phase": phase, "current_local_index": current_view,
                                     "observation_local_index": view, "step": step+1,
                                     **loss_numbers(losses)})
        map_parameters = {key: value.detach() for key, value in map_parameters.items()}
        if not all(torch.isfinite(value).all() for value in map_parameters.values()):
            raise RuntimeError("Map parameters became nonfinite; refusing to export a valid map")
        torch.cuda.synchronize()
        return {"phase": phase, "current_local_index": current_view,
                "window_local_indices": window, "window_frame_ids": [selected[i].frame_id for i in window],
                "steps_requested": steps, "steps_completed": step+1,
                "seconds": time.perf_counter()-start_map,
                "point_count": len(map_parameters["means"]), "failure_flags": failures}

    with torch.no_grad():
        first_initial_loss = loss_numbers(data_loss(estimated_w2c[0], 0))
        save_rgb(render(estimated_w2c[0])[0], preview_directory / "frame000_initial_map.png")
    warmup_event = optimize_map([0], args.warmup_steps, 0, "warmup_mapping")
    mapping_events.append(warmup_event)
    any_failure |= bool(warmup_event["failure_flags"])
    with torch.no_grad():
        first_loss = loss_numbers(data_loss(estimated_w2c[0], 0))
        save_rgb(images_gpu[0], preview_directory / "frame000_target.png")
        save_rgb(render(estimated_w2c[0])[0], preview_directory / "frame000_map.png")
    frame_metrics.append({"local_index": 0, "frame_id": selected[0].frame_id,
                          "rgb_timestamp": selected[0].rgb_timestamp,
                          "initialization": "identity world gauge", "initial_loss": first_initial_loss,
                          "optimized_loss": first_loss, "initial_w2c": np.eye(4).tolist(),
                          "estimated_w2c": np.eye(4).tolist(), "estimated_c2w": np.eye(4).tolist(),
                          "failure_flags": warmup_event["failure_flags"], "new_points_added": 0,
                          "point_count": len(map_parameters["means"]), "tracking_seconds": 0.})
    persist_progress("running")
    torch.cuda.reset_peak_memory_stats()
    online_start = time.perf_counter()
    for view in range(1, len(selected)):
        current_view = view
        frame_start = time.perf_counter()
        if args.pose_initializer == "constant-velocity" and view >= 2:
            previous_c2w = rigid_camera_inverse(estimated_w2c[-1])
            before_previous_c2w = rigid_camera_inverse(estimated_w2c[-2])
            predicted_c2w_raw = previous_c2w @ rigid_camera_inverse(before_previous_c2w) @ previous_c2w
            predicted_c2w, motion_prediction_projection = canonicalize_rigid_camera(predicted_c2w_raw)
            initial_w2c_raw = rigid_camera_inverse(predicted_c2w)
            initialization_method = "constant-velocity from previous two estimates"
        else:
            initial_w2c_raw = estimated_w2c[-1].detach().clone()
            motion_prediction_projection = None
            initialization_method = "previous estimate"
        initial_w2c, initializer_projection = canonicalize_rigid_camera(initial_w2c_raw)
        if any(value.requires_grad for value in map_parameters.values()):
            raise RuntimeError("Map must be frozen while tracking")
        translation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
        rotation = torch.nn.Parameter(torch.zeros(3, device="cuda"))

        def pose(twist=None):
            vector = torch.cat((translation, rotation)) if twist is None else twist
            return se3_exp(vector) @ initial_w2c

        flags = []
        with torch.no_grad():
            initial_loss = loss_numbers(data_loss(initial_w2c, view))
        if pose_gradient_checks is None:
            try:
                pose_gradient_checks = {"status": "passed", "local_index": view,
                                        **pose_gradient_probe(lambda twist: data_loss(pose(twist), view)["objective"])}
            except RuntimeError as error:
                pose_gradient_checks = {"status": "failed", "local_index": view, "reason": str(error)}
                flags.append("pose_gradient_finite_difference_check_failed")
        optimizer = torch.optim.Adam([{"params": [translation], "lr": args.pose_lr_translation},
                                      {"params": [rotation], "lr": args.pose_lr_rotation}], eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
        best_loss, best_pose, best_step = initial_loss["objective"], initial_w2c.clone(), 0
        selected_raw_pose = initial_w2c.clone()
        _, selected_projection = canonicalize_rigid_camera(initial_w2c)
        selected_raw_loss = initial_loss
        for step in range(args.tracking_steps):
            candidate = pose()
            losses = data_loss(candidate, view)
            objective = losses["objective"]
            if not torch.isfinite(objective):
                flags.append("tracking_nonfinite_objective")
                break
            if float(objective.detach()) < best_loss:
                canonical_candidate, correction = canonicalize_rigid_camera(candidate)
                with torch.no_grad():
                    canonical_losses = loss_numbers(data_loss(canonical_candidate, view))
                if canonical_losses["objective"] < best_loss:
                    best_loss, best_pose, best_step = canonical_losses["objective"], canonical_candidate, step
                    selected_raw_pose = candidate.detach().clone()
                    selected_projection = correction
                    selected_raw_loss = loss_numbers(losses)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            if any(value.grad is None or not torch.isfinite(value.grad).all()
                   for value in (translation, rotation)):
                flags.append("tracking_missing_or_nonfinite_gradient")
                break
            torch.nn.utils.clip_grad_norm_([translation, rotation], max_norm=1.)
            optimizer.step()
            scheduler.step()
            if step % 10 == 0 or step == args.tracking_steps-1:
                loss_history.append({"phase": "tracking", "current_local_index": view,
                                     "observation_local_index": view, "step": step+1,
                                     **loss_numbers(losses)})
        with torch.no_grad():
            last_pose = pose()
            if torch.isfinite(last_pose).all():
                canonical_last_pose, last_projection = canonicalize_rigid_camera(last_pose)
                last_loss = loss_numbers(data_loss(canonical_last_pose, view))
                if math.isfinite(last_loss["objective"]) and last_loss["objective"] < best_loss:
                    best_loss, best_pose, best_step = last_loss["objective"], canonical_last_pose, args.tracking_steps
                    selected_raw_pose = last_pose.clone()
                    selected_projection = last_projection
                    selected_raw_loss = loss_numbers(data_loss(last_pose, view))
            optimized_loss = loss_numbers(data_loss(best_pose, view))
        initial_c2w = rigid_camera_inverse(initial_w2c).cpu().numpy()
        best_c2w = rigid_camera_inverse(best_pose).cpu().numpy()
        previous_c2w = rigid_camera_inverse(estimated_w2c[-1]).cpu().numpy()
        delta = pose_delta(initial_c2w, best_c2w)
        step_delta = pose_delta(previous_c2w, best_c2w)
        if optimized_loss["coverage"] < args.failure_min_coverage:
            flags.append("tracking_low_valid_depth_coverage")
        if step_delta["translation_m"] > args.failure_max_step_m:
            flags.append("tracking_step_translation_exceeds_threshold")
        if step_delta["rotation_degrees"] > args.failure_max_step_deg:
            flags.append("tracking_step_rotation_exceeds_threshold")
        # Rejected poses propagate the valid previous estimate and skip mapping.
        tracking_rejected = bool(flags)
        if tracking_rejected:
            accepted_pose = estimated_w2c[-1].clone()
        else:
            accepted_pose = best_pose.detach()
        estimated_w2c.append(accepted_pose)
        torch.cuda.synchronize()
        tracking_seconds = time.perf_counter()-frame_start
        new_points = 0
        mapping_event = None
        if view % args.keyframe_interval == 0 and not flags:
            with torch.no_grad():
                _, _, alpha = render(accepted_pose)
            new_mask = masks[view] & (alpha.detach().cpu().numpy() < args.alpha_threshold)
            free_points = args.max_points-len(map_parameters["means"])
            cap = min(args.max_new_points, free_points)
            v_grid, u_grid = np.mgrid[0:args.height:args.new_point_stride, 0:args.width:args.new_point_stride]
            candidate_count = int(new_mask[v_grid, u_grid].sum())
            if cap >= 100 and candidate_count >= 100:
                # Only current sensor depth and estimated pose can initialize additions.
                estimated_frame = replace(sensor_frames[view], c2w=rigid_camera_inverse(accepted_pose).cpu().double().numpy())
                additions, _ = initialize_from_training_depth(
                    [estimated_frame], images[view:view+1], depths[view:view+1], new_mask[None], [0], K,
                    cap, args.new_point_stride, args.seed+view)
                new_points = len(additions["means"])
                added_parameters = {"means": torch.from_numpy(additions["means"]).cuda(),
                                    "log_scales": torch.from_numpy(additions["scales"]).cuda().log(),
                                    "quats": torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(new_points, 1),
                                    "color_logits": torch.logit(torch.from_numpy(additions["colors"]).cuda().clamp(.001, .999)),
                                    "opacity_logits": torch.full((new_points,), math.log(.8/.2), device="cuda")}
                map_parameters = {key: torch.cat((value, added_parameters[key])).detach()
                                  for key, value in map_parameters.items()}
                source_frame_ids.extend([selected[view].frame_id]*new_points)
            if len(map_parameters["means"]) > args.max_points:
                raise RuntimeError("Gaussian count exceeded configured cap")
            keyframes.append(view)
            window = [0]+[i for i in keyframes[-(args.keyframe_window-1):] if i != 0]
            mapping_event = optimize_map(window, args.mapping_steps, view, "online_mapping")
            mapping_event.update({"new_points_added": new_points,
                                  "uncovered_candidate_pixels_at_sampling_stride": candidate_count,
                                  "point_addition_camera_source": "current accepted estimated pose"})
            mapping_events.append(mapping_event)
            flags.extend(mapping_event["failure_flags"])
        with torch.no_grad():
            save_rgb(images_gpu[view], preview_directory / f"frame{view:03d}_target.png")
            save_rgb(render(accepted_pose)[0], preview_directory / f"frame{view:03d}_estimated_map.png")
            after_mapping_loss = loss_numbers(data_loss(accepted_pose, view))
        accepted_c2w = rigid_camera_inverse(accepted_pose).cpu().numpy()
        frame_metrics.append({"local_index": view, "frame_id": selected[view].frame_id,
                              "rgb_timestamp": selected[view].rgb_timestamp,
                              "depth_timestamp": selected[view].depth_timestamp,
                              "rgb_depth_abs_dt_s": abs(selected[view].rgb_timestamp-selected[view].depth_timestamp),
                              "initialization": initialization_method,
                              "motion_prediction_rigid_projection": motion_prediction_projection,
                              "initializer_rigid_projection": initializer_projection,
                              "selected_pose_rigid_projection": selected_projection,
                              "selected_raw_w2c_before_projection": selected_raw_pose.cpu().tolist(),
                              "initial_w2c": initial_w2c.cpu().tolist(), "candidate_optimized_w2c": best_pose.cpu().tolist(),
                              "estimated_w2c": accepted_pose.cpu().tolist(), "estimated_c2w": accepted_c2w.tolist(),
                              "initial_loss": initial_loss, "optimized_loss_before_mapping": optimized_loss,
                              "selected_raw_candidate_loss_before_projection": selected_raw_loss,
                              "accepted_pose_loss_after_mapping": after_mapping_loss,
                              "best_step_selected_by_sensor_loss_only": best_step,
                              "initial_to_candidate_delta": delta, "previous_to_candidate_delta": step_delta,
                              "pose_rejected": tracking_rejected, "failure_flags": flags,
                              "new_points_added": new_points, "point_count": len(map_parameters["means"]),
                              "tracking_seconds": tracking_seconds, "frame_total_seconds": time.perf_counter()-frame_start})
        any_failure |= bool(flags)
        if view % args.keyframe_interval == 0 or view == len(selected)-1:
            persist_progress("running")
        print(f"frame={view+1}/{len(selected)} id={selected[view].frame_id} "
              f"loss={initial_loss['objective']:.5f}->{optimized_loss['objective']:.5f} "
              f"coverage={optimized_loss['coverage']:.3f} points={len(map_parameters['means'])} "
              f"failures={flags}", flush=True)

    torch.cuda.synchronize()
    online_seconds = time.perf_counter()-online_start
    estimate_array = np.stack([rigid_camera_inverse(pose).cpu().double().numpy() for pose in estimated_w2c])
    # This is the first consumer of actual GT poses since data association.
    gt_array = np.stack([frame.c2w for frame in selected])
    trajectory = evaluate_trajectory(estimate_array, gt_array)
    for row, frame in zip(trajectory["per_view"], selected):
        row.update({"frame_id": frame.frame_id, "rgb_timestamp": frame.rgb_timestamp,
                    "rgb_gt_abs_dt_s": abs(frame.rgb_timestamp-frame.gt_timestamp)})
    for row in trajectory["adjacent_rpe"]:
        i, j = row["from_local_index"], row["to_local_index"]
        row.update({"from_frame_id": selected[i].frame_id, "to_frame_id": selected[j].frame_id,
                    "dt_s": selected[j].rgb_timestamp-selected[i].rgb_timestamp})
    timestamps = [frame.rgb_timestamp for frame in selected]
    trajectory_tum(output / "trajectory_estimated.txt", timestamps, estimate_array)
    trajectory_tum(output / "trajectory_groundtruth.txt", timestamps, gt_array)
    trajectory_tum(output / "trajectory_groundtruth_anchored.txt", timestamps,
                   np.array(trajectory["anchored_truth_c2w"]))
    aligned_to_absolute_gt = gt_array[0][None] @ np.array(trajectory["aligned_estimated_c2w"])
    trajectory_tum(output / "trajectory_estimated_se3_aligned.txt", timestamps, aligned_to_absolute_gt)
    checkpoint = {"parameters": {key: value.detach().cpu() for key, value in map_parameters.items()},
                  "implementation_version": IMPLEMENTATION_VERSION,
                  "configuration": config, "K": K_gpu.cpu(),
                  "estimated_w2c": torch.stack(estimated_w2c).cpu(),
                  "estimated_c2w": torch.from_numpy(estimate_array),
                  "evaluation_only_gt_c2w": torch.from_numpy(gt_array),
                  "frame_ids": [frame.frame_id for frame in selected],
                  "source_frame_ids_per_gaussian": source_frame_ids,
                  "keyframe_local_indices": keyframes, "width": args.width, "height": args.height,
                  "world_gauge": "first estimated camera identity", "camera_convention": metadata["camera"]["convention"]}
    torch.save(checkpoint, output / "online_map.pt")
    with (output / "loss.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(loss_history[0]))
        writer.writeheader()
        writer.writerows(loss_history)
    write_json(output / "trajectory_evaluation.json", trajectory)
    metrics = {"experiment_id": args.run,
               "status": "completed_with_tracking_or_mapping_failures" if any_failure else "completed",
               **metadata, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "frames_processed": len(frame_metrics),
               "failure_frame_count": sum(bool(row["failure_flags"]) for row in frame_metrics),
               "initial_point_count": init_metadata["point_count"],
               "final_point_count": len(map_parameters["means"]),
               "frame_metrics": frame_metrics, "mapping_events": mapping_events,
               "pose_gradient_checks": pose_gradient_checks, "trajectory": trajectory,
               "online_loop_seconds_excludes_warmup": online_seconds,
               "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "limitations": [f"{args.frames}-frame teaching prototype, not complete TUM benchmark SLAM",
                               "V2 rigid camera projection and causal progress checkpoints differ from V1; results from different source hashes are not same-code ablations",
                               "V1 frame-81 geometry failure lacks the actual failing matrix; bottom-row drift is a reproducible numeric risk, not a confirmed root cause of that historical failure",
                               "No global BA, loop closure, relocalization, robust motion model or dynamic-object rejection",
                               "GT is nearest to RGB time, not interpolated; RGB/depth mismatch <=20 ms is uncorrected",
                               "Registered RGB/depth pixels and provided desk calibration assumed; no distortion correction",
                               "Map window fixes estimated cameras; early pose errors can be absorbed into Gaussian map",
                               "Pose candidates minimize image/depth losses, not GT errors; lower loss does not guarantee better trajectory",
                               "Failed tracking reuses previous estimate and skips point addition; all failures are retained in trajectory and metrics",
                               "ATE uses one rigid SE3 alignment and never estimates scale; raw first-camera anchored errors also reported",
                               "Masked depth losses require alpha coverage and exclude holes; coverage loss discourages but does not eliminate mask gaming"],
               "artifacts": {"checkpoint": "online_map.pt", "configuration": "configuration.json",
                             "loss_history": "loss.csv", "evaluation": "trajectory_evaluation.json",
                             "estimated_trajectory": "trajectory_estimated.txt", "groundtruth_trajectory": "trajectory_groundtruth.txt"},
               "total_seconds_before_metrics_export": time.perf_counter()-start}
    write_json(output / "metrics.json", metrics)
    persist_progress("completed_with_tracking_or_mapping_failures" if any_failure else "completed")
    _FAILURE_WRITER = None
    print(json.dumps({"output": str(output), "status": metrics["status"],
                      "failure_frame_count": metrics["failure_frame_count"],
                      "final_point_count": metrics["final_point_count"],
                      "pose_gradient_checks": pose_gradient_checks,
                      "trajectory_summary": {key: value for key, value in trajectory.items()
                                             if key not in ("per_view", "adjacent_rpe", "anchored_truth_c2w", "aligned_estimated_c2w")},
                      "online_loop_seconds": online_seconds}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _FAILURE_WRITER is not None:
            try:
                _FAILURE_WRITER("failed", error)
            except BaseException as diagnostic_error:
                print(f"Could not save failure diagnostic state: {diagnostic_error}", flush=True)
        raise
