"""Small fixed-GT-pose TUM RGB-D Gaussian mapping experiment using gsplat.

Known camera poses, train-only depth initialization, RGB plus expected-depth
supervision and fixed Gaussian count. This does not perform tracking or SLAM.
Use --check-only for CPU data/geometry preparation without loading CUDA code.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image, ImageDraw

from tum_rgbd import (DEPTH_UNITS_PER_METER, RAW_K, associate_frames,
                      initialize_from_training_depth, load_frames,
                      scaled_intrinsics, select_split)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", default="tum-rgbd-gsplat-001")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--points", type=int, default=30000)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--train-frames", type=int, default=10)
    parser.add_argument("--heldout-frames", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--init-stride", type=int, default=2)
    parser.add_argument("--max-association-dt", type=float, default=.02)
    parser.add_argument("--max-gt-dt", type=float, default=.02)
    parser.add_argument("--min-depth", type=float, default=.2)
    parser.add_argument("--max-depth", type=float, default=5.)
    parser.add_argument("--depth-weight", type=float, default=.1)
    parser.add_argument("--alpha-weight", type=float, default=.01)
    parser.add_argument("--alpha-threshold", type=float, default=.5)
    parser.add_argument("--means-lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Run name must contain only ASCII letters, digits, dashes or underscores")
    if args.steps < 1 or args.points < 100 or not 0 < args.alpha_threshold < 1:
        parser.error("Require steps >=1, points >=100 and 0<alpha-threshold<1")
    if args.depth_weight < 0 or args.alpha_weight < 0 or args.means_lr <= 0:
        parser.error("Loss weights must be nonnegative and means-lr positive")
    return args


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_rgb(values: np.ndarray, path: Path) -> None:
    Image.fromarray(np.round(np.clip(values, 0, 1)*255).astype(np.uint8)).save(path)


def depth_preview(depth: np.ndarray, valid: np.ndarray, max_depth: float) -> Image.Image:
    t = np.clip(depth/max_depth, 0, 1)
    knots = np.array([0., .25, .5, .75, 1.])
    palette = np.array([[25, 45, 125], [25, 130, 175], [45, 190, 145],
                        [180, 215, 75], [250, 235, 80]], dtype=float)
    colors = np.stack([np.interp(t, knots, palette[:, c]) for c in range(3)], axis=-1).astype(np.uint8)
    colors[~valid] = 0
    return Image.fromarray(colors)


def main() -> None:
    args = parse_arguments()
    total_start = time.perf_counter()
    output = ROOT / "outputs" / args.run
    if output.exists() and not args.check_only:
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    frames, association = associate_frames(args.dataset, args.max_association_dt, args.max_gt_dt)
    selected, train_indices, test_indices = select_split(
        frames, args.start_frame, args.frame_stride, args.train_frames, args.heldout_frames)
    K = scaled_intrinsics(args.width, args.height)
    images, depths, valid_masks = load_frames(args.dataset, selected, args.width, args.height,
                                             args.min_depth, args.max_depth)
    initial, initialization = initialize_from_training_depth(
        selected, images, depths, valid_masks, train_indices, K,
        args.points, args.init_stride, args.seed)
    train_frame_ids = [selected[i].frame_id for i in train_indices]
    test_frame_ids = [selected[i].frame_id for i in test_indices]
    if set(initial["source_frame_ids"]) != set(train_frame_ids):
        raise ValueError("Initialization source IDs must be exactly the training frames")
    if set(initial["source_frame_ids"]) & set(test_frame_ids):
        raise ValueError("Held-out depth entered initialization")
    configuration = {**vars(args), "dataset": args.dataset.resolve().as_posix()}
    metadata = {
        "configuration": configuration,
        "scope": "fixed known-GT-pose RGB-D Gaussian mapping; no tracking or SLAM",
        "dataset": "TUM RGB-D freiburg1_desk",
        "association": association,
        "selected_frames": [{**frame.metadata(), "split": "train" if i in train_indices else "heldout",
                             "valid_depth_fraction": float(valid_masks[i].mean())}
                            for i, frame in enumerate(selected)],
        "split": {"train_local_indices": train_indices, "heldout_local_indices": test_indices,
                  "train_frame_ids": train_frame_ids, "heldout_frame_ids": test_frame_ids,
                  "rgb_and_depth_files_disjoint": True,
                  "sequence_duration_seconds": selected[-1].rgb_timestamp-selected[0].rgb_timestamp,
                  "heldout_protocol": "RGB/depth used only under no_grad for evaluation and image export; GT supplies evaluation camera; no optimization or initialization from held-out frames"},
        "camera": {"convention": "OpenCV +X right,+Y down,+Z forward; TUM GT is column-vector c2w in meters, quaternion xyzw; gsplat Gaussian quaternion wxyz",
                   "raw_K": RAW_K.tolist(), "resized_K": K.tolist(),
                   "K_scaling": "PIL integer pixel centers: u_new=(u_old+0.5)*sx-0.5; K_new=A@K_raw",
                   "resolution": [args.width, args.height], "raw_resolution": [640, 480],
                   "depth_raw_units_per_meter": DEPTH_UNITS_PER_METER,
                   "rgb_resize": "PIL bilinear, sRGB values, no gamma linearization",
                   "depth_resize": "PIL nearest, zeros preserved",
                   "pose_timestamp_reference": "nearest GT to RGB timestamp, not depth timestamp",
                   "distortion_corrected": False, "crop_applied": False},
        "initialization": initialization,
        "supervision": {"renderer": "gsplat rasterization render_mode=RGB+ED, packed=False",
                        "rgb_loss": "mean squared error over all RGB pixels",
                        "depth_loss": "mean absolute expected-camera-Z error in meters, where reference depth valid and detached alpha>threshold",
                        "alpha_loss": "mean (1-alpha)^2 on valid reference depth pixels; discourages evading depth loss through reduced coverage",
                        "depth_mask": "min_depth<=GT_depth<=max_depth, finite RGB+ED prediction, detached alpha>threshold",
                        "alpha_threshold": args.alpha_threshold,
                        "depth_weight": args.depth_weight, "alpha_weight": args.alpha_weight,
                        "color_channels": "constant RGB per Gaussian; no spherical harmonics",
                        "optimization": "Adam; learn means/scales/quaternion/RGB/opacity; fixed count; no densification/pruning/camera updates"},
        "cpu_preparation_seconds": time.perf_counter()-total_start,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(Path(__file__).with_name("tum_rgbd.py").read_bytes()).hexdigest(),
        "dataset_attribution": "TUM RGB-D Benchmark; Sturm et al., IROS 2012; CC BY 4.0",
        "sources": ["https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats",
                    "https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license",
                    "https://github.com/spla-tam/SplaTAM/blob/main/datasets/gradslam_datasets/tum.py",
                    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"],
    }
    if args.check_only:
        print(json.dumps({"status": "cpu_checks_passed", **metadata}, ensure_ascii=False, indent=2, allow_nan=False))
        return

    # Import only after CPU checks, so --check-only never loads CUDA/JIT.
    import torch
    from gsplat import rasterization
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; use --check-only for CPU preparation")
    output.mkdir(parents=True, exist_ok=False)
    view_directory = output / "views"
    view_directory.mkdir()
    write_json(output / "configuration.json", metadata)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    images_gpu = torch.from_numpy(images).to(device)
    depths_gpu = torch.from_numpy(depths).to(device)
    masks_gpu = torch.from_numpy(valid_masks).to(device)
    c2w_gpu = torch.from_numpy(np.stack([frame.c2w for frame in selected])).float().to(device)
    w2c_gpu = torch.linalg.inv(c2w_gpu).contiguous()
    K_gpu = torch.from_numpy(K).float().to(device)[None].contiguous()
    count = initialization["point_count"]
    means = torch.nn.Parameter(torch.from_numpy(initial["means"]).to(device))
    log_scales = torch.nn.Parameter(torch.from_numpy(initial["scales"]).to(device).log())
    quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device=device).repeat(count, 1))
    color_logits = torch.nn.Parameter(torch.logit(torch.from_numpy(initial["colors"]).to(device).clamp(.001, .999)))
    opacity_logits = torch.nn.Parameter(torch.full((count,), math.log(.8/.2), device=device))
    parameters = {"means": means, "log_scales": log_scales, "quats": quats,
                  "color_logits": color_logits, "opacity_logits": opacity_logits}
    learning_rates = {"means": args.means_lr, "log_scales": .003, "quats": .001,
                      "color_logits": .01, "opacity_logits": .01}
    optimizer = torch.optim.Adam([{"params": [value], "lr": learning_rates[name]}
                                  for name, value in parameters.items()], eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)

    def render(view: int):
        colors_depth, alpha, _ = rasterization(
            means, torch.nn.functional.normalize(quats, dim=-1),
            log_scales.exp(), opacity_logits.sigmoid(), color_logits.sigmoid(),
            w2c_gpu[view:view+1], K_gpu, args.width, args.height,
            packed=False, render_mode="RGB+ED", near_plane=.1,
            far_plane=args.max_depth+1., backgrounds=torch.zeros(1, 3, device=device))
        return colors_depth[0, ..., :3], colors_depth[0, ..., 3], alpha[0, ..., 0]

    def evaluate(stage: str) -> tuple[list[dict], float]:
        started = time.perf_counter()
        scores = []
        with torch.no_grad():
            for view, frame in enumerate(selected):
                rgb, depth, alpha = render(view)
                if not (torch.isfinite(rgb).all() and torch.isfinite(depth).all() and torch.isfinite(alpha).all()):
                    raise RuntimeError(f"Nonfinite {stage} render at frame {frame.frame_id}")
                valid = masks_gpu[view]
                covered = valid & (alpha > args.alpha_threshold) & (depth > 0)
                valid_count, covered_count = int(valid.sum()), int(covered.sum())
                mse = float(((rgb-images_gpu[view])**2).mean())
                depth_rmse = float(torch.sqrt(((depth[covered]-depths_gpu[view][covered])**2).mean())) if covered_count else None
                # Supplement the coverage-conditioned score with a penalty for holes.
                all_valid_prediction = torch.where(alpha > args.alpha_threshold, depth, torch.zeros_like(depth))
                depth_rmse_with_holes = float(torch.sqrt(((all_valid_prediction[valid]-depths_gpu[view][valid])**2).mean()))
                scores.append({"local_index": view, "frame_id": frame.frame_id,
                               "rgb_timestamp": frame.rgb_timestamp,
                               "split": "train" if view in train_indices else "heldout",
                               "rgb_mse": mse, "color_psnr_db": -10*math.log10(max(mse, 1e-12)),
                               "masked_depth_rmse_m": depth_rmse,
                               "depth_rmse_all_valid_uncovered_as_zero_m": depth_rmse_with_holes,
                               "valid_depth_pixels": valid_count,
                               "covered_valid_depth_pixels": covered_count,
                               "valid_depth_coverage_fraction": covered_count/valid_count,
                               "mean_alpha_on_valid_depth": float(alpha[valid].mean())})
                prefix = f"frame{frame.frame_id:04d}"
                rgb_array, depth_array, alpha_array = [value.detach().cpu().numpy() for value in (rgb, depth, alpha)]
                save_rgb(rgb_array, view_directory / f"{prefix}_{stage}_rgb.png")
                Image.fromarray(np.round(np.clip(alpha_array, 0, 1)*255).astype(np.uint8)).save(view_directory / f"{prefix}_{stage}_alpha.png")
                Image.fromarray(np.round(np.clip(depth_array*DEPTH_UNITS_PER_METER, 0, 65535)).astype(np.uint16)).save(view_directory / f"{prefix}_{stage}_depth_u16.png")
                depth_preview(depth_array, alpha_array > args.alpha_threshold, args.max_depth).save(view_directory / f"{prefix}_{stage}_depth_preview.png")
                if stage == "initial":
                    save_rgb(images[view], view_directory / f"{prefix}_target_rgb.png")
                    Image.fromarray(np.round(np.clip(depths[view]*DEPTH_UNITS_PER_METER, 0, 65535)).astype(np.uint16)).save(view_directory / f"{prefix}_target_depth_u16.png")
                    depth_preview(depths[view], valid_masks[view], args.max_depth).save(view_directory / f"{prefix}_target_depth_preview.png")
        torch.cuda.synchronize()
        return scores, time.perf_counter()-started

    initial_scores, initial_eval_seconds = evaluate("initial")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    history = []
    for step in range(args.steps):
        view = random.choice(train_indices)
        predicted_rgb, predicted_depth, alpha = render(view)
        rgb_loss = ((predicted_rgb-images_gpu[view])**2).mean()
        depth_mask = (masks_gpu[view] & (alpha.detach() > args.alpha_threshold)
                      & torch.isfinite(predicted_depth) & (predicted_depth.detach() > 0))
        covered_count = int(depth_mask.sum())
        depth_loss = ((predicted_depth[depth_mask]-depths_gpu[view][depth_mask]).abs().mean()
                      if covered_count else predicted_depth.sum()*0.)
        alpha_loss = ((1-alpha[masks_gpu[view]])**2).mean()
        loss = rgb_loss+args.depth_weight*depth_loss+args.alpha_weight*alpha_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at optimization step {step+1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            gradients = {name: {"finite": bool(value.grad is not None and torch.isfinite(value.grad).all()),
                                 "max_abs": float(value.grad.abs().max()) if value.grad is not None else None}
                         for name, value in parameters.items()}
            if not all(value["finite"] for value in gradients.values()):
                raise RuntimeError("First-step gradients are missing or nonfinite")
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), max_norm=1.)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            log_scales.clamp_(math.log(.001), math.log(.15))
            opacity_logits.clamp_(-10, 10)
        if step % 25 == 0 or step == args.steps-1:
            history.append({"step": step+1, "training_frame_id": selected[view].frame_id,
                            "loss": float(loss.detach()), "rgb_mse": float(rgb_loss.detach()),
                            "depth_mae_m": float(depth_loss.detach()), "alpha_loss": float(alpha_loss.detach()),
                            "covered_valid_depth_pixels": covered_count,
                            "valid_depth_pixels": int(masks_gpu[view].sum())})
        if step % 100 == 0 or step == args.steps-1:
            print(f"step={step+1}/{args.steps} frame={selected[view].frame_id} "
                  f"rgb_mse={float(rgb_loss.detach()):.6f} depth_mae_m={float(depth_loss.detach()):.5f} "
                  f"coverage={covered_count/int(masks_gpu[view].sum()):.3f}", flush=True)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-training_start
    peak_allocated_mib = torch.cuda.max_memory_allocated()/1024**2
    peak_reserved_mib = torch.cuda.max_memory_reserved()/1024**2
    final_scores, final_eval_seconds = evaluate("optimized")
    if not all(torch.isfinite(value).all() for value in parameters.values()):
        raise RuntimeError("Nonfinite optimized Gaussian parameters")

    def summarize(scores: list[dict], split: str) -> dict:
        chosen = [score for score in scores if score["split"] == split]
        valid_depth_scores = [score["masked_depth_rmse_m"] for score in chosen if score["masked_depth_rmse_m"] is not None]
        total_covered = sum(score["covered_valid_depth_pixels"] for score in chosen)
        total_valid = sum(score["valid_depth_pixels"] for score in chosen)
        # Pixel-weighted RMSE is comparable when frame-validity sizes differ.
        squared_error = sum(score["masked_depth_rmse_m"]**2*score["covered_valid_depth_pixels"]
                            for score in chosen if score["masked_depth_rmse_m"] is not None)
        return {"view_count": len(chosen),
                "mean_color_psnr_db": float(np.mean([score["color_psnr_db"] for score in chosen])),
                "mean_per_view_masked_depth_rmse_m": float(np.mean(valid_depth_scores)) if valid_depth_scores else None,
                "pooled_masked_depth_rmse_m": math.sqrt(squared_error/total_covered) if total_covered else None,
                "valid_depth_coverage_fraction": total_covered/total_valid,
                "mean_per_view_depth_rmse_uncovered_as_zero_m": float(np.mean([score["depth_rmse_all_valid_uncovered_as_zero_m"] for score in chosen]))}

    metrics = {"experiment_id": args.run, "status": "completed", **metadata,
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "initial_per_view": initial_scores, "optimized_per_view": final_scores,
               "initial_summary": {split: summarize(initial_scores, split) for split in ("train", "heldout")},
               "optimized_summary": {split: summarize(final_scores, split) for split in ("train", "heldout")},
               "first_step_gradients": gradients, "learning_rates": learning_rates,
               "timing": {"initial_evaluation_includes_first_cuda_or_jit_call_seconds": initial_eval_seconds,
                          "optimization_loop_excludes_initial_compile_and_final_evaluation_seconds": training_seconds,
                          "final_evaluation_seconds": final_eval_seconds,
                          "elapsed_before_checkpoint_export_seconds": time.perf_counter()-total_start},
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "peak_torch_allocated_mib_during_optimization": peak_allocated_mib,
               "peak_torch_reserved_mib_during_optimization": peak_reserved_mib,
               "memory_scope": "PyTorch tensors during optimization; excludes other processes and driver allocations",
               "limitations": ["GT cameras fixed for both training and evaluation; no tracking, localization, loop closure, SLAM, ATE or RPE claim",
                               "Tiny short-clip teaching experiment; held-out frames are interleaved nearby views, not an independent scene or complete benchmark",
                               "RGB/depth temporal mismatch <=20 ms by default is uncorrected; GT nearest to RGB time; moving camera still causes registration error",
                               "Registered depth/RGB pixel correspondence is assumed; no lens distortion correction or calibration optimization",
                               "Fixed Gaussian count and constant RGB; no SH, densification, pruning, dynamic-scene handling or pose refinement",
                               "Coverage-conditioned depth RMSE excludes holes; coverage and a hole-penalized metric are also reported",
                               "Gaussian expected depth may average surfaces near boundaries; depth sensor noise and low-resolution sampling affect scores",
                               "Coordinate inverse checks validate arithmetic, not physical calibration or accurate cross-frame map alignment"],
               "artifacts": {"checkpoint": "gaussians.pt", "configuration": "configuration.json",
                             "loss_history": "loss.csv", "comparison": "comparison.png", "views": "views/",
                             "tracking_bundle": "tracking_data.npz"}}
    checkpoint = {"parameters": {name: value.detach().cpu() for name, value in parameters.items()},
                  "configuration": configuration, "learning_rates": learning_rates,
                  "K": K_gpu.detach().cpu(), "c2w": c2w_gpu.detach().cpu(), "w2c": w2c_gpu.detach().cpu(),
                  "frame_ids": [frame.frame_id for frame in selected],
                  "train_local_indices": train_indices, "heldout_local_indices": test_indices,
                  "train_frame_ids": train_frame_ids, "heldout_frame_ids": test_frame_ids,
                  "initialization_source_frame_ids": torch.from_numpy(initial["source_frame_ids"]),
                  "selected_frame_metadata": metadata["selected_frames"],
                  "width": args.width, "height": args.height,
                  "camera_convention": metadata["camera"]["convention"],
                  "depth_raw_units_per_meter": DEPTH_UNITS_PER_METER,
                  "render_mode": "RGB+ED", "point_count": count}
    torch.save(checkpoint, output / "gaussians.pt")
    np.savez_compressed(output / "tracking_data.npz", images=images,
                        depths=depths, valid_depth_masks=valid_masks,
                        w2c=w2c_gpu.detach().cpu().numpy(), intrinsics=K.astype(np.float32),
                        frame_ids=np.array([frame.frame_id for frame in selected], dtype=np.int64),
                        train_local_indices=np.array(train_indices, dtype=np.int64),
                        heldout_local_indices=np.array(test_indices, dtype=np.int64))
    with (output / "loss.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(output / "metrics.json", metrics)
    # Same held-out camera shows target, initial map and optimized map, RGB and depth.
    frame = selected[test_indices[len(test_indices)//2]]
    prefix = f"frame{frame.frame_id:04d}"
    columns = [("target", "Held-out target"), ("initial", "Depth initialization"),
               ("optimized", "After optimization")]
    panel = Image.new("RGB", (720, 420), (245, 247, 250))
    draw = ImageDraw.Draw(panel)
    for column, (stage, title) in enumerate(columns):
        draw.text((column*240+8, 8), title, fill=(20, 30, 45))
        for row, suffix in enumerate(("rgb", "depth_preview")):
            with Image.open(view_directory / f"{prefix}_{stage}_{suffix}.png") as source:
                panel.paste(source.resize((240, 180), Image.Resampling.NEAREST), (column*240, 30+row*190))
    draw.text((8, 405), f"Known-GT-pose mapping; held-out frame {frame.frame_id}; depth colors 0..{args.max_depth:g} m; black=invalid/uncovered", fill=(20, 30, 45))
    panel.save(output / "comparison.png")
    (output / "README.md").write_text(
        "# TUM known-pose RGB-D Gaussian mapping\n\n"
        "This teaching run uses train-only depth initialization and fixed GT cameras. "
        "It does not estimate camera poses or perform SLAM.\n\n"
        "See configuration.json for timestamps, split, K resizing, quaternion order and loss mask. "
        "See metrics.json for initial/optimized RGB PSNR, covered-valid-depth RMSE, coverage and hole-penalized RMSE for every view. "
        "Masked depth RMSE must always be read with coverage.\n\n"
        "Depth PNGs use uint16/5000 meters. Previews use a fixed 0..max_depth range; black means invalid reference or alpha below threshold. "
        "gaussians.pt includes optimized parameters, K, c2w/w2c, frame IDs and train-only initialization provenance. "
        "tracking_data.npz contains RGB/depth, intrinsics, OpenCV w2c and explicit train/held-out indices for subsequent camera recovery experiments.\n\n"
        "Data: TUM RGB-D Benchmark, Sturm et al., IROS 2012, CC BY 4.0.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "completed", "point_count": count,
                      "initial_summary": metrics["initial_summary"],
                      "optimized_summary": metrics["optimized_summary"], "timing": metrics["timing"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
