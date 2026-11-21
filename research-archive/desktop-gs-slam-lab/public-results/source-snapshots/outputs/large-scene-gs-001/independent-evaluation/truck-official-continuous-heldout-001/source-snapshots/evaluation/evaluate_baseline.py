"""Independent evaluation of the *post-update* frozen Truck GS baseline.

Does not instantiate the official trainer, modify its files, install packages,
or reconstruct/renormalize cameras. All output goes to independent-evaluation.
Run GPU mode only after the training owner has confirmed training has ended.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "outputs/large-scene-gs-001"
PREP = TRACK / "data-preparation"
OUTPUT = TRACK / "independent-evaluation"
COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    require(np.all(np.isfinite(x)) and x.size > 0, "Nonfinite or empty numeric summary")
    return {"count": int(x.size), "min": float(x.min()), "mean": float(x.mean()),
            "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def input_audit():
    manifest_path, split_path, files_path = PREP / "input-manifest.json", PREP / "split.json", PREP / "data-files.json"
    manifest, split, files = read_json(manifest_path), read_json(split_path), read_json(files_path)
    require(manifest["status"] == "ready_for_isolated_official_gsplat_single_scene_baseline", "Complete input audit is not ready")
    require(digest(split_path) == manifest["split_sha256"], "Frozen split hash mismatch")
    require(digest(files_path) == manifest["file_manifest_sha256"], "Data-file manifest hash mismatch")
    require(split == manifest["split"], "Split content differs from input manifest")
    geometry_path = Path(manifest["colmap_audit"]["global_geometry_arrays_path"])
    require(digest(geometry_path) == manifest["colmap_audit"]["global_geometry_arrays_sha256"], "Independent source geometry hash mismatch")
    with np.load(geometry_path, allow_pickle=False) as f:
        original = {k: f[k].copy() for k in f.files}
    names = original["image_names"].tolist()
    require(names == sorted(names) and len(names) == 251 and len(set(names)) == len(names), "Unexpected complete Truck image identity")
    frozen_test = np.arange(len(names)) % 8 == 0
    require(np.array_equal(frozen_test, original["test_mask"]), "Original geometry split mask mismatch")
    require([n for i, n in enumerate(names) if frozen_test[i]] == split["test_names"], "Frozen test names differ from independent ordering")
    require([n for i, n in enumerate(names) if not frozen_test[i]] == split["train_names"], "Frozen train names differ from independent ordering")
    require(len(split["test_names"]) == 32 and len(split["train_names"]) == 219, "Unexpected split counts")
    scene = Path(manifest["scene_path"])
    verified = []
    for row in files:
        path = scene.joinpath(*Path(row["path"]).parts)
        require(path.resolve().is_relative_to(scene.resolve()), "Data-file manifest escaped source scene")
        require(path.is_file() and path.stat().st_size == row["bytes"], f"Missing or resized source file: {path}")
        actual = digest(path)
        require(actual == row["sha256"], f"Data SHA256 mismatch: {path}")
        verified.append({"path": str(path), "bytes": row["bytes"], "sha256": actual})
    images = read_json(PREP / "image-audit.json")
    require([i["name"] for i in images] == names, "Decoded image audit order mismatch")
    image_by_name = {i["name"]: i for i in images}
    decoded_test = []
    for name in split["test_names"]:
        row = image_by_name[name]
        path = Path(manifest["training_images_path"]) / row["resized_name"]
        with Image.open(path) as im:
            im.load()
            require(im.mode == "RGB", f"Expected derived RGB PNG: {path}")
            pixels = np.asarray(im).copy()
        require(pixels.shape == (row["resized_height"], row["resized_width"], 3), "Decoded test dimensions changed")
        decoded_test.append({"name": name, "path": str(path), "width": row["resized_width"], "height": row["resized_height"], "sha256": digest(path)})
    report = {"status": "passed_frozen_input_hash_and_test_decode_audit", "manifest_path": str(manifest_path),
              "input_manifest_sha256": digest(manifest_path), "split_sha256": digest(split_path),
              "source_geometry_sha256": digest(geometry_path), "data_files_manifest_sha256": digest(files_path),
              "verified_data_file_count": len(verified), "test_image_count": len(decoded_test),
              "decoded_test_images": decoded_test, "verified_files": verified,
              "camera_convention": "COLMAP/OpenCV C2W; camera +X right, +Y down, +Z forward",
              "heldout_scope": split["used_for_camera_sfm"],
              "intrinsics_size_correction": manifest["colmap_audit"]["training_K_correction_note"],
              "official_vs_exact_K_max_difference_px": manifest["colmap_audit"]["official_parser_K_max_abs_difference_px"]}
    return manifest, split, original, image_by_name, report


def geometry_audit(training, config, original, split):
    path = training / "parser_geometry.npz"
    with np.load(path, allow_pickle=False) as f:
        parsed = {k: f[k].copy() for k in f.files}
    names = original["image_names"].tolist()
    require(parsed["image_names"].tolist() == names, "Saved parser image identity/order mismatch")
    require(np.array_equal(parsed["camera_ids"], original["camera_ids"]), "Saved parser camera identity mismatch")
    train_indices, test_indices = np.asarray(parsed["train_indices"]), np.asarray(parsed["test_indices"])
    require(np.array_equal(test_indices, np.flatnonzero(original["test_mask"])), "Saved parser test indices differ")
    require(np.array_equal(train_indices, np.flatnonzero(~original["test_mask"])), "Saved parser train indices differ")
    require(read_json(training / "split.json") == {"train_names": split["train_names"], "test_names": split["test_names"]}, "Training split copy mismatch")
    for key in ("camtoworlds", "global_transform", "points", "K"):
        require(np.all(np.isfinite(parsed[key])), f"Nonfinite saved parser geometry: {key}")
    T = parsed["global_transform"].astype(np.float64)
    require(T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1], atol=1e-12), "Invalid global homogeneous transform")
    A, translation = T[:3, :3], T[:3, 3]
    singular_values = np.linalg.svd(A, compute_uv=False)
    scale = float(singular_values.mean())
    require(scale > 0 and np.allclose(singular_values, scale, atol=1e-8, rtol=1e-8), "Global transform is not isotropic Sim(3)")
    rotation = A / scale
    require(abs(np.linalg.det(rotation) - 1) < 1e-8 and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8), "Global Sim(3) contains reflection or invalid rotation")
    if not config["normalize_world_space"]:
        require(np.allclose(T, np.eye(4), atol=1e-12), "Normalization disabled but recorded transform is not identity")
    source_c2w = original["camera_to_world"]
    expected = source_c2w.copy()
    expected[:, :3, :3] = rotation @ source_c2w[:, :3, :3]
    expected[:, :3, 3] = source_c2w[:, :3, 3] @ A.T + translation
    c2w_error = float(np.max(np.abs(expected - parsed["camtoworlds"])))
    require(c2w_error < 1e-7, "Saved C2W does not equal recorded global Sim(3) applied to original COLMAP poses")
    # Official parser explicitly rounds initial sparse point storage to float32
    # before applying the global float64 similarity transform.
    raw_points = original["points_xyz"].astype(np.float32).astype(np.float64)
    expected_points = raw_points @ A.T + translation
    point_error = float(np.max(np.abs(expected_points - parsed["points"])))
    require(parsed["points"].shape == raw_points.shape and point_error < 1e-6, "Saved parser sparse points/order or global transform mismatch")
    expected_K = original["official_parser_training_pixel_K"]
    k_error = float(np.max(np.abs(expected_K - parsed["K"])))
    require(parsed["K"].shape == (251, 3, 3) and k_error < 1e-8, "Saved K differs from audited actual-size official parser formula")
    require(np.all(parsed["K"][:, [0, 1], [0, 1]] > 0), "Nonpositive focal length")
    require(np.allclose(parsed["K"][:, 2], [0, 0, 1], atol=1e-12), "Invalid K last row")
    R = parsed["camtoworlds"][:, :3, :3]
    rotation_error = float(np.max(np.abs(np.transpose(R, (0, 2, 1)) @ R - np.eye(3))))
    require(rotation_error < 1e-7, "Saved camera rotations are not orthonormal")
    pid_to_index = {int(pid): i for i, pid in enumerate(original["point_ids"])}
    offsets, observed_ids = original["registered_point_observation_offsets"], original["registered_point_observation_ids"]
    projection_errors, depth_scale_errors, rays = [], [], 0
    for i in test_indices:
        ids = observed_ids[offsets[i]:offsets[i + 1]]
        # Deterministic bounded sample of stored associated points for every test camera.
        chosen = ids[np.linspace(0, len(ids) - 1, min(1024, len(ids)), dtype=int)]
        point_indices = np.array([pid_to_index[int(pid)] for pid in chosen])
        old_w2c = np.linalg.inv(source_c2w[i])
        new_w2c = np.linalg.inv(parsed["camtoworlds"][i])
        old_cam = raw_points[point_indices] @ old_w2c[:3, :3].T + old_w2c[:3, 3]
        new_cam = parsed["points"][point_indices] @ new_w2c[:3, :3].T + new_w2c[:3, 3]
        require(np.all(old_cam[:, 2] > 0) and np.all(new_cam[:, 2] > 0), "Stored observed point has nonpositive depth under global transform")
        old_uvh = old_cam @ parsed["K"][i].T
        new_uvh = new_cam @ parsed["K"][i].T
        errors = np.linalg.norm(old_uvh[:, :2] / old_uvh[:, 2:] - new_uvh[:, :2] / new_uvh[:, 2:], axis=1)
        projection_errors.extend(errors.tolist())
        depth_scale_errors.extend(np.abs(new_cam[:, 2] - scale * old_cam[:, 2]).tolist())
        rays += len(chosen)
    require(max(projection_errors) < 1e-5, "Sim(3) projection invariance failed")
    require(max(depth_scale_errors) < 1e-6, "Sim(3) positive-depth scaling failed")
    report = {"status": "passed_independent_c2w_K_sim3_projection_audit", "parser_geometry_sha256": digest(path),
              "global_transform": T.tolist(), "global_scale": scale, "global_rotation_determinant": float(np.linalg.det(rotation)),
              "global_similarity_singular_values": singular_values.tolist(), "C2W_max_abs_difference": c2w_error,
              "sparse_point_transform_max_abs_difference": point_error, "K_max_abs_difference_px": k_error,
              "camera_rotation_orthonormal_max_abs_error": rotation_error,
              "projection_sample_point_camera_pairs": rays, "projection_sample_cameras": len(test_indices),
              "Sim3_projection_max_difference_px": max(projection_errors),
              "Sim3_depth_scale_max_difference": max(depth_scale_errors),
              "point_initial_precision": "Original sparse xyz cast to float32, then transformed, matching explicit official parser operation",
              "K_policy": "Use recorded official parser K, including model integer-dimension rounding; do not substitute exact-ratio K or rescale twice"}
    return parsed, report


def state_audit(training, configuration, outcome, expected_updates, cpu_only):
    import torch
    path = training / "final_state.pt"
    require(outcome["status"] in ("training_completed", "preflight_passed"), "Training has not completed successfully")
    require(cpu_only or outcome["status"] == "training_completed", "Preflight cannot be used as a quality baseline")
    state_sha = digest(path)
    require(state_sha == outcome["final_state_sha256"], "Final-state hash differs from completed training outcome")
    state = torch.load(path, map_location="cpu", weights_only=True)
    updates = int(state["completed_updates"])
    require(updates == configuration["config"]["max_steps"] == outcome["completed_updates"], "Completed update counts disagree")
    require(cpu_only or updates == expected_updates, f"Expected complete {expected_updates}-update baseline")
    require(state["step"] == updates - 1, "Final-state step index mismatch")
    require(state["state_timing"] == "after final optimizer and densification updates", "Require post-update final_state.pt")
    splats = state["splats"]
    expected_keys = {"means", "quats", "scales", "opacities", "sh0", "shN"}
    require(set(splats) == expected_keys, "Unsupported appearance or tensor fields in saved splats")
    count = len(splats["means"])
    require(count > 0 and count == outcome["gaussian_count"], "Gaussian count differs from training outcome")
    degree = configuration["config"]["sh_degree"]
    shapes = {"means": (count, 3), "quats": (count, 4), "scales": (count, 3),
              "opacities": (count,), "sh0": (count, 1, 3), "shN": (count, (degree + 1)**2 - 1, 3)}
    tensors = {}
    for key, expected_shape in shapes.items():
        tensor = splats[key]
        require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == expected_shape, f"Tensor shape/type mismatch: {key}")
        require(tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(), f"Expected finite float32 tensor: {key}")
        tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "min": float(tensor.min()), "max": float(tensor.max()), "bytes": tensor.numel() * tensor.element_size()}
    norms = torch.linalg.vector_norm(splats["quats"], dim=1)
    require(torch.isfinite(norms).all().item() and (norms > 1e-12).all().item(), "Zero or invalid quaternion norm")
    decoded_scales, decoded_opacity = torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"])
    require(torch.isfinite(decoded_scales).all().item() and (decoded_scales > 0).all().item(), "Scale decoding overflow/underflow")
    require(torch.isfinite(decoded_opacity).all().item() and ((decoded_opacity >= 0) & (decoded_opacity <= 1)).all().item(), "Invalid decoded opacity")
    rendering_source = ROOT / "repos/gsplat/gsplat/rendering.py"
    require("wxyz convension" in rendering_source.read_text(encoding="utf-8"), "Fixed renderer quaternion convention evidence missing")
    report = {"status": "passed_post_update_tensor_audit", "final_state_path": str(path), "final_state_sha256": state_sha,
              "state_timing": state["state_timing"], "completed_updates": updates, "gaussian_count": count,
              "tensors": tensors, "total_saved_tensor_bytes": sum(t["bytes"] for t in tensors.values()),
              "quaternion_convention": "wxyz, scalar first, documented by fixed gsplat rendering.py; renderer internally normalizes nonunit quaternions",
              "quaternion_norm_stats": summarize(norms.numpy()),
              "scale_storage": "Natural log standard deviations; render with torch.exp", "decoded_scale_stats": summarize(decoded_scales.numpy()),
              "opacity_storage": "Logits; render with torch.sigmoid", "decoded_opacity_stats": summarize(decoded_opacity.numpy()),
              "rendering_source_sha256": digest(rendering_source)}
    return splats, report


def quality_and_render(training, cfg, parsed, splats, split, image_rows, output):
    import torch
    from gsplat.rendering import rasterization
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import torchmetrics.functional.image.lpips as lpips_source

    runtime = read_json(training / "runtime.json")
    fixed = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "numpy": "1.26.4",
             "gsplat": "1.5.3", "torchmetrics": "1.7.1"}
    versions = {name: importlib.metadata.version(name) for name in fixed}
    require(versions == fixed, f"Evaluation package versions differ from frozen protocol: {versions}")
    require(all(runtime["packages"][name] == value for name, value in versions.items()), "Training/evaluation runtime version mismatch")
    alex_path = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    linear_path = Path(lpips_source.__file__).parent / "lpips_models/alex.pth"
    require(alex_path.is_file() and linear_path.is_file(), "Required already-installed AlexNet/LPIPS weights missing; evaluation never downloads weights")
    weight_files = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in (alex_path, linear_path)]
    require(torch.cuda.is_available(), "CUDA unavailable for authorized render pass")
    device = torch.device("cuda:0")
    tensor_gpu = {k: v.to(device) for k, v in splats.items()}
    cameras_gpu = torch.from_numpy(parsed["camtoworlds"].astype(np.float32)).to(device)
    K_gpu = torch.from_numpy(parsed["K"].astype(np.float32)).to(device)
    test_indices = parsed["test_indices"].astype(int)
    targets = []
    target_paths = []
    for name in split["test_names"]:
        path = Path(cfg["data_dir"]) / f"images_{cfg['data_factor']}" / image_rows[name]["resized_name"]
        with Image.open(path) as im:
            im.load()
            targets.append(np.asarray(im).copy())
        target_paths.append(path)

    @torch.no_grad()
    def render(index, height, width):
        return rasterization(
            means=tensor_gpu["means"], quats=tensor_gpu["quats"],
            scales=torch.exp(tensor_gpu["scales"]), opacities=torch.sigmoid(tensor_gpu["opacities"]),
            colors=torch.cat([tensor_gpu["sh0"], tensor_gpu["shN"]], dim=1),
            viewmats=torch.linalg.inv(cameras_gpu[index:index + 1]), Ks=K_gpu[index:index + 1],
            width=width, height=height, sh_degree=cfg["sh_degree"], near_plane=cfg["near_plane"], far_plane=cfg["far_plane"],
            packed=cfg["packed"], absgrad=cfg["strategy"]["absgrad"], sparse_grad=cfg["sparse_grad"],
            rasterize_mode="antialiased" if cfg["antialiased"] else "classic", camera_model=cfg["camera_model"],
            distributed=False, with_ut=cfg["with_ut"], with_eval3d=cfg["with_eval3d"])

    # Three untimed render warmups, including synchronization. No model/camera
    # transfer, file I/O, image clamping or metric network runs enter timers.
    for j in range(3):
        h, w = targets[j].shape[:2]
        torch.cuda.synchronize()
        warm = render(int(test_indices[j]), h, w)
        torch.cuda.synchronize()
        del warm
    prediction_arrays, timings = [], []
    for j, index in enumerate(test_indices):
        h, w = targets[j].shape[:2]
        torch.cuda.reset_peak_memory_stats()
        resident_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        started = time.perf_counter()
        colors, alphas, info = render(int(index), h, w)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        require(tuple(colors.shape) == (1, h, w, 3) and torch.isfinite(colors).all().item(), "Invalid rendered RGB tensor")
        require(torch.isfinite(alphas).all().item(), "Nonfinite rendered alpha")
        raw = colors[0].cpu().numpy().copy()
        prediction_arrays.append(np.clip(raw, 0, 1))
        timings.append({"name": split["test_names"][j], "seconds": seconds, "resident_allocated_bytes": resident_allocated,
                        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                        "render_peak_allocated_above_resident_bytes": peak_allocated - resident_allocated,
                        "unclipped_rgb_min": float(raw.min()), "unclipped_rgb_max": float(raw.max()),
                        "clipped_rgb_fraction": float(np.mean((raw < 0) | (raw > 1)))})
        del colors, alphas, info, raw
    timing_report = {"status": "measured_32_full_map_test_camera_renders", "warmup_count": 3,
                     "timed_camera_count": len(timings), "seconds": summarize([r["seconds"] for r in timings]),
                     "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in timings),
                     "max_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in timings),
                     "timing_boundary": "CUDA sync before/after; parameter exp/sigmoid/SH concatenate, C2W inverse and full-map gsplat rasterization included; transfers, decode, clamping, image writing and metrics excluded",
                     "memory_scope": "PyTorch CUDA allocated/reserved, including resident full GS tensors/cameras and raster temporaries; not total GPU process/driver memory",
                     "scope": "Full unpartitioned single-map rendering; no LOD selection, hierarchy traversal, streaming or block swapping",
                     "per_image": timings}
    save(output / "render-timings.json", timing_report)
    np.savez(output / "predictions.npz", image_names=np.array(split["test_names"]), rgb=np.stack(prediction_arrays))
    # Allocate perceptual networks only after rendering timing/memory collection.
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    rows = []
    paired = output / "paired"
    paired.mkdir()
    predicted_dir = output / "predicted"
    predicted_dir.mkdir()
    for j, (name, pred, truth_uint8) in enumerate(zip(split["test_names"], prediction_arrays, targets)):
        truth = truth_uint8.astype(np.float64) / 255.0
        mse = float(np.mean((pred.astype(np.float64) - truth)**2))
        require(mse > 0 and np.isfinite(mse), "Unexpected exact-perfect/nonfinite reconstruction; revise JSON representation before claiming PSNR")
        psnr = float(-10 * np.log10(mse))
        pred_t = torch.from_numpy(pred).permute(2, 0, 1)[None].to(device)
        truth_t = torch.from_numpy(truth_uint8).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            ssim_value = float(ssim(pred_t, truth_t).cpu())
            lpips_value = float(lpips(pred_t, truth_t).cpu())
        ssim.reset()
        lpips.reset()
        require(np.isfinite(ssim_value) and np.isfinite(lpips_value), "Nonfinite SSIM or LPIPS")
        image_pred = (pred * 255).astype(np.uint8)
        prediction_path = predicted_dir / f"{j:02d}-{Path(name).stem}.png"
        pair_path = paired / f"{j:02d}-{Path(name).stem}.png"
        Image.fromarray(image_pred, "RGB").save(prediction_path)
        Image.fromarray(np.concatenate([truth_uint8, image_pred], axis=1), "RGB").save(pair_path)
        rows.append({"name": name, "test_index": int(test_indices[j]), "width": truth_uint8.shape[1], "height": truth_uint8.shape[0],
                     "mse_numpy_float64": mse, "psnr_numpy_db": psnr, "ssim_torchmetrics": ssim_value, "lpips_alex_torchmetrics": lpips_value,
                     "target_path": str(target_paths[j]), "target_sha256": digest(target_paths[j]),
                     "prediction_path": str(prediction_path), "prediction_sha256": digest(prediction_path),
                     "paired_path": str(pair_path), "paired_sha256": digest(pair_path)})
        del pred_t, truth_t
    metrics = {"status": "completed_independent_post_update_heldout_evaluation", "test_image_count": len(rows),
               "psnr_numpy_mean_per_image_db": float(np.mean([r["psnr_numpy_db"] for r in rows])),
               "ssim_mean_per_image": float(np.mean([r["ssim_torchmetrics"] for r in rows])),
               "lpips_alex_mean_per_image": float(np.mean([r["lpips_alex_torchmetrics"] for r in rows])),
               "psnr_aggregation": "NumPy float64 pixel RGB MSE and -10log10 per image; arithmetic mean of 32 image PSNR values, not PSNR of pooled MSE",
               "photometric_scope": "All RGB pixels, original derived PNG target; clamped float32 rendered RGB before PNG quantization; black default background; no masked or color-corrected metric",
               "ssim_protocol": "torchmetrics 1.7.1 StructuralSimilarityIndexMeasure(data_range=1.0), defaults, independently reset per image",
               "lpips_protocol": "torchmetrics 1.7.1 LearnedPerceptualImagePatchSimilarity(alex, normalize=True); input [0,1], fixed pretrained weights, independently reset per image",
               "package_versions": versions, "weight_files": weight_files,
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "per_image": rows,
               "render_timing_summary": {k: v for k, v in timing_report.items() if k != "per_image"}}
    official_path = training / "stats" / f"val_step{cfg['max_steps']-1:04d}.json"
    if official_path.exists():
        official = read_json(official_path)
        diff = {"psnr": metrics["psnr_numpy_mean_per_image_db"] - official["psnr"],
                "ssim": metrics["ssim_mean_per_image"] - official["ssim"],
                "lpips": metrics["lpips_alex_mean_per_image"] - official["lpips"]}
        checks = {"same_gaussian_count": int(official["num_GS"]) == len(splats["means"]),
                  "psnr_agrees_within_0.005_db": abs(diff["psnr"]) < 0.005,
                  "ssim_agrees_within_0.0005": abs(diff["ssim"]) < 0.0005,
                  "lpips_agrees_within_0.0005": abs(diff["lpips"]) < 0.0005}
        metrics["official_final_eval_comparison"] = {"path": str(official_path), "sha256": digest(official_path), "official": official,
                                                     "independent_minus_official": diff, "checks": checks}
        if not all(checks.values()):
            metrics["status"] = "completed_independent_evaluation_official_disagreement_requires_review"
    save(output / "per-image-metrics.json", rows)
    save(output / "metrics.json", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-run", default="truck-baseline-001")
    p.add_argument("--evaluation-run", required=True)
    p.add_argument("--expected-updates", type=int, default=30000)
    p.add_argument("--inputs-only", action="store_true")
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args()
    require(all(x.replace("-", "").replace("_", "").isalnum() for x in (args.training_run, args.evaluation_run)), "Unsafe run identifier")
    output = OUTPUT / args.evaluation_run
    output.mkdir(parents=True, exist_ok=False)
    started, clock = now(), time.perf_counter()
    outcome = {"status": "starting_independent_audit", "started_utc": started,
               "training_run": args.training_run, "cpu_only": args.cpu_only, "inputs_only": args.inputs_only,
               "evaluation_script_sha256": digest(Path(__file__).resolve())}
    save(output / "outcome.json", outcome)
    try:
        manifest, split, original, images, inputs_report = input_audit()
        save(output / "input-audit.json", inputs_report)
        if args.inputs_only:
            outcome.update(status="passed_cpu_input_audit_gpu_not_started", ended_utc=now(), elapsed_seconds=time.perf_counter()-clock)
            save(output / "outcome.json", outcome)
            print(json.dumps(outcome, ensure_ascii=False))
            return
        training = TRACK / "training" / args.training_run
        configuration, training_outcome = read_json(training / "configuration.json"), read_json(training / "outcome.json")
        cfg = configuration["config"]
        require(configuration["gsplat_commit"] == COMMIT, "Unexpected fixed official trainer revision")
        require(configuration["trainer_sha256"] == digest(ROOT / "repos/gsplat/examples/simple_trainer.py"), "Trainer source hash changed after configuration")
        require(configuration["input_manifest_sha256"] == inputs_report["input_manifest_sha256"], "Training input manifest identity differs")
        require(configuration["split_sha256"] == inputs_report["split_sha256"], "Training split identity differs")
        require(Path(cfg["data_dir"]).resolve() == Path(manifest["scene_path"]).resolve(), "Training uses another data root")
        require(cfg["data_factor"] == manifest["data_factor"] == 4 and cfg["test_every"] == 8, "Training sampling/factor changed")
        require(not any(cfg[k] for k in ("pose_opt", "app_opt", "use_bilateral_grid", "depth_loss")) and cfg["pose_noise"] == 0,
                "Independent evaluator does not support enabled pose/appearance/bilateral/depth experiments")
        require(cfg["camera_model"] == "pinhole" and cfg["patch_size"] is None, "Require full undistorted pinhole frames")
        parsed, geo_report = geometry_audit(training, cfg, original, split)
        save(output / "geometry-audit.json", geo_report)
        splats, state_report = state_audit(training, configuration, training_outcome, args.expected_updates, args.cpu_only)
        save(output / "state-audit.json", state_report)
        provenance = {"training_path": str(training), "final_state_sha256": state_report["final_state_sha256"],
                      "source_hashes": {str(training / n): digest(training / n) for n in ("configuration.json", "runtime.json", "outcome.json", "split.json", "parser_geometry.npz")},
                      "input_manifest_sha256": inputs_report["input_manifest_sha256"], "split_sha256": inputs_report["split_sha256"],
                      "evaluation_script_sha256": digest(Path(__file__).resolve()), "saved_model_state": "post-final-optimizer-and-strategy-update",
                      "preupdate_official_checkpoint_used": False, "renormalization_applied_by_evaluation": False}
        save(output / "provenance.json", provenance)
        if args.cpu_only:
            outcome.update(status="passed_complete_cpu_geometry_and_state_audit_gpu_not_started", ended_utc=now(), elapsed_seconds=time.perf_counter()-clock)
        else:
            metrics = quality_and_render(training, cfg, parsed, splats, split, images, output)
            outcome.update(status=metrics["status"], ended_utc=now(), elapsed_seconds=time.perf_counter()-clock,
                           gaussian_count=state_report["gaussian_count"], completed_updates=state_report["completed_updates"],
                           psnr_numpy_mean_per_image_db=metrics["psnr_numpy_mean_per_image_db"],
                           ssim_mean_per_image=metrics["ssim_mean_per_image"], lpips_alex_mean_per_image=metrics["lpips_alex_mean_per_image"])
            timings = metrics["render_timing_summary"]["seconds"]
            report = (f"# Truck 单块最终模型独立评测\n\n使用完成 {state_report['completed_updates']:,} 次更新后保存的 final_state.pt，"
                      f"包含 {state_report['gaussian_count']:,} 个高斯。未使用官方保存于最后更新前的 checkpoint。\n\n"
                      "输入、固定 219／32 划分、相机 C2W、内参与全局 Sim(3) 投影一致性以及模型张量均通过独立审计。"
                      "相机和稀疏点来自完整场景的公开 SfM，留出只针对 GS 图像优化与新视角合成评测。\n\n"
                      f"32 张留出逐图 NumPy PSNR 平均 {metrics['psnr_numpy_mean_per_image_db']:.6f} dB；"
                      f"固定 TorchMetrics SSIM {metrics['ssim_mean_per_image']:.6f}、Alex LPIPS {metrics['lpips_alex_mean_per_image']:.6f}。"
                      "指标使用 PNG 量化前、裁至 [0,1] 的渲染浮点值。实际左右配对 PNG 的左侧为输入留出图，右侧为预测。\n\n"
                      f"3 次热身后，32 个相机各计时一次，CUDA 前后同步；p50 {timings['p50']*1000:.4f} ms、p95 {timings['p95']*1000:.4f} ms。"
                      "计时包含参数解码、相机求逆与完整单地图光栅化，排除数据传输、指标网络和图片保存。"
                      "该计时没有 LOD 选择、层级遍历、分块换入或流式加载成本。allocated／reserved 是 PyTorch 分配器口径。\n\n"
                      "逐图误差、固定依赖及权重哈希见 metrics.json；相机检查见 geometry-audit.json；张量检查见 state-audit.json；"
                      "每帧耗时和显存见 render-timings.json；模型与输入身份见 provenance.json。\n")
            (output / "report.md").write_text(report, encoding="utf-8")
        save(output / "outcome.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False))
    except Exception as exc:
        outcome.update(status="failed_preserved_independent_evaluation", error=repr(exc), traceback=traceback.format_exc(),
                       ended_utc=now(), elapsed_seconds=time.perf_counter()-clock)
        save(output / "outcome.json", outcome)
        raise


if __name__ == "__main__":
    main()
