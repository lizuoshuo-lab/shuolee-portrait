"""Check one TUM RGB-D frame's coordinate math; this is not SLAM/training."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk"
OUTPUT = ROOT / "outputs" / "tum-geometry-001"
FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
DEPTH_SCALE = 5000.0
MAX_DT = 0.08  # Matches the upstream SplaTAM TUM loader's association threshold.


def rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid ground-truth quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def write_binary_ply(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(len(positions), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for axis, key in enumerate(("x", "y", "z")):
        vertices[key] = positions[:, axis]
    for channel, key in enumerate(("red", "green", "blue")):
        vertices[key] = colors[:, channel]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment TUM RGB-D fr1_desk CC BY 4.0; single-frame math check\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("xb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())


def save_preview(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                 output: Path, rgb_t: float, depth_t: float) -> None:
    lo, hi = np.quantile(depth[valid], [0.01, 0.99])
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-12), 0, 1)
    # Piecewise color scale for a raw-data preview, not a publication plot.
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    palette = np.array([[25, 45, 125], [25, 130, 175], [45, 190, 145],
                        [180, 215, 75], [250, 235, 80]], dtype=float)
    colored = np.stack([np.interp(normalized, knots, palette[:, c])
                        for c in range(3)], axis=-1).astype(np.uint8)
    colored[~valid] = 0
    height, width = depth.shape
    canvas = Image.new("RGB", (2 * width, height + 62), (18, 22, 29))
    canvas.paste(Image.fromarray(rgb), (0, 62))
    canvas.paste(Image.fromarray(colored), (width, 62))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"RGB timestamp {rgb_t:.6f}", fill="white")
    draw.text((12, 27), "Color and depth are timestamp-associated, not simultaneous.", fill=(190, 200, 215))
    draw.text((width + 12, 8), f"Depth timestamp {depth_t:.6f}; uint16 / 5000 = meters", fill="white")
    draw.text((width + 12, 27), f"Display p01..p99: {lo:.3f}..{hi:.3f} m; black = invalid", fill=(190, 200, 215))
    draw.text((width + 12, 44), "Blue = nearer; yellow = farther; no distortion correction", fill=(190, 200, 215))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("Stride must be >= 1")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {OUTPUT}")

    rgb_rows, depth_rows, gt_rows = [rows(DATASET / f"{key}.txt")
                                    for key in ("rgb", "depth", "groundtruth")]
    depth_times = np.array([float(row[0]) for row in depth_rows])
    gt_times = np.array([float(row[0]) for row in gt_rows])
    selected = None
    for rgb_index, rgb_row in enumerate(rgb_rows):
        rgb_t = float(rgb_row[0])
        depth_index = int(np.argmin(np.abs(depth_times - rgb_t)))
        gt_index = int(np.argmin(np.abs(gt_times - rgb_t)))
        if abs(depth_times[depth_index] - rgb_t) < MAX_DT and abs(gt_times[gt_index] - rgb_t) < MAX_DT:
            selected = (rgb_index, depth_index, gt_index)
            break
    if selected is None:
        raise RuntimeError("No RGB/depth/GT association meets the threshold")
    rgb_index, depth_index, gt_index = selected
    rgb_row, depth_row, gt_row = rgb_rows[rgb_index], depth_rows[depth_index], gt_rows[gt_index]
    rgb_t, depth_t, gt_t = float(rgb_row[0]), float(depth_row[0]), float(gt_row[0])
    rgb = np.array(Image.open(DATASET / rgb_row[1]).convert("RGB"))
    raw_depth = np.array(Image.open(DATASET / depth_row[1]))
    if rgb.shape[:2] != raw_depth.shape or raw_depth.shape != (480, 640):
        raise ValueError("Unexpected raw TUM image dimensions")
    depth = raw_depth.astype(np.float64) / DEPTH_SCALE
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("No valid depth pixels")

    # Pixel centers follow the usual integer-coordinate pinhole convention.
    v_grid, u_grid = np.mgrid[0:depth.shape[0]:args.stride, 0:depth.shape[1]:args.stride]
    sampled_valid = valid[v_grid, u_grid]
    u = u_grid[sampled_valid].astype(np.float64)
    v = v_grid[sampled_valid].astype(np.float64)
    z = depth[v.astype(int), u.astype(int)]
    camera_xyz = np.column_stack(((u - CX) * z / FX, (v - CY) * z / FY, z))
    colors = rgb[v.astype(int), u.astype(int)]

    pose = np.array(gt_row[1:], dtype=np.float64)
    rotation = quaternion_xyzw_to_matrix(pose[3:])
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3], c2w[:3, 3] = rotation, pose[:3]
    w2c = np.linalg.inv(c2w)
    camera_h = np.column_stack((camera_xyz, np.ones(len(camera_xyz))))
    world_h = (c2w @ camera_h.T).T
    restored_h = (w2c @ world_h.T).T
    restored = restored_h[:, :3] / restored_h[:, 3:4]
    roundtrip_error = np.linalg.norm(restored - camera_xyz, axis=1)
    reprojected = np.column_stack((FX * restored[:, 0] / restored[:, 2] + CX,
                                  FY * restored[:, 1] / restored[:, 2] + CY))
    reprojection_error = np.linalg.norm(reprojected - np.column_stack((u, v)), axis=1)
    if not (np.isfinite(world_h).all() and np.all(restored[:, 2] > 0)):
        raise RuntimeError("Invalid transformed geometry")
    if roundtrip_error.max() >= 1e-10 or reprojection_error.max() >= 1e-8:
        raise RuntimeError("Coordinate arithmetic failed the numerical tolerance")
    quantile_levels = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
    quantiles = np.quantile(depth[valid], quantile_levels)
    metrics = {
        "experiment_id": "tum-geometry-001",
        "status": "passed",
        "kind": "single_frame_rgbd_geometry_math_check",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": "TUM RGB-D freiburg1_desk",
        "dataset_relative_path": DATASET.relative_to(ROOT).as_posix(),
        "association": {"method": "First RGB with nearest depth and nearest GT, both strictly <80 ms, matching SplaTAM TUM loader", "max_dt_s": MAX_DT, "rgb_index": rgb_index, "depth_index": depth_index, "gt_index": gt_index, "rgb_path": rgb_row[1], "depth_path": depth_row[1], "rgb_timestamp": rgb_t, "depth_timestamp": depth_t, "gt_timestamp": gt_t, "rgb_depth_abs_dt_s": abs(rgb_t-depth_t), "rgb_gt_abs_dt_s": abs(rgb_t-gt_t), "depth_gt_abs_dt_s": abs(depth_t-gt_t)},
        "camera": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "depth_raw_units_per_meter": DEPTH_SCALE, "width": 640, "height": 480, "convention": "+X right, +Y down, +Z forward; TUM GT quaternion qx qy qz qw; c2w column-vector transform", "distortion_corrected": False, "crop_applied": False},
        "depth": {"valid_pixels": int(valid.sum()), "total_pixels": int(valid.size), "valid_ratio": float(valid.mean()), "quantiles_m": {str(q): float(value) for q, value in zip(quantile_levels, quantiles)}},
        "pointcloud": {"sampling_stride_pixels": args.stride, "point_count": len(camera_xyz), "coordinate_frame": "TUM ground-truth world", "file": "first_pair_world.ply", "storage": "binary_little_endian float32 XYZ and uint8 RGB", "color_assumption": "Use RGB at the same pixel location as registered depth; temporal mismatch and residual spatial calibration errors are not compensated."},
        "pose": {"gt_quaternion_raw_norm": float(np.linalg.norm(pose[3:])), "quaternion_normalized": True, "rotation_determinant": float(np.linalg.det(rotation)), "rotation_orthogonality_max_abs_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))), "c2w": c2w.tolist(), "w2c": w2c.tolist()},
        "numerical_checks": {"arithmetic_dtype": "float64 before float32 PLY serialization", "roundtrip_max_l2_m": float(roundtrip_error.max()), "roundtrip_mean_l2_m": float(roundtrip_error.mean()), "reprojection_max_l2_pixels": float(reprojection_error.max()), "reprojection_mean_l2_pixels": float(reprojection_error.mean()), "roundtrip_tolerance_m": 1e-10, "reprojection_tolerance_pixels": 1e-8},
        "limitations": ["This is data loading and single-frame coordinate algebra, not training, localization, SLAM, or reconstruction accuracy evaluation.", "The round trip uses a transform and its own inverse. Near-zero error confirms numerical consistency only, not that intrinsics, depth scale, timestamps, or sensor calibration are physically correct.", "RGB, depth, and ground truth are nearest-timestamp associated. No synchronization correction, motion compensation, or pose interpolation is applied.", "Lens distortion is not corrected and depth and RGB are assumed to share pixel registration. Coloring can be misaligned.", "Only one frame is checked. No cross-frame calibration, tracking, map quality, ATE, or RPE claim is supported."],
        "sources": ["https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats", "https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license", "https://github.com/spla-tam/SplaTAM/blob/main/configs/data/TUM/freiburg1_desk.yaml", "https://github.com/spla-tam/SplaTAM/blob/main/datasets/gradslam_datasets/tum.py"],
    }
    OUTPUT.mkdir(parents=True)
    write_binary_ply(OUTPUT / "first_pair_world.ply", world_h[:, :3], colors)
    save_preview(rgb, depth, valid, OUTPUT / "preview_rgb_depth.png", rgb_t, depth_t)
    with (OUTPUT / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(OUTPUT), "association": metrics["association"],
                      "depth": metrics["depth"], "point_count": len(camera_xyz),
                      "numerical_checks": metrics["numerical_checks"]}, indent=2))


if __name__ == "__main__":
    main()
