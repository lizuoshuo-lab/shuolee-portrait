"""Reusable TUM RGB-D association, camera loading and train-only initialization.

TUM ground truth stores OpenCV camera-to-world poses as tx ty tz qx qy qz qw.
No pose estimation, pose interpolation or RGB/depth motion compensation occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

RAW_K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
RAW_SIZE = (640, 480)
DEPTH_UNITS_PER_METER = 5000.0


def read_rows(path: Path, columns: int) -> list[list[str]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(len(row) != columns for row in rows):
        raise ValueError(f"Unexpected TUM table format: {path}")
    stamps = np.array([float(row[0]) for row in rows], dtype=np.float64)
    if not np.isfinite(stamps).all() or not np.all(np.diff(stamps) > 0):
        raise ValueError(f"Timestamps must be finite and strictly increasing: {path}")
    return rows


def pose_from_tum(row: list[str]) -> np.ndarray:
    values = np.array(row[1:], dtype=np.float64)
    q = values[3:]
    if not np.isfinite(values).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid TUM quaternion or translation")
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("Ground-truth pose rotation is not orthonormal")
    if abs(np.linalg.det(rotation) - 1) > 1e-10:
        raise ValueError("Ground-truth pose rotation determinant is not +1")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3], c2w[:3, 3] = rotation, values[:3]
    return c2w


@dataclass(frozen=True)
class Frame:
    frame_id: int  # Ordinal after all association filters, not raw RGB index.
    rgb_index: int
    depth_index: int
    gt_index: int
    rgb_timestamp: float
    depth_timestamp: float
    gt_timestamp: float
    rgb_path: str
    depth_path: str
    c2w: np.ndarray
    gt_quaternion_norm_before_normalization: float

    def metadata(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "rgb_index": self.rgb_index,
                "depth_index": self.depth_index, "gt_index": self.gt_index,
                "rgb_timestamp": self.rgb_timestamp,
                "depth_timestamp": self.depth_timestamp,
                "gt_timestamp": self.gt_timestamp,
                "rgb_depth_abs_dt_s": abs(self.rgb_timestamp-self.depth_timestamp),
                "rgb_gt_abs_dt_s": abs(self.rgb_timestamp-self.gt_timestamp),
                "rgb_path": self.rgb_path, "depth_path": self.depth_path,
                "gt_quaternion_norm_before_normalization":
                    self.gt_quaternion_norm_before_normalization,
                "c2w": self.c2w.tolist(), "w2c": np.linalg.inv(self.c2w).tolist()}


def associate_frames(dataset: Path, max_depth_dt: float = .02,
                     max_gt_dt: float = .02, frame_rate: float = 32.) -> tuple[list[Frame], dict]:
    """Nearest timestamp association with unique depth files and a 32 Hz cap.

    The upstream SplaTAM loader uses 80 ms and can reuse a depth file. This
    teaching experiment deliberately uses tighter 20 ms and uniqueness to
    prevent held-out RGB images sharing depth supervision with training views.
    Poses are nearest samples at RGB time; they are not interpolated.
    """
    if min(max_depth_dt, max_gt_dt, frame_rate) <= 0:
        raise ValueError("Association thresholds and frame rate must be positive")
    rgb_rows = read_rows(dataset / "rgb.txt", 2)
    depth_rows = read_rows(dataset / "depth.txt", 2)
    gt_rows = read_rows(dataset / "groundtruth.txt", 8)
    depth_times = np.array([float(row[0]) for row in depth_rows])
    gt_times = np.array([float(row[0]) for row in gt_rows])
    associated = []
    used_depths: set[int] = set()
    previous_rgb = -float("inf")
    threshold_passes = rate_skips = repeated_depth_skips = 0
    for i, row in enumerate(rgb_rows):
        timestamp = float(row[0])
        j = int(np.argmin(np.abs(depth_times-timestamp)))
        k = int(np.argmin(np.abs(gt_times-timestamp)))
        if abs(depth_times[j]-timestamp) >= max_depth_dt or abs(gt_times[k]-timestamp) >= max_gt_dt:
            continue
        threshold_passes += 1
        if timestamp-previous_rgb <= 1./frame_rate:
            rate_skips += 1
            continue
        if j in used_depths:
            repeated_depth_skips += 1
            continue
        c2w = pose_from_tum(gt_rows[k])
        associated.append(Frame(len(associated), i, j, k, timestamp,
                                float(depth_times[j]), float(gt_times[k]), row[1],
                                depth_rows[j][1], c2w,
                                float(np.linalg.norm(np.array(gt_rows[k][4:], dtype=float)))))
        used_depths.add(j)
        previous_rgb = timestamp
    if not associated:
        raise ValueError("No RGB/depth/GT frames passed association")
    stats = {"method": "nearest depth and nearest GT at RGB timestamp; chronological filtering",
             "depth_max_abs_dt_s_strict": max_depth_dt,
             "gt_max_abs_dt_s_strict": max_gt_dt,
             "frame_rate_upper_limit_hz": frame_rate,
             "unique_depth_files": True, "pose_interpolation": False,
             "raw_rgb_count": len(rgb_rows), "raw_depth_count": len(depth_rows),
             "raw_gt_count": len(gt_rows), "threshold_passes": threshold_passes,
             "rate_skips": rate_skips, "repeated_depth_skips": repeated_depth_skips,
             "associated_count": len(associated),
             "difference_from_splatam_default": "20 ms instead of 80 ms; prohibit depth reuse"}
    return associated, stats


def select_split(frames: list[Frame], start: int = 30, stride: int = 4,
                 train_count: int = 10, heldout_count: int = 5) -> tuple[list[Frame], list[int], list[int]]:
    if min(stride, train_count, heldout_count) <= 0 or start < 0:
        raise ValueError("Invalid short-sequence selection")
    count = train_count + heldout_count
    selected_ids = list(range(start, start+stride*count, stride))
    if selected_ids[-1] >= len(frames):
        raise ValueError(f"Requested frame {selected_ids[-1]}, only {len(frames)} associated frames")
    selected = [frames[i] for i in selected_ids]
    # Interleaved evaluation samples, evenly distributed through the short clip.
    heldout = np.floor((np.arange(heldout_count)+1)*count/heldout_count).astype(int)-1
    test_indices = heldout.tolist()
    train_indices = [i for i in range(count) if i not in set(test_indices)]
    if len(train_indices) != train_count or len(set(test_indices)) != heldout_count:
        raise ValueError("Invalid split cardinality")
    for field in ("rgb_index", "depth_index", "rgb_path", "depth_path"):
        train_values = {getattr(selected[i], field) for i in train_indices}
        test_values = {getattr(selected[i], field) for i in test_indices}
        if train_values & test_values:
            raise ValueError(f"Training and held-out data overlap in {field}")
    return selected, train_indices, test_indices


def scaled_intrinsics(width: int, height: int) -> np.ndarray:
    if min(width, height) < 8:
        raise ValueError("Output resolution is too small")
    sx, sy = width/RAW_SIZE[0], height/RAW_SIZE[1]
    # PIL resize maps integer pixel centers by u_new=(u_old+.5)*scale-.5.
    transform = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    return transform @ RAW_K


def safe_data_path(dataset: Path, relative_path: str) -> Path:
    target = (dataset / relative_path).resolve()
    if not target.is_relative_to(dataset.resolve()) or not target.is_file():
        raise ValueError(f"Missing or unsafe TUM index path: {relative_path}")
    return target


def load_frames(dataset: Path, frames: list[Frame], width: int, height: int,
                min_depth: float = .2, max_depth: float = 5.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < min_depth < max_depth:
        raise ValueError("Depth range must be positive and increasing")
    images, depths, masks = [], [], []
    for frame in frames:
        with Image.open(safe_data_path(dataset, frame.rgb_path)) as source:
            if source.size != RAW_SIZE:
                raise ValueError("TUM RGB image dimensions differ from desk calibration")
            rgb = np.array(source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                           dtype=np.float32)/255.
        with Image.open(safe_data_path(dataset, frame.depth_path)) as source:
            raw = np.array(source)
            if source.size != RAW_SIZE or raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError("Expected 640x480 integer TUM depth PNG")
            if raw.min() < 0 or raw.max() > 65535:
                raise ValueError("Depth PNG exceeds uint16 range")
            # Nearest sampling preserves zeros and does not blend depth discontinuities.
            depth = np.array(source.resize((width, height), Image.Resampling.NEAREST),
                             dtype=np.float32)/DEPTH_UNITS_PER_METER
        valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        if not np.isfinite(rgb).all() or not np.isfinite(depth).all() or valid.mean() < .1:
            raise ValueError(f"Insufficient finite RGB-D data for frame {frame.frame_id}")
        images.append(rgb)
        depths.append(depth)
        masks.append(valid)
    return np.stack(images), np.stack(depths), np.stack(masks)


def initialize_from_training_depth(frames: list[Frame], images: np.ndarray,
                                   depths: np.ndarray, masks: np.ndarray,
                                   train_indices: list[int], K: np.ndarray,
                                   max_points: int = 30000, pixel_stride: int = 2,
                                   seed: int = 0) -> tuple[dict[str, np.ndarray], dict]:
    """Back-project training pixels only; fixed bounded count, no map fusion."""
    if max_points < len(train_indices) or pixel_stride < 1:
        raise ValueError("Invalid point cap or initialization pixel stride")
    rng = np.random.default_rng(seed)
    height, width = depths.shape[1:]
    v_grid, u_grid = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    candidates, counts = [], []
    for view in train_indices:
        valid = masks[view, v_grid, u_grid]
        v, u = v_grid[valid], u_grid[valid]
        counts.append(len(u))
        candidates.append((view, u, v))
    count = min(max_points, sum(counts))
    if count < 100:
        raise ValueError("Fewer than 100 valid initialization points")
    # Each training frame contributes fairly; redistribute any small-frame shortage.
    quotas = np.minimum(np.array(counts), count//len(train_indices))
    remainder = count-int(quotas.sum())
    while remainder:
        for i in range(len(quotas)):
            if quotas[i] < counts[i] and remainder:
                quotas[i] += 1
                remainder -= 1
    means, colors, scales, source_views = [], [], [], []
    max_roundtrip = max_reprojection = 0.
    for (view, u_all, v_all), quota in zip(candidates, quotas):
        chosen = rng.choice(len(u_all), int(quota), replace=False)
        u, v = u_all[chosen], v_all[chosen]
        z = depths[view, v, u].astype(np.float64)
        camera = np.column_stack(((u-K[0, 2])*z/K[0, 0],
                                  (v-K[1, 2])*z/K[1, 1], z))
        c2w = frames[view].c2w
        world = camera @ c2w[:3, :3].T+c2w[:3, 3]
        w2c = np.linalg.inv(c2w)
        restored = world @ w2c[:3, :3].T+w2c[:3, 3]
        projection = np.column_stack((K[0, 0]*restored[:, 0]/restored[:, 2]+K[0, 2],
                                      K[1, 1]*restored[:, 1]/restored[:, 2]+K[1, 2]))
        max_roundtrip = max(max_roundtrip, float(np.linalg.norm(restored-camera, axis=1).max()))
        max_reprojection = max(max_reprojection, float(np.linalg.norm(projection-np.column_stack((u, v)), axis=1).max()))
        if not np.all(restored[:, 2] > 0):
            raise ValueError("Training points project behind their source camera")
        footprint = np.clip(.8*pixel_stride*z/min(K[0, 0], K[1, 1]), .002, .08)
        means.append(world)
        colors.append(images[view, v, u])
        scales.append(np.repeat(footprint[:, None], 3, axis=1))
        source_views.append(np.full(len(u), frames[view].frame_id, dtype=np.int64))
    if max_roundtrip > 1e-10 or max_reprojection > 1e-8:
        raise ValueError("Double-precision initialization coordinate check failed")
    arrays = {"means": np.concatenate(means).astype(np.float32),
              "colors": np.concatenate(colors).astype(np.float32),
              "scales": np.concatenate(scales).astype(np.float32),
              "source_frame_ids": np.concatenate(source_views)}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite initialization")
    return arrays, {"point_count": count, "maximum_points": max_points,
                    "pixel_stride": pixel_stride,
                    "training_frame_ids": [frames[i].frame_id for i in train_indices],
                    "candidate_counts_by_training_frame": counts,
                    "selected_counts_by_training_frame": quotas.tolist(),
                    "roundtrip_max_m_float64": max_roundtrip,
                    "reprojection_max_pixels_float64": max_reprojection,
                    "validation_scope": "Coordinate algebra only; does not validate physical calibration or multi-frame alignment"}
