"""Explicit local HSfM interface adapter; upstream sources remain unmodified.

Coordinates are OpenCV-style zero-based raster centers unless explicitly named
edge or COLMAP. No learned subject parameters, image warps or camera fitting.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RasterTransform:
    original_wh: tuple
    resized_wh: tuple
    crop_xyxy: tuple

    @property
    def output_wh(self):
        x0, y0, x1, y1 = self.crop_xyxy
        return x1 - x0, y1 - y0

    @property
    def centers(self):
        sx, sy = np.asarray(self.resized_wh) / np.asarray(self.original_wh)
        x0, y0, _, _ = self.crop_xyxy
        return np.array([[sx, 0., (sx - 1.) / 2. - x0],
                         [0., sy, (sy - 1.) / 2. - y0], [0., 0., 1.]])

    @property
    def edges(self):
        sx, sy = np.asarray(self.resized_wh) / np.asarray(self.original_wh)
        x0, y0, _, _ = self.crop_xyxy
        return np.array([[sx, 0., -x0], [0., sy, -y0], [0., 0., 1.]])

    def intrinsics(self, K_centers):
        return self.centers @ validate_K(K_centers)


def dust3r_raster_transform(original_wh, long_edge=512, square_ok=False):
    w, h = original_wh
    if min(w, h) <= 0 or long_edge != 512:
        raise ValueError('This adapter is registered only for positive-size 512 input')
    nw, nh = (int(round(x * long_edge / max(w, h))) for x in (w, h))
    cx, cy = nw // 2, nh // 2
    hw, hh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if not square_ok and nw == nh:
        hh = int(3 * hw / 4)
    return RasterTransform((w, h), (nw, nh), (cx-hw, cy-hh, cx+hw, cy+hh))


def map_pixels(xy, matrix):
    p = np.asarray(xy, dtype=np.float64)
    q = np.concatenate((p, np.ones(p.shape[:-1] + (1,))), axis=-1) @ np.asarray(matrix).T
    return q[..., :2] / q[..., 2:]


def validate_K(K):
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Invalid K')
    if min(K[0, 0], K[1, 1]) <= 0 or abs(np.linalg.det(K)) < 1e-12:
        raise ValueError('Singular or nonpositive intrinsics')
    return K


def colmap_to_raster_K(K_corner):
    return np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]]) @ validate_K(K_corner)


def invert_pose(pose):
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (4, 4) or not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1]):
        raise ValueError('Pose must be finite SE(3)')
    R = p[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1., atol=1e-6):
        raise ValueError('Pose rotation is not SO(3)')
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p[:3, 3]
    return out


def project_pinhole(points_world, K_centers, c2w):
    w2c = invert_pose(c2w)
    p = np.asarray(points_world) @ w2c[:3, :3].T + w2c[:3, 3]
    if np.any(p[..., 2] <= 0):
        raise ValueError('Projection contains nonpositive depth')
    q = p @ validate_K(K_centers).T
    return q[..., :2] / q[..., 2:], p[..., 2]


def bbox_area_xyxy(box):
    b = np.asarray(box, dtype=np.float64)
    size = b[..., 2:] - b[..., :2]
    if b.shape[-1] != 4 or not np.isfinite(b).all() or np.any(size <= 0):
        raise ValueError('Invalid xyxy box')
    return np.prod(size, axis=-1)


def estimate_root_translation(joints_root_relative, joints_2d, K_centers, confidence, threshold=.5):
    """Weighted linear perspective translation, known orientation/shape only.

    Every 3D row and 2D row must have the SAME named anatomical landmark.
    No assumption that row zero is pelvis, nor that all joints share depth.
    Used only as initializer. A low residual does not validate the body prior.
    """
    X, u = np.asarray(joints_root_relative, dtype=float), np.asarray(joints_2d, dtype=float)
    c = np.asarray(confidence, dtype=float)
    if X.shape != (len(u), 3) or u.shape[-1] != 2 or c.shape != (len(u),):
        raise ValueError('Joint dimensions disagree')
    keep = np.isfinite(X).all(1) & np.isfinite(u).all(1) & np.isfinite(c) & (c >= threshold)
    if keep.sum() < 4:
        raise ValueError('Insufficient visible same-name landmarks')
    X, u, c = X[keep], u[keep], c[keep]
    xy1 = np.c_[u, np.ones(len(u))] @ np.linalg.inv(validate_K(K_centers)).T
    n = xy1[:, :2] / xy1[:, 2:]
    A = np.zeros((len(n), 2, 3))
    A[:, 0, 0] = 1
    A[:, 1, 1] = 1
    A[:, :, 2] = -n
    b = n * X[:, 2:] - X[:, :2]
    weight = np.sqrt(np.clip(c, 0, 1))[:, None, None]
    A = (A * weight).reshape(-1, 3)
    b = (b * weight[:, :, 0]).reshape(-1)
    root, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    if rank != 3 or sv[0] / sv[-1] > 1e6 or np.any((X + root)[:, 2] <= 0):
        raise ValueError('Degenerate or behind-camera root initialization')
    return root, dict(used=int(keep.sum()), condition_number=float(sv[0]/sv[-1]),
                      algebraic_rmse=float(np.sqrt(np.mean((A @ root - b)**2))))


def strict_name_mapping(source_names, target_names):
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        raise ValueError('Duplicate landmark name; cannot establish unique mapping')
    shared = [n for n in target_names if n in source_names]
    return (shared, np.array([source_names.index(n) for n in shared], dtype=int),
            np.array([target_names.index(n) for n in shared], dtype=int))
