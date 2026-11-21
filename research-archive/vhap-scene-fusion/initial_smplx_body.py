"""Initialize a shared SMPL-X body in the calibrated scene world."""

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch


# COCO anatomical shoulder/hip labels correspond to these SMPL-X skeletal joints.
TORSO = {"left_shoulder": 16, "right_shoulder": 17, "left_hip": 1, "right_hip": 2}


def _triangulate(xy, reliable, K, world_from_camera, centers, tolerance):
    if len(xy) < 4 or not np.isfinite(xy[reliable]).all():
        raise ValueError("Insufficient finite native observations for body triangulation")
    observed = np.flatnonzero(reliable)
    if len(observed) < 4:
        raise ValueError(f"Only {len(observed)} reliable observations of a torso joint")
    camera_from_world = world_from_camera.transpose(0, 2, 1)
    P = np.concatenate((camera_from_world, -(camera_from_world @ centers[..., None])), axis=2)
    rays_cv = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(K).T
    rays_world = (world_from_camera @ rays_cv[..., None])[..., 0]
    rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
    equations = P[:, :2] - rays_cv[:, :2, None] / rays_cv[:, 2:3, None] * P[:, 2:3]

    def solve(indices):
        _, _, vectors = np.linalg.svd(equations[indices].reshape(-1, 4), full_matrices=False)
        point = vectors[-1]
        return point[:3] / point[3] if abs(point[3]) > 1e-10 else None

    def errors(point):
        camera = (camera_from_world @ (point - centers)[..., None])[..., 0]
        projected = camera @ K.T
        valid_depth = camera[:, 2] > 0
        pixel = projected[:, :2] / np.where(valid_depth, projected[:, 2], 1.)[:, None]
        error = np.linalg.norm(pixel - xy, axis=1) / tolerance
        return np.where(reliable & valid_depth, error, np.inf)

    # Test at most 24 temporal anchors, while scoring every reliable observation.
    anchors = observed[np.linspace(0, len(observed) - 1, min(len(observed), 24), dtype=int)]
    best, inliers = (0, -np.inf), None
    for a, b in combinations(anchors, 2):
        if rays_world[a] @ rays_world[b] > np.cos(np.deg2rad(4.)):
            continue
        point = solve([a, b])
        if point is None:
            continue
        error = errors(point)
        support = np.flatnonzero(error <= 1.)
        if support.size:
            rank = (support.size, -float(np.median(error[support])))
            if rank > best:
                best, inliers = rank, support
    if inliers is None or len(inliers) < max(4, np.ceil(.50 * len(observed))):
        raise ValueError(f"No stable triangulation: {best[0]}/{len(observed)} reliable views agree")
    if np.min(rays_world[inliers] @ rays_world[inliers].T) > np.cos(np.deg2rad(4.)):
        raise ValueError("Torso triangulation inliers have insufficient parallax")
    point = solve(inliers)
    if point is None or (errors(point) <= 1.).sum() < max(4, np.ceil(.50 * len(observed))):
        raise ValueError("Refitted torso triangulation did not retain its support")
    return point


def initialize(model, body_archive, init_records, scale, device, K):
    """Return rotation, world translation and pose for Xw = scale * R @ Xmodel + t."""
    if isinstance(body_archive, (str, Path)):
        with np.load(body_archive) as archive:
            return initialize(model, archive, init_records, scale, device, K)
    K = np.asarray(K, dtype=np.float64)
    names = [record["name"] for record in init_records]
    if body_archive["names"].tolist() != names:
        raise ValueError("Native body observations must match the registered camera frame names and order")
    if not np.isfinite(scale) or scale <= 0 or K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Expected a positive fixed model-to-world scale and finite native K")
    joint_names = list(body_archive["joint_names"])
    ids = [joint_names.index(name) for name in TORSO]
    xy = np.asarray(body_archive["keypoints"], dtype=np.float64)[:, ids]
    scores = np.asarray(body_archive["scores"], dtype=np.float64)[:, ids]
    flips = np.asarray(body_archive["flip_disagreement"], dtype=np.float64)[:, ids]
    boxes = np.asarray(body_archive["boxes"], dtype=np.float64)
    height = boxes[:, 3] - boxes[:, 1]
    reliable = (scores >= .45) & (flips <= np.maximum(3., height * .025)[:, None])
    reliable &= np.isfinite(xy).all(axis=2) & np.isfinite(height[:, None]) & (height[:, None] > 0)
    tolerance = np.maximum(3., height * .03)
    world_from_virtual = np.asarray([r["world_from_virtual"] for r in init_records])
    virtual_from_native = np.asarray([r["virtual_from_native_cv"] for r in init_records])
    rotations = world_from_virtual @ virtual_from_native
    centers = np.asarray([r["camera_center_world"] for r in init_records])
    if (not np.isfinite(centers).all() or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)):
        raise ValueError("Registered native cameras must carry valid world rotations and centers")
    estimates = []
    for i, name in enumerate(TORSO):
        try:
            estimates.append(_triangulate(xy[:, i], reliable[:, i], K, rotations, centers, tolerance))
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    target = np.asarray(estimates)

    body_pose = np.zeros((21, 3), dtype=np.float32)
    # SMPL-X body_pose excludes root; lower the left/right arms from the neutral T-pose.
    body_pose[[16 - 1, 17 - 1], 2] = np.deg2rad([-65., 65.])
    body_pose = body_pose.reshape(63)
    with torch.no_grad():
        joints = model(body_pose=torch.as_tensor(body_pose, device=device)[None]).joints
        source = scale * joints[0, list(TORSO.values())].detach().cpu().numpy().astype(np.float64)
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    a, b = source - source_center, target - target_center
    u, singular, vh = np.linalg.svd(a.T @ b)
    if singular[1] <= 1e-3 * singular[0] or singular[0] <= 0:
        raise ValueError("Triangulated torso does not constrain an upright orientation")
    proper = np.eye(3)
    proper[2, 2] = np.linalg.det(vh.T @ u.T)
    rotation = vh.T @ proper @ u.T
    translation = target_center - rotation @ source_center
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    torso_height = np.linalg.norm(target[:2].mean(axis=0) - target[2:].mean(axis=0))
    if not np.isfinite(residual).all() or residual.max() > .35 * torso_height:
        raise ValueError(f"Fixed-scale torso fit disagrees with calibrated 3D joints: "
                         f"max error {residual.max():.5g}, torso height {torso_height:.5g}")
    return cv2.Rodrigues(rotation)[0].reshape(3), translation, body_pose
