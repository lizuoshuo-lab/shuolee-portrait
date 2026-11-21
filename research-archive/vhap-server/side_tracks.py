"""Track native head texture and triangulate it from calibrated pose anchors."""

from itertools import combinations

import cv2
import numpy as np


_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 28, .01))
_MIN_ANCHORS = 3
_ANCHOR_ERROR_PX = 2.25
_MIN_BASELINE_DEG = 1.5
_MAX_ACTIVE_TRACKS = 112
_MAX_TARGET_TRACKS = 60
_MAX_ANCHOR_DISTANCE = 20
_MAX_ANCHOR_SECONDS = 7.5
_SUPPORT_KERNEL = np.ones((7, 7), np.uint8)


def _inside(mask, xy):
    h, w = mask.shape
    valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < w - .5)
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] < h - .5)
    idx = np.flatnonzero(valid)
    x, y = np.rint(xy[idx]).astype(int).T
    valid[idx] = mask[y, x] > 0
    return valid


def _advance(before, image, xy, mask):
    next_xy, found, error = cv2.calcOpticalFlowPyrLK(before, image, xy, None, **_LK)
    if next_xy is None:
        return xy[:0], np.empty(0, dtype=int)
    valid = (found[:, 0] > 0) & (error[:, 0] <= 24)
    valid &= _inside(mask, next_xy[:, 0])
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return xy[:0], idx
    back_xy, back_found, _ = cv2.calcOpticalFlowPyrLK(image, before, next_xy[idx], None, **_LK)
    if back_xy is None:
        return xy[:0], idx[:0]
    fb_error = np.linalg.norm((back_xy - xy[idx])[:, 0], axis=1)
    idx = idx[(back_found[:, 0] > 0) & (fb_error <= .9)]
    return next_xy[idx], idx


def _sweep(names, grayscale, support, anchors):
    tracks, active = [], []
    xy = np.empty((0, 1, 2), dtype=np.float32)
    previous = None
    for name in names:
        image, mask = grayscale[name], support[name]
        if len(xy):
            xy, surviving = _advance(previous, image, xy, mask)
            active = [active[i] for i in surviving]
            for track, point in zip(active, xy[:, 0]):
                track[name] = point.copy()
        previous = image
        if name not in anchors:
            continue
        capacity = min(44, _MAX_ACTIVE_TRACKS - len(active))
        if capacity <= 0:
            continue
        fresh_mask = mask.copy()
        for u, v in np.rint(xy[:, 0]).astype(int):
            cv2.circle(fresh_mask, (u, v), 7, 0, -1)
        seeds = cv2.goodFeaturesToTrack(image, maxCorners=capacity, qualityLevel=.012,
                                        minDistance=7, mask=fresh_mask, blockSize=5)
        if seeds is None:
            continue
        fresh = [{name: point.copy()} for point in seeds[:, 0]]
        active.extend(fresh)
        tracks.extend(fresh)
        xy = np.concatenate((xy, seeds))
    return tracks


def _solve(projections, xy):
    equations = (xy[:, :, None] * projections[:, 2:3] - projections[:, :2]).reshape(-1, 4)
    norms = np.linalg.norm(equations[:, :3], axis=1, keepdims=True)
    _, _, vh = np.linalg.svd(equations / np.maximum(norms, 1e-12), full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) <= 1e-10:
        return None
    return homogeneous[:3] / homogeneous[3]


def _reprojection(projections, xy, point):
    rays = projections[:, :, :3] @ point + projections[:, :, 3]
    valid = rays[:, 2] > 1e-8
    error = np.full(len(rays), np.inf)
    error[valid] = np.linalg.norm(rays[valid, :2] / rays[valid, 2:3] - xy[valid], axis=1)
    return error


def _triangulate(track, anchors, inverse, bounds):
    names = [name for name in track if name in anchors]
    if len(names) < _MIN_ANCHORS:
        return None
    xy = np.array([track[name] for name in names], dtype=np.float64)
    proj = np.array([anchors[name] for name in names])
    directions = np.array([inverse[name] @ [*point, 1.] for name, point in zip(names, xy)])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    pairs = sorted(combinations(range(len(names)), 2), key=lambda pair: -angles[pair])[:12]
    best = None
    for pair in pairs:
        if angles[pair] < _MIN_BASELINE_DEG:
            break
        sample = list(pair)
        point = _solve(proj[sample], xy[sample])
        if point is None:
            continue
        error = _reprojection(proj, xy, point)
        inlier = error <= _ANCHOR_ERROR_PX
        count = int(inlier.sum())
        if count < max(_MIN_ANCHORS, (len(names) + 1) // 2):
            continue
        candidate = (-count, float(np.median(error[inlier])), inlier)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    inlier = best[2]
    point = _solve(proj[inlier], xy[inlier])
    if point is None or not np.isfinite(point).all() or not ((point >= bounds[0]) & (point <= bounds[1])).all():
        return None
    error = _reprojection(proj[inlier], xy[inlier], point)
    baseline = float(np.max(angles[np.ix_(inlier, inlier)]))
    if not (error <= _ANCHOR_ERROR_PX).all() or baseline < _MIN_BASELINE_DEG:
        return None
    info = {"anchor_names": [name for name, keep in zip(names, inlier) if keep],
            "anchor_reprojection_px": float(np.sqrt(np.mean(error ** 2))),
            "baseline_deg": baseline}
    return point, info


def track_features(names, grayscale, support, anchor_projections, bounds, *, timestamps=None, targets=None):
    """Return non-anchor observations with point provenance, in FLAME-local coordinates.

    Anchor projections map that same local frame to native-image pixel centers.
    Target observations contribute no equations to their own 3D triangulation.
    Timestamps in seconds keep the allowed travel independent of image sampling.
    """
    names = list(names)
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not (np.isfinite(bounds).all() and (bounds[1] > bounds[0]).all()):
        raise ValueError("Expected finite lower and upper FLAME-local head bounds")
    if len(names) != len(set(names)) or not set(anchor_projections).issubset(names):
        raise ValueError("Track frame names must be unique and cover the pose anchors")
    candidates = {name: [] for name in (names if targets is None else targets)}
    if not set(candidates).issubset(names):
        raise ValueError("Track targets must appear in the input sequence")
    frame_order = {name: index for index, name in enumerate(names)}
    if timestamps is not None:
        values = np.array([timestamps[name] for name in names], dtype=np.float64)
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise ValueError("Track timestamps must be finite and strictly increasing")
    if len(anchor_projections) >= _MIN_ANCHORS:
        anchors = {}
        inside = {}
        shape = grayscale[names[0]].shape
        for name in names:
            image, mask = grayscale[name], support[name]
            if len(shape) != 2 or image.shape != shape or image.dtype != np.uint8 or mask.shape != shape:
                raise ValueError(f"Native grayscale or head support size disagrees for {name}")
            inside[name] = cv2.erode(np.ascontiguousarray(mask, dtype=np.uint8), _SUPPORT_KERNEL,
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if name in anchor_projections:
                projection = np.asarray(anchor_projections[name], dtype=np.float64)
                if projection.shape != (3, 4) or not np.isfinite(projection).all():
                    raise ValueError(f"Invalid pose anchor projection for {name}")
                anchors[name] = projection
        inverse = {name: np.linalg.inv(proj[:, :3]) for name, proj in anchors.items()}
        tracks = _sweep(names, grayscale, inside, anchors)
        tracks.extend(_sweep(reversed(names), grayscale, inside, anchors))
        for track in tracks:
            fit = _triangulate(track, anchors, inverse, bounds)
            if fit is None:
                continue
            point, info = fit
            for target, xy in track.items():
                if target in anchors or target not in candidates:
                    continue
                distance = min(abs(frame_order[target] - frame_order[anchor]) for anchor in info["anchor_names"])
                measurement = dict(info, nearest_anchor_frames=distance)
                if timestamps is not None:
                    seconds = min(abs(timestamps[target] - timestamps[anchor]) for anchor in info["anchor_names"])
                    if seconds > _MAX_ANCHOR_SECONDS:
                        continue
                    measurement["nearest_anchor_seconds"] = float(seconds)
                elif distance > _MAX_ANCHOR_DISTANCE:
                    continue
                candidates[target].append((point, xy, measurement))

    results = {}
    for name, options in candidates.items():
        quality = lambda obs: (-len(obs[2]["anchor_names"]), obs[2]["anchor_reprojection_px"],
                               obs[2].get("nearest_anchor_seconds", obs[2]["nearest_anchor_frames"]))
        picked = []
        for point, xy, info in sorted(options, key=quality):
            if all(np.linalg.norm(xy - previous[1]) >= 5 for previous in picked):
                picked.append((point, xy, info))
            if len(picked) == _MAX_TARGET_TRACKS:
                break
        results[name] = {"points3d": np.asarray([row[0] for row in picked], dtype=np.float64).reshape(-1, 3),
                         "points2d": np.asarray([row[1] for row in picked], dtype=np.float64).reshape(-1, 2),
                         "metadata": [row[2] for row in picked]}
    return results
