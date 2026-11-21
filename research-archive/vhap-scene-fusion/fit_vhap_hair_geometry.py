"""Refine the registered scalp against native training-view silhouette geometry."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch


@dataclass
class View:
    frame: object
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray
    contour: np.ndarray
    normals: np.ndarray
    tree: object
    bottom: int


def _array(tensor):
    return tensor.detach().cpu().numpy()


def _observed(frame):
    K, (width, height), R, C = frame.rasterizer.calibration
    y, x = frame.crop
    start = np.array([x.start, y.start])
    observed = _array(frame.masks["head_rgb"][y, x]).astype(np.uint8)
    allowed = _array(frame.masks["head_allowed"][y, x]).astype(np.uint8)
    contours, _ = cv2.findContours(observed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    raw = max(contours, key=cv2.contourArea)
    if len(raw) < 24 or cv2.contourArea(raw) < 100:
        return None
    xy = raw[:, 0].astype(float)
    tangents = np.roll(xy, -4, axis=0) - np.roll(xy, 4, axis=0)
    normals = np.sign(cv2.contourArea(raw, oriented=True)) * tangents[:, [1, 0]] * [1, -1]
    norm = np.linalg.norm(normals, axis=1)
    col, row = xy.astype(int).T
    bottom = int(row.min() + .65 * (row.max() - row.min() + 1))
    interior = cv2.erode(allowed, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    keep = (row < bottom) & (norm > 0) & (interior[row, col] > 0)
    xy += start
    keep &= ((xy >= 3) & (xy < np.array([width, height]) - 3)).all(1)
    # Reject the contour where the coarse head gate itself cut off the person.
    if keep.sum() < 12:
        return None
    contour = xy[keep]
    return View(frame, K, R, C, contour, normals[keep] / norm[keep, None], cKDTree(contour), bottom + y.start)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _mesh_edges(faces, movable):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edge_keys = np.sort(edges, axis=1)
    unique, membership, count = np.unique(edge_keys, axis=0, return_inverse=True, return_counts=True)
    # Two incident triangles let face winding determine the outward image normal.
    selected = (count == 2) & movable[unique].any(1)
    groups = np.argsort(membership, kind="stable")
    starts = np.cumsum(count) - count
    uses = groups[starts[selected, None] + [0, 1]]
    opposing = np.concatenate((faces[:, 2], faces[:, 0], faces[:, 1]))
    return unique[selected], uses % len(faces), opposing[uses]


def _occluded(points, triangles):
    """Test projected triangle union at pixels just outside each candidate edge."""
    lower, upper = triangles.min(1), triangles.max(1)
    covered = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        inside_box = ((point >= lower) & (point <= upper)).all(1)
        tri = triangles[inside_box]
        if len(tri):
            side = _cross(np.roll(tri, -1, axis=1) - tri, point - tri)
            covered[i] = ((side >= 0).all(1) | (side <= 0).all(1)).any()
    return covered


def _match(mesh, view, faces, adjacency, max_distance):
    vertices = _array(mesh.vertices_world(view.frame)).astype(float)
    camera = (vertices - view.C) @ view.R
    if not np.isfinite(camera).all() or (camera[:, 2] <= 0).any():
        return None
    projection = camera @ view.K.T
    xy = projection[:, :2] / projection[:, 2:]
    tri = camera[faces]
    winding = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = (winding * tri[:, 0]).sum(1) < 0
    edges, incident, opposing = adjacency
    keep = front[incident[:, 0]] != front[incident[:, 1]]
    edges, incident, opposing = (item[keep] for item in (edges, incident, opposing))
    if not len(edges):
        return None
    midpoint_camera = camera[edges].mean(1)
    p = midpoint_camera @ view.K.T
    midpoint = p[:, :2] / p[:, 2:]
    edge = xy[edges[:, 1]] - xy[edges[:, 0]]
    length = np.linalg.norm(edge, axis=1)
    normal = edge[:, [1, 0]] * [1, -1] / np.maximum(length, 1e-8)[:, None]
    third = opposing[np.arange(len(edges)), (~front[incident[:, 0]]).astype(int)]
    direction = ((xy[third] - midpoint) * normal).sum(1)
    normal *= np.where(direction > 0, -1, 1)[:, None]
    _, near = view.tree.query(midpoint)
    observed, target_normal = view.contour[near], view.normals[near]
    delta = midpoint - observed
    normal_error = (delta * target_normal).sum(1)
    tangent_error = abs(_cross(delta, target_normal))
    keep = ((midpoint[:, 1] < view.bottom) & (length > .65) & (abs(direction) > .05)
            & (np.linalg.norm(delta, axis=1) <= max_distance) & ((normal * target_normal).sum(1) >= .65)
            & (tangent_error <= np.maximum(2.5, .45 * abs(normal_error))))
    # A facing transition hidden by a different surface is not an outer silhouette.
    candidates = np.flatnonzero(keep)
    visible = ~_occluded(midpoint[candidates] + 1.25 * normal[candidates], xy[faces])
    selected = candidates[visible]
    if len(selected) < 8:
        return None
    return edges[selected], observed[selected], target_normal[selected]


def _pairs(mesh, views, faces, adjacency, max_distance):
    pairs = []
    like = mesh.offset
    tensor = lambda values: torch.as_tensor(values, dtype=like.dtype, device=like.device)
    for view in views:
        match = _match(mesh, view, faces, adjacency, max_distance)
        if match is not None:
            edges, observed, normals = match
            ids = torch.as_tensor(edges, dtype=torch.long, device=like.device)
            pairs.append((view.frame, ids, tensor(observed), tensor(normals), tensor(view.K), tensor(view.R), tensor(view.C)))
    return pairs


def _residual(mesh, pairs):
    values = []
    for frame, edges, observed, normals, K, R, C in pairs:
        midpoints = mesh.vertices_world(frame)[edges].mean(1)
        camera = (midpoints - C) @ R
        pixels = camera @ K.T
        error = ((pixels[:, :2] / pixels[:, 2:] - observed) * normals).sum(1)
        values.append(error)
    return values


def _objective(values):
    # Equal view weighting and a smooth robust penalty in native pixel units.
    return torch.stack([4 * ((1 + (v / 4).square()).sqrt() - 1).mean() for v in values]).mean()


def _summary(pairs, values):
    count = {pair[0].name: len(error) for pair, error in zip(pairs, values, strict=True)}
    absolute = torch.cat(values).abs() if values else None
    return dict(frames=count, correspondences=sum(count.values()), objective=float(_objective(values)) if values else None,
                mean_abs_normal_px=float(absolute.mean()) if values else None,
                p90_abs_normal_px=float(torch.quantile(absolute, .9)) if values else None)


def _face_normals(mesh, frame, faces):
    a, b, c = mesh.vertices_world(frame)[faces].unbind(1)
    return torch.linalg.cross(b - a, c - a)


def fit_hair_outline(mesh, train, *, max_steps=40, max_match_px=24., max_change=.008):
    """Fit only scalp offsets from registered training cameras, preserving all other parameters."""
    train = tuple(train)
    if len({frame.name for frame in train}) != len(train) or not 1 <= max_steps <= 120:
        raise ValueError("Expected distinct training frames and a bounded number of iterations")
    if not np.isfinite([max_match_px, max_change]).all() or max_match_px <= 0 or not 0 < max_change <= mesh.max_offset:
        raise ValueError("Invalid native-pixel match radius or tracker-unit offset bound")
    faces = _array(mesh.faces)
    scalp = mesh.editable & mesh.region_mask("scalp") & ~mesh.region_mask("neck")
    active = scalp[mesh.editable]
    affected = mesh.faces[scalp[mesh.faces].any(1)]
    adjacency = _mesh_edges(faces, _array(scalp))
    views = [view for frame in train if (view := _observed(frame)) is not None]
    starting = mesh.offset.detach().clone()
    was_trainable = mesh.offset.requires_grad
    starting_grad = mesh.offset.grad
    accepted = False
    report = dict(accepted=False, training_views=len(train), usable_observation_views=len(views),
                  max_match_px=max_match_px, max_change_tracker_units=max_change)

    with torch.no_grad():
        pairs = _pairs(mesh, views, faces, adjacency, max_match_px)
        old = _summary(pairs, _residual(mesh, pairs))
    report["before"] = old
    if len(pairs) < 4 or old["correspondences"] < 40:
        report["reason"] = "insufficient_contour_correspondences"
        return report

    # Avoid carrying Adam history from RGB optimization. Clamp the physical
    # change around the restored checkpoint and preserve neck parameters exactly.
    lower, upper = ((starting.tanh() + sign * max_change / mesh.max_offset).clamp(-.99999, .99999).atanh()
                    for sign in (-1, 1))
    with torch.no_grad():
        original_normals = [_face_normals(mesh, frame, affected) for frame in train]
    try:
        with torch.enable_grad():
            mesh.offset.requires_grad_(True)
            optimizer = torch.optim.Adam([mesh.offset], lr=.035)
            for _ in range(max_steps):
                loss = _objective(_residual(mesh, pairs)) + mesh.regularizer()
                gradient, = torch.autograd.grad(loss, mesh.offset)
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("Nonfinite scalp silhouette gradient")
                mesh.offset.grad = gradient.masked_fill(~active, 0)
                optimizer.step()
                with torch.no_grad():
                    mesh.offset.copy_(torch.where(active, torch.maximum(lower, torch.minimum(upper, mesh.offset)), starting))
        with torch.no_grad():
            fixed = _summary(pairs, _residual(mesh, pairs))
            refreshed = _pairs(mesh, views, faces, adjacency, max_match_px)
            new = _summary(refreshed, _residual(mesh, refreshed))
            changes = ((mesh.offset.tanh() - starting.tanh()) * mesh.max_offset).abs()
            comparable = all(new["frames"].get(name, 0) >= .8 * n for name, n in old["frames"].items())
            geometry_rejections = []
            for frame, normal in zip(train, original_normals, strict=True):
                result = _face_normals(mesh, frame, affected)
                nondegenerate = normal.norm(dim=1) > 1e-12
                bad = ((normal * result).sum(1) <= 0) | (result.norm(dim=1) < .25 * normal.norm(dim=1))
                if not torch.isfinite(result).all() or (bad & nondegenerate).any():
                    geometry_rejections.append(frame.name)
            accepted = (not geometry_rejections and comparable and fixed["objective"] < old["objective"]
                        and new["objective"] < old["objective"] and new["mean_abs_normal_px"] < old["mean_abs_normal_px"])
            report.update(accepted=accepted, after_fixed_matches=fixed, after_refreshed_matches=new,
                          proposed_max_change_tracker_units=float(changes.max()), proposed_moved_vertices=int((changes > 1e-7).sum()),
                          geometry_rejection_frames=geometry_rejections,
                          reason="improved" if accepted else "geometry_rejection_or_contour_regression")
    finally:
        with torch.no_grad():
            if not accepted:
                mesh.offset.copy_(starting)
            mesh.offset.grad = starting_grad
            mesh.offset.requires_grad_(was_trainable)
    return report
