"""Repair the VHAP lower seam with visible collars and registered body geometry."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from export_head_neck_frames import boundary_loops


def unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-8)


def shoulder_axes(joints):
    lateral = unit(joints[:, 16] - joints[:, 17])
    up = joints[:, 12] - joints[:, 9]
    up = unit(up - (up * lateral).sum(axis=-1, keepdims=True) * lateral)
    return np.stack((lateral, up, np.cross(lateral, up)), axis=-1)


def mouth_protection(vertices, faces, ring):
    openings = [loop for loop in boundary_loops(faces) if not np.isin(loop, ring).any()]
    if len(openings) != 1:
        raise ValueError("Expected the native mouth opening apart from the lower neck ring")
    protected = np.unique(faces[np.isin(faces, openings[0]).any(axis=1)])
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]])), axis=1), axis=0)
    a, b = edges.T
    lengths = np.median(np.linalg.norm(vertices[:, a] - vertices[:, b], axis=-1), axis=0)
    graph = coo_matrix((lengths, (a, b)), shape=(vertices.shape[1],) * 2).tocsr()
    distance = dijkstra(graph + graph.T, indices=protected, directed=False, min_only=True)
    s = np.clip(distance / .075, 0, 1)
    return s*s*(3 - 2*s), protected


def read_observations(frame_source, target_file, names):
    with ZipFile(frame_source) as source, np.load(target_file, allow_pickle=False) as target:
        manifest = source.read("initialization.json")
        records = json.loads(manifest)["frames"]
        if (str(target["initialization_sha256"].item()) != sha256(manifest).hexdigest()
                or [r["name"] for r in records] != names or target["names"].tolist() != names):
            raise ValueError("Inputs have different camera or frame provenance")
        masks, curves = [], []
        seam = target["collar"]
        matte_hashes = target["matte_sha256"].tolist()
        for i in range(len(names)):
            alpha = source.read(f"alpha_maps/{i:08d}.jpg")
            if matte_hashes[i] != sha256(alpha).hexdigest():
                raise ValueError(f"Unregistered matte at {names[i]}")
            matte = cv2.imdecode(np.frombuffer(alpha, np.uint8), cv2.IMREAD_GRAYSCALE)
            masks.append(matte)
            y, x = np.where(seam[i])
            # One robust column-center per 2 pixels, no extrapolation between components.
            xy = np.array([(col, np.median(y[(x >= col) & (x < col + 2)]))
                           for col in np.unique(x // 2 * 2)], dtype=np.float64).reshape(-1, 2)
            curves.append(xy if len(xy) >= 12 and abs(records[i]["baseline_yaw_deg"]) <= 65 else xy[:0])
    return records, masks, curves


def fit_ring(head, body, masks, curves, records):
    vertices = head["vertices_world"]
    ring = head["neck_boundary_vertex_ids"]
    joints = body["joints_world"].astype(np.float64)
    basis = shoulder_axes(joints)
    origin = joints[:, 12, None]
    local = (vertices - origin) @ basis
    median = np.median(local[:, ring], axis=0)
    center = np.median(median, axis=0)
    # A continuous layer taper preserves the existing triangles; pulling binary rims
    # independently collapsed small triangles along the cut.
    y = np.median(local[..., 1], axis=0)
    upper = np.quantile(y[head["neck_vertex_ids"]], .83)
    taper = np.clip((upper - y) / (upper - center[1]), 0., 1.)
    protection, mouth = mouth_protection(vertices, head["faces"], ring)
    taper = taper**2 * (3 - 2*taper) * protection
    radial = median[:, [0, 2]] - center[[0, 2]]
    radial_neck = unit(radial)
    cameras = head["native_camera_to_world_rotation"]
    centers = head["native_camera_center"]
    projection = head["virtual_crop_homographies"] @ head["native_K"]
    eye_line = unit((centers[:, None] - origin) @ basis)[:, 0, [0, 2]]
    visibility = eye_line @ radial_neck.T > .15
    observed = np.asarray([len(c) > 0 for c in curves])
    validation = np.arange(len(masks)) % 5 == 1
    use = observed & ~validation
    if use.sum() < 10 or (observed & validation).sum() < 5:
        raise ValueError("Too few independently sampled visible collars")

    signed = []
    for mask in masks:
        foreground = (mask >= 128).astype(np.uint8)
        # The artificial bottom crop is well below the actual neckline. No score at that edge.
        signed.append(cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
                      - cv2.distanceTransform(foreground, cv2.DIST_L2, 5))
    samples = np.arange(0., 1., .25)[None, None, :, None]
    factor = np.array([.32, .06, .07, .08, .05, .06, .06])

    def deform(c, ids):
        xscale, depth_shift, height_shift, xshift, tilt, depth_scale, lift_sides = c
        v = local[:, ids]
        relative = v[..., [0, 2]] - center[[0, 2]]
        direction = unit(relative)
        delta = np.stack((xscale * relative[..., 0] + xshift,
                          height_shift + tilt * direction[..., 1] + lift_sides * direction[..., 0]**2,
                          depth_shift + depth_scale * relative[..., 1]), axis=-1)
        return vertices[:, ids] + (delta * taper[ids][None, :, None]) @ basis.transpose(0, 2, 1)

    def pixels(v):
        samples3 = (v[:, :, None, :] * (1 - samples) + np.roll(v, -1, axis=1)[:, :, None] * samples)
        p = ((samples3.reshape(len(v), -1, 3) - centers[:, None]) @ cameras) @ projection.transpose(0, 2, 1)
        if (p[..., 2] <= 0).any():
            raise ValueError("Neckline must remain in front of every camera")
        return p[..., :2] / p[..., 2:]

    visible = np.repeat(visibility, samples.size, axis=1)

    def error(c, indices):
        xy = pixels(deform(c, ring))
        collar, outside = [], []
        for i in indices:
            p = xy[i]
            outside.append(np.maximum(map_coordinates(signed[i], p.T[::-1], order=1, mode="nearest"), 0.))
            if observed[i]:
                distances = cdist(p[visible[i]], curves[i])
                collar.append(np.r_[distances.min(axis=0), distances.min(axis=1)])
        return collar, outside

    train_indices = np.flatnonzero(~validation)

    def objective(c):
        collar, outside = error(c, train_indices)
        a = np.concatenate([v / np.sqrt(len(v)) for v in collar])
        b = np.concatenate([v / np.sqrt(len(v)) for v in outside])
        return np.r_[1.4 * a, 1.0 * b, 2.2 * c / factor]

    initial = np.zeros(7)
    fitted = least_squares(objective, initial, loss="soft_l1", f_scale=.6, max_nfev=60,
                           bounds=([-0.63, -.058, -.04, -.09, -.024, -.48, -.024],
                                   [.05, .095, .04, .09, .024, .15, .024]))
    if not fitted.success:
        raise RuntimeError(f"Rim fit stalled: {fitted.message}")
    band = np.flatnonzero(taper)
    faces = head["faces"]
    local_faces = faces[np.isin(faces, band).any(axis=1)]
    a = vertices[:, local_faces]
    reference_normal = np.cross(a[:, :, 1] - a[:, :, 0], a[:, :, 2] - a[:, :, 0])
    after = vertices.copy()
    # Find the longest safe step in the fitted direction. Never accept new folds
    # or a near-collapsed triangle to improve a projected rim measurement.
    def safe_step(fraction):
        after[:, band] = deform(fitted.x * fraction, band)
        p = after[:, local_faces]
        n = np.cross(p[:, :, 1] - p[:, :, 0], p[:, :, 2] - p[:, :, 0])
        return ((unit(reference_normal) * unit(n)).sum(axis=-1).min() >= .25
                and (np.linalg.norm(n, axis=-1) / np.linalg.norm(reference_normal, axis=-1)).min() >= .35)

    lo, hi = 0., 1.
    if safe_step(1.):
        lo = 1.
    else:
        for _ in range(16):
            middle = (lo + hi) / 2
            if safe_step(middle):
                lo = middle
            else:
                hi = middle
        safe_step(lo)
    final = fitted.x * lo
    if lo < .1:
        raise RuntimeError("The rim target conflicts with the mesh topology")
    if not np.array_equal(after[:, mouth], vertices[:, mouth]):
        raise RuntimeError("The seam edit moved protected mouth vertices")
    result = {}
    for label, ids in (("fit", train_indices), ("check", np.flatnonzero(validation)),
                       ("profiles", np.array([i for i, r in enumerate(records) if abs(r["baseline_yaw_deg"]) >= 80]))):
        evaluations = []
        for parameters in (initial, final):
            collar, outside = error(parameters, ids)
            evaluations.append(dict(collar_frames=len(collar),
                                    typical_collar_distance_px=float(np.median([np.median(a) for a in collar])) if collar else None,
                                    median_leaking_fraction=float(np.median([(a > 1).mean() for a in outside])),
                                    farthest_background_px=[float(a.max()) for a in outside]))
        result[label] = {"frames": [records[i]["name"] for i in ids], **dict(zip(("before", "after"), evaluations))}
    return after, result, final.tolist(), lo


def run(args):
    cv2.setNumThreads(1)
    with np.load(args.head, allow_pickle=False) as archive:
        head = {k: archive[k] for k in archive.files}
    with np.load(args.body, allow_pickle=False) as archive:
        body = {k: archive[k] for k in ("names", "joints_world", "native_K", "native_camera_center", "native_camera_to_world_rotation", "scene_scale")}
    for k in ("names", "native_K", "native_camera_center", "native_camera_to_world_rotation", "scene_scale"):
        if not np.array_equal(head[k], body[k]):
            raise ValueError(f"Body and head disagree on {k}")
    names = head["names"].tolist()
    records, masks, curves = read_observations(args.frames, args.targets, names)
    before = head["vertices_world"]
    faces = head["faces"]
    after, measurements, parameters, step_fraction = fit_ring(head, body, masks, curves, records)
    changed = np.any(before != after, axis=(0, 2))
    local = faces[changed[faces].any(axis=1)]
    normals = []
    for p in (before, after):
        triangles = p[:, local]
        normals.append(np.cross(triangles[:, :, 1]-triangles[:, :, 0], triangles[:, :, 2]-triangles[:, :, 0]))
    rotation_cosine = (unit(normals[0]) * unit(normals[1])).sum(axis=-1)
    area_ratio = np.linalg.norm(normals[1], axis=-1) / np.linalg.norm(normals[0], axis=-1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(frames=len(names), optimized=parameters, accepted_step_fraction=step_fraction, measurements=measurements,
                  changed_vertices=int(changed.sum()), unchanged_vertices=int((~changed).sum()),
                  min_triangle_cosine=float(rotation_cosine.min()),
                  triangles_rotated_more_than_90=int((rotation_cosine < 0).sum()),
                  area_ratio_percentiles=np.percentile(area_ratio, [0, 1, 50, 99, 100]).tolist())
    (args.output / "lower-seam-fit.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    head["vertices_world"] = after
    np.savez_compressed(args.output / "head-neck-lower-seam.npz", **head)
    print(json.dumps({k: v for k, v in report.items() if k != "measurements"}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("head", "body", "frames", "targets", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
