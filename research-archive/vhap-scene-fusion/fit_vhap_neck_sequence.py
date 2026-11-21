"""Jointly refine visible neck contours with fixed cameras and compensated head pose."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, file_hash, mesh_intersections, project
from prepare_video_avatar import setup_vhap_cameras


CONTOUR_WEIGHT = 4.
HEAD_P95_LIMIT = .50
FACE_LANDMARK_LIMIT = 1.
POSE_KEYS = ("neck_pose", "rotation", "translation")


def load_observations(args, records):
    current = json.loads(args.targets.read_bytes())
    historical = json.loads(args.native_fallback.read_bytes())
    source = json.loads((args.scene / "source-frames.json").read_bytes())
    expected = file_hash(args.sequence / "initialization.json")
    supplemental = json.loads(args.supplement.read_bytes()) if args.supplement else None
    if supplemental is not None and (supplemental["source"]["initialization_sha256"] != expected
                                     or supplemental["source"]["video_sha256"] != source["video_sha256"]):
        raise ValueError("Supplemental contours refer to a different source or camera")
    if current["source"]["initialization_sha256"] != expected:
        raise ValueError("Current contours were computed with a different crop")
    if any(x["source"]["video_sha256"] != source["video_sha256"] for x in (current, historical)):
        raise ValueError("Contour observations refer to a different source video")
    observations = []
    for i, (record, src) in enumerate(zip(records, source["frames"], strict=True)):
        name = record["name"]
        image = args.scene / "orig_undist/images" / name
        crop = args.sequence / "images" / f"{i:08d}.jpg"
        if name != src["name"] or i != int(record["timestep"]) or src["native_sha256"] != file_hash(image):
            raise ValueError(f"Source frame identity changed at {name}")
        now, old = (x["frames"][name] for x in (current, historical))
        extra = supplemental["frames"].get(name) if supplemental is not None else None
        matte_hash = file_hash(args.scene / "alpha_maps" / name)
        for row in (now, old, *((extra,) if extra is not None else ())):
            if (row["native_sha256"] != src["native_sha256"] or row["raw_matte_sha256"] != matte_hash
                    or row["timestamp_s"] != record["timestamp_s"]):
                raise ValueError(f"Detached contour observations at {name}")
        current_crop_hash = file_hash(crop)
        if now["virtual_crop_sha256"] != current_crop_hash or (extra is not None
                and extra.get("virtual_crop_sha256", current_crop_hash) != current_crop_hash):
            raise ValueError(f"Current contour crop changed at {name}")
        warp = np.asarray(record["warp"])
        yaw = float(record["baseline_yaw_deg"])
        for side in ("left", "right"):
            points = [p for p in now["points"] if p["side"] == side]
            provider = "current_crop"
            if not points:
                points = [p for p in old["points"] if p["side"] == side]
                provider = "native_fallback"
            if not points and extra is not None:
                points = [p for p in extra["points"] if p["side"] == side]
                provider = "native_anterior_supplement"
            if not points:
                continue
            xy = np.array([point["native_xy"] for point in points])
            projected = cv2.perspectiveTransform(xy[None], warp)[0]
            if provider != "native_fallback" and not np.allclose(projected, [p["xy"] for p in points], atol=1e-5, rtol=0):
                raise ValueError(f"Current crop transform does not reproduce detections at {name}")
            sign = -1 if side == "left" else 1
            region = ("anterior" if sign == np.sign(-yaw) else "posterior") if abs(yaw) >= 55. else "lateral"
            for point, (x, y), crop_xy in zip(points, xy, projected, strict=True):
                if (crop_xy >= 2).all() and (crop_xy <= 509).all():
                    observations.append((i, sign, float(x), float(y), point["split"], region, provider))
    return observations


def contour_edges(faces, mask_file):
    with mask_file.open("rb") as handle:
        masks = pickle.load(handle, encoding="latin1")
    edges, incidence = np.unique(np.sort(np.concatenate(
        [faces[:, [i, (i + 1) % 3]] for i in range(3)]), axis=1), axis=0, return_counts=True)
    excluded = np.concatenate([masks[k] for k in ("boundary", "face", "left_ear", "right_ear")])
    neck = np.setdiff1d(masks["neck"], excluded)
    eligible = (incidence == 2) & np.isin(edges, neck).all(axis=1)
    return edges, eligible, np.unique(np.concatenate([masks[k] for k in ("face", "left_ear", "right_ear")])), masks


def neck_shape_fields(flame, canonical, faces, edges, masks):
    """Create two sagittal fields with zero displacement at every skeletal and seam support."""
    template = np.median(canonical, axis=0)
    fixed = np.ones(len(template), dtype=bool)
    fixed[masks["neck"]] = False
    for name in ("boundary", "face", "left_ear", "right_ear", "lips"):
        fixed[masks[name]] = True
    face_edges = np.sort(np.concatenate([faces[:, [i, (i + 1) % 3]] for i in range(3)]), axis=1)
    boundary, incidence = np.unique(face_edges, axis=0, return_counts=True)
    fixed[boundary[incidence == 1].ravel()] = True
    regressor = flame.J_regressor.cpu().numpy()
    fixed[np.any(regressor != 0., axis=0)] = True
    fixed[faces[flame.full_lmk_faces_idx.cpu().numpy().reshape(-1)[:68]].ravel()] = True
    lengths = np.linalg.norm(template[edges[:, 0]] - template[edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, edges.T), shape=(len(template), len(template))).tocsr()
    distance = dijkstra(graph, directed=False, indices=np.flatnonzero(fixed), min_only=True, limit=.020)
    fade = .5 - .5 * np.cos(np.pi * np.minimum(distance / .020, 1.))
    z = template[:, 2] - regressor[1] @ template[:, 2]
    blend = np.clip((z + .018) / .036, 0., 1.)
    blend = blend**2 * (3. - 2. * blend)
    field = np.zeros((2, len(template), 3), dtype=template.dtype)
    field[:, :, 2] = np.stack((fade * blend, fade * (1. - blend)))
    if (not np.isfinite(field).all() or np.any(field[:, fixed]) or not (np.max(field, axis=(1, 2)) > .5).all()
            or np.any(np.einsum("jv,kvd->kjd", regressor, field))):
        raise ValueError("Neck shape fields failed anatomical or skeletal support checks")
    return field


def face_area_check(base, candidate, faces):
    def normals(verts):
        p, q, r = verts[:, faces].transpose(2, 0, 1, 3)
        return np.cross(q - p, r - p)

    original, changed = (normals(verts) for verts in (base, candidate))
    area = np.linalg.norm(changed, axis=-1) / np.maximum(np.linalg.norm(original, axis=-1), 1e-12)
    opposed = int((np.sum(original * changed, axis=-1) <= 0.).sum())
    return float(area.min()), float(area.max()), opposed


def profile_values(pixels, depths, world, observations, edges, eligible):
    results = np.full(len(observations), np.nan)
    groups = {}
    for j, (frame, side, _, y, *_) in enumerate(observations):
        groups.setdefault((frame, side), []).append((j, y))
    for (frame, side), group in groups.items():
        ids, ys = zip(*group, strict=True)
        x, _ = mesh_intersections(pixels[frame], depths[frame], world[frame], edges, eligible,
                                  np.asarray(ys), side)
        results[list(ids)] = x
    return results


def fit_pose_and_shape(values, observations, bounds, pitch_samples, start, shape_slope, shape_range, args):
    frames = np.asarray([x[0] for x in observations])
    fit = np.asarray([x[4] == "fit" for x in observations])
    target = np.asarray([x[2] for x in observations])
    counts = Counter((x[0], x[1]) for x, accepted in zip(observations, fit, strict=True) if accepted)
    views = Counter(label for _, label in {(x[0], x[5]) for x, accepted in zip(observations, fit, strict=True) if accepted})
    local_solution = args.local_mesh or args.freeze_previous_mesh
    front_balance = min(2., max(1., views["posterior"] / max(1, views["anterior"]))) if local_solution else 1.
    weight = np.asarray([np.sqrt(CONTOUR_WEIGHT * (front_balance if x[5] == "anterior" else 1.)
                                 / max(1, counts[x[0], x[1]])) for x in observations])[fit]
    x = frames[fit]
    n = len(bounds)
    if len(np.unique(x)) < n // 2:
        raise ValueError("Too few frames have a visible anatomical neck contour")
    m = len(shape_range)
    sparse = lil_matrix((len(x) + n + n - 1 + n - 2 + m, n + m), dtype=bool)
    sparse[np.arange(len(x)), x] = True
    offset = len(x)
    sparse[offset + np.arange(n), np.arange(n)] = True
    offset += n
    for k in range(n - 1):
        sparse[offset + k, k:k + 2] = True
    offset += n - 1
    for k in range(n - 2):
        sparse[offset + k, k:k + 3] = True
    if m:
        sparse[:len(x), n:] = True
    for k in range(m):
        sparse[offset + n - 2 + k, n + k] = True
    points = np.flatnonzero(fit)

    def residual(solution):
        delta, shape = solution[:n], solution[n:]
        intervals = (delta[x] - pitch_samples[0]) / (pitch_samples[1] - pitch_samples[0])
        lower = intervals.astype(int).clip(0, len(pitch_samples) - 2)
        frac = intervals - lower
        line = values[lower, points] * (1 - frac) + values[lower + 1, points] * frac
        error = line + shape @ shape_slope[:, points] - target[fit]
        robust = np.sign(error) * np.sqrt(18. * (np.sqrt(1. + (error / 3.)**2) - 1.))
        velocity, accel = (1.4, 2.3) if local_solution else (.95, 1.6)
        return np.concatenate((weight * robust, args.pitch_anchor * delta, velocity * np.diff(delta),
                               accel * np.diff(delta, n=2), .22 * shape))

    limits = np.r_[bounds, shape_range]
    initial = np.r_[np.clip(start, -bounds, bounds), np.zeros(m)]
    result = least_squares(residual, initial, bounds=(-limits, limits),
                           jac_sparsity=sparse.tocsr(), max_nfev=70)
    if not result.success:
        raise RuntimeError(f"Joint neck solve did not converge: {result.message}")
    return result.x[:n], result.x[n:], float(result.cost), front_balance


def neck_metrics(before, after, observations):
    target = np.asarray([point[2] for point in observations])
    result = {}
    for label in ("all", "anterior", "posterior", "lateral"):
        for split in ("fit", "check"):
            keep = np.array([p[4] == split and (label == "all" or p[5] == label) for p in observations])
            valid = keep & np.isfinite(after)
            if valid.any():
                result[f"{label}_{split}"] = {
                    "points": int(keep.sum()), "lost_after": int(keep.sum() - valid.sum()),
                    "median_before_after_px": [float(np.median(np.abs(p[valid]-target[valid]))) for p in (before, after)],
                    "p90_before_after_px": [float(np.quantile(np.abs(p[valid]-target[valid]), .90)) for p in (before, after)],
                }
    return result


def fit_shared_pitch(sample_pitch, before, observations, records, pitch, bounds, tolerance, temporal, side_only=False):
    """Estimate one local pitch from posterior skin in both signed view sectors."""
    lower, upper = pitch - bounds, pitch + bounds
    confidence = np.ones_like(pitch)
    if side_only:
        yaw = np.abs([record["baseline_yaw_deg"] for record in records])
        weight = np.clip((yaw - 35.) / 20., 0., 1.)
        confidence = weight ** 2 * (3. - 2. * weight)
    active = confidence > 0
    if not active.any():
        raise ValueError("No observed side sectors for the neck pitch solve")
    effective_bound = bounds[active] / confidence[active]
    lo = float(np.max(pitch[active] - effective_bound) - tolerance)
    hi = float(np.min(pitch[active] + effective_bound) + tolerance)
    if lo > hi:
        minimum = (lo - hi) * .5 + tolerance
        raise ValueError(f"Front protection requires at least ±{minimum:.2f} degrees around a shared pitch")
    target = np.asarray([point[2] for point in observations])
    sectors = {-1: {}, 1: {}}
    for j, (index, _, _, _, split, region, _) in enumerate(observations):
        if split == "fit" and region == "posterior" and np.isfinite(before[j]):
            sector = int(np.sign(records[index]["baseline_yaw_deg"]))
            sectors[sector].setdefault(index, []).append(j)
    if any(len(rows) < 2 for rows in sectors.values()):
        raise ValueError("Shared pitch needs visible fitting contours in both side sectors")

    def correction(value):
        return np.clip(pitch + confidence * (value - pitch), lower, upper) - pitch

    def objective(value):
        measured = sample_pitch(correction(value))
        error = np.where(np.isfinite(measured), np.abs(measured - target), np.abs(before - target) + 24.)
        robust = np.sqrt(1. + (error / 3.) ** 2) - 1.
        return float(np.mean([np.mean([robust[indices].mean() for indices in frames.values()])
                              for frames in sectors.values()]))

    shared = lo
    if lo < hi:
        candidates = np.linspace(lo, hi, 9)
        best = int(np.argmin([objective(value) for value in candidates]))
        bracket = candidates[max(0, best - 1)], candidates[min(len(candidates) - 1, best + 1)]
        result = minimize_scalar(objective, bounds=bracket, method="bounded", options={"xatol": .05})
        shared = min((float(candidates[best]), float(result.x)), key=objective)
    fitted = shared
    controls = None
    if temporal:
        time = np.asarray([record["timestamp_s"] for record in records])
        knots = np.linspace(time[0], time[-1], 5)

        def trajectory(values):
            return shared + np.interp(time, knots, values)

        def smooth_objective(values):
            return (objective(trajectory(values)) + .006 * np.mean(values ** 2)
                    + .012 * np.mean(np.diff(values, n=2) ** 2))

        zero = np.zeros(len(knots))
        fit = minimize(smooth_objective, zero, method="Powell",
                       bounds=[(-tolerance, tolerance)] * len(knots),
                       options={"maxfev": 120, "xtol": .08, "ftol": .002})
        controls = fit.x if smooth_objective(fit.x) < smooth_objective(zero) else zero
        fitted = trajectory(controls)
    delta = correction(fitted)
    return delta, objective(fitted), {
        "center_deg": shared, "max_deviation_deg": float(np.abs((pitch + delta - shared)[confidence == 1]).max()),
        "front_pitch_unchanged": bool(np.all(delta[confidence == 0] == 0)),
        "temporal_control_offsets_deg": None if controls is None else controls.tolist(),
        "fit_frames_by_yaw_sign": {str(sign): len(frames) for sign, frames in sectors.items()},
        "pose_component": "FLAME neck rotation vector X",
    }


def motion(rotvec):
    transforms = Rotation.from_rotvec(rotvec)
    step = np.degrees((transforms[:-1].inv() * transforms[1:]).magnitude())
    return {"p95_deg": float(np.quantile(step, .95)), "max_deg": float(step.max())}


@torch.no_grad()
def run(args):
    initialization = args.sequence / "initialization.json"
    context = json.loads(initialization.read_bytes())
    records = context["frames"]
    names = [r["name"] for r in records]
    scale = float(context["scene_scale"])
    tape = np.load(args.registered, allow_pickle=False)
    if (tape["filenames"].tolist() != names or tape["initialization_sha256"].item() != file_hash(initialization)
            or tape["checkpoint_sha256"].item() != file_hash(args.checkpoint)):
        raise ValueError("Registered geometry, initial poses and camera manifest differ")
    observations = load_observations(args, records)
    held_frames = set()
    for block in args.holdout_blocks:
        first, last = (int(index) for index in block.split(":"))
        if first < 0 or last >= len(names) or first > last:
            raise ValueError(f"Bad held-out frame block: {block}")
        held_frames.update(range(first, last + 1))
    observations = [(i, side, x, y, "check" if i in held_frames else split, region, provider)
                    for i, side, x, y, split, region, provider in observations]
    setup_vhap_cameras(camera_gauge(records, scale)[2])
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "native"),
        "--no-async-func",
    ])
    tracker = GlobalTracker(cfg)
    old = np.load(args.checkpoint, allow_pickle=False)
    if tracker.n_timesteps != len(names) or old["timestep_id"].tolist() != list(tracker.dataset.timestep_ids):
        raise ValueError("VHAP read a different roster")
    tracker.load_from_tracked_flame_params(args.checkpoint)
    faces = tracker.flame.faces.cpu().numpy()
    if not np.array_equal(faces, tape["faces"]):
        raise ValueError("Stock VHAP faces differ from registered geometry")
    edges, eligible, protect, masks = contour_edges(faces, args.flame_masks)
    ids = torch.arange(len(names), device=tracker.device)

    def replay():
        chunks = [tracker.forward_flame(batch)[:3] for batch in ids.split(cfg.batch_size)]
        return tuple(torch.cat(tensors).cpu().numpy() for tensors in zip(*chunks, strict=True))

    base, canonical, lm_base = replay()
    if not np.allclose(base, tape["vertices_tracker"], rtol=0., atol=1e-6):
        raise RuntimeError("Stock VHAP replay did not reproduce the registered geometry")
    if args.freeze_previous_mesh:
        with np.load(args.prior_checkpoint, allow_pickle=False) as prior_npz:
            prior_static = prior_npz["static_offset"]
        displacement = (prior_static - old["static_offset"])[0]
        regressor = tracker.flame.J_regressor.cpu().numpy()
        if np.any(displacement[protect]) or not np.allclose(regressor @ displacement, 0., atol=1e-8, rtol=0):
            raise ValueError("The fixed previous mesh would move the head or skeletal supports")
        tracker.static_offset.copy_(torch.as_tensor(prior_static).to(tracker.static_offset))
        base, canonical, lm_base = replay()
    joints = np.einsum("jv,nvc->njc", tracker.flame.J_regressor[:2].cpu().numpy(), canonical)
    original_offset = tracker.static_offset.clone()
    fields = neck_shape_fields(tracker.flame, canonical, faces, edges, masks) if args.local_mesh else None
    field_tensor = torch.from_numpy(fields).to(original_offset) if args.local_mesh else None
    moved_vertices = np.any(fields, axis=(0, 2)) if args.local_mesh else np.zeros(base.shape[1], dtype=bool)
    local_faces = faces[moved_vertices[faces].any(axis=1)]
    bound_shape = np.full(2 if args.local_mesh else 0, 9.)
    arm = joints[:, 1] - joints[:, 0]
    root = Rotation.from_rotvec(old["rotation"]).as_matrix()
    neck = Rotation.from_rotvec(old["neck_pose"]).as_matrix()
    composed = root @ neck
    world_basis = tape["tracker_to_world_rotation"].T * scale
    world_origin = tape["scene_origin"]
    cameras = [camera(record) for record in records]

    def shift(amount, mesh_mm=None):
        angles = old["neck_pose"].copy()
        angles[:, 0] += np.radians(amount)
        neck_next = Rotation.from_rotvec(angles).as_matrix()
        root_next = composed @ neck_next.transpose(0, 2, 1)
        translation = old["translation"] + np.einsum("nij,nj->ni", root - root_next, arm)
        rotation = Rotation.from_matrix(root_next).as_rotvec()
        if args.side_pitch_only:
            frozen = amount == 0
            rotation[frozen] = old["rotation"][frozen]
            translation[frozen] = old["translation"][frozen]
        for field, value in zip(POSE_KEYS, (angles, rotation, translation), strict=True):
            tensor = getattr(tracker, field)
            tensor.copy_(torch.as_tensor(value).to(tensor))
        if field_tensor is not None:
            tracker.static_offset.copy_(original_offset)
            if mesh_mm is not None:
                coeff = torch.as_tensor(mesh_mm).to(original_offset) * .001
                tracker.static_offset.add_(torch.einsum("k,kvc->vc", coeff, field_tensor)[None])
        return replay()

    def screen(track):
        world = track @ world_basis + world_origin
        coords, depth = zip(*(project(points, *view) for points, view in zip(world, cameras, strict=True)), strict=True)
        return np.asarray(coords), np.asarray(depth), world

    def sample(track):
        xy, z, world = screen(track)
        return profile_values(xy, z, world, observations, edges, eligible)

    start = np.zeros(len(names))
    prior_stats = None
    if args.prior_report is not None:
        prior = json.loads(args.prior_report.read_bytes())
        if (prior["source"]["initialization_sha256"] != file_hash(initialization)
                or Path(prior["source"]["checkpoint"]) != args.checkpoint
                or set(prior["frame_results"]) != set(names)
                or not args.prior_checkpoint.is_file()):
            raise ValueError("Previous pose report refers to a different registered run")
        start = np.asarray([prior["frame_results"][name]["pitch_delta_deg"] for name in names])
        previous, _, _ = shift(start)
        with np.load(args.prior_checkpoint, allow_pickle=False) as prior_npz:
            compared = (*POSE_KEYS, "static_offset") if args.freeze_previous_mesh else POSE_KEYS
            if (Path(prior["checkpoint"]) != args.prior_checkpoint
                    or any(not np.allclose(prior_npz[key], getattr(tracker, key).cpu().numpy(), atol=2e-7, rtol=0.)
                           for key in compared)
                    or any(not np.array_equal(prior_npz[key], old[key]) for key in old.files if key not in compared)):
                raise ValueError("Previous pose replay did not reproduce its checkpoint")
        prior_stats = sample(previous)
    else:
        previous = base

    def safe_area(values):
        low, high, flips = values
        return low >= .65 and high <= 1.60 and not flips

    if args.local_mesh:
        # With the regressor fixed, a blend of fields is affine for each frozen pose.
        corners = []
        for front, back in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            posed, _, _ = shift(start, bound_shape * (front, back))
            corners.append(posed - previous)
        factor = 1.
        if not all(safe_area(face_area_check(previous, previous + delta, local_faces)) for delta in corners):
            lo, hi = 0., 1.
            for _ in range(12):
                mid = (lo + hi) / 2
                if all(safe_area(face_area_check(previous, previous + mid * delta, local_faces)) for delta in corners):
                    lo = mid
                else:
                    hi = mid
            factor = lo * .98
        bound_shape *= factor
        if (bound_shape < .5).any():
            raise RuntimeError("The shared neck mesh has no safe useful deformation range")

    original = sample(base)
    face0 = screen(base[:, protect])[0]
    lmk0 = screen(lm_base[:, 17:68])[0]
    grid = []
    max_move = np.zeros(len(names))
    max_lmk = np.zeros_like(max_move)
    pitch_samples = np.linspace(-args.pitch_limit, args.pitch_limit, 5)
    for angle in pitch_samples:
        if angle:
            posed, _, lm = shift(np.full(len(names), angle))
            grid.append(sample(posed))
            if abs(angle) == args.pitch_limit:
                move = np.linalg.norm(screen(posed[:, protect])[0] - face0, axis=2)
                lmk = np.linalg.norm(screen(lm[:, 17:68])[0] - lmk0, axis=2)
                max_move = np.maximum(max_move, np.quantile(move, .95, axis=1))
                max_lmk = np.maximum(max_lmk, lmk.max(axis=1))
        else:
            grid.append(original)
    grid = np.asarray(grid)
    observed = list(observations)
    keep = np.isfinite(original) if args.shared_pitch is not None else np.isfinite(grid).all(axis=0)
    shape_slope = np.empty((len(bound_shape), len(observations)))
    for k in range(len(bound_shape)):
        step = .5 * np.eye(len(bound_shape))[k]
        plus = sample(shift(start, step)[0])
        minus = sample(shift(start, -step)[0])
        shape_slope[k] = plus - minus
    keep &= np.isfinite(shape_slope).all(axis=0)
    if prior_stats is not None:
        keep &= np.isfinite(prior_stats)
        prior_stats = prior_stats[keep]
    observations = [point for point, ok in zip(observed, keep, strict=True) if ok]
    grid = grid[:, keep]
    if not all(any(p[5] == region and p[4] == "check" for p in observations)
               for region in ("anterior", "posterior")):
        raise ValueError("Need independently scored anterior and posterior contour evidence")
    bounds = np.minimum.reduce((np.full(len(names), args.pitch_limit),
                                .95 * args.pitch_limit * HEAD_P95_LIMIT / max_move.clip(min=1e-5),
                                .95 * args.pitch_limit * FACE_LANDMARK_LIMIT / max_lmk.clip(min=1e-5)))
    frontal = np.array([abs(row["baseline_yaw_deg"]) <= 30. for row in records])
    bounds[frontal] = np.minimum(bounds[frontal], args.frontal_pitch_limit)
    initial = start.copy()
    initial[list(held_frames)] = 0.
    shared = None
    if args.shared_pitch is None:
        delta, mesh_mm, cost, front_balance = fit_pose_and_shape(
            grid, observations, bounds, pitch_samples, initial, shape_slope[:, keep], bound_shape, args)
    else:
        delta, cost, shared = fit_shared_pitch(
            lambda correction: sample(shift(correction)[0]), grid[2], observations, records,
            np.degrees(old["neck_pose"][:, 0]).astype(float), bounds, args.shared_pitch,
            args.shared_pitch_temporal, args.side_pitch_only)
        mesh_mm = np.empty(0)
        front_balance = 1.
    mesh_check = None
    if args.local_mesh:
        pose_only, _, _ = shift(delta)
    final, _, lm = shift(delta, mesh_mm)
    if args.local_mesh:
        mesh_check = face_area_check(pose_only, final, local_faces)
        if not safe_area(mesh_check) or np.max(np.abs(final[:, ~moved_vertices] - pose_only[:, ~moved_vertices])) > 2e-6:
            raise RuntimeError(f"Exact posed local mesh failed the triangle or protected-vertex guard: {mesh_check}")
    after = sample(final)
    face = np.linalg.norm(screen(final[:, protect])[0] - face0, axis=2)
    lmk = np.linalg.norm(screen(lm[:, 17:68])[0] - lmk0, axis=2)
    face_max_p95 = float(np.quantile(face, .95, axis=1).max())
    before_motion = motion(old["neck_pose"])
    after_motion = motion(tracker.neck_pose.cpu().numpy())
    correction_step = float(np.max(np.abs(np.diff(delta))))
    metrics = neck_metrics(grid[2], after, observations)
    against_prior = neck_metrics(prior_stats, after, observations) if prior_stats is not None else None
    held_metrics = neck_metrics(grid[2] if prior_stats is None else prior_stats, after,
                                [(i, side, x, y, split if i in held_frames else "exclude", region, provider)
                                 for i, side, x, y, split, region, provider in observations])
    # Use each row's frozen witness set. Disappearing edges cannot count as an improvement.
    checks = [metrics.get(f"{region}_check") for region in ("anterior", "posterior")]
    accepted = (all(x and x["lost_after"] == 0 and x["median_before_after_px"][1] < x["median_before_after_px"][0] for x in checks)
                and face_max_p95 <= HEAD_P95_LIMIT and lmk.max() <= FACE_LANDMARK_LIMIT
                and after_motion["max_deg"] <= before_motion["max_deg"] + .75 and correction_step <= 2.)
    if shared is not None:
        side_checks = {}
        for sign in (-1, 1):
            rows = [p[:4] + ("exclude",) + p[5:] if np.sign(records[p[0]]["baseline_yaw_deg"]) != sign else p
                    for p in observations]
            side_checks[str(sign)] = neck_metrics(grid[2], after, rows).get("posterior_check")
        shared["posterior_check_by_yaw_sign"] = side_checks
        accepted = (
            all(x and x["lost_after"] == 0 and x["median_before_after_px"][1] < x["median_before_after_px"][0]
                and x["p90_before_after_px"][1] <= x["p90_before_after_px"][0] for x in side_checks.values())
            and checks[0] and checks[0]["lost_after"] == 0
            and checks[0]["median_before_after_px"][1] <= checks[0]["median_before_after_px"][0] + 1.
            and face_max_p95 <= HEAD_P95_LIMIT and lmk.max() <= FACE_LANDMARK_LIMIT
            and after_motion["max_deg"] <= before_motion["max_deg"] + .75)
    if held_frames and (args.local_mesh or args.freeze_previous_mesh):
        held_sides = [held_metrics.get(f"{region}_check") for region in ("anterior", "posterior")]
        accepted &= all(x and x["lost_after"] == 0 and x["median_before_after_px"][1] < x["median_before_after_px"][0]
                        for x in held_sides)
    saved_fields = (*POSE_KEYS, "static_offset") if args.local_mesh or args.freeze_previous_mesh else POSE_KEYS
    report = {
        "status": "accepted_pose_candidate" if accepted else "review_only",
        "frames": len(names), "matched_frames": len({x[0] for x in observations}), "fit_frames": len({x[0] for x in observations if x[4] == "fit"}),
        "detected_points": len(observed), "stable_visible_points": len(observations),
        "supervision": dict(contour_weight=CONTOUR_WEIGHT, anterior_balance=front_balance,
                            per_frame_side_normalized=True, native_integer_pixels=True,
                            supplemental_points=sum(p[6] == "native_anterior_supplement" for p in observations)),
        "source": dict(checkpoint=str(args.checkpoint), initialization_sha256=file_hash(initialization),
                       observations_sha256=file_hash(args.targets), native_fallback_sha256=file_hash(args.native_fallback)),
        "updated_fields": list(saved_fields), "objective": cost, "pitch_limit_deg": args.pitch_limit,
        "frontal_pitch_limit_deg": args.frontal_pitch_limit, "pitch_anchor": args.pitch_anchor,
        "previous_neck_mesh_frozen": args.freeze_previous_mesh,
        "shape_offset_mm": mesh_mm.tolist(), "shape_ranges_mm": bound_shape.tolist(),
        "shape_axis": "canonical +Z anterior; modes are anterior and posterior neck",
        "shape_affected_vertices": int(moved_vertices.sum()), "shape_area_ratio_and_flips": mesh_check,
        "held_frames": [names[i] for i in sorted(held_frames)], "held_frame_metrics": held_metrics,
        "max_frame_head_vertex_p95_px": face_max_p95, "max_inner_face_landmark_px": float(lmk.max()),
        "neck_step_before": before_motion, "neck_step_after": after_motion, "max_adjacent_pitch_correction_delta_deg": correction_step,
        "native_contour_residuals": metrics, "residuals_against_previous_pose_only": against_prior,
        "previous_checkpoint": str(args.prior_checkpoint) if args.prior_checkpoint is not None else None,
        "supplemental_sha256": file_hash(args.supplement) if args.supplement is not None else None,
        "frame_results": {},
    }
    if shared is not None:
        report["shared_pitch"] = shared
    for i, name in enumerate(names):
        rows = [j for j, point in enumerate(observations) if point[0] == i]
        report["frame_results"][name] = {
            "yaw_deg": records[i]["baseline_yaw_deg"], "pitch_delta_deg": float(delta[i]),
            "limit_deg": float(bounds[i]),
            "flame_neck_x_deg": [float(np.degrees(old["neck_pose"][i, 0])), float(np.degrees(old["neck_pose"][i, 0])) + delta[i]],
            "protected_head_p95_px": float(np.quantile(face[i], .95)),
            "contours": [
                {"side": "left" if observations[j][1] == -1 else "right", "region": observations[j][5],
                 "native_xy": list(observations[j][2:4]), "split": observations[j][4], "provider": observations[j][6],
                 "before_x": float(grid[2, j]), "previous_x": float(prior_stats[j]) if prior_stats is not None else None,
                 "after_x": float(after[j]) if np.isfinite(after[j]) else None}
                for j in rows],
        }
    report_path = args.output / "neck-joint-fit.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tracker.timestep = int(old["n_processed_frames"])
    tracker.evaluate(make_visualization=True, epoch=0)
    tracker.tb_writer.close()
    image_grid = tracker.out_dir / "eval_0/image_grid"
    checkpoint = tracker.out_dir / "tracked_flame_params_0.npz"
    with np.load(checkpoint, allow_pickle=False) as new:
        if (set(new.files) != set(old.files)
                or any(not np.array_equal(old[key], new[key]) for key in old.files if key not in saved_fields)
                or (fields is not None and not np.array_equal(old["static_offset"][:, ~moved_vertices],
                                                              new["static_offset"][:, ~moved_vertices]))
                or (args.freeze_previous_mesh and not np.array_equal(new["static_offset"], prior_static))
                or {p.name for p in image_grid.glob("*.jpg")} != {f"frame_{i:05d}.jpg" for i in range(len(names))}):
            raise RuntimeError("Failed exact checkpoint or full native-grid verification")
    report.update(image_grid=str(image_grid), checkpoint=str(checkpoint))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], image_grid, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "sequence", "checkpoint", "registered", "targets", "native-fallback", "flame-masks", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--pitch-limit", type=float, default=4.)
    parser.add_argument("--frontal-pitch-limit", type=float, default=1.)
    parser.add_argument("--pitch-anchor", type=float, default=.30)
    parser.add_argument("--shared-pitch", type=float, metavar="MAX_DEVIATION_DEG",
                        help="Fit one absolute local pitch from posterior skin on both sides.")
    parser.add_argument("--shared-pitch-temporal", action="store_true",
                        help="Allow a smooth residual within the shared pitch allowance.")
    parser.add_argument("--side-pitch-only", action="store_true",
                        help="Apply side-derived pitch to profiles, with a smooth transition and frozen frontal pitch.")
    parser.add_argument("--local-mesh", action="store_true")
    parser.add_argument("--freeze-previous-mesh", action="store_true")
    parser.add_argument("--holdout-blocks", nargs="*", default=[])
    cv2.setNumThreads(1)
    options = parser.parse_args()
    if (not 0. < options.frontal_pitch_limit <= options.pitch_limit <= 12.
            or not np.isfinite(options.pitch_anchor) or options.pitch_anchor < 0.
            or (options.prior_report is None) != (options.prior_checkpoint is None)
            or (options.freeze_previous_mesh and (options.local_mesh or options.prior_report is None))):
        parser.error("Use finite pitch limits and provide both previous files when freezing the previous mesh")
    if options.shared_pitch is not None and (
        not np.isfinite(options.shared_pitch) or options.shared_pitch < 0
        or options.local_mesh or options.prior_report is not None
    ):
        parser.error("Shared pitch fits one checkpoint and requires a finite nonnegative deviation")
    if options.shared_pitch_temporal and (options.shared_pitch is None or options.shared_pitch <= 0):
        parser.error("A smooth shared-pitch residual requires a positive pitch allowance")
    if options.side_pitch_only and options.shared_pitch is None:
        parser.error("Side-only pitch requires a shared pitch solve")
    run(options)
