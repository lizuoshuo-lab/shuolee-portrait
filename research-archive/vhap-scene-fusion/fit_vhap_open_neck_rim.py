"""Refine the native lower FLAME rim with aligned collar and person evidence."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares, minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from export_head_neck_frames import boundary_loops
from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, file_hash, project
from fit_side_frames import alpha_maps
from fit_vhap_lower_seam import read_observations
from fit_vhap_neck_sequence import contour_edges, face_area_check, neck_metrics, profile_values
from guided_vhap_input import _native_cameras
from prepare_video_avatar import setup_vhap_cameras
from review_smplx_head_neck import project as project_camera


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value**2 * (3. - 2. * value)


def rim_fields(flame, canonical, faces, edges, masks, attach=False):
    template = np.median(canonical, axis=0)
    loops = boundary_loops(faces)
    ring = min(loops, key=lambda loop: np.median(template[loop, 1]))
    if len(ring) != 30:
        raise ValueError("Expected the 30-edge native FLAME lower neck boundary")
    regressor = flame.J_regressor.cpu().numpy()
    fixed = np.any(regressor != 0., axis=0)
    for group in ("face", "left_ear", "right_ear", "lips"):
        fixed[masks[group]] = True
    lmk = flame.full_lmk_faces_idx.cpu().numpy().reshape(-1)[:68]
    fixed[faces[lmk].ravel()] = True
    lengths = np.linalg.norm(template[edges[:, 0]] - template[edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, edges.T), shape=(len(template), len(template))).tocsr()
    width = .090 if attach else .052
    from_rim = dijkstra(graph, directed=False, indices=ring, min_only=True, limit=width)
    from_fixed = dijkstra(graph, directed=False, indices=np.flatnonzero(fixed), min_only=True, limit=.015)
    fade = smoothstep(from_fixed / .015)
    z = template[:, 2] - regressor[1] @ template[:, 2]
    taper = smoothstep(1. - from_rim / width) * fade
    changes = np.zeros((4 if attach else 2, len(template), 3), dtype=template.dtype)
    if attach:
        center = np.mean(template[ring], axis=0)
        radius = np.max(np.abs(template[ring] - center), axis=0).clip(min=.01)
        radial = np.clip((template - center) / radius, -1., 1.)
        changes[0, :, 0] = -taper * radial[:, 0]
        changes[1, :, 2] = -taper * radial[:, 2] * smoothstep(-radial[:, 2] * 3.)
        changes[2, :, 2] = -taper * radial[:, 2] * smoothstep(radial[:, 2] * 3.)
        changes[3, :, 1] = taper
    else:
        changes[0, :, 1] = taper
        changes[1, :, 2] = smoothstep(1. - from_rim / .025) * fade * smoothstep((-z - .070) / .020)
    if np.any(changes[:, fixed]) or np.any(np.einsum("jv,kvc->kjc", regressor, changes)):
        raise ValueError("Lower rim motion would move protected regions or joints")
    return ring, changes


def read_body_contacts(args, names, initialization_hash):
    if args.body_contact is None:
        return {}
    with np.load(args.body_contact, allow_pickle=False) as source:
        uv, z, frames = source["uv"], source["z"], source["names"]
        if (source["initialization_sha256"].item() != initialization_hash
                or source["scene_sha256"].item() != file_hash(args.body_scene)
                or uv.shape != (len(frames), 2) or z.shape != (len(frames),)
                or not np.isfinite(uv).all() or not np.isfinite(z).all() or (z <= 0).any()
                or not set(frames.tolist()) <= set(names)):
            raise ValueError("Body contacts must match the exact native cameras and scene layer")
        return {i: (uv[frames == name], z[frames == name]) for i, name in enumerate(names)
                if np.count_nonzero(frames == name) >= 4}


@torch.no_grad()
def run(args):
    prior = json.loads(args.prior_report.read_bytes())
    checkpoint = Path(prior["checkpoint"])
    init = args.sequence / "initialization.json"
    context = json.loads(init.read_bytes())
    names = [r["name"] for r in context["frames"]]
    tape = np.load(args.registered, allow_pickle=False)
    init_hash = file_hash(init)
    if (tape["filenames"].tolist() != names or tape["initialization_sha256"].item() != init_hash
            or prior["source"]["initialization_sha256"] != init_hash
            or prior["frames"] != len(names)):
        raise ValueError("Expected the same fitted frame list and calibrated cameras")
    records, _, curves = read_observations(args.frames, args.targets, names)
    if records != context["frames"]:
        raise ValueError("Collar observations use different frame records")
    setup_vhap_cameras(camera_gauge(records, context["scene_scale"])[2])
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "native"),
        "--no-async-func",
    ])
    tracker = GlobalTracker(cfg)
    original = np.load(checkpoint, allow_pickle=False)
    if tracker.n_timesteps != len(names) or original["timestep_id"].tolist() != list(tracker.dataset.timestep_ids):
        raise ValueError("VHAP frame order mismatch")
    tracker.load_from_tracked_flame_params(checkpoint)
    original_offset = tracker.static_offset.clone()
    faces = tracker.flame.faces.cpu().numpy()
    if not np.array_equal(faces, tape["faces"]):
        raise ValueError("Rim connectivity differs from the registered mesh")
    edges, eligible, protect, regions = contour_edges(faces, args.flame_masks)
    ids = torch.arange(len(names), device=tracker.device)

    def replay():
        chunks = [tracker.forward_flame(group)[:2] for group in ids.split(cfg.batch_size)]
        return tuple(torch.cat(items).cpu().numpy() for items in zip(*chunks, strict=True))

    base, canonical = replay()
    if args.attach_rim and (tape["checkpoint_sha256"].item() != file_hash(checkpoint)
                            or not np.allclose(base, tape["vertices_tracker"], atol=1e-6, rtol=0.)):
        raise ValueError("Body attachment requires an exact replay of the current registered geometry")
    ring, directions = rim_fields(tracker.flame, canonical, faces, edges, regions, args.attach_rim)
    field = torch.as_tensor(directions).to(original_offset)
    affected = np.any(directions, axis=(0, 2))
    incident = faces[affected[faces].any(axis=1)]
    if args.attach_rim:
        lower_mm, max_mm = np.array([0., 0., 0., -20.]), np.array([40., 55., 22., 20.])
    else:
        lower_mm, max_mm = np.zeros(2), np.array([22., 10.])

    def evaluate(coefficients):
        value = torch.as_tensor(coefficients).to(original_offset) * .001
        tracker.static_offset.copy_(original_offset + torch.einsum("p,pvc->vc", value, field)[None])
        return replay()[0]

    derivatives = np.stack([evaluate(offset) - base for offset in np.eye(len(directions))])
    scale_basis = tape["tracker_to_world_rotation"].T * float(tape["scene_scale"])
    origin = tape["scene_origin"]
    k, r, c = (np.asarray(x) for x in zip(*(camera(record) for record in records), strict=True))
    crop_matrix = np.asarray([r["warp"] for r in records]) @ k
    fractions = np.arange(0., 1., .25)[None, None, :, None]

    def samples(vertices):
        rim = vertices[:, ring]
        points = rim[:, :, None]*(1 - fractions) + np.roll(rim, -1, axis=1)[:, :, None]*fractions
        return points.reshape(len(vertices), -1, 3)

    verts_camera = (base @ scale_basis + origin - c[:, None]) @ r
    camera_deltas = (derivatives @ scale_basis) @ r[None]
    base_samples = samples(verts_camera)
    gradients = np.stack([samples(delta) for delta in camera_deltas])
    native_k, size, native_views = _native_cameras(args.scene)
    if (not np.allclose(k, native_k, rtol=0., atol=1e-6)
            or any(not np.allclose(r[i], native_views[name][0], atol=1e-6, rtol=0)
                   or not np.allclose(c[i], native_views[name][1], atol=1e-6, rtol=0) for i, name in enumerate(names))):
        raise ValueError("Physical person alpha requires the registered native camera")
    _, _, parity, remap, raw_size = alpha_maps(args.scene, size, native_k, [names[0]])
    signed = []
    for i, name in enumerate(names):
        uv, _ = project_camera(base_samples[i], native_k)
        left, top = (np.floor(uv.min(axis=0)).astype(int) - 48).clip(0, size)
        right, bottom = (np.ceil(uv.max(axis=0)).astype(int) + 49).clip(0, size)
        raw = cv2.imread(str(args.scene / "alpha_maps" / name), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape[::-1] != raw_size:
            raise ValueError(f"Wrong native person matte at {name}")
        region = np.s_[top:bottom, left:right]
        physical = cv2.remap(raw, remap[0][region], remap[1][region], cv2.INTER_LINEAR)
        foreground = (physical >= 128).astype(np.uint8)
        dist = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5) - cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        signed.append((dist, np.array([left, top])))
    center = verts_camera[:, ring].mean(axis=1)
    up = Rotation.from_rotvec(original["rotation"]).apply([0., 1., 0.]) @ scale_basis
    up_camera = (up[:, None] @ r)[:, 0] / float(tape["scene_scale"])
    radial = base_samples - center[:, None]
    covector = -base_samples
    radial -= (radial * up_camera[:, None]).sum(axis=2, keepdims=True) * up_camera[:, None]
    covector -= (covector * up_camera[:, None]).sum(axis=2, keepdims=True) * up_camera[:, None]
    visible = (radial * covector).sum(axis=2) > .15 * np.linalg.norm(radial, axis=2) * np.linalg.norm(covector, axis=2)
    held = {names.index(name) for name in prior["held_frames"]}
    train_ids = [i for i in range(len(names)) if i not in held]
    contacts = read_body_contacts(args, names, init_hash)
    if args.body_contact and (sum(i in contacts for i in train_ids) < 3 or sum(i in contacts for i in held) < 2):
        raise ValueError("Body attachment needs several supported fitting and held collar views")
    if sum(bool(len(curves[i])) for i in train_ids) < 10 or sum(bool(len(curves[i])) for i in held) < 3:
        raise ValueError("Need independent collar curves across the sequence")

    def raster_projection(xyz, matrix):
        p = xyz @ matrix.transpose(0, 2, 1)
        return p[..., :2] / p[..., 2:]

    def distances(coefficients, roster):
        xyz = base_samples + np.einsum("p,pnvc->nvc", coefficients, gradients)
        uv = raster_projection(xyz, crop_matrix)
        native_uv = raster_projection(xyz, k)
        outside = []
        collar = []
        depth_gaps = []
        for i in roster:
            dist, corner = signed[i]
            outside.append(np.maximum(map_coordinates(dist, (native_uv[i] - corner).T[::-1], order=1, mode="nearest"), 0.))
            if len(curves[i]) and visible[i].any():
                gaps = cdist(uv[i, visible[i]], curves[i])
                collar.append(gaps.min(axis=0) if args.attach_rim else np.r_[gaps.min(axis=0), gaps.min(axis=1)])
            if i in contacts and visible[i].any():
                xy, z = contacts[i]
                closest = cdist(xy, native_uv[i, visible[i]]).argmin(axis=1)
                depth_gaps.append(np.abs(z - xyz[i, visible[i], 2][closest]) / float(tape["scene_scale"]))
        return collar, outside, depth_gaps

    def errors(coefficients):
        collar, outside, depth_gaps = distances(coefficients, train_ids)
        depth = [np.maximum(v - .012, 0.) * 120. / np.sqrt(len(v)) for v in depth_gaps]
        weight = 4. if args.attach_rim else 2.
        return np.concatenate(
            [v / np.sqrt(len(v)) for v in collar]
            + [weight * v / np.sqrt(len(v)) for v in outside]
            + [coefficients / (16. if args.attach_rim else 7.)]
            + depth)

    fit = least_squares(errors, np.ones(len(directions)), bounds=(lower_mm, max_mm),
                        diff_step=.01 if args.attach_rim else None,
                        loss="soft_l1", f_scale=.8, max_nfev=70 if args.attach_rim else 60)
    if not fit.success:
        raise RuntimeError(f"Lower rim solve failed: {fit.message}")

    def safe(vertices):
        low, high, opposed = face_area_check(base, vertices, incident)
        return low >= (.35 if args.attach_rim else .65) and high <= (2.10 if args.attach_rim else 1.6) and not opposed

    update = np.einsum("p,pnvc->nvc", fit.x, derivatives)
    factor = 1.
    if not safe(base + update):
        low, high = 0., 1.
        for _ in range(14):
            mid = (low + high) / 2.
            if safe(base + mid * update):
                low = mid
            else:
                high = mid
        factor = .99 * low
    applied = fit.x * factor
    if args.attach_rim and factor < 1.:
        triangles = base[:, incident]
        delta_triangles = derivatives[:, :, incident]

        def normals(v):
            return np.cross(v[:, :, 1] - v[:, :, 0], v[:, :, 2] - v[:, :, 0])

        original_normal = normals(triangles)
        original_area = np.linalg.norm(original_normal, axis=-1).clip(min=1e-12)
        original_unit = original_normal / original_area[..., None]

        def feasible(coefficients):
            normal = normals(triangles + np.einsum("p,pnfvc->nfvc", coefficients, delta_triangles))
            area = np.linalg.norm(normal, axis=-1)
            ratio = area / original_area
            orientation = np.sum(normal * original_unit, axis=2) / area.clip(min=1e-12)
            return np.array([ratio.min() - .36, 2.08 - ratio.max(), orientation.min() - .05])

        def robust_cost(coefficients):
            residual = errors(coefficients)
            return float(np.sum(np.sqrt(1. + (residual / .8) ** 2) - 1.))

        start = applied * .98
        constrained = minimize(robust_cost, start, method="SLSQP",
                               bounds=list(zip(lower_mm, max_mm, strict=True)),
                               constraints={"type": "ineq", "fun": feasible},
                               options={"maxiter": 30, "eps": .03, "ftol": 1e-5})
        if (np.isfinite(constrained.x).all() and np.min(feasible(constrained.x)) >= -1e-5
                and robust_cost(constrained.x) < robust_cost(applied)):
            applied = constrained.x
    final = evaluate(applied)
    if not safe(final) or np.any(base[:, protect] != final[:, protect]) or np.any(base[:, ~affected] != final[:, ~affected]):
        raise RuntimeError("Exact lower rim replay failed mesh shape or head protection checks")

    def group_measures(roster):
        out = []
        for coeff in (np.zeros(len(directions)), applied):
            collar, outside, depth_gaps = distances(coeff, roster)
            out.append({
                "collar_frames": len(collar),
                "median_collar_crop_px": float(np.median([np.median(v) for v in collar])) if collar else None,
                "outer_rim_p90_max_native_px": float(np.quantile([v.max() for v in outside], .9)),
                "max_outside_native_px": float(np.max(outside)),
                "supported_collar_depth_median_flame_mm": float(np.median(np.concatenate(depth_gaps)) * 1000.) if depth_gaps else None,
            })
        return out

    witness = json.loads(args.neck_checks.read_bytes()) if args.neck_checks else prior
    if args.neck_checks and (
        witness["source"]["initialization_sha256"] != init_hash
        or Path(witness["source"]["checkpoint"]).resolve() != checkpoint.resolve()
    ):
        raise ValueError("Neck check points must use the current baseline and registered cameras")
    observations = [(i, -1 if row["side"] == "left" else 1, *row["native_xy"], row["split"], row["region"], row["provider"])
                    for i, name in enumerate(names) for row in witness["frame_results"][name]["contours"]]

    def sample(verts):
        world = verts @ scale_basis + origin
        xy, z = zip(*(project(v, K, R, C) for v, K, R, C in zip(world, k, r, c, strict=True)), strict=True)
        return profile_values(np.asarray(xy), np.asarray(z), world, observations, edges, eligible)

    before, after = sample(base), sample(final)
    if not np.isfinite(before).all():
        raise RuntimeError("Existing accepted neck observations did not replay")
    contour = neck_metrics(before, after, observations)
    check = group_measures(sorted(held))
    accepted = (check[1]["median_collar_crop_px"] <= check[0]["median_collar_crop_px"] - .2
                and check[1]["outer_rim_p90_max_native_px"] <= check[0]["outer_rim_p90_max_native_px"]
                and np.any(final != base)
                and all(not data["lost_after"] and data["median_before_after_px"][1] <= data["median_before_after_px"][0] + .35
                        for key, data in contour.items() if key.endswith("_check")))
    if args.attach_rim and contacts:
        accepted &= (
            check[1]["supported_collar_depth_median_flame_mm"]
            <= check[0]["supported_collar_depth_median_flame_mm"] + 2.
        )
    report = {
        "status": "accepted_rim_candidate" if accepted else "review_only", "frames": len(names),
        "rim_vertices": len(ring),
        "coefficient_order": (["side_inset", "posterior_inset", "anterior_inset", "rim_up"]
                              if args.attach_rim else ["rim_up", "posterior_rim_forward"]),
        "coefficients_flame_mm": applied.tolist(), "fit_before_triangle_guard_flame_mm": fit.x.tolist(),
        "geometry_limit_fraction": factor, "triangles": face_area_check(base, final, incident),
        "current_checkpoint": str(checkpoint), "held_frames": [names[i] for i in sorted(held)],
        "native_matte_parity": parity,
        "held": check, "all": group_measures(list(range(len(names)))), "neck_contour": contour,
        "views": {name: group_measures([index]) for index, name in enumerate(names)},
        "protected_head_vertex_delta": 0., "modified_vertices": int(affected.sum()),
        "neck_checks_sha256": file_hash(args.neck_checks) if args.neck_checks else None,
    }
    if args.attach_rim:
        report["scene_contact_counts"] = {names[i]: len(xy) for i, (xy, _) in contacts.items()}
        report["posterior_checks"] = [
            {"frame": names[i], "uv": [x, y], "before_px": float(abs(b - x)),
             "after_px": float(abs(a - x)) if np.isfinite(a) else None}
            for (i, _, x, y, split, region, _), b, a in zip(observations, before, after, strict=True)
            if split == "check" and region == "posterior"
        ]
    path = args.output / "rim-fit.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tracker.timestep = int(original["n_processed_frames"])
    tracker.evaluate(make_visualization=True, epoch=0)
    tracker.tb_writer.close()
    evaluated = tracker.out_dir / "eval_0/image_grid"
    if len(list(evaluated.glob("*.jpg"))) != len(names):
        raise RuntimeError("Incomplete native grid")
    saved = tracker.out_dir / "tracked_flame_params_0.npz"
    with np.load(saved, allow_pickle=False) as candidate:
        if set(candidate.files) != set(original.files) or any(
                not np.array_equal(candidate[k], original[k]) for k in original.files if k != "static_offset"):
            raise RuntimeError("The lower-rim candidate changed a frozen checkpoint field")
    report.update(image_grid=str(evaluated), checkpoint=str(saved))
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], evaluated, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "sequence", "frames", "targets", "registered", "prior-report", "flame-masks", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--attach-rim", action="store_true", help="Fit a shared tapered neck-base contraction.")
    parser.add_argument("--body-contact", type=Path)
    parser.add_argument("--body-scene", type=Path)
    parser.add_argument("--neck-checks", type=Path, help="Baseline report with additional native check contours.")
    cv2.setNumThreads(1)
    args = parser.parse_args()
    if bool(args.body_contact) != bool(args.body_scene) or (args.body_contact and not args.attach_rim):
        parser.error("Body contact evidence requires both --body-scene and --attach-rim")
    run(args)
