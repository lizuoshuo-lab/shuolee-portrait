"""Fit shared lower-neck Gaussian support on the posed registered training views."""

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import torch
from roma import (
    quat_product, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz,
    rotmat_to_unitquat, unitquat_to_rotmat,
)
from utils.graphics_utils import compute_face_orientation

from export_head_neck_frames import boundary_loops
from fit_side_frames import alpha_maps
from fuse_scene import read_gaussians, render, save_jpg, to_bgr
from msv2_scene_engine import make_rasterizer
from train_joint import combine


def camera(row):
    k = np.array([[row["fx"], 0., row["cx"] - .5],
                  [0., row["fy"], row["cy"] - .5], [0., 0., 1.]])
    return k, (row["width"], row["height"]), np.array(row["rotation"]), np.array(row["position"])


def project(points, k, r, c):
    xy = (points - c) @ r @ k.T
    if np.any(xy[:, 2] <= 0.):
        raise ValueError("The neck evidence projects behind the camera")
    return xy[:, :2] / xy[:, 2:]


def panel(image, label):
    height = 352
    canvas = cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
    canvas = cv2.copyMakeBorder(canvas, 30, 0, 0, 2, cv2.BORDER_CONSTANT, value=(22, 22, 22))
    cv2.putText(canvas, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .44, (242, 242, 242), 1, cv2.LINE_AA)
    return canvas


def run(args):
    cv2.setNumThreads(1)
    with np.load(args.geometry / "shared-neck-geometry.npz", allow_pickle=False) as tape:
        frames = [Path(name).stem for name in tape["frames"]]
        faces, old, new = tape["faces"], tape["old"], tape["vertices"]
        basis, scale, origin, rim = tape["basis"], float(tape["scale"]), tape["origin"], tape["rim"]
    fit_report = json.loads((args.geometry / "shared-neck.json").read_bytes())
    if set(frames) != {Path(name).stem for name in fit_report["fit_frames"] + fit_report["held_frames"]}:
        raise ValueError("Fit report and geometry use different frame sets")
    held = {Path(name).stem for name in fit_report["held_frames"]}
    ref = frames.index(args.pose)
    with args.flame_masks.open("rb") as reader:
        regions = pickle.load(reader, encoding="latin1")
    protect = np.unique(np.concatenate([regions[key] for key in
                                        ("face", "left_ear", "right_ear", "lips", "left_eyeball", "right_eyeball")]))
    ring, = [loop for loop in boundary_loops(faces) if set(loop) == set(rim)]
    edges = np.unique(np.sort(np.concatenate([faces[:, a] for a in ([0, 1], [1, 2], [2, 0])]), axis=1), axis=0)
    length = np.linalg.norm(old[ref, edges[:, 0]] - old[ref, edges[:, 1]], axis=1)
    graph = coo_matrix((length, edges.T), shape=(old.shape[1], old.shape[1])).tocsr()
    distance = dijkstra(graph, directed=False, indices=ring, min_only=True)[faces].mean(axis=1)
    band = (distance <= .12) & ~np.isin(faces, protect).any(axis=1)
    ids = np.flatnonzero(band)
    feather = np.clip((.12 - distance[ids]) / .04, 0., 1.)
    feather = feather**2 * (3. - 2. * feather)
    full = read_gaussians(args.snapshot / "point_cloud.ply")
    source = tuple(field[:-len(faces)] for field in full)
    actor = tuple(field[-len(faces):] for field in full)
    actor_positions, colors, opacity, sizes, quaternions = actor
    rows = json.loads((args.snapshot / "cameras.json").read_bytes())
    if set(frames) != {row["img_name"] for row in rows}:
        raise ValueError("Snapshot cameras must match the complete posed sequence")
    old_world = old[ref] @ basis.T * scale + origin
    np.testing.assert_allclose(actor_positions.cpu().numpy(), old_world[faces].mean(1), rtol=0., atol=2e-6)
    def gpu(value):
        return torch.as_tensor(value, device="cuda", dtype=torch.float32)

    topo = torch.as_tensor(faces, device="cuda", dtype=torch.long)
    origin, basis, weights = gpu(origin), gpu(basis), gpu(feather)
    old_rot, old_scale = compute_face_orientation(gpu(old[ref]), topo, return_scale=True)

    def pose(vertices):
        v = gpu(vertices)
        rot, metric = compute_face_orientation(v, topo, return_scale=True)
        motion = (basis @ rot) @ (basis @ old_rot).transpose(-1, -2)
        q = quat_product(rotmat_to_unitquat(motion), quat_wxyz_to_xyzw(quaternions))
        return (v @ basis.T * scale + origin)[topo].mean(1), colors, opacity, sizes * (metric / old_scale), quat_xyzw_to_wxyz(q)

    def ellipsoids(posed, vertices):
        rot = unitquat_to_rotmat(quat_wxyz_to_xyzw(posed[4][ids]))
        frame, _ = compute_face_orientation(gpu(vertices), topo, return_scale=True)
        normals = (basis @ frame)[ids, :, 1]
        # Transform in the current triangle's tangent plane and normal and
        # rasterize the exact transformed covariance.
        # Mesh normals follow triangle winding. Resolve their outward sign from
        # each posed neck's rim center in the plane normal to its neck axis.
        rim_center = gpu(vertices[ring].mean(0)) @ basis.T * scale + origin
        axis = (gpu(vertices[protect].mean(0)) @ basis.T * scale + origin) - rim_center
        axis = axis / torch.linalg.vector_norm(axis)
        radial = posed[0][ids] - rim_center
        radial = radial - (radial @ axis)[:, None] * axis
        outward = normals * torch.sign((normals * radial).sum(1))[:, None]
        alignment = (rot.transpose(-1, -2) @ normals[..., None]).squeeze(-1).square()
        return (normals[:, :, None] * normals[:, None, :], outward, alignment,
                (rot * posed[3][ids, None, :].square()) @ rot.transpose(-1, -2))

    def adjust(posed, ellipsoid, amount):
        normal, outward, _, covariance = ellipsoid
        tangent_gain = 1. - weights * amount[0]
        normal_gain = 1. - weights * amount[1]
        transform = tangent_gain[:, None, None] * torch.eye(3, device="cuda")
        transform = transform + (normal_gain - tangent_gain)[:, None, None] * normal
        vals, vecs = torch.linalg.eigh(transform @ covariance @ transform)
        vecs[:, :, 0] *= torch.linalg.det(vecs).sign()[:, None]
        sigmas, orientation = posed[3].clone(), posed[4].clone()
        sigmas[ids] = vals.clamp_min(1e-18).sqrt()
        orientation[ids] = quat_xyzw_to_wxyz(rotmat_to_unitquat(vecs))
        centers = posed[0]
        if len(amount) == 3:
            centers = centers.clone()
            centers[ids] -= (scale * .001 * amount[2]) * weights[:, None] * outward
        return (centers, *posed[1:3], sigmas, orientation)

    def adjust_per_face(posed, ellipse, values):
        _, outward, alignment, _ = ellipse
        tangent, normal, inset = ((values.sigmoid() * bounds[:, None]) * weights).unbind()
        gain = 1. - tangent[:, None] * (1. - alignment) - normal[:, None] * alignment
        scales = posed[3].index_copy(0, band_indices, posed[3][ids] * gain)
        centers = posed[0].index_copy(0, band_indices, posed[0][ids] - outward * (scale * .001 * inset[:, None]))
        return centers, *posed[1:3], scales, posed[4]

    labels = gpu(np.r_[np.zeros(len(source[0])), band])[:, None].expand(-1, 3).contiguous()

    def visible(raster, head):
        xyz, _, alpha, radii, quats = combine(source, head)
        out, _ = raster(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None, colors_precomp=labels,
                        opacities=alpha, scales=radii, rotations=quats, cov3D_precomp=None)
        return out[0]

    fixed = pose(new[ref])
    fixed_ellipsoid = ellipsoids(fixed, new[ref])
    first_k, size, _, _ = camera(rows[0])
    _, _, matte_parity, maps, raw_size = alpha_maps(args.scene, size, first_k, [args.pose + ".jpg"])
    observations = {}
    for row in rows:
        name = row["img_name"]
        index = frames.index(name)
        k, size, r, c = camera(row)
        np.testing.assert_allclose(k, first_k, rtol=0., atol=1e-8)
        posed = pose(new[index])
        probe = torch.cat((actor_positions[ids], fixed[0][ids], posed[0][ids])).cpu().numpy()
        xy = project(probe, k, r, c)
        lo = np.floor(xy.min(0) - [48, 56]).astype(int).clip(0, size)
        hi = np.ceil(xy.max(0) + [48, 44]).astype(int).clip(0, size)
        x, y = lo
        X, Y = hi
        raw = cv2.imread(str(args.scene / "alpha_maps" / (name + ".jpg")), 0)
        if raw is None or raw.shape[::-1] != raw_size:
            raise ValueError(f"Missing native matte: {name}")
        matte = cv2.remap(raw, maps[0][y:Y, x:X], maps[1][y:Y, x:X], cv2.INTER_LINEAR) >= 128
        outside = cv2.distanceTransform(1 - matte.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        inside = cv2.distanceTransform(matte.astype(np.uint8), cv2.DIST_L2, 5)
        k[:2, 2] -= lo
        raster = make_rasterizer(k, tuple(hi-lo), r, c)
        baseline = visible(raster, posed)
        core = (gpu(inside) > 3.) & (baseline >= .35)
        exterior = gpu(outside) > 3.
        if int(core.sum()) < 40 or int(exterior.sum()) < 40:
            raise ValueError(f"Insufficient core/exterior evidence: {name}")
        spill_weight = gpu(np.clip((outside - 3.) / 6., 0., 1.))
        observations[name] = (raster, posed, ellipsoids(posed, new[index]), baseline, core, exterior,
                              spill_weight, outside, (x, y, X, Y))

    training = [item for name, item in observations.items() if name not in held]
    def objective(values):
        errors = []
        for raster, posed, ellipse, prior, core, _, weight, _, _ in training:
            alpha = visible(raster, adjust(posed, ellipse, values))
            base = prior[core].sum()
            loss = ((prior[core] - alpha[core]).clamp_min(0.).sum() / base - .015).clamp_min(0.)
            errors.append((alpha * weight).sum() / base + 65. * loss.square())
        per_view = torch.stack(errors)
        return float(.5 * per_view.mean() + .5 * per_view.topk(max(1, len(errors)//5)).values.mean()
                     ) + .025 * np.square(values[:2]).sum() + .002 * np.square(values[2:]).sum()

    # One shared offset in model millimetres. Limit it to the immediate surface
    # neighborhood, and keep every center outside the anatomical band frozen.
    fit_summary = {}
    if args.per_face:
        # Every training view contributes each epoch, without a singled-out
        # pose or per-camera parameters. Coverage penalizes empty seams.
        band_indices = torch.as_tensor(ids, device="cuda", dtype=torch.long)
        bounds = gpu([.32, .55, 4.])
        with torch.enable_grad():
            params = torch.full((3, len(ids)), -3., device="cuda", requires_grad=True)
            trainer = torch.optim.Adam([params], lr=.065)
            order = np.random.default_rng(17)
            for _ in range(5):
                for index in order.permutation(len(training)):
                    raster, posed, ellipse, prior, core, _, weights_exterior, _, _ = training[index]
                    alpha = visible(raster, adjust_per_face(posed, ellipse, params))
                    core_alpha = prior[core].sum()
                    lost_coverage = ((prior[core]-alpha[core]).clamp_min(0.).sum()/core_alpha - .015).clamp_min(0.)
                    # Penalize apparent support near and above the review
                    # alpha threshold, increasing with exterior distance.
                    spill = (alpha.square() * weights_exterior).sum() / core_alpha
                    loss = spill + 65. * lost_coverage.square() + .004 * params.sigmoid().square().mean()
                    trainer.zero_grad(set_to_none=True)
                    loss.backward()
                    if params.grad is None or not torch.isfinite(params.grad).all():
                        raise RuntimeError("The rasterizer did not return finite covariance/position gradients")
                    trainer.step()
        values = params.detach()
        def fitted(posed, ellipse):
            return adjust_per_face(posed, ellipse, values)

        amplitudes = (values.sigmoid() * bounds[:, None] * weights).cpu().numpy()
        fit_summary = {"fitted_faces": len(ids), "steps": 5 * len(training), "per_face": True,
                       "amplitude_max_tangent_normal_inset_flame_mm": amplitudes.max(1).tolist()}
    else:
        limits = [(0., .32), (0., .55)] + ([(0., 4.)] if args.fit_inset else [])
        initial = np.zeros(len(limits))
        fit = minimize(objective, initial, method="Powell", bounds=limits,
                       options={"maxfev": 60, "xtol": .008, "ftol": .001})
        if objective(initial) < fit.fun:
            raise RuntimeError("No feasible scale improvement")
        def fitted(posed, ellipse):
            return adjust(posed, ellipse, fit.x)

        fit_summary = {"covariance_reduction": fit.x[:2].tolist(),
                       "normal_inset_flame_mm": float(fit.x[2]) if args.fit_inset else 0.,
                       "evaluations": fit.nfev, "objective_initial": objective(initial), "objective_fitted": fit.fun}
    after_fixed = fitted(fixed, fixed_ellipsoid)
    assert torch.equal(after_fixed[0][~band], fixed[0][~band])
    assert torch.equal(after_fixed[1], fixed[1]) and torch.equal(after_fixed[2], fixed[2])
    stats = {}
    for name, (raster, posed, ellipse, prior, core, _, _, outside, roi) in observations.items():
        after = fitted(posed, ellipse)
        after_alpha = visible(raster, after)
        def metrics(alpha_tensor):
            alpha = alpha_tensor.cpu().numpy()
            support = alpha >= .10
            excess = support & (outside > 3.)
            _, _, components, _ = cv2.connectedComponentsWithStats(excess.astype(np.uint8), connectivity=8)
            return {"overflow_px": int(excess.sum()),
                    "largest_component_px": int(components[1:, cv2.CC_STAT_AREA].max()) if len(components) > 1 else 0,
                    "max_distance_px": float(outside[support].max(initial=0.))}

        core_ratio = float(after_alpha[core].sum() / prior[core].sum())
        stats[name] = {"fixed_pose": [metrics(visible(raster, head)) for head in (fixed, after_fixed)],
                       "native_pose": [metrics(prior), metrics(after_alpha)],
                       "native_core_alpha_ratio": core_ratio}
        if name in held:
            x, y, X, Y = roi
            raw = cv2.imread(str(args.scene / "orig_undist/images" / (name + ".jpg")))[y:Y, x:X]
            truth = gpu(raw[:, :, ::-1].copy()).permute(2, 0, 1) / 255.
            stats[name]["native_core_rgb_mae_255"] = [
                255. * float((render(raster, combine(source, head)) - truth).abs()[:, core].mean())
                for head in (posed, after)]
    args.output.mkdir(exist_ok=False)
    worst = max(stats, key=lambda name: stats[name]["fixed_pose"][1]["largest_component_px"])
    for name in dict.fromkeys([args.pose, *args.review, worst]):
        raster, *_, (x, y, X, Y) = observations[name]
        raw = cv2.imread(str(args.scene / "orig_undist/images" / (name + ".jpg")))[y:Y, x:X]
        columns = [panel(raw, "Same-time source" if name == args.pose else "Source, different time")]
        for head, label in zip((actor, fixed, after_fixed), ("Previous", "Before Gaussian fit", "After Gaussian fit")):
            columns.append(panel(to_bgr(render(raster, combine(source, head))), label))
        save_jpg(args.output / f"{args.pose}-at-{name}.jpg", np.hstack(columns))
    summary = {key: [max(row["fixed_pose"][i][key] for row in stats.values()) for i in range(2)]
               for key in ("largest_component_px", "max_distance_px")}
    summary["min_held_native_core_alpha_ratio"] = min(stats[name]["native_core_alpha_ratio"] for name in held)
    summary["max_held_native_core_rgb_mae_increase_255"] = max(
        stats[name]["native_core_rgb_mae_255"][1] - stats[name]["native_core_rgb_mae_255"][0] for name in held)
    summary["max_held_native_pose_px"] = [
        max(stats[name]["native_pose"][i]["max_distance_px"] for name in held) for i in range(2)]
    summary["native_pose_held_views_over_3px"] = [
        sum(stats[name]["native_pose"][i]["max_distance_px"] > 3. for name in held) for i in range(2)]
    report = {"pose": args.pose, "same_time_camera": args.pose, "n_fit": len(training), "n_held": len(held),
              "working_resolution": list(camera(rows[0])[1]),
              "mask_distance_px": 3., "visible_alpha_threshold": .10,
              "held_visible_support_within_3px": summary["native_pose_held_views_over_3px"][1] == 0,
              **fit_summary, "matte_remap_iou": matte_parity, "worst_camera": worst,
              "summary": summary, "views": stats}
    (args.output / "footprint.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({**fit_summary, **summary, **{name: stats[name] for name in [args.pose, *args.review, worst]}},
                     indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("geometry", "snapshot", "scene", "flame-masks", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--review", nargs="*", default=[])
    parser.add_argument("--fit-inset", action="store_true")
    parser.add_argument("--per-face", action="store_true")
    with torch.no_grad():
        run(parser.parse_args())

