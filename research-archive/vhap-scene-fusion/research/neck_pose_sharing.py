"""Read-only pose-sharing experiment on a registered VHAP run, in the native camera raster."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, mesh_intersections, project
from fit_vhap_neck_sequence import contour_edges
from prepare_video_avatar import setup_vhap_cameras


def read(path):
    return json.loads(path.read_bytes())


def stats(values):
    values = np.asarray(values, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    return {"count": len(values), "missing": int(len(values) - len(finite)),
            "over_3": int(np.count_nonzero(finite > 3)), **(
                dict(zip(("median", "p95", "max"), np.percentile(finite, [50, 95, 100]).tolist()))
                if len(finite) else {})}


def spread(rotations, mask):
    center = rotations[mask].mean()
    return dict(zip(("median_deg", "p95_deg", "max_deg"), np.percentile(
        np.rad2deg((center.inv() * rotations).magnitude()), [50, 95, 100]).tolist()))


def shoulder_check(observations, names, views, train):
    if observations["names"].tolist() != names:
        raise ValueError("Native RTMW order differs from camera order")
    report = {}
    for label in ("left_shoulder", "right_shoulder"):
        index = observations["joint_names"].tolist().index(label)
        observed = observations["keypoints"][:, index]
        disagreement = observations["flip_disagreement"][:, index]
        rays, centers = [], []
        for xy, (k, rotation, center) in zip(observed, views):
            ray = rotation @ np.linalg.solve(k, np.r_[xy, 1.])
            rays.append(ray / np.linalg.norm(ray))
            centers.append(center)
        plane = np.eye(3)[None] - np.einsum("ij,ik->ijk", rays, rays)
        w = 1. / np.sqrt(1. + disagreement[train]**2)
        a = plane[train] * w[:, None, None]
        b = np.einsum("nij,nj->ni", a, np.asarray(centers)[train])
        initial = np.linalg.lstsq(a.reshape(-1, 3), b.ravel(), rcond=None)[0]

        def residual(point):
            return np.stack([project(point[None], *view)[0][0] for view in views]) - observed

        fit = least_squares(lambda p: (residual(p)[train] * w[:, None]).ravel(),
                            initial, loss="huber", f_scale=3.)
        if not fit.success:
            raise RuntimeError("Static shoulder triangulation did not converge")
        err = np.linalg.norm(residual(fit.x), axis=1)
        report[label] = {"static_point_held_error_px": stats(err[~train]),
                         "held_detector_flip_disagreement_px": stats(disagreement[~train])}
    return report


class NeckContours:
    def __init__(self, fit, names, views, edges, eligible):
        self.names, self.views, self.edges, self.eligible = names, views, edges, eligible
        self.held = set(fit["held_frames"])
        self.observations = [[q for q in fit["frame_results"][n]["contours"]
                              if q["split"] == "check" and q["after_x"] is not None] for n in names]

    def sample(self, world, view_id, points=None):
        if points is None:
            points = self.observations[view_id]
        xy, z = project(world, *self.views[view_id])
        pred = np.full(len(points), np.nan)
        for label, sign in (("left", -1), ("right", 1)):
            ids = [j for j, p in enumerate(points) if p["side"] == label]
            if ids:
                rows = np.array([points[j]["native_xy"][1] for j in ids])
                pred[ids] = mesh_intersections(xy, z, world, self.edges, self.eligible, rows, sign)[0]
        return pred

    def evaluate(self, world, reference, baseline=False):
        observed, held, fixed, cross, replay = [], [], [], [], []
        frame_stats = {}
        for i, (name, pts) in enumerate(zip(self.names, self.observations)):
            if not pts:
                continue
            target = np.array([q["native_xy"][0] for q in pts])
            pred = self.sample(world[i], i)
            errors = np.abs(pred - target)
            observed.extend(errors)
            if baseline:
                replay.extend(np.abs(pred - [q["after_x"] for q in pts]))
            if name not in self.held:
                continue
            held.extend(errors)
            ref_errors = np.abs(self.sample(world[reference], i) - target)
            fixed.extend(ref_errors)
            frame_stats[name] = {"own": stats(errors), "fixed": stats(ref_errors)}
            for j, surface in enumerate(world):
                if i != j:
                    cross.extend(np.abs(self.sample(surface, i) - target))
        if baseline and (not np.isfinite(replay).all() or max(replay) > .02):
            raise RuntimeError("Baseline failed to reproduce the saved neck contour coordinates")
        return {"own_all": stats(observed), "own_held": stats(held), "fixed_held": stats(fixed),
                "all_other_poses_held": stats(cross), "by_held_frame": frame_stats}


def outline_panel(output, scene, name, view, images, faces):
    raw = cv2.imread(str(scene / "orig_undist/images" / name))
    if raw is None:
        raise IOError(name)
    clouds = [project(points, *view)[0] for _, points in images]
    all_xy = np.vstack(clouds)
    lo = np.maximum(np.floor(all_xy.min(0) - [14, 16]).astype(int), 0)
    hi = np.minimum(np.ceil(all_xy.max(0) + [15, 20]).astype(int), raw.shape[1::-1])

    def panel(img, title):
        crop = img[lo[1]:hi[1], lo[0]:hi[0]]
        tile = cv2.copyMakeBorder(crop, 29, 0, 0, 0, cv2.BORDER_CONSTANT, value=(33, 29, 26))
        cv2.putText(tile, title, (3, 19), cv2.FONT_HERSHEY_SIMPLEX, .40, (244, 244, 244), 1, cv2.LINE_AA)
        return tile

    panels = [panel(raw, "Source " + Path(name).stem)]
    for (label, _), xy in zip(images, clouds):
        alpha = np.zeros(raw.shape[:2], np.uint8)
        for tri in np.rint(xy).astype(np.int32)[faces]:
            cv2.fillConvexPoly(alpha, tri, 255)
        contour = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        frame = raw.copy()
        cv2.drawContours(frame, contour, -1, (0, 225, 255), 1, cv2.LINE_AA)
        panels.append(panel(frame, label))
    # Two rows keep original-pose and fixed-pose outlines readable at native crop size.
    grid = np.vstack([np.hstack([panels[0], *panels[start::2]]) for start in (1, 2)])
    if not cv2.imwrite(str(output / (Path(name).stem + ".png")), grid):
        raise IOError("Could not save native-pixel review")


@torch.no_grad()
def run(args):
    root, out = args.run.resolve(strict=True), args.output.resolve()
    seq = root / "head-neck"
    init = read(seq / "initialization.json")
    records = init["frames"]
    names = [r["name"] for r in records]
    fit = read(root / "neck-pitch/neck-joint-fit.json")
    train = ~np.isin(names, fit["held_frames"])
    ref = names.index(read(root / "split.json")["reference"])
    receipt = read(root / "rear-rgb/result.json")
    checkpoint = Path(receipt["checkpoint"]).resolve(strict=True)
    checkpoint.relative_to(root)
    registration = root / "registered-head/registered-head.npz"
    reg = np.load(registration, allow_pickle=False)
    if (reg["filenames"].tolist() != names or reg["initialization_sha256"].item() != sha256((seq/"initialization.json").read_bytes()).hexdigest()
            or reg["checkpoint_sha256"].item() != sha256(checkpoint.read_bytes()).hexdigest()):
        raise ValueError("Registered geometry and VHAP source differ")
    for f in args.review:
        if f not in names:
            raise ValueError(f"Unregistered review view {f}")
    out.mkdir(parents=True, exist_ok=False)
    plan = {"input": str(root), "checkpoint_sha256": reg["checkpoint_sha256"].item(),
            "cases": ["baseline", "shared_local_neck", "shared_root_orientation", "shared_root_and_pivot",
                      "half_shared_root_and_pivot"],
            "pool": "Equal-weight non-held frames, SO(3) chordal mean, median root-joint position.",
            "splits": {"nonheld": np.asarray(names)[train].tolist(), "held": fit["held_frames"],
                       "fixed_pose": names[ref]},
            "scores": "Same physical check rows as baseline, count missing intersections. Cross-pose is a stress test.",
            "no_model_output": True}
    if args.anatomical:
        plan["cases"] = ["baseline", "shared_pitch_face_protected", "shared_collar_surface"]
        plan["pitch"] = "Set the repo's local axis-angle X coordinate to the median across nonheld frames; preserve head orientation and pivot by FK."
        plan["collar"] = "Mean registered positions of homologous nonheld vertices, blended by the stock FLAME lower-root skinning weight within the neck mask. Protect all facial landmark triangles, face and ears."
    if args.temporal:
        plan["cases"] = ["baseline","shared_pitch_face_protected","smooth_shared_pitch"]
        plan["temporal"] = "Five equally spaced linear-interpolation knots, each within 2 degrees of the nonheld median local pitch. Optimize native fit contours only, equal total weight per frame, Huber 3px. Preserve head orientation/pivot. One candidate, no tuning on check rows."
    (out/"plan.json").write_text(json.dumps(plan, indent=2))
    setup_vhap_cameras(camera_gauge(records, init["scene_scale"])[2])
    views = [camera(r) for r in records]
    results = {}
    samples = {name: [] for name in args.review}
    with TemporaryDirectory() as temp:
        cfg = tyro.cli(BaseTrackingConfig, args=["--data.root-folder", str(seq.parent), "--data.sequence", seq.name,
                       "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", temp, "--no-async-func"])
        tracker = GlobalTracker(cfg)
        try:
            tracker.load_from_tracked_flame_params(checkpoint)
            ids = torch.arange(len(names), device=tracker.device)

            def forward():
                chunks = [tracker.forward_flame(i)[:3] for i in ids.split(cfg.batch_size)]
                return [torch.cat(k).cpu().numpy() for k in zip(*chunks)]

            base, canonical, lmk = forward()
            np.testing.assert_allclose(base, reg["vertices_tracker"], rtol=0, atol=1e-6)
            faces = tracker.flame.faces.cpu().numpy()
            np.testing.assert_array_equal(faces, reg["faces"])
            edges, eligible, protected, masks = contour_edges(faces, args.flame_masks)
            contours = NeckContours(fit, names, views, edges, eligible)
            root_r = Rotation.from_rotvec(tracker.rotation.cpu().numpy())
            neck_r = Rotation.from_rotvec(tracker.neck_pose.cpu().numpy())
            t = tracker.translation.cpu().numpy().copy()
            head = root_r * neck_r
            root_mean, neck_mean = root_r[train].mean(), neck_r[train].mean()
            joints = np.einsum("jv,nvc->njc", tracker.flame.J_regressor[:2].cpu().numpy(), canonical)
            arm = joints[:,1] - joints[:,0]
            pivot = joints[:,0] + t
            pivot_center = np.median(pivot[train], axis=0)
            stock_weights = tracker.flame.lbs_weights.cpu().numpy()
            collar = np.zeros(len(stock_weights), dtype=np.float32)
            collar[masks["neck"]] = stock_weights[masks["neck"], 0]
            collar[protected] = 0.
            collar[faces[tracker.flame.full_lmk_faces_idx.cpu().numpy()].ravel()] = 0.
            static_collar = base[train].mean(axis=0)
            affected_faces = faces[(collar[faces] > 0).any(axis=1)]
            basis = reg["tracker_to_world_rotation"].T * float(reg["scene_scale"])
            origin = reg["scene_origin"]
            gt = np.load(seq/"landmark2d/STAR.npz")["face_landmark_2d"][:, 17:68]
            crop_size = np.array(tracker.image_size)[::-1]
            gt_pixels = [cv2.perspectiveTransform(np.ascontiguousarray((g[:, :2]*crop_size-.5)[None]),
                         np.linalg.inv(row["warp"]))[0] for g,row in zip(gt,records)]
            face_ref = [project(points[17:68] @ basis + origin, *view)[0] for points,view in zip(lmk,views)]
            data = {"shared_shape": list(tracker.shape.shape), "static_offset": list(tracker.static_offset.shape),
                    "dynamic_offset": cfg.model.use_dynamic_offset, "root_spread": spread(root_r, train),
                    "neck_spread": spread(neck_r, train), "head_spread": spread(head, train),
                    "root_pivot_distance_from_train_median_flame_mm": stats(1000*np.linalg.norm(pivot - pivot_center,axis=1)),
                    "valid_held_points_by_region": dict(Counter(q["region"] for n,qs in zip(names,contours.observations)
                        if n in contours.held for q in qs)), "shoulders": shoulder_check(np.load(root/"body-observations.npz"), names, views, train)}
            data["collar_weight_nonzero_vertices"] = int((collar > 0).sum())
            data["collar_weight_above_half_vertices"] = int((collar > .5).sum())
            data["held_face_landmark_count_per_image"] = {n: int(np.sum(g[:,2] > 0)) for n,g in zip(names,gt) if n in contours.held}
            if args.temporal:
                time = np.array([row["timestamp_s"] for row in records])
                knots = np.linspace(time.min(),time.max(),5)
                basis_time = np.column_stack([np.interp(time,knots,vec) for vec in np.eye(5)])
                center_deg = np.median(np.rad2deg(neck_r.as_rotvec()[train,0]))
                fit_rows = [[q for q in fit["frame_results"][name]["contours"]
                             if q["split"]=="fit" and q["after_x"] is not None] if train[i] else []
                            for i,name in enumerate(names)]
                target = np.array([q["native_xy"][0] for rows in fit_rows for q in rows])
                row_ids = np.repeat(np.arange(len(names)),[len(rows) for rows in fit_rows])
                if not len(target) or not np.all(train[row_ids]):
                    raise ValueError("Invalid training witnesses")
                scale = np.sqrt(np.mean(np.bincount(row_ids)[np.unique(row_ids)]) / np.bincount(row_ids)[row_ids])
                original_x = np.array([q["after_x"] for rows in fit_rows for q in rows])

                def predict(pitch):
                    angles = neck_r.as_rotvec()
                    angles[:,0] = np.deg2rad(pitch)
                    local = Rotation.from_rotvec(angles)
                    rigid = head * local.inv()
                    translation = t + root_r.apply(arm) - rigid.apply(arm)
                    for attr, value in (("neck_pose",angles),("rotation",rigid.as_rotvec()),("translation",translation)):
                        tensor = getattr(tracker,attr)
                        tensor.copy_(torch.as_tensor(value).to(tensor))
                    world = forward()[0] @ basis + origin
                    x = np.concatenate([contours.sample(world[i],i,rows) for i,rows in enumerate(fit_rows) if rows])
                    missing = ~np.isfinite(x)
                    x[missing] = target[missing] + np.where(original_x[missing] >= target[missing],1,-1) * (
                        np.abs(original_x[missing] - target[missing]) + 24.)
                    return x

                def residual(values):
                    return (predict(center_deg+basis_time @ values) - target)*scale

                def jacobian(values):
                    pitch = center_deg + basis_time @ values
                    dx = (predict(pitch+.05) - predict(pitch-.05))/.1
                    return (dx*scale)[:,None]*basis_time[row_ids]

                solution = least_squares(residual,np.zeros(5),jac=jacobian,bounds=(-2.,2.),
                                         loss="huber",f_scale=3.,max_nfev=30)
                soft_pitch = center_deg + basis_time @ solution.x
                data["temporal_pitch_fit"] = {"fit_rows":len(target),"fit_frames":int(len(np.unique(row_ids))),
                    "global_center_deg":float(center_deg),"residual_knots_deg":solution.x.tolist(),
                    "fitted_range_deg":[float(soft_pitch.min()),float(soft_pitch.max())],
                    "status":int(solution.status),"solver_message":solution.message,
                    "function_evaluations":int(solution.nfev),"cost":float(solution.cost)}
            for case in plan["cases"]:
                rt, nt, trans = root_r, neck_r, t
                if case == "shared_local_neck":
                    nt = Rotation.from_rotvec(np.tile(neck_mean.as_rotvec(), (len(names),1)))
                elif case in ("shared_pitch_face_protected","smooth_shared_pitch"):
                    angles = neck_r.as_rotvec()
                    angles[:,0] = np.deg2rad(soft_pitch) if case=="smooth_shared_pitch" else np.median(angles[train,0])
                    nt = Rotation.from_rotvec(angles)
                    rt = head * nt.inv()
                    trans = t + root_r.apply(arm) - rt.apply(arm)
                elif case not in ("baseline", "shared_collar_surface"):
                    fraction = .5 if case == "half_shared_root_and_pivot" else 1.
                    rt = root_r * Rotation.from_rotvec(fraction * (root_r.inv() * root_mean).as_rotvec())
                    nt = rt.inv() * head
                    trans = t + root_r.apply(arm) - rt.apply(arm)
                    if case != "shared_root_orientation":
                        # Interpolate the actual lower joint, not an arbitrary camera-space translation.
                        trans = (1-fraction)*trans + fraction*(pivot_center - joints[:,0])
                if case == "baseline":
                    vertices, landmarks = base, lmk
                elif case == "shared_collar_surface":
                    vertices = base + collar[None,:,None]*(static_collar - base)
                    landmarks = lmk
                else:
                    for attr, val in (("rotation",rt.as_rotvec()),("neck_pose",nt.as_rotvec()),("translation",trans)):
                        tensor = getattr(tracker, attr)
                        tensor.copy_(torch.as_tensor(val).to(tensor))
                    vertices, _, landmarks = forward()
                world = vertices @ basis + origin
                scores = contours.evaluate(world, ref, baseline=case=="baseline")
                if case == "shared_collar_surface":
                    def normals(mesh):
                        tri = mesh[:, affected_faces]
                        return np.cross(tri[:,:,1]-tri[:,:,0], tri[:,:,2]-tri[:,:,0])
                    before, after = normals(base), normals(vertices)
                    area = np.linalg.norm(before,axis=2)
                    good = area > np.finfo(np.float32).eps ** 2
                    ratio = np.linalg.norm(after,axis=2)[good]/area[good]
                    scores["collar_triangles"] = {"nondegenerate_pose_faces": int(good.sum()),
                        "normal_dot_nonpositive": int(np.sum((before*after).sum(axis=-1)[good] <= 0)),
                        "area_ratio_min_p05_p95_max": np.percentile(ratio,[0,5,95,100]).tolist()}
                projections = np.asarray([project(points[17:68] @ basis + origin, *v)[0] for points,v in zip(landmarks,views)])
                visible = gt[:,:,2] > 0
                scores["face_shift_relative_baseline_px"] = stats(np.linalg.norm(projections-face_ref,axis=2))
                scores["visible_inner_face_STAR_error_px"] = stats(np.linalg.norm(projections-gt_pixels,axis=2)[visible])
                scores["visible_inner_face_STAR_held_px"] = stats(np.linalg.norm(projections-gt_pixels,axis=2)[visible & ~train[:,None]])
                results[case] = scores
                if case in ("baseline","shared_root_orientation","half_shared_root_and_pivot","shared_pitch_face_protected","shared_collar_surface","smooth_shared_pitch"):
                    tag = {"baseline":"Base","shared_root_orientation":"Root","half_shared_root_and_pivot":"Half root+pos",
                           "shared_pitch_face_protected":"Pitch","shared_collar_surface":"Collar","smooth_shared_pitch":"Soft pitch"}[case]
                    for name, panels in samples.items():
                        i = names.index(name)
                        panels.extend([(tag+" own",world[i].copy()), (tag+" fixed",world[ref].copy())])
                print(case, scores["own_held"], scores["fixed_held"], flush=True)
        finally:
            tracker.tb_writer.close()
    for name, panels in samples.items():
        outline_panel(out,root/"scene",name,views[names.index(name)],panels,faces)
    (out/"result.json").write_text(json.dumps({"source": data, "cases": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for arg in ("run","output","flame-masks"):
        parser.add_argument("--"+arg,type=Path,required=True)
    parser.add_argument("--review",nargs="*",default=[])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--anatomical",action="store_true")
    group.add_argument("--temporal",action="store_true")
    run(parser.parse_args())
