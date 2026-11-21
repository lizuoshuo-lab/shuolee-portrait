"""Place the shared FLAME head in registered frames without facial landmarks."""

from argparse import ArgumentParser
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp
import torch

from fuse_scene import (FIT_LANDMARKS, local_head, make_rasterizer, prepare_scene,
                        project, render, to_bgr, world_gaussians)
from profile_pose import PROFILE_MIN_YAW, fit_profile_pose
from profile_rgb_check import compare_profile_rgb
from profile_shape import PROFILE_LIMITS, profile_basis, profile_weights
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from side_masks import profile_mask
from side_tracks import track_features
from train_joint import JointParameters, combine


def camera_matrix(K, camera, pose, scale):
    R, C = camera
    A, t = pose
    return K @ np.column_stack((scale * R.T @ A, R.T @ (t - C)))


def world_priors(names, time, anchors):
    ordered = sorted(anchors, key=time.__getitem__)
    times = np.array([time[n] for n in ordered])
    rotations = Rotation.from_matrix(np.array([anchors[n][0] for n in ordered]))
    positions = np.array([anchors[n][1] for n in ordered])
    interp = Slerp(times, rotations)
    priors = {}
    for name in names:
        t = float(np.clip(time[name], times[0], times[-1]))
        center = np.array([np.interp(t, times, positions[:, j]) for j in range(3)])
        priors[name] = (interp(t).as_matrix(), center)
    return priors


def alpha_maps(scene, size, K, names):
    candidates = []
    for directory in (scene / "colmap_bg/sparse").iterdir():
        if (directory / "images.bin").is_file():
            images = read_extrinsics_binary(directory / "images.bin")
            candidates.append((len(images), directory))
    if not candidates:
        raise ValueError("Missing original COLMAP camera for the undistortion map.")
    cameras = read_intrinsics_binary(max(candidates)[1] / "cameras.bin")
    if len(cameras) != 1:
        raise ValueError("Expected one source camera.")
    camera = next(iter(cameras.values()))
    if camera.model != "OPENCV":
        raise ValueError(f"Unsupported source distortion model: {camera.model}")
    fx, fy, cx, cy, *distortion = camera.params
    source_K = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1.]])
    mx, my = cv2.initUndistortRectifyMap(source_K, np.array(distortion), None, K, size, cv2.CV_32FC1)
    masks, gray = {}, {}
    parity = []
    for name in names:
        raw = cv2.imread(str(scene / "alpha_maps" / name), 0)
        coarse = cv2.imread(str(scene / "headmask_undist/images" / Path(name).with_suffix(".png")), 0)
        image = cv2.imread(str(scene / "orig_undist/images" / name), 0)
        if raw is None or coarse is None or image is None:
            raise ValueError(f"Missing per-frame RGB or mask: {name}")
        if raw.shape[::-1] != (camera.width, camera.height) or image.shape[::-1] != size:
            raise ValueError(f"Unexpected calibration raster: {name}")
        warped = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
        # Cross-check the warp on the exact mask previously run through COLMAP.
        binary = cv2.imread(str(scene / "person_masks" / name), 0)
        delivered = cv2.imread(str(scene / "mask_undist/images" / name), 0) >= 128
        check = cv2.remap(binary, mx, my, cv2.INTER_LINEAR) >= 128
        overlap = (check & delivered).sum() / max(1, (check | delivered).sum())
        if overlap < .99:
            raise ValueError(f"Mask and native pixel coordinates disagree for {name}: {overlap:.4f}")
        parity.append(float(overlap))
        masks[name] = profile_mask(warped, coarse)
        gray[name] = image
    return masks, gray, min(parity), (mx, my), (camera.width, camera.height)


def mask_scores(alpha, evidence):
    crop = evidence["crop"]
    mask = alpha[crop] >= .5
    target = evidence["target"][crop]
    valid = evidence["valid"][crop]
    mask_area = (mask | target) & valid
    iou = (mask & target & valid).sum() / max(1, mask_area.sum())
    kernel = np.ones((3, 3), np.uint8)
    boundary_support = evidence["boundary_valid"][crop].astype(np.uint8)
    inside = cv2.erode(boundary_support, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    def contour(binary):
        return binary & ~cv2.erode(binary.astype(np.uint8), kernel).astype(bool) & inside

    a, b = contour(mask), contour(target)
    if not a.any() or not b.any():
        return {"iou": float(iou), "edge_p50_px": None, "edge_p90_px": None}
    dist = np.concatenate((cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)[b],
                           cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)[a]))
    return {"iou": float(iou), "edge_p50_px": float(np.median(dist)), "edge_p90_px": float(np.percentile(dist, 90))}


def contour_shift(alpha, evidence):
    crop = evidence["crop"]
    image = alpha[crop]
    target = evidence["target"][crop].astype(np.float32)
    valid = evidence["valid"][crop]
    width, height = image.shape[::-1]
    limit = min(20., max(6., .12 * np.count_nonzero(target.any(axis=0))))

    def error(shift):
        m = np.eye(2, 3, dtype=np.float32)
        m[:, 2] = shift
        moved = cv2.warpAffine(image, m, (width, height), flags=cv2.INTER_LINEAR)
        return float(np.abs(moved - target)[valid].mean() + .020 * np.square(shift / limit).sum())

    result = minimize(error, np.zeros(2), method="Powell", bounds=[(-limit, limit)] * 2,
                      options={"maxiter": 16, "xtol": .15, "ftol": 1e-4})
    return result.x if result.fun < error(np.zeros(2)) - .001 else np.zeros(2)


def track_pose(tracks, K, camera, prior, scale):
    xyz, uv = tracks["points3d"], tracks["points2d"]
    count = len(xyz)
    if count < 14:
        return None, {"track_count": count, "reason": "insufficient_tracks"}
    # Split complete tracks by observed location before solving the pose.
    order = np.lexsort((uv[:, 1], uv[:, 0]))
    test = order[::4]
    train = np.setdiff1d(order, test)
    if cv2.contourArea(cv2.convexHull(uv[train].astype(np.float32))) < 350:
        return None, {"track_count": count, "reason": "insufficient_image_coverage"}
    R, C = camera
    A, t = prior
    r0 = cv2.Rodrigues(R.T @ A)[0]
    t0 = (R.T @ (t - C))[:, None]
    ok, rv, tv, strong = cv2.solvePnPRansac(
        np.ascontiguousarray(xyz[train] * scale), np.ascontiguousarray(uv[train]), K, None,
        r0.copy(), t0.copy(), True, iterationsCount=150, reprojectionError=3.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or strong is None or len(strong) < max(9, .65 * len(train)):
        return None, {"track_count": count, "reason": "pnp_inlier_support"}
    X = xyz[test] * scale
    before = np.linalg.norm(cv2.projectPoints(X, r0, t0, K, None)[0][:, 0] - uv[test], axis=1)
    after = np.linalg.norm(cv2.projectPoints(X, rv, tv, K, None)[0][:, 0] - uv[test], axis=1)
    A_new, t_new = R @ cv2.Rodrigues(rv)[0], C + R @ tv[:, 0]
    angle = float(np.degrees(Rotation.from_matrix(A_new @ A.T).magnitude()))
    distance = float(np.linalg.norm(t_new - t) / scale)
    report = {"track_count": count, "fit_inliers": len(strong), "test_tracks": len(test),
              "track_test_median_before_px": float(np.median(before)),
              "track_test_median_after_px": float(np.median(after)), "track_test_p90_after_px": float(np.percentile(after, 90)),
              "rotation_delta_deg": angle, "translation_delta_flame": distance}
    if (angle > 14 or distance > .055 or np.median(after) > 3.0
            or np.percentile(after, 90) > 5.0 or np.median(after) >= np.median(before) - .20):
        return None, {**report, "reason": "independent_track_check"}
    return (A_new, t_new), {**report, "reason": "accepted"}


def blind_error(landmarks, observed, K, scale, camera, pose):
    R, C = camera
    A, t = pose
    pixel = project((landmarks[FIT_LANDMARKS] @ A.T * scale + t - C) @ R, K)
    errors = np.linalg.norm(pixel - observed[FIT_LANDMARKS], axis=1)
    return {"median_px": float(np.median(errors)), "p90_px": float(np.percentile(errors, 90))}


def pose_json(pose):
    return {"rotation": pose[0].tolist(), "translation": pose[1].tolist()}


def review_row(image, initial, final, evidence, name, method, *, compare=False):
    sy, sx = evidence["crop"]
    original = image[sy, sx]
    overlay = original.copy()
    valid = evidence["boundary_valid"][sy, sx]
    target = evidence["target"][sy, sx].astype(np.uint8)
    source_edge = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    overlay[source_edge & valid] = [75, 230, 70]
    panels = []
    entries = [(original, name + " original"), (initial[sy, sx], "Before" if compare else "World prior"),
               (final[sy, sx], method)]
    if not compare:
        entries.append((overlay, "Scored contour"))
    for item, label in entries:
        height, width = item.shape[:2]
        scale = min(278 / width, 238 / height)
        resized = cv2.resize(item, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        panel = np.full((266, 286, 3), 19, dtype=np.uint8)
        y, x = (238 - resized.shape[0]) // 2 + 25, (286 - resized.shape[1]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(panel, label, (8, 18), 0, .45, (250, 250, 250), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def run(args):
    result = json.loads((args.run / "model/metrics.json").read_text())
    settings = {key: Path(value) for key, value in result["input_paths"].items()}
    settings["export_frame"] = result["reference"]
    source_manifest = json.loads((settings["scene"] / "source-frames.json").read_text())
    time = {record["name"]: record["timestamp_s"] for record in source_manifest["frames"]}
    names = sorted(time, key=time.__getitem__)
    K, size, registered, detected, xy, track, actor, keep, _, scene = prepare_scene(SimpleNamespace(**settings))
    cameras = {name: registered[Path(name).stem] for name in names}
    masks, gray, parity, maps, source_size = alpha_maps(settings["scene"], size, K, names)
    train = result["training_frames"]
    hold = result["holdout_frames"]
    scale = result["scale"]
    anchors = {name: tuple(np.asarray(result["placement"][name][k]) for k in ("rotation", "translation"))
               for name in train}
    priors = world_priors(names, time, anchors)
    if not args.review:
        # Fixed target angles for reproducible review. Results do not select themselves.
        estimated = {}
        for name in names:
            if name in anchors:
                continue
            A, t = priors[name]
            view = (cameras[name][1] - t) @ A
            estimated[name] = np.degrees(np.arctan2(view[0], view[2]))
        for angle in (45, 70, 85, -45, -70, -85):
            candidates = [n for n, yaw in estimated.items()
                          if n not in args.review and np.sign(angle) * yaw >= 40]
            if candidates:
                args.review.append(min(candidates, key=lambda name: (abs(estimated[name] - angle), name)))
    # Keep the original trained shared model. Use median articulation when a frame has no landmarks.
    train_index = [detected.index(name) for name in train]
    template = {key: value[train_index].median(dim=0, keepdim=True).values for key, value in track.items()}
    local_head(actor, track, 0)
    bindings = actor.binding[keep].long()
    initial_size = actor.face_scaling[bindings].reshape(-1, 1, 1).clone()
    landmarks = local_head(actor, template, 0)
    local = world_gaussians(actor, np.eye(3), np.zeros(3), 1., keep)
    axes = actor.face_orien_mat[bindings] * actor.face_scaling[bindings].reshape(-1, 1, 1) / initial_size
    params = JointParameters(scene, local)
    checkpoint = torch.load(args.run / "model/training_state.pt", weights_only=True, map_location="cpu")
    params.load_state_dict(checkpoint["parameters"])
    local = params.head(SimpleNamespace(head=local, axes=axes), geometry=True)
    fitted_scene = params.scene(scene)
    profile_local = local
    if args.profile_refinement:
        report = json.loads(args.profile_refinement.read_text())
        coefficients = report["scalp_and_posterior_neck_flame_units"]
        if (not report["shape_candidate_passed_contour_comparison"] or len(coefficients) != 2
                or not all(lo <= value <= hi for value, (lo, hi) in zip(coefficients, PROFILE_LIMITS))):
            raise ValueError("Expected a validated profile shape correction.")
        weights, protected = profile_weights(actor, settings["flame_boundary_mask"])
        basis = profile_basis(actor, weights)
        vertices = actor.verts + (basis * basis.new_tensor(coefficients)[None, :, None]).sum(dim=1)[None]
        if not torch.isfinite(vertices).all() or not torch.equal(vertices[:, protected], actor.verts[:, protected]):
            raise RuntimeError("Invalid deformation or movement of protected landmarks.")
        actor.update_mesh_properties(vertices, actor.verts_cano.clone())
        shape = world_gaussians(actor, np.eye(3), np.zeros(3), 1., keep)
        axes = actor.face_orien_mat[bindings] * actor.face_scaling[bindings].reshape(-1, 1, 1) / initial_size
        profile_local = params.head(SimpleNamespace(head=shape, axes=axes), geometry=True)
    profile_center = profile_local[0].mean(dim=0).cpu().numpy()
    bounds = local[0].cpu().numpy()
    projections = {n: camera_matrix(K, cameras[n], p, scale) for n, p in anchors.items()}
    local_bounds = (bounds.min(axis=0) - .025, bounds.max(axis=0) + .025)
    support = {n: masks[n]["target"] & masks[n]["valid"] for n in names}
    diagnostics = {}
    if args.dense:
        from dense_side_tracks import dense_feature_tracks
        tracks, diagnostics = dense_feature_tracks(Path(source_manifest["video"]), source_manifest["frames"],
                                                    gray, support, projections, local_bounds, maps, source_size)
    else:
        tracks = track_features(names, gray, support, projections, local_bounds)

    def head_at(pose, model=local):
        A, t = pose
        xyz, feature, opacity, sizes, rotation = model
        shifted = scale * xyz @ xyz.new_tensor(A.T) + xyz.new_tensor(t)
        wxyz = rotation.cpu().numpy()
        q = (Rotation.from_matrix(A) * Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
        return shifted, feature, opacity, sizes * scale, xyz.new_tensor(q)

    reviews, profile_reviews, records = {}, {}, {}
    selected = set(args.review)
    detected_xy = dict(zip(detected, xy))
    for name in names:
        if name in anchors:
            records[name] = {"mode": "front_landmarks", **pose_json(anchors[name])}
            continue
        camera, prior, evidence = cameras[name], priors[name], masks[name]
        R, C = camera
        raster = make_rasterizer(K, size, R, C)
        head0 = head_at(prior)
        alpha0 = render(raster, head0, alpha_only=True).cpu().numpy()
        s0 = mask_scores(alpha0, evidence)
        pose, proof = track_pose(tracks[name], K, camera, prior, scale)
        mode, scores = "world_prior", s0
        if pose is not None:
            candidate = mask_scores(render(raster, head_at(pose), alpha_only=True).cpu().numpy(), evidence)
            if candidate["iou"] >= s0["iou"] - .025:
                mode, scores = "texture_pose", candidate
            else:
                proof["reason"] = "contour_conflict"
        if mode != "texture_pose":
            pose = prior
            shift = contour_shift(alpha0, evidence)
            if np.linalg.norm(shift) >= .2:
                depth = float(((head0[0].cpu().numpy() - C) @ R)[:, 2].mean())
                delta = R @ np.array([shift[0] * depth / K[0, 0], shift[1] * depth / K[1, 1], 0.])
                proposal = (prior[0], prior[1] + delta)
                candidate = mask_scores(render(raster, head_at(proposal), alpha_only=True).cpu().numpy(), evidence)
                if candidate["iou"] > s0["iou"] + .003:
                    pose, mode, scores = proposal, "contour_translation", candidate
        baseline, baseline_mode = pose, mode
        view = (C - pose[1]) @ pose[0]
        baseline_yaw = float(np.degrees(np.arctan2(view[0], view[2])))
        profile = None
        comparison_model = local
        if mode != "texture_pose" and abs(baseline_yaw) >= PROFILE_MIN_YAW:
            comparison_model = profile_local

            def full_alpha(pose):
                return render(raster, head_at(pose, profile_local), alpha_only=True)

            baseline_scores = mask_scores(full_alpha(baseline).cpu().numpy(), evidence)
            proposal, profile = fit_profile_pose(baseline, camera, K, scale, profile_center, evidence,
                                                 lambda p: full_alpha(p)[evidence["crop"]].contiguous().cpu().numpy())
            scores = baseline_scores
            if profile["accepted"]:
                candidate = mask_scores(full_alpha(proposal).cpu().numpy(), evidence)
                if (candidate["iou"] >= baseline_scores["iou"]
                        and candidate["edge_p90_px"] is not None and baseline_scores["edge_p90_px"] is not None
                        and candidate["edge_p90_px"] <= baseline_scores["edge_p90_px"] + .5):
                    pose, mode, scores = proposal, "profile_silhouette_pose", candidate
                else:
                    profile.update(accepted=False, reason="native_boundary_check")
            profile.update(baseline_mode=baseline_mode, baseline_yaw_deg=baseline_yaw,
                           baseline=pose_json(baseline), silhouette_baseline=baseline_scores)
        view = (C - pose[1]) @ pose[0]
        record = {"mode": mode, "camera_relative_yaw_deg": float(np.degrees(np.arctan2(view[0], view[2]))),
                  "prior": pose_json(prior), **pose_json(pose),
                  "silhouette_before": s0, "silhouette_after": scores, "texture": proof,
                  "silhouette_supported": scores["iou"] >= .72 and scores["edge_p90_px"] is not None
                  and scores["edge_p90_px"] <= 14}
        record["angular_training_candidate"] = mode == "texture_pose" and record["silhouette_supported"]
        if profile is not None:
            record["profile"] = profile
        if name in hold:
            record["blind_facial_points"] = {
                "prior": blind_error(landmarks, detected_xy[name], K, scale, camera, prior),
                "recovered": blind_error(landmarks, detected_xy[name], K, scale, camera, pose),
            }
        records[name] = record
        if name in selected:
            original = cv2.imread(str(settings["scene"] / "orig_undist/images" / name))
            rgb0 = to_bgr(render(raster, combine(fitted_scene, head0)))
            rgb1 = to_bgr(render(raster, combine(fitted_scene, head_at(pose, comparison_model))))
            reviews[name] = review_row(original, rgb0, rgb1, evidence, name, mode)
            cv2.imwrite(str(args.output / (Path(name).stem + "-source-and-side.jpg")),
                        np.hstack([cv2.resize(i, (380, 676)) for i in (original, rgb0, rgb1)]))
            if profile is not None:
                before = to_bgr(render(raster, combine(fitted_scene, head_at(baseline, comparison_model))))
                crop = evidence["crop"]
                profile["rgb_check"] = compare_profile_rgb(original[crop], before[crop], rgb1[crop],
                                                            evidence["target"][crop] & evidence["valid"][crop])
                label = (f"After: yaw {profile['tested_yaw_delta_deg']:+.1f} deg" if profile["accepted"]
                         else "After: pose retained")
                profile_reviews[name] = review_row(original, before, rgb1, evidence, name, label, compare=True)
        print(f"{name} {mode} prior {s0['iou']:.3f} final {scores['iou']:.3f} tracks {proof['track_count']}", flush=True)

    if reviews:
        cv2.imwrite(str(args.output / "side-review.jpg"), np.vstack([reviews[name] for name in args.review if name in reviews]))
    if profile_reviews and not cv2.imwrite(str(args.output / "profile-before-after.jpg"),
                                           np.vstack([profile_reviews[name] for name in args.review if name in profile_reviews])):
        raise OSError("Could not write profile-before-after.jpg")
    payload = {"scale": scale, "mask_mapping_min_iou": parity, "source_run": str(args.run),
               "dense_texture_tracking": args.dense, "tracking_diagnostics": diagnostics,
               "profile_refinement": str(args.profile_refinement) if args.profile_refinement else None,
               "review_frames": list(reviews),
               "propagated_articulation": {key: value[0].cpu().tolist() for key, value in template.items()},
               "orientation_modes": {"front_landmarks": "existing verified frame fit",
                                     "texture_pose": "native texture with held-out tracks",
                                     "profile_silhouette_pose": "bounded yaw and image-plane translation, tested on reserved contour bands",
                                     "contour_translation": "world orientation propagated; camera-plane position refined",
                                     "world_prior": "world orientation and position propagated"},
               "frames": records}
    (args.output / "side-poses.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", nargs="*", default=[])
    parser.add_argument("--dense", action="store_true", help="Track additional decoded source frames between registered views.")
    parser.add_argument("--profile-refinement", type=Path, help="Apply a validated shared shape to the profile fallback.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    with torch.no_grad():
        run(args)
