"""Test image-based head orientation in frames without a reliable texture pose."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from fit_side_frames import alpha_maps, contour_shift, mask_scores, pose_json, review_row
from fuse_scene import local_head, make_rasterizer, prepare_scene, render, to_bgr, world_gaussians
from learned_profile_pose import load_profile_pose, predict_profile_rotations
from prepare_video_avatar import virtual_crop
from profile_rgb_check import compare_profile_rgb
from profile_shape import profile_basis, profile_weights
from train_joint import JointParameters, combine


def image_rotations(weights, names, scene_root, masks, K, cameras):
    crops, supports, to_world = [], [], []
    for name in names:
        source = cv2.imread(str(scene_root / "orig_undist/images" / name))
        mask = masks[name]["target"].astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        warp, virtual_rotation, _, _ = virtual_crop(corners, K, 384)
        crops.append(cv2.warpPerspective(source, warp, (384, 384)))
        supports.append(cv2.warpPerspective(mask, warp, (384, 384), flags=cv2.INTER_NEAREST) != 0)
        to_world.append(cameras[Path(name).stem][0] @ virtual_rotation.T)
    model = load_profile_pose(weights, device="cuda")
    matrices = predict_profile_rotations(model, crops, supports)
    return dict(zip(names, to_world @ np.asarray(matrices), strict=True))


def calibration(estimated, placements, train, hold):
    known = {name: np.array(placements[name]["rotation"]) for name in train + hold}
    # Upstream axes are right/down/in. The fixed right factor aligns those with FLAME.
    relative = np.stack([estimated[name].T @ known[name] for name in train])
    axis = Rotation.from_matrix(relative).mean().as_matrix()
    errors = {name: float(np.degrees(Rotation.from_matrix(known[name].T @ estimated[name] @ axis).magnitude()))
              for name in hold}
    trusted = len(errors) >= 3 and np.median(list(errors.values())) < 10. and max(errors.values()) < 17.
    return axis, errors, bool(trusted)


@torch.no_grad()
def run(args):
    metrics = json.loads((args.run / "model/metrics.json").read_text())
    prior = json.loads(args.baseline.read_text())
    settings = SimpleNamespace(export_frame=metrics["reference"],
                               **{key: Path(value) for key, value in metrics["input_paths"].items()})
    K, size, cameras, _, _, track, actor, keep, _, scene = prepare_scene(settings)
    train, hold = metrics["training_frames"], metrics["holdout_frames"]
    targets = [name for name, frame in sorted(prior["frames"].items())
               if frame["mode"] in ("contour_translation", "world_prior")
               and abs(frame["camera_relative_yaw_deg"]) >= 65.]
    review_names = args.review or sorted(targets, key=lambda n: -abs(prior["frames"][n]["camera_relative_yaw_deg"]))[:4]
    names = list(dict.fromkeys(train + hold + targets))
    masks, _, _, _, _ = alpha_maps(settings.scene, size, K, names)
    estimated = image_rotations(args.weights, names, settings.scene, masks, K, cameras)
    axis, errors, trusted = calibration(estimated, metrics["placement"], train, hold)
    print(f"Calibration held out (deg): {errors}; trusted: {trusted}", flush=True)

    joints = {key: torch.as_tensor([value], device="cuda", dtype=torch.float32)
              for key, value in prior["propagated_articulation"].items()}
    binding = actor.binding[keep].long()
    local_head(actor, track, 0)
    reference_size = actor.face_scaling[binding].reshape(-1, 1, 1).clone()
    local_head(actor, joints, 0)
    if prior.get("profile_refinement"):
        shape = json.loads(Path(prior["profile_refinement"]).read_text())
        weights, _ = profile_weights(actor, settings.flame_boundary_mask)
        basis = profile_basis(actor, weights)
        delta = (basis * basis.new_tensor(shape["scalp_and_posterior_neck_flame_units"])[None, :, None]).sum(dim=1)
        actor.update_mesh_properties(actor.verts + delta[None], actor.verts_cano.clone())
    local = world_gaussians(actor, np.eye(3), np.zeros(3), 1., keep)
    axes = actor.face_orien_mat[binding] * actor.face_scaling[binding].reshape(-1, 1, 1) / reference_size
    params = JointParameters(scene, local)
    state = torch.load(args.run / "model/training_state.pt", weights_only=True, map_location="cpu")
    params.load_state_dict(state["parameters"])
    xyz, features, opacity, sizes, quats = params.head(SimpleNamespace(head=local, axes=axes), geometry=True)
    local_rotation = Rotation.from_quat(quats.cpu().numpy()[:, [1, 2, 3, 0]])
    fitted_scene = params.scene(scene)
    pivot = xyz.mean(dim=0).cpu().numpy()
    scale = prior["scale"]

    def head_at(pose):
        A, t = pose
        q = (Rotation.from_matrix(A) * local_rotation).as_quat()[:, [3, 0, 1, 2]]
        return scale * xyz @ xyz.new_tensor(A.T) + xyz.new_tensor(t), features, opacity, sizes * scale, xyz.new_tensor(q)

    reviews, results = {}, {}
    for name in targets:
        original = cv2.imread(str(settings.scene / "orig_undist/images" / name))
        if original is None:
            raise OSError(f"Missing source frame {name}")
        row = prior["frames"][name]
        old = tuple(np.asarray(row[key]) for key in ("rotation", "translation"))
        R, C = cameras[Path(name).stem]
        raster = make_rasterizer(K, size, R, C)
        evidence = masks[name]
        crop = evidence["crop"]
        A, t = old
        relative = A.T @ estimated[name] @ axis
        yaw = float(np.degrees(np.arctan2(relative[0, 2] - relative[2, 0], relative[0, 0] + relative[2, 2])))
        applied = float(np.clip(yaw, -18., 18.))
        depth = (R.T @ (t + scale * A @ pivot - C))[2]
        before = to_bgr(render(raster, combine(fitted_scene, head_at(old))))
        s0 = mask_scores(render(raster, head_at(old), alpha_only=True).cpu().numpy(), evidence)
        support = (evidence["target"] & evidence["valid"])[crop]

        def sample(yaw_delta):
            B = A @ Rotation.from_rotvec([0, np.deg2rad(yaw_delta), 0]).as_matrix()
            shift = t + scale * (A - B) @ pivot
            pixel = contour_shift(render(raster, head_at((B, shift)), alpha_only=True).cpu().numpy(), evidence)
            shift += R[:, :2] @ (pixel * depth / [K[0, 0], K[1, 1]])
            pose = (B, shift)
            rgb = to_bgr(render(raster, combine(fitted_scene, head_at(pose))))
            edge = mask_scores(render(raster, head_at(pose), alpha_only=True).cpu().numpy(), evidence)
            quality = compare_profile_rgb(original[crop], before[crop], rgb[crop], support)
            info = {"yaw_deg": yaw_delta, "translation_px_at_pivot": pixel.tolist(), "quality": quality,
                    "contour": edge, "pose": pose_json(pose)}
            gain = quality.get("delta", {}).get("gradient_zncc", 0.)
            passes = (gain > .015 and quality["delta"]["local_luma_zncc"] >= -.01
                      and edge["iou"] >= s0["iou"] - .025 and s0["edge_p90_px"] is not None
                      and edge["edge_p90_px"] is not None and edge["edge_p90_px"] <= s0["edge_p90_px"] + 2.)
            return pose, rgb, info, gain, passes

        translated = sample(0.)
        rotated = sample(applied)
        rotation_passes = (trusted and 2. <= abs(yaw) <= 24. and rotated[4]
                           and rotated[3] > max(0., translated[3]) + .015)
        chosen = rotated if rotation_passes else translated if translated[4] else None
        mode = "image_profile_pose" if rotation_passes else "profile_translation" if chosen else row["mode"]
        results[name] = {"mode": mode, "tracked_points": row["texture"]["track_count"],
                         "image_pose_yaw_delta_deg": yaw, "contour_before": s0,
                         "translation_only": translated[2], "image_rotation": rotated[2]}
        print(f"{name} RGB yaw {yaw:+.1f}; {mode}; gradient translation {translated[3]:+.3f},"
              f" rotation {rotated[3]:+.3f}", flush=True)
        if chosen:
            (B, t_candidate), _, info, _, _ = chosen
            row.update(info["pose"], mode=mode, silhouette_after=info["contour"])
            view = (C - t_candidate) @ B
            row["camera_relative_yaw_deg"] = float(np.degrees(np.arctan2(view[0], view[2])))
        if name in review_names:
            after = chosen[1] if chosen else before
            label = (f"After: yaw {applied:+.1f} deg" if rotation_passes else "After: position refined" if chosen
                     else "After: pose retained")
            reviews[name] = review_row(original, before, after, evidence, name, label, compare=True)
    if (not set(review_names) <= set(reviews) or
            (review_names and not cv2.imwrite(str(args.output / "profile-before-after.jpg"),
                                              np.vstack([reviews[n] for n in review_names])))):
        raise RuntimeError("Missing requested profile comparison images.")
    prior["image_profile_pose"] = {"baseline": str(args.baseline), "weights_sha256": sha256(args.weights.read_bytes()).hexdigest(),
                                   "calibration_frames": train, "heldout_rotation_errors_deg": errors,
                                   "calibration_passed": trusted, "review_frames": review_names, "frames": results}
    prior["orientation_modes"].update(image_profile_pose="RGB-derived yaw validated against translation only",
                                      profile_translation="alignment improved at the previous yaw")
    (args.output / "side-poses.json").write_text(json.dumps(prior, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("run", "baseline", "weights", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--review", nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    run(args)
