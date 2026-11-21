"""Prepare matched VHAP inputs and geometric or 6DRepNet360 pose seeds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch

from fit_side_frames import alpha_maps, world_priors
from fuse_scene import SCALE_LANDMARKS, align_rigid, estimate_scale
from image_profile_pose import calibration
from learned_profile_pose import load_profile_pose, predict_profile_rotations
from prepare_video_avatar import virtual_crop, write_jpeg
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def _native_cameras(scene):
    sparse = scene / "orig_undist/sparse"
    cameras = read_intrinsics_binary(sparse / "cameras.bin")
    if len(cameras) != 1 or next(iter(cameras.values())).model != "PINHOLE":
        raise ValueError("Guided VHAP requires one native COLMAP PINHOLE camera")
    camera = next(iter(cameras.values()))
    fx, fy, cx, cy = camera.params
    K = np.array([[fx, 0., cx - .5], [0., fy, cy - .5], [0., 0., 1.]])
    views = {}
    for image in read_extrinsics_binary(sparse / "images.bin").values():
        if image.camera_id != camera.id or image.name in views:
            raise ValueError(f"Ambiguous native camera for {image.name}")
        R = image.qvec2rotmat().T
        views[image.name] = (R, -R @ image.tvec)
    return K, (int(camera.width), int(camera.height)), views


def _template_and_root():
    # Archive-only equivalent to measure_video_frames.face_template; avoids
    # importing STAR or VHAP into the Gaussian runtime.
    assets = Path.home() / "GaussianAvatars/flame_model/assets/flame"
    with (assets / "generic_model.pkl").open("rb") as file:
        model = pickle.load(file, encoding="latin1")
    embedding = np.load(assets / "landmark_embedding_with_eyes.npy", allow_pickle=True).item()
    faces = np.asarray(embedding["full_lmk_faces_idx"]).reshape(-1)[:68]
    weights = np.asarray(embedding["full_lmk_bary_coords"]).reshape(-1, 3)[:68]
    vertices = np.asarray(model["v_template"])
    roi = vertices[np.asarray(model["f"])[faces]]
    root = np.asarray(model["J_regressor"] @ vertices)[0]
    return np.sum(roi * weights[..., None], axis=1), root


def _head_support(scene, name, size):
    matte = cv2.imread(str(scene / "mask_undist/images" / name), cv2.IMREAD_GRAYSCALE)
    head = cv2.imread(str(scene / "headmask_undist/images" / Path(name).with_suffix(".png")),
                      cv2.IMREAD_GRAYSCALE)
    if matte is None or head is None or matte.shape != size[::-1] or head.shape != matte.shape:
        raise ValueError(f"Missing aligned head support for {name}")
    support = np.minimum(matte, head)
    x, y, w, h = cv2.boundingRect((support >= 128).astype(np.uint8))
    if min(w, h) < 12:
        raise ValueError(f"Head mask is too small in {name}")
    corners = np.array([[x, y], [x + w - 1, y], [x, y + h - 1], [x + w - 1, y + h - 1]], dtype=np.float64)
    return support, corners


def _anchors(template, xy, train, holdout, K, cameras):
    fitted, centers, offsets = {}, [], []
    for name in train + holdout:
        A, t, points, _ = align_rigid(template, xy[name], K)
        R, C = cameras[name]
        fitted[name] = (R @ A, C, R @ t)
        if name in train:
            centers.append(C)
            offsets.append(points[SCALE_LANDMARKS] @ R.T)
    scale, _ = estimate_scale(np.asarray(centers)[:, None], np.asarray(offsets))
    poses = {name: (A, C + scale * offset) for name, (A, C, offset) in fitted.items()}
    return scale, poses


def _pose_cv(world_to_virtual, camera_center, scale, pose):
    rotation, translation = pose
    return {"rotation": (world_to_virtual @ rotation).tolist(),
            "translation": (world_to_virtual @ (translation - camera_center) / scale).tolist()}


def prepare(scene: Path, selection: Path, landmarks: Path, weights: Path, output: Path, size: int):
    """Create vhap-input and return its directory.

    initialization.json stores model-origin rotation/translation in each virtual
    OpenCV camera. A downstream VHAP seed adds its FLAME root-pivot correction.
    """
    selection = json.loads(selection.read_text(encoding="utf-8"))
    train, holdout = list(selection["train"]), list(selection["holdout"])
    manifest = json.loads((scene / "source-frames.json").read_text(encoding="utf-8"))
    source = sorted(manifest["frames"], key=lambda record: record["timestamp_s"])
    names = [record["name"] for record in source]
    time = {record["name"]: float(record["timestamp_s"]) for record in source}
    K, native_size, cameras = _native_cameras(scene)
    if (size < 128 or len(set(train)) != len(train) or len(set(holdout)) != len(holdout)
            or len(train) < 3 or not holdout or len(names) != len(time)
            or set(train) & set(holdout) or not set(train + holdout) <= set(names)
            or not set(names) <= set(cameras)
            or not np.isfinite(list(time.values())).all() or not (np.diff(list(time.values())) > 0).all()):
        raise ValueError("Invalid or inconsistent registered-video and frontal split records")
    with np.load(landmarks) as archive:
        observed = archive["names"].astype(str).tolist()
        if (set(observed) != set(train + holdout) or len(set(observed)) != len(observed)
                or not np.allclose(archive["K"], K, atol=1e-9, rtol=0)
                or archive["xy"].shape != (len(observed), 68, 2) or not np.isfinite(archive["xy"]).all()):
            raise ValueError("Only the originally selected real frontal STAR landmarks can be used")
        xy = dict(zip(observed, archive["xy"], strict=True))
        hashes = dict(zip(observed, archive["native_sha256"].astype(str), strict=True))

    template, root = _template_and_root()
    scale, fitted = _anchors(template, xy, train, holdout, K, cameras)
    priors = world_priors(names, time, {name: fitted[name] for name in train})
    observed_mattes, _, matte_parity, _, _ = alpha_maps(scene, native_size, K, names)
    ratios = []
    for name in train:
        _, corners = _head_support(scene, name, native_size)
        _, _, K_head, _ = virtual_crop(corners, K, size)
        _, _, K_face, _ = virtual_crop(xy[name], K, size)
        ratios.append(K_face[0, 0] / K_head[0, 0])
    # Equivalent occupancy to the real frontal face crop, capped to keep the head in view.
    zoom = float(np.clip(np.median(ratios), 1.0, 2.0))
    mid = (size - 1) / 2
    tighter = np.array([[zoom, 0., (1 - zoom) * mid], [0., zoom, (1 - zoom) * mid], [0., 0., 1.]])

    seq = output / "vhap-input"
    (seq / "images").mkdir(parents=True)
    (seq / "alpha_maps").mkdir()
    (seq / "landmark2d").mkdir()
    images, masks, camera_records, star, bbox = [], [], [], [], []
    for index, row in enumerate(source):
        name = row["name"]
        if Path(name).name != name:
            raise ValueError(f"Expected a registered image basename: {name}")
        path = scene / "orig_undist/images" / name
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != row["native_sha256"] or (name in hashes and digest != hashes[name]):
            raise ValueError(f"Native photometric observation changed: {name}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != native_size[::-1]:
            raise ValueError(f"Unexpected native color raster: {name}")
        _, corners = _head_support(scene, name, native_size)
        H, V, virtual_K, corners_2d = virtual_crop(corners, K, size)
        H = tighter @ H
        head_pixels = mid + zoom * (corners_2d - mid)
        bbox.append([*head_pixels.min(axis=0), *head_pixels.max(axis=0)])
        crop = cv2.warpPerspective(image, H, (size, size), flags=cv2.INTER_LINEAR)
        alpha = observed_mattes[name]["target"].astype(np.uint8) * 255
        matte = cv2.warpPerspective(alpha, H, (size, size), flags=cv2.INTER_LINEAR)
        if int((matte >= 128).sum()) < 500:
            raise ValueError(f"Missing foreground in virtual head crop: {name}")
        landmarks_uv = np.full((68, 3), -1., dtype=np.float32)
        landmarks_uv[:, 2] = 0.
        if name in xy:
            points = cv2.perspectiveTransform(xy[name][None].astype(np.float64), H)[0]
            if (points < -.5).any() or (points > size - .5).any():
                raise ValueError(f"Real STAR observations leave the virtual crop: {name}")
            landmarks_uv[:, :2] = (points + .5) / size
            landmarks_uv[:, 2] = 1.
        R, C = cameras[name]
        record = {"name": name, "timestep": index, "timestamp_s": time[name], "train": name in train,
                  "holdout": name in holdout, "has_landmarks": name in xy,
                  "warp": H.tolist(), "virtual_from_native_cv": V.tolist(),
                  "world_from_virtual": (R @ V.T).tolist(), "camera_center_world": C.tolist(),
                  "K_integer_pixel_centers": (tighter @ virtual_K).tolist()}
        write_jpeg(seq / "images" / f"{index:08d}.jpg", crop)
        write_jpeg(seq / "alpha_maps" / f"{index:08d}.jpg", matte)
        images.append(crop)
        masks.append(matte >= 128)
        camera_records.append(record)
        star.append(landmarks_uv)

    network = load_profile_pose(weights, device="cuda" if torch.cuda.is_available() else "cpu")
    matrices = predict_profile_rotations(network, images, masks)
    estimated = {record["name"]: np.asarray(record["world_from_virtual"]) @ matrix
                 for record, matrix in zip(camera_records, matrices, strict=True)}
    known = {name: {"rotation": A.tolist()} for name, (A, _) in fitted.items()}
    basis, errors, trusted = calibration(estimated, known, train, holdout)

    poses = []
    for record in camera_records:
        name = record["name"]
        R_virtual = np.asarray(record["world_from_virtual"]).T
        center = np.asarray(record["camera_center_world"])
        initial = priors[name]
        A, t = initial
        candidate = estimated[name] @ basis
        use_network = trusted and name not in train
        guided = (candidate, t + scale * (A - candidate) @ root) if use_network else initial
        view = (center - t) @ A
        yaw = float(np.degrees(np.arctan2(view[0], view[2])))
        poses.append({**record, "baseline_yaw_deg": yaw,
                      "baseline": _pose_cv(R_virtual, center, scale, initial),
                      "guided": _pose_cv(R_virtual, center, scale, guided), "guided_from_network": use_network})

    conditioning = {"network": "6DRepNet360 Full Rotation", "weights_sha256": sha256(weights.read_bytes()).hexdigest(),
                    "frame_count": len(names), "frontal_anchors": train, "calibration_holdout": holdout,
                    "calibration_passed": trusted, "heldout_rotation_errors_deg": errors,
                    "rotation_basis": basis.tolist(), "crop_zoom": zoom, "raw_matte_min_remap_iou": matte_parity,
                    "guide_frame_count": sum(record["guided_from_network"] for record in poses)}
    (seq / "cameras.json").write_text(json.dumps(camera_records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (seq / "initialization.json").write_text(json.dumps({
        "scene_scale": scale, "root_flame_units": root.tolist(), "conditioning": conditioning,
        "frames": poses,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez(seq / "landmark2d/STAR.npz", face_landmark_2d=np.asarray(star, dtype=np.float32),
             bounding_box=np.asarray(bbox, dtype=np.float32))
    return seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("scene", "selection", "landmarks", "weights", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    print(prepare(**vars(parser.parse_args())))
