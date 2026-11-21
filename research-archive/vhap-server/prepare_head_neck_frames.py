"""Reframe a registered VHAP sequence to include the full neck and upper shoulders."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from fit_side_frames import alpha_maps
from guided_vhap_input import _native_cameras
from prepare_video_avatar import virtual_crop, write_jpeg


def run(args):
    initialization = json.loads((args.sequence / "initialization.json").read_text())
    frames = initialization["frames"]
    manifest = json.loads((args.scene / "source-frames.json").read_text())
    names = [r["name"] for r in frames]
    if names != [r["name"] for r in manifest["frames"]]:
        raise ValueError("Guided frames must match the registered source timeline")
    if args.body_observations:
        with np.load(args.body_observations) as archive:
            if archive["names"].tolist() != names:
                raise ValueError("Body observations and VHAP frames disagree")
            shoulder_ids = [archive["joint_names"].tolist().index(n) for n in ("left_shoulder", "right_shoulder")]
            shoulders = archive["keypoints"][:, shoulder_ids]
            quality = archive["scores"][:, shoulder_ids]
            consistency = archive["flip_disagreement"][:, shoulder_ids]
            boxes = archive["boxes"]
    K, native_size, _ = _native_cameras(args.scene)
    _, _, parity, remap, _ = alpha_maps(args.scene, native_size, K, names[:1])
    with np.load(args.sequence / "landmark2d/STAR.npz") as archive:
        landmarks = archive["face_landmark_2d"].copy()
    if landmarks.shape != (len(names), 68, 3):
        raise ValueError("The source must provide the original 68-point VHAP observations")
    for folder in ("images", "alpha_maps", "landmark2d"):
        (args.output / folder).mkdir(parents=True, exist_ok=False)

    records, boxes_out = [], []
    size = args.size
    center = (size - 1) / 2
    zoom = 1.92
    zoom_matrix = np.array([[zoom, 0., (1. - zoom) * center],
                            [0., zoom, (1. - zoom) * center], [0., 0., 1.]])
    for index, (old, src) in enumerate(zip(frames, manifest["frames"], strict=True)):
        name = old["name"]
        native_path = args.scene / "orig_undist/images" / name
        if sha256(native_path.read_bytes()).hexdigest() != src["native_sha256"]:
            raise ValueError(f"Registered input was modified: {name}")
        image = cv2.imread(str(native_path))
        coarse = cv2.imread(str(args.scene / "headmask_undist/images" / Path(name).with_suffix(".png")), 0)
        matte = cv2.imread(str(args.scene / "alpha_maps" / name), 0)
        alpha = cv2.remap(matte, *remap, interpolation=cv2.INTER_LINEAR)
        x, top, width, height = cv2.boundingRect((coarse >= 128).astype(np.uint8))
        if args.body_observations:
            person_width, person_height = boxes[index, 2:] - boxes[index, :2]
            reliable = (quality[index] > .5) & (consistency[index] < max(7., .06 * person_width))
        else:
            _, _, _, person_height = cv2.boundingRect((alpha >= 128).astype(np.uint8))
            reliable = np.zeros(2, dtype=bool)
            if person_height <= height:
                raise ValueError(f"Person matte does not cover the neck crop at {name}")
        shoulder_y = (float(shoulders[index, reliable, 1].max()) if reliable.any()
                      else top + height + .055 * person_height)
        bottom = int(np.clip(max(top + height, shoulder_y + .025 * person_height),
                             top + height, min(native_size[1] - 1, top + .37 * person_height)))
        pad = max(4, round(.11 * width))
        rectangle = np.array([[x - pad, top - 3], [x + width + pad, top - 3],
                              [x - pad, bottom], [x + width + pad, bottom]], dtype=np.float64)
        warp, camera_rotation, crop_K, crop_corners = virtual_crop(rectangle, K, size)
        warp = zoom_matrix @ warp
        new_K = zoom_matrix @ crop_K
        crop_corners = (crop_corners - center) * zoom + center
        crop = cv2.warpPerspective(image, warp, (size, size), flags=cv2.INTER_LINEAR)
        alpha[bottom + 1:] = 0
        crop_alpha = cv2.warpPerspective(alpha, warp, (size, size), flags=cv2.INTER_LINEAR)
        if not np.all((crop_corners >= 0) & (crop_corners < size)) or (crop_alpha >= 128).sum() < 500:
            raise ValueError(f"Incomplete head and neck coverage at {name}")
        old_to_new = camera_rotation @ np.asarray(old["virtual_from_native_cv"]).T
        record = {**old, "warp": warp.tolist(), "virtual_from_native_cv": camera_rotation.tolist(),
                  "K_integer_pixel_centers": new_K.tolist(),
                  "world_from_virtual": (np.asarray(old["world_from_virtual"]) @ old_to_new.T).tolist(),
                  "native_head_neck_bottom_px": bottom, "reliable_shoulder_count": int(reliable.sum())}
        for mode in ("baseline", "guided"):
            record[mode] = {key: (old_to_new @ np.asarray(old[mode][key])).tolist()
                            for key in ("rotation", "translation")}
        if old["has_landmarks"]:
            same_rays = warp @ np.linalg.inv(np.asarray(old["warp"]))
            pixels = landmarks[index, :, :2].astype(np.float64) * args.original_size - .5
            projection = cv2.perspectiveTransform(pixels[None], same_rays)[0]
            if not np.all((projection >= -.5) & (projection < size - .5)):
                raise ValueError(f"A trusted facial landmark left the reframed image at {name}")
            landmarks[index, :, :2] = (projection + .5) / size
        write_jpeg(args.output / "images" / f"{index:08d}.jpg", crop)
        write_jpeg(args.output / "alpha_maps" / f"{index:08d}.jpg", crop_alpha)
        boxes_out.append([*crop_corners.min(axis=0), *crop_corners.max(axis=0)])
        records.append(record)
    np.savez(args.output / "landmark2d/STAR.npz", face_landmark_2d=landmarks,
             bounding_box=np.asarray(boxes_out, dtype=np.float32))
    (args.output / "cameras.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    initialization.update(frames=records, framing={"kind": "head-neck-upper-shoulders",
                                                  "native_matte_remap_iou": parity,
                                                  "body_landmarks_use": "crop bounds only" if args.body_observations else None})
    (args.output / "initialization.json").write_text(json.dumps(initialization, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "sequence", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--body-observations", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--original-size", type=int, default=512)
    cv2.setNumThreads(1)
    run(parser.parse_args())
