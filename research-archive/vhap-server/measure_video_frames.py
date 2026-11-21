"""Measure candidate head frames in the undistorted scene camera and select a split."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import pickle
import struct

import cv2
import numpy as np
from vhap.util.landmark_detector_star import LandmarkDetectorSTAR

from select_head_frames import select_head_frames

POSE_POINTS = np.arange(17, 48)


def read_pinhole(scene):
    path = scene / "orig_undist/sparse/cameras.bin"
    with path.open("rb") as file:
        count, _, model, width, height = struct.unpack("<QiiQQ", file.read(32))
        if count != 1 or model != 1:
            raise ValueError("The first video pipeline expects one COLMAP PINHOLE camera.")
        fx, fy, cx, cy = struct.unpack("<dddd", file.read(32))
    return np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.]]), (width, height)


def face_template(flame):
    with (flame / "generic_model.pkl").open("rb") as stream:
        model = pickle.load(stream, encoding="latin1")
    embedding = np.load(flame / "landmark_embedding_with_eyes.npy", allow_pickle=True).item()
    faces = np.asarray(embedding["full_lmk_faces_idx"]).reshape(-1)[:68]
    weights = np.asarray(embedding["full_lmk_bary_coords"]).reshape(-1, 3)[:68]
    triangles = model["v_template"][model["f"][faces]]
    return (triangles * weights[..., None]).sum(axis=1)


def score_face(xy, image, mask, template, K):
    height, width = image.shape[:2]
    pixels = np.rint(xy).astype(np.int32)
    in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    visible = np.zeros(len(xy), dtype=bool)
    visible[in_bounds] = mask[pixels[in_bounds, 1], pixels[in_bounds, 0]] >= 128
    per_eye = visible[36:48].reshape(2, 6).sum(axis=1)
    if per_eye.min() < 5:
        return {"status": "eye_landmarks_outside_person_mask", "eye_points_in_matte": per_eye.tolist(),
                "visibility": float(visible[17:].mean())}
    left = xy[36:42].mean(axis=0)
    right = xy[42:48].mean(axis=0)
    eye_span = np.linalg.norm(left - right)
    if visible[17:].mean() < 0.75 or eye_span < 12:
        return {"status": "face_too_small_or_outside_person_mask", "visibility": float(visible[17:].mean())}
    ids = POSE_POINTS[visible[POSE_POINTS]]
    if len(ids) < 16:
        return {"status": "insufficient_inner_landmarks"}
    success, rv, t = cv2.solvePnP(template[ids], xy[ids], K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not success:
        return {"status": "pnp_failed"}
    rv, t = cv2.solvePnPRefineLM(template[ids], xy[ids], K, None, rv, t)
    rotation = cv2.Rodrigues(rv)[0]
    camera_in_head = -(rotation.T @ t[:, 0])
    if camera_in_head[2] <= 0 or (template @ rotation.T + t[:, 0])[:, 2].min() <= 0:
        return {"status": "pnp_nonfrontal_solution"}
    predicted = cv2.projectPoints(template[ids], rv, t, K, None)[0][:, 0]
    error = np.linalg.norm(predicted - xy[ids], axis=1)
    yaw = float(np.degrees(np.arctan2(camera_in_head[0], camera_in_head[2])))
    lo = np.maximum(0, np.floor(xy[17:].min(axis=0))).astype(int)
    hi = np.minimum([width, height], np.ceil(xy[17:].max(axis=0))).astype(int)
    gray = cv2.cvtColor(image[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(cv2.resize(gray, (72, 72), interpolation=cv2.INTER_AREA), cv2.CV_32F).var()
    return {
        "status": "measured", "yaw_deg": yaw, "face_score": float(min(1, eye_span / 36)),
        "reprojection_px": float(np.median(error)), "reprojection_p90_px": float(np.percentile(error, 90)),
        "blur_score": float(sharpness), "visibility": float(visible[17:].mean()),
        "eye_distance_px": float(eye_span), "eye_points_in_matte": per_eye.tolist(),
    }


def save_contact_sheet(path, names, landmarks, paths, result):
    held = set(result["holdout"])
    panels = []
    for name in names:
        xy = landmarks[name]
        image = cv2.imread(str(paths[name]))
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        radius = int(0.85 * max(hi - lo))
        center = ((lo + hi) * 0.5 - np.array([0, 0.16 * radius])).astype(int)
        left, top = np.maximum(center - radius, 0)
        right, bottom = center + radius
        crop = cv2.resize(image[top:bottom, left:right], (192, 176), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(crop, 0, 26, 0, 0, cv2.BORDER_CONSTANT, value=(23, 27, 32))
        yaw = result["decisions"][name]["yaw_deg"]
        tag = "HOLD" if name in held else "TRAIN"
        color = (86, 201, 255) if name in held else (244, 244, 244)
        cv2.putText(panel, f"{tag} {Path(name).stem}  {yaw:+.1f} deg", (7, 194),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        panels.append(panel)
    columns = 5
    panels.extend(np.zeros_like(panels[0]) for _ in range((-len(panels)) % columns))
    sheet = np.vstack([np.hstack(panels[i:i + columns]) for i in range(0, len(panels), columns)])
    if not cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise OSError(f"Could not write {path}")


def run(args):
    K, size = read_pinhole(args.scene)
    manifest = json.loads((args.scene / "source-frames.json").read_text())
    timestamp = {row["name"]: row["timestamp_s"] for row in manifest["frames"]}
    directory = args.scene / "orig_undist/images"
    template = face_template(args.flame)
    detector = LandmarkDetectorSTAR()
    records, landmarks, paths = [], {}, {}
    for path in sorted(directory.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None or image.shape[1::-1] != size:
            raise ValueError(f"Unexpected native raster: {path}")
        mask = cv2.imread(str(args.scene / "mask_undist/images" / path.name), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f"Missing matching alpha for {path.name}")
        bbox, points = detector.detect_single_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        row = {"name": path.name, "timestamp_s": timestamp[path.name],
               "yaw_deg": None, "face_score": 0., "reprojection_px": None, "status": "no_face"}
        if bbox[-1] > 0 and (points[:, 2] > 0).all():
            xy = points[:, :2].astype(np.float64) * size
            row.update(score_face(xy, image, mask, template, K))
            landmarks[path.name] = xy
            paths[path.name] = path
        records.append(row)
        print(f"{path.name} {row['status']} yaw={row['yaw_deg']}", flush=True)
    result = select_head_frames(records, train_count=args.train_count, holdout_count=args.holdout_count,
                                holdout_block_seconds=1.2)
    for row in records:
        result["decisions"][row["name"]].update({
            key: row[key] for key in ("eye_points_in_matte", "eye_distance_px", "reprojection_p90_px")
            if key in row
        })
    result["measurement"] = "STAR + calibrated PINHOLE PnP with frontal FLAME template"
    result["native_size"] = list(size)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    names = sorted(result["train"] + result["holdout"], key=timestamp.__getitem__)
    if len(result["train"]) < 6 or len(result["holdout"]) < 2:
        raise RuntimeError("Insufficient qualified frontal frames. See selection.json for the amount and reasons.")
    save_contact_sheet(args.output / "selection-preview.jpg", names, landmarks, paths, result)
    np.savez(args.output / "selected-landmarks.npz", names=names, K=K,
             xy=np.array([landmarks[name] for name in names]),
             native_sha256=[sha256(paths[name].read_bytes()).hexdigest() for name in names])
    print(f"Selected {len(result['train'])} fitting images and {len(result['holdout'])} held out", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for option in ("scene", "output", "flame"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=4)
    cv2.setNumThreads(1)
    run(parser.parse_args())
