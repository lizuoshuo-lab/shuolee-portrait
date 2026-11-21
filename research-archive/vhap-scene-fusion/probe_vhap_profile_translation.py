"""Test fixed-camera head translation against existing RTMW side observations."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from fit_native_skin_profile import camera, file_hash
from review_smplx_head_neck import mesh_overlay, project


def describe(errors):
    error = np.linalg.norm(errors, axis=-1)
    return {key: float(value) for key, value in zip(
        ("median", "p90", "max"), np.quantile(error, (0.5, 0.9, 1.0)), strict=True)}


def visible_points(camera_space, points, faces, K):
    xy, z = project(camera_space, K)
    projected, depths = project(points, K)
    start, middle, stop = xy[faces].transpose(1, 0, 2)
    u, v = middle - start, stop - start
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    live = np.abs(det) > 1e-8
    keep = []
    for pos, depth in zip(projected, depths, strict=True):
        delta = pos - start[live]
        b = (delta[:, 0] * v[live, 1] - delta[:, 1] * v[live, 0]) / det[live]
        c = (u[live, 0] * delta[:, 1] - u[live, 1] * delta[:, 0]) / det[live]
        a = 1 - b - c
        inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        t = np.column_stack((a, b, c))[inside]
        nearest = np.sum(t / z[faces[live][inside]], axis=1).max(initial=0)
        keep.append(not nearest or depth - 1 / nearest <= .001)
    return np.asarray(keep)


def run(args):
    initialization = args.sequence / "initialization.json"
    rows = json.loads(initialization.read_text())["frames"]
    names = [row["name"] for row in rows]
    state = np.load(args.registered, allow_pickle=False)
    rtmw = np.load(args.rtmw, allow_pickle=False)
    star = np.load(args.sequence / "landmark2d/STAR.npz", allow_pickle=False)["face_landmark_2d"]
    if (names != rtmw["names"].tolist() or names != state["filenames"].tolist()
            or state["initialization_sha256"].item() != file_hash(initialization)
            or rtmw["joint_names"][3:5].tolist() != ["left_ear", "right_ear"]
            or rtmw["joint_names"][23:91].tolist() != [f"face-{i}" for i in range(68)]
            or star.shape != (len(names), 68, 3)):
        raise ValueError("The observations and registered FLAME frames do not share one frame order")
    # Read the full FLAME barycentric landmark table from the pinned VHAP asset.
    embedding = np.load(args.embedding, allow_pickle=True).item()
    asset = args.embedding.with_name("head_template_mesh.obj")
    asset_faces = np.array([[int(value.split("/")[0]) - 1 for value in line.split()[1:]]
                            for line in asset.read_text().splitlines() if line.startswith("f ")])
    faces = state["faces"]
    if not np.array_equal(faces[:len(asset_faces)], asset_faces):
        raise ValueError("The landmark table and registered mesh have different face ordering")
    indices = embedding["full_lmk_faces_idx"].reshape(-1)[17:68]
    if indices.min() < 0 or indices.max() >= len(asset_faces):
        raise ValueError("Landmark embeddings reference triangles outside the verified head template")
    weights = embedding["full_lmk_bary_coords"].reshape(-1, 3)[17:68]
    transform = state["tracker_to_world_rotation"]
    scale = float(state["scene_scale"])
    surface = state["vertices_tracker"] @ transform.T * scale + state["scene_origin"]
    face_points = np.einsum("fk,nfkj->nfj", weights, surface[:, faces[indices]])
    # Official SMPL-X/FLAME shared vertices for the COCO left/right ear proxies.
    ear_points = surface[:, [6, 616]]
    landmarks = np.concatenate((face_points, ear_points), axis=1)
    matrices = [camera(row) for row in rows]
    columns = np.r_[40:91, 3:5]
    raw = rtmw["keypoints"][:, columns]
    quality = (rtmw["scores"][:, columns] > 0) & (rtmw["flip_disagreement"][:, columns] <= 3.)
    focus = np.r_[np.array([30, 33, 51, 57]) - 17, 51:53]
    for i, ((k, r, c), row) in enumerate(zip(matrices, rows, strict=True)):
        if abs(row["baseline_yaw_deg"]) < 55.:
            continue
        cam = (surface[i] - c) @ r
        xyz = (landmarks[i, focus] - c) @ r
        triangles = cam[faces[indices[focus[:4]]]]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        quality[i, focus[:4]] &= np.einsum("ij,ij->i", normal, xyz[:4]) < 0.
        quality[i, focus] &= visible_points(cam, xyz, faces, k)
        quality[i, 51 + int(np.argmax(xyz[-2:, 2]))] = False
    before = np.stack([project((points - c) @ r, k)[0] for points, (k, r, c)
                       in zip(landmarks, matrices, strict=True)])
    supports = {}
    for sign in (-1, 1):
        side = [i for i, row in enumerate(rows) if sign * row["baseline_yaw_deg"] >= 55.]
        fit = [i for j, i in enumerate(side) if j % 3 != 1 and Path(names[i]).stem != "00850"]
        supports[sign] = (fit, [i for i in side if i not in fit])
    # Fix depth and shape. Fit just nose tip and upper lip; keep subnasale and
    # lower lip apart to see whether a simple translation moves them together.
    fit_mask = np.zeros(quality.shape, dtype=bool)
    for fit, _ in supports.values():
        fit_mask[np.ix_(fit, np.array([30, 51]) - 17)] = True
    ear_mask = np.zeros_like(fit_mask)
    for fit, _ in supports.values():
        ear_mask[np.ix_(fit, [51, 52])] = True
    fit_mask &= quality
    ear_mask &= quality
    if any(fit_mask[ids].sum() < 8 for ids, _ in supports.values()):
        raise ValueError("Not enough repeatable side landmark detections")
    if any(ear_mask[ids].sum() < 3 for ids, _ in supports.values()):
        raise ValueError("Not enough visible ear observations on both sides")

    def projected(xy):
        shift = transform[:, :2] @ xy * scale
        return np.stack([project((points + shift - c) @ r, k)[0]
                         for points, (k, r, c) in zip(landmarks, matrices, strict=True)])

    def fit_translation(mask):
        return least_squares(lambda value: (projected(value) - raw)[mask].ravel(), np.zeros(2),
                             bounds=(-.008, .008), loss="soft_l1", f_scale=2., max_nfev=24)

    without_ears = fit_translation(fit_mask)
    with_ears = fit_translation(fit_mask | ear_mask)
    after_without_ears = projected(without_ears.x)
    after = projected(with_ears.x)
    outcomes = (before, after_without_ears, after)
    provenance = {key: file_hash(path) for key, path in {
        "checkpoint_registration": args.registered, "track_initialization": initialization,
        "raw_body_landmarks": args.rtmw, "flame_embedding": args.embedding}.items()}
    report = {"observation": "RTMW repeatability is a detector control, with no subpixel truth claim",
              "comparison_order": ["original", "nose_lips", "nose_lips_visible_ears"],
              "candidate_translation_tracker_xy": [without_ears.x.tolist(), with_ears.x.tolist()],
              "source_sha256": provenance, "fit_ear_observations": int(ear_mask.sum()),
              "fit_face_observations": int(fit_mask.sum()),
              "fit": [describe((xy - raw)[fit_mask | ear_mask]) for xy in outcomes],
              "holdout_side": {}, "sequence_frames": len(names)}
    for sign, (fit, hold) in supports.items():
        groups = {}
        for key, ids, columns in (("held_landmark_families", fit, np.array([33, 57]) - 17),
                                  ("held_frames_nose", hold, np.array([30, 33]) - 17),
                                  ("held_frames_lips", hold, np.array([51, 57]) - 17),
                                  ("held_frames_ears", hold, [51, 52])):
            selection = np.ix_(ids, columns)
            valid = quality[selection]
            if not valid.any():
                raise ValueError(f"No repeatable {key} observations for side {sign}")
            groups[key] = {"count": int(valid.sum()),
                           "comparison": [describe((xy - raw)[selection][valid]) for xy in outcomes]}
        report["holdout_side"][str(sign)] = {"fit": [names[i] for i in fit], "hold": [names[i] for i in hold], **groups}
    front = np.array([i for i, row in enumerate(rows)
                      if abs(row["baseline_yaw_deg"]) <= 30. and row["has_landmarks"]])
    front_parts = []
    for i in front:
        crop = cv2.imread(str(args.sequence / "images" / f"{i:08d}.jpg"))
        if crop is None:
            raise OSError(f"Missing STAR source crop for {names[i]}")
        observed = star[i, 17:68, :2].astype(float) * crop.shape[1::-1] - .5
        native_target = cv2.perspectiveTransform(observed[None], np.linalg.inv(rows[i]["warp"]))[0]
        native_error = np.stack([xy[i, :51] - native_target for xy in outcomes])
        front_parts.append(native_error[:, star[i, 17:68, 2] > 0])
    report["front_star"] = [describe(x) for x in np.concatenate(front_parts, axis=1)]
    report["acceptance"] = "Diagnostic candidate only. The RTMW ear is an anatomical proxy."
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("00290.jpg", "00560.jpg", "00850.jpg"):
        i = names.index(name)
        report.setdefault("sample", {})[name] = {
            "landmark_ids": [30, 33, 51, 57],
            "ear_valid": quality[i, 51:].tolist() if abs(rows[i]["baseline_yaw_deg"]) >= 55. else None,
            "ear_signed_xy": [(xy[i, 51:] - raw[i, 51:]).tolist() for xy in outcomes],
            "nose_lips_valid": quality[i, np.array([30, 33, 51, 57]) - 17].tolist(),
            "before_signed_xy": (before[i, np.array([30, 33, 51, 57]) - 17]
                                 - raw[i, np.array([30, 33, 51, 57]) - 17]).tolist(),
            "after_signed_xy": (after[i, np.array([30, 33, 51, 57]) - 17]
                                - raw[i, np.array([30, 33, 51, 57]) - 17]).tolist()}
        img = cv2.imread(str(args.sequence / "images" / f"{i:08d}.jpg"))
        k, r, c = matrices[i]
        crop_k = np.asarray(rows[i]["warp"]) @ k
        shifts = (np.zeros(3), transform[:, :2] @ without_ears.x * scale,
                  transform[:, :2] @ with_ears.x * scale)
        strips = [img]
        for shift in shifts:
            strips.append(mesh_overlay(img, (surface[i] + shift - c) @ r, crop_k, faces))
        labeled = []
        for image, label in zip(strips, ("Source", "Original FLAME", "Nose / lips", "Nose / lips / visible ears"), strict=True):
            image = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(image, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .56, (245, 245, 245), 1, cv2.LINE_AA)
            labeled.append(image)
        if not cv2.imwrite(str(args.output / f"{Path(name).stem}-translation-control.jpg"), np.hstack(labeled)):
            raise OSError(f"Could not write translation review for {name}")
    (args.output / "translation-control.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "registered", "rtmw", "embedding", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
