"""Estimate actual NTU VIRAL stereo intrinsics/extrinsics from its public board ZIP.

No ROS, CUDA, package installation, flight bag, or FAST-LIVO2 replay is required.
Calibration-board geometry is 9x6 inner corners, 0.08 m squares per NTU docs.
Every fifth sorted shared image ID is held out BEFORE corner detection/optimization.
Held-out board pose is estimated from 27 alternating corners; reprojection is
evaluated on the other 27 corners. Left and right intrinsics use k1,k2,p1,p2;
k3 is fixed to zero to match published NTU radial-tangential model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.stdout.reconfigure(encoding="utf-8")
LAB = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = LAB / "outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001"
SOURCE_URL = "https://researchdata.ntu.edu.sg/api/access/datafile/58998"
BOARD = (9, 6)
SQUARE_M = 0.08


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_opencv_yaml(path):
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("tag:yaml.org,2002:opencv-matrix", lambda loader, node: loader.construct_mapping(node, deep=True))
    text = path.read_text(encoding="utf-8-sig")
    return yaml.load(text.replace("%YAML:1.0", "", 1), Loader=Loader)


def error_statistics(vector):
    a = np.asarray(vector, dtype=np.float64)
    return {"points": len(a), "euclidean_pixel_rmse": float(np.sqrt(np.mean(np.sum(a * a, axis=1)))), "mean_euclidean_pixel_error": float(np.mean(np.linalg.norm(a, axis=1))), "p95_euclidean_pixel_error": float(np.quantile(np.linalg.norm(a, axis=1), .95)), "max_euclidean_pixel_error": float(np.max(np.linalg.norm(a, axis=1)))}


def project(obj, rvec, tvec, k, d):
    return cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)


def intrinsic_from_yaml(cam):
    p, d = cam["projection_parameters"], cam["distortion_parameters"]
    return np.asarray([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float64), np.asarray([d["k1"], d["k2"], d["p1"], d["p2"], 0], dtype=np.float64)


def heldout_errors(obj, corners, k, d, pose_mask):
    ok, rv, tv = cv2.solvePnP(obj[pose_mask], corners[pose_mask], k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("Held-out solvePnP failed")
    predicted = project(obj, rv, tv, k, d)
    return corners[~pose_mask].reshape(-1, 2) - predicted[~pose_mask], rv, tv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_OUTPUT / "calib_stereo.zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--published-calibration-dir", type=Path, default=LAB / "outputs/fast-livo2-data-discovery-001/rtp02-prefix-calibration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    events = []
    def log(event):
        events.append({"elapsed_s": time.perf_counter() - started, **event})
        print(json.dumps(events[-1], ensure_ascii=False), flush=True)
    with zipfile.ZipFile(args.archive) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Archive CRC failure: {corrupt}")
        members = {side: {} for side in ["left", "right"]}
        for name in archive.namelist():
            parts = Path(name).parts
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                for side in members:
                    if side in parts:
                        ident = Path(name).stem
                        if ident in members[side]:
                            raise RuntimeError(f"Duplicate image ID: {side}/{ident}")
                        members[side][ident] = name
        ids = sorted(set(members["left"]) & set(members["right"]))
        split = {ident: "heldout" if index % 5 == 0 else "train" for index, ident in enumerate(ids)}
        if len(ids) < 15:
            raise RuntimeError(f"Too few stereo pairs: {len(ids)}")
        object_points = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE_M
        gx, gy = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2).T
        pose_mask = ((gx + gy) % 2 == 0)
        corners = {side: {} for side in members}
        rows = []
        image_size = None
        qa_saved = set()
        for side in members:
            for index, ident in enumerate(ids):
                name = members[side][ident]
                raw = archive.read(name)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot decode image: {name}")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                size = (gray.shape[1], gray.shape[0])
                if image_size is None:
                    image_size = size
                if size != image_size:
                    raise RuntimeError(f"Inconsistent image dimensions: {name}")
                found, detected = cv2.findChessboardCorners(gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
                method = "classic+cornerSubPix"
                if found:
                    detected = cv2.cornerSubPix(gray, detected, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))
                else:
                    found, detected = cv2.findChessboardCornersSB(gray, BOARD, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                    method = "SB-fallback" if found else "no-corners"
                row = {"side": side, "image_id": ident, "split_predeclared_before_detection": split[ident], "zip_member": name, "image_bytes": len(raw), "image_sha256": hashlib.sha256(raw).hexdigest(), "detected": bool(found), "method": method, "corner_count": len(detected) if found else 0}
                rows.append(row)
                if found:
                    corners[side][ident] = detected.astype(np.float32)
                    key = (side, split[ident])
                    if key not in qa_saved:
                        cv2.drawChessboardCorners(image, BOARD, detected, True)
                        cv2.imwrite(str(args.output / f"{side}-{split[ident]}-corners-{ident}.png"), image)
                        qa_saved.add(key)
                if index % 25 == 0 or index == len(ids) - 1:
                    log({"event": "corner_detection", "side": side, "images_processed": index + 1, "images_total": len(ids), "detected_so_far": len(corners[side])})
    with (args.output / "image-splits-and-detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output / "detected-corners.npz", object_points_m=object_points, heldout_pose_corner_mask=pose_mask, **{side + "__" + ident: value for side in corners for ident, value in corners[side].items()})
    published = {side: read_opencv_yaml(args.published_calibration_dir / f"camera_{side}.yaml") for side in members}
    models = {}
    training_errors = {}
    heldout = {}
    per_image = []
    flags = cv2.CALIB_FIX_K3
    for side in members:
        train_ids = [ident for ident in ids if split[ident] == "train" and ident in corners[side]]
        test_ids = [ident for ident in ids if split[ident] == "heldout" and ident in corners[side]]
        if len(train_ids) < 10 or len(test_ids) < 3:
            raise RuntimeError(f"Insufficient detected train/heldout images: {side}/{len(train_ids)}/{len(test_ids)}")
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera([object_points] * len(train_ids), [corners[side][ident] for ident in train_ids], image_size, None, None, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
        k_pub, d_pub = intrinsic_from_yaml(published[side])
        models[side] = {"K": k, "D": d, "K_published": k_pub, "D_published": d_pub, "train_ids": train_ids, "heldout_ids": test_ids, "opencv_training_rms": rms}
        residuals = []
        for ident, rv, tv in zip(train_ids, rvecs, tvecs):
            residual = corners[side][ident].reshape(-1, 2) - project(object_points, rv, tv, k, d)
            residuals.append(residual)
            per_image.append({"side": side, "image_id": ident, "split": "train", "model": "reestimated", **error_statistics(residual)})
        training_errors[side] = error_statistics(np.concatenate(residuals))
        heldout[side] = {}
        for model_name, model_k, model_d in [("reestimated", k, d), ("published", k_pub, d_pub)]:
            residuals = []
            for ident in test_ids:
                residual, rv, tv = heldout_errors(object_points, corners[side][ident], model_k, model_d, pose_mask)
                residuals.append(residual)
                per_image.append({"side": side, "image_id": ident, "split": "heldout", "model": model_name, **error_statistics(residual)})
            heldout[side][model_name] = error_statistics(np.concatenate(residuals))
        log({"event": "intrinsic_estimated", "side": side, "train_images": len(train_ids), "heldout_images": len(test_ids), "training_rms_pixels": rms, "heldout": heldout[side]})
    joint_train_ids = [ident for ident in ids if split[ident] == "train" and all(ident in corners[side] for side in members)]
    joint_test_ids = [ident for ident in ids if split[ident] == "heldout" and all(ident in corners[side] for side in members)]
    stereo_rms, _, _, _, _, stereo_r, stereo_t, essential, fundamental = cv2.stereoCalibrate([object_points] * len(joint_train_ids), [corners["left"][ident] for ident in joint_train_ids], [corners["right"][ident] for ident in joint_train_ids], models["left"]["K"].copy(), models["left"]["D"].copy(), models["right"]["K"].copy(), models["right"]["D"].copy(), image_size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
    t_i_cl = np.asarray(published["left"]["T_Body2Cam"]["data"]).reshape(4, 4)
    t_i_cr = np.asarray(published["right"]["T_Body2Cam"]["data"]).reshape(4, 4)
    t_cr_cl_published = np.linalg.inv(t_i_cr) @ t_i_cl
    stereo_heldout = {"left": [], "right": []}
    for ident in joint_test_ids:
        _, rv, tv = heldout_errors(object_points, corners["left"][ident], models["left"]["K"], models["left"]["D"], pose_mask)
        left_r = cv2.Rodrigues(rv)[0]
        right_r = stereo_r @ left_r
        right_t = stereo_r @ tv + stereo_t
        right_rv = cv2.Rodrigues(right_r)[0]
        for side, side_rv, side_tv in [("left", rv, tv), ("right", right_rv, right_t)]:
            residual = corners[side][ident][~pose_mask].reshape(-1, 2) - project(object_points, side_rv, side_tv, models[side]["K"], models[side]["D"])[~pose_mask]
            stereo_heldout[side].append(residual)
    stereo = {"training_shared_images": len(joint_train_ids), "heldout_shared_images": len(joint_test_ids), "training_rms_pixels": float(stereo_rms), "T_right_left_rotation": stereo_r.tolist(), "T_right_left_translation_m": stereo_t.ravel().tolist(), "baseline_norm_m": float(np.linalg.norm(stereo_t)), "published_T_right_left": t_cr_cl_published.tolist(), "published_baseline_norm_m": float(np.linalg.norm(t_cr_cl_published[:3, 3])), "translation_delta_norm_m_vs_published": float(np.linalg.norm(stereo_t.ravel() - t_cr_cl_published[:3, 3])), "rotation_delta_angle_deg_vs_published": float(np.rad2deg(np.arccos(np.clip((np.trace(stereo_r @ t_cr_cl_published[:3, :3].T) - 1) / 2, -1, 1)))), "heldout_pose_fit_on_left_27_corners_only": True, "heldout_other_27_corners_errors": {side: error_statistics(np.concatenate(stereo_heldout[side])) for side in stereo_heldout}, "does_not_estimate_camera_IMU_or_LiDAR_extrinsics_or_time_offsets": True}
    with (args.output / "per-image-reprojection-errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0])); writer.writeheader(); writer.writerows(per_image)
    arrays = {}
    for side in models:
        arrays.update({side + "_" + name: value for name, value in models[side].items() if isinstance(value, np.ndarray)})
    np.savez_compressed(args.output / "calibration-model.npz", **arrays, stereo_rotation=stereo_r, stereo_translation_m=stereo_t, essential=essential, fundamental=fundamental)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, side in zip(axes, ["left", "right"]):
        for label, color in [("reestimated", "#2563eb"), ("published", "#b45309")]:
            subset = [x for x in per_image if x["side"] == side and x["split"] == "heldout" and x["model"] == label]
            ax.plot(range(len(subset)), [x["euclidean_pixel_rmse"] for x in subset], "o-", label=label, color=color, markersize=3)
        ax.set(title=f"{side}: held-out images", xlabel="Held-out image order", ylabel="27 evaluation corners RMSE (pixels)")
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(args.output / "heldout-reprojection-errors.png", dpi=160); plt.close(fig)
    metrics = {
        "status": "completed_actual_public_board_stereo_calibration_and_heldout_evaluation",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "duration_s": time.perf_counter() - started,
        "scope": "Actual published NTU VIRAL board images. Camera intrinsics and stereo camera extrinsics re-estimated; no camera-IMU, LiDAR-camera, clock, flight trajectory, or FAST-LIVO2 replay calibration.",
        "archive": str(args.archive), "archive_bytes": args.archive.stat().st_size, "archive_sha256": sha256(args.archive), "archive_all_member_CRC_test_passed": True,
        "source_url": SOURCE_URL, "source_dataset_license": "NTU VIRAL CC BY-NC-SA 4.0; non-commercial academic dataset", "source_documentation": "https://ntu-aris.github.io/ntu_viral_dataset/sensor_calibration.html",
        "audit_script": str(Path(__file__).resolve()), "audit_script_sha256": sha256(__file__), "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "board_inner_corners_width_height": list(BOARD), "square_size_m_documented_not_independently_measured": SQUARE_M,
        "raw_image_size_width_height": list(image_size),
        "split_rule_predeclared": "sorted shared image IDs, index % 5 == 0 heldout, all others train; split before detection, no test-based model selection/outlier deletion",
        "all_stereo_pair_ids": ids, "declared_train_pair_ids": [i for i in ids if split[i] == "train"], "declared_heldout_pair_ids": [i for i in ids if split[i] == "heldout"],
        "detected_images_per_side": {side: len(corners[side]) for side in corners}, "failed_detection_rows": [x for x in rows if not x["detected"]],
        "distortion_model": "k1,k2,p1,p2; fixed k3=0", "opencv_calibration_flags": flags,
        "heldout_pose_fit_corner_indices": np.flatnonzero(pose_mask).tolist(), "heldout_evaluation_corner_indices": np.flatnonzero(~pose_mask).tolist(),
        "heldout_pose_fit_corners_disjoint_from_evaluation_corners": True,
        "per_camera": {side: {"train_image_ids": model["train_ids"], "heldout_image_ids": model["heldout_ids"], "K_reestimated": model["K"].tolist(), "D_reestimated_k1_k2_p1_p2_k3": model["D"].ravel().tolist(), "K_published": model["K_published"].tolist(), "D_published_k1_k2_p1_p2_k3": model["D_published"].ravel().tolist(), "training_cv_rms_pixels": float(model["opencv_training_rms"]), "training_explicit_error": training_errors[side], "heldout_other_27_corners": heldout[side]} for side, model in models.items()},
        "stereo": stereo,
        "limitations": ["Same calibration sequence, interleaved image split; not cross-session generalization", "Image pose inferred from 27 heldout corners; evaluates remaining 27 at that estimated pose", "No manual observation deletion or iterative configuration/model search", "Stereo translation magnitude depends on documented 80 mm square scale", "Does not overwrite FAST-LIVO2 official configuration", "No IMU/LiDAR/time calibration or hardware synchronization demonstrated"],
        "original_FAST_configuration_modified": False,
    }
    artifacts = [args.output / name for name in ["image-splits-and-detections.csv", "detected-corners.npz", "per-image-reprojection-errors.csv", "calibration-model.npz", "heldout-reprojection-errors.png"]]
    metrics["artifacts"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in artifacts]
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log({"event": "completed", "metrics": str(args.output / "metrics.json"), "stereo": stereo})
    (args.output / "run-log.json").write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(__file__, args.output / "calibration_source.py")


if __name__ == "__main__":
    main()
