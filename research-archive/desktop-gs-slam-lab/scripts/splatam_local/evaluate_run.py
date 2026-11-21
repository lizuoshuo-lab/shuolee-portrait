"""Independent SplaTAM trajectory evaluation from raw TUM ground truth.

Official params store w2c and scalar-first quaternions.  Rigid SE(3)
alignment never estimates scale.  Upstream's mislabeled mean ATE is not reused.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from audit_dataset import LAB, PIN, associate, relative_gt, save, select_frames, sha, write_rows

FREIBURG1 = dict(image_height=480, image_width=640, fx=517.3, fy=516.5,
                cx=318.6, cy=255.3, crop_edge=8, png_depth_scale=5000.0)


def statistics(values, unit):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid metric values")
    return dict(count=len(values), unit=unit, rmse=float(np.sqrt(np.mean(values ** 2))),
                mean=float(np.mean(values)), median=float(np.median(values)),
                max=float(values.max()), min=float(values.min()), std=float(values.std()))


def cameras_from_params(params):
    q = np.asarray(params["cam_unnorm_rots"], dtype=np.float64)
    t = np.asarray(params["cam_trans"], dtype=np.float64)
    if q.ndim != 3 or q.shape[:2] != (1, 4) or t.shape != (1, 3, q.shape[2]):
        raise ValueError(f"Unexpected official camera shapes: {q.shape}, {t.shape}")
    q, t = q[0].T, t[0].T
    if len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(t).all() or np.any(np.linalg.norm(q, axis=1) < 1e-12):
        raise ValueError("Non-finite camera values or zero-norm quaternion")
    w2c = np.repeat(np.eye(4)[None], len(q), axis=0)
    w2c[:, :3, :3] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    w2c[:, :3, 3] = t
    return np.linalg.inv(w2c)


def align_rigid(estimated, reference):
    """Kabsch with det(R)=+1, scale fixed to 1, estimated c2w -> GT gauge."""
    x, y = estimated[:, :3, 3], reference[:, :3, 3]
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    u, singular, vh = np.linalg.svd((x - xm).T @ (y - ym))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vh.T @ u.T)
    rotation = vh.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = ym - rotation @ xm
    return transform[None] @ estimated, transform, singular


def trajectory_errors(estimated, reference):
    aligned, transform, singular = align_rigid(estimated, reference)
    ate_raw = np.linalg.norm(estimated[:, :3, 3] - reference[:, :3, 3], axis=1)
    ate_aligned = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    gt_delta = np.linalg.inv(reference[:-1]) @ reference[1:]
    pred_delta = np.linalg.inv(estimated[:-1]) @ estimated[1:]
    rpe = np.linalg.inv(gt_delta) @ pred_delta
    rpe_translation = np.linalg.norm(rpe[:, :3, 3], axis=1)
    rpe_rotation = np.rad2deg(Rotation.from_matrix(rpe[:, :3, :3]).magnitude())
    return dict(
        ate_se3_translation=statistics(ate_aligned, "m"),
        ate_first_camera_gauge_translation=statistics(ate_raw, "m"),
        rpe_1frame_translation=statistics(rpe_translation, "m"),
        rpe_1frame_rotation=statistics(rpe_rotation, "deg"),
        alignment=dict(type="SE3 rigid Kabsch, scale fixed to 1", scale=1.0,
                       transform_estimated_to_reference=transform.tolist(),
                       positional_covariance_singular_values=singular.tolist(),
                       determinant_rotation=float(np.linalg.det(transform[:3, :3])),
                       positional_covariance_rank=int(np.linalg.matrix_rank((estimated[:, :3, 3] - estimated[:, :3, 3].mean(axis=0)).T @
                                                                           (reference[:, :3, 3] - reference[:, :3, 3].mean(axis=0))))),
    ), aligned, (ate_raw, ate_aligned, rpe_translation, rpe_rotation)


def config_gate(config):
    if config.get("tracking", {}).get("use_gt_poses") is not False:
        raise ValueError("GT leakage gate: tracking.use_gt_poses must be explicitly false")
    if int(config["tracking"].get("num_iters", 0)) <= 0 or int(config.get("mapping", {}).get("num_iters", 0)) <= 0:
        raise ValueError("Genuine tracking and mapping must both optimize")
    if int(config.get("map_every", 0)) <= 0:
        raise ValueError("Invalid mapping interval")
    tracking_lrs = config["tracking"].get("lrs", {})
    mapping_lrs = config["mapping"].get("lrs", {})
    if not any(float(tracking_lrs.get(name, 0)) > 0 for name in ("cam_unnorm_rots", "cam_trans")):
        raise ValueError("No camera parameter has a positive tracking learning rate")
    if not any(float(mapping_lrs.get(name, 0)) > 0 for name in ("means3D", "rgb_colors", "log_scales", "logit_opacities")):
        raise ValueError("No Gaussian parameter has a positive mapping learning rate")


def resolved_input_config(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Expected one literal config assignment in input_config.py")
    value = ast.literal_eval(assignments[0].value)
    if not isinstance(value, dict):
        raise ValueError("input_config.py config is not a dictionary")
    return value


def check_saved_gt(saved, reference):
    saved = np.asarray(saved, dtype=np.float64)
    expected = np.linalg.inv(reference)
    if saved.shape != expected.shape or not np.isfinite(saved).all():
        raise ValueError("Saved GT count/shape/finite mismatch")
    max_difference = float(np.max(np.abs(saved - expected)))
    if max_difference > 2e-5:
        raise ValueError(f"Saved GT disagrees with independently reconstructed raw GT: max {max_difference}")
    return max_difference


def native_inputs_gate(execution, raw_lists, selected_images, sensor_snapshot_hash=None):
    before = execution.get("native_input_sha256_before")
    after = execution.get("native_input_sha256_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Native training input hashes before/after are missing")
    if execution.get("inputs_unchanged_during_run") is not True or before != after:
        raise ValueError("Native training inputs changed during execution")
    if before.get("raw_lists") != raw_lists:
        raise ValueError("Independent Windows raw TUM lists differ from native training inputs")
    if before.get("selected_images") != selected_images:
        raise ValueError("Independent Windows selected images differ from native training inputs")
    sensor_hash = before.get("sensor_config_sha256")
    if not isinstance(sensor_hash, str) or len(sensor_hash) != 64 or any(ch not in "0123456789abcdef" for ch in sensor_hash):
        raise ValueError("Native sensor configuration SHA256 is missing or malformed")
    if sensor_snapshot_hash is not None and sensor_hash != sensor_snapshot_hash:
        raise ValueError("Sensor configuration snapshot differs from native training input")
    return sensor_hash


def calibration_values(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("dataset_name") != "tum" or not isinstance(config.get("camera_params"), dict):
        raise ValueError("Invalid TUM calibration YAML")
    camera = config["camera_params"]
    if any(camera.get(key) != value for key, value in FREIBURG1.items()):
        raise ValueError("Calibration YAML differs from the public Freiburg1 sensor parameters")
    if camera.get("distortion") is not None:
        raise ValueError("Unexpected active distortion correction differs from the pinned public configuration")
    return camera


def check_saved_intrinsics(params, config, camera):
    height, width = config["data"]["desired_image_height"], config["data"]["desired_image_width"]
    # Match upstream scale_intrinsics float32 arithmetic exactly. crop_edge is
    # present in YAML but the pinned loader does not actually crop or shift K.
    expected = np.eye(3, dtype=np.float32)
    expected[0, 0], expected[1, 1] = camera["fx"], camera["fy"]
    expected[0, 2], expected[1, 2] = camera["cx"], camera["cy"]
    expected[0, [0, 2]] *= width / camera["image_width"]
    expected[1, [1, 2]] *= height / camera["image_height"]
    saved = np.asarray(params["intrinsics"], dtype=np.float64)
    if saved.shape != (3, 3) or not np.isfinite(saved).all():
        raise ValueError("Invalid saved camera intrinsic matrix")
    max_difference = float(np.max(np.abs(saved - expected)))
    if max_difference > 2e-5:
        raise ValueError(f"Saved K differs from independent public calibration: max {max_difference}")
    if int(np.asarray(params["org_width"]).item()) != width or int(np.asarray(params["org_height"]).item()) != height:
        raise ValueError("Saved image dimensions differ from actual input configuration")
    return dict(public_freiburg1_sensor=FREIBURG1, expected_scaled_k=expected.tolist(),
                saved_k=saved.tolist(), max_matrix_difference=max_difference, tolerance=2e-5,
                crop_edge_scope="Pinned loader records crop_edge=8 but does not apply an image crop or principal-point shift")


def tum(path, stamps, poses):
    values = np.column_stack((stamps, poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()))
    np.savetxt(path, values, fmt="%.9f", header="timestamp tx ty tz qx qy qz qw; c2w; metres")


def self_test(output):
    """Algorithm controls are synthetic and never counted as a real run."""
    rng = np.random.default_rng(7)
    reference = np.repeat(np.eye(4)[None], 12, axis=0)
    reference[:, :3, 3] = rng.normal(size=(12, 3))
    reference[:, :3, :3] = Rotation.from_rotvec(rng.normal(scale=0.2, size=(12, 3))).as_matrix()
    rigid = np.eye(4)
    rigid[:3, :3] = Rotation.from_euler("xyz", [12, -30, 47], degrees=True).as_matrix()
    rigid[:3, 3] = [1, -2, 0.5]
    transformed = np.linalg.inv(rigid)[None] @ reference
    metric, _, _ = trajectory_errors(transformed, reference)
    controls = dict(
        exact_rigid_transform_recovers_ate=metric["ate_se3_translation"]["rmse"] < 1e-12,
        rigid_global_transform_preserves_relative_motion=metric["rpe_1frame_translation"]["rmse"] < 1e-12 and metric["rpe_1frame_rotation"]["rmse"] < 1e-10,
        rmse_is_not_mean=abs(statistics([1., 3.], "m")["rmse"] - np.sqrt(5)) < 1e-12 and statistics([1., 3.], "m")["mean"] == 2.,
    )
    scaled = reference.copy()
    scaled[:, :3, 3] *= 2
    scale_metric, _, _ = trajectory_errors(scaled, reference)
    controls["scale_error_is_not_absorbed"] = scale_metric["ate_se3_translation"]["rmse"] > 0.5 and scale_metric["alignment"]["scale"] == 1.
    w2c = np.linalg.inv(reference)
    q_xyzw = Rotation.from_matrix(w2c[:, :3, :3]).as_quat()
    params = dict(cam_unnorm_rots=q_xyzw[:, [3, 0, 1, 2]].T[None], cam_trans=w2c[:, :3, 3].T[None])
    controls["wxyz_w2c_conversion_recovers_c2w"] = np.max(np.abs(cameras_from_params(params) - reference)) < 1e-12
    valid_config = dict(tracking=dict(use_gt_poses=False, num_iters=200, lrs=dict(cam_trans=.002)),
                        mapping=dict(num_iters=30, lrs=dict(means3D=.0001)), map_every=1)
    config_gate(valid_config)
    controls["accepts_explicit_gt_disabled_positive_optimization"] = True
    controls["saved_float32_gt_roundoff_is_tolerated"] = check_saved_gt(w2c.astype(np.float32), reference) < 2e-5
    controls["selection_order_matches_loader"] = [row["retained_index"] for row in select_frames(
        [dict(retained_index=i) for i in range(10)], dict(start=2, end=9, stride=2, num_frames=3))] == [2, 4, 6]
    for label, operation in (
        ("rejects_zero_quaternion", lambda: cameras_from_params(dict(cam_unnorm_rots=np.zeros((1, 4, 2)), cam_trans=np.zeros((1, 3, 2))))),
        ("rejects_nonfinite_translation", lambda: cameras_from_params(dict(cam_unnorm_rots=params["cam_unnorm_rots"], cam_trans=np.full((1, 3, 12), np.nan)))),
        ("rejects_gt_pose_tracking", lambda: config_gate(dict(tracking=dict(use_gt_poses=True, num_iters=200), mapping=dict(num_iters=30), map_every=1))),
        ("rejects_zero_camera_optimization", lambda: config_gate(dict(tracking=dict(use_gt_poses=False, num_iters=200, lrs=dict(cam_trans=0.)),
                                                                    mapping=valid_config["mapping"], map_every=1))),
        ("rejects_camera_shape_mismatch", lambda: cameras_from_params(dict(cam_unnorm_rots=params["cam_unnorm_rots"], cam_trans=np.zeros((1, 3, 11))))),
        ("rejects_independent_gt_mismatch", lambda: check_saved_gt(w2c + np.eye(4)[None] * .01, reference)),
    ):
        try:
            operation()
            controls[label] = False
        except ValueError:
            controls[label] = True
    identity = np.repeat(np.eye(4)[None], 2, axis=0)
    rotate = identity.copy()
    rotate[1, :3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    rotation_metric, _, _ = trajectory_errors(rotate, identity)
    controls["rotation_rpe_units_are_degrees"] = abs(rotation_metric["rpe_1frame_rotation"]["rmse"] - 30) < 1e-10
    raw_lists = {name: "a"*64 for name in ("rgb.txt", "depth.txt", "groundtruth.txt")}
    images = {"rgb/1.png": "b"*64, "depth/1.png": "c"*64}
    native = dict(raw_lists=raw_lists, selected_images=images, sensor_config_sha256="d"*64)
    execution = dict(native_input_sha256_before=copy.deepcopy(native), native_input_sha256_after=copy.deepcopy(native),
                     inputs_unchanged_during_run=True)
    controls["accepts_matching_native_windows_inputs"] = native_inputs_gate(execution, raw_lists, images, "d"*64) == "d"*64
    intrinsic_params = dict(intrinsics=np.array([[258.65, 0, 159.3], [0, 258.25, 127.65], [0, 0, 1]], dtype=np.float32),
                            org_width=320, org_height=240)
    image_config = dict(data=dict(desired_image_height=240, desired_image_width=320))
    controls["saved_intrinsics_match_public_resize"] = check_saved_intrinsics(intrinsic_params, image_config, FREIBURG1)["max_matrix_difference"] == 0
    incorrect_cropped = copy.deepcopy(intrinsic_params)
    incorrect_cropped["intrinsics"][0, 2] -= 8
    altered_native = copy.deepcopy(execution)
    altered_native["native_input_sha256_after"]["selected_images"]["rgb/1.png"] = "e"*64
    for label, operation in (
        ("rejects_missing_native_input_evidence", lambda: native_inputs_gate({}, raw_lists, images)),
        ("rejects_native_training_input_change", lambda: native_inputs_gate(altered_native, raw_lists, images)),
        ("rejects_windows_vs_native_list_mismatch", lambda: native_inputs_gate(execution, dict(raw_lists, **{"rgb.txt": "f"*64}), images)),
        ("rejects_windows_vs_native_image_mismatch", lambda: native_inputs_gate(execution, raw_lists, dict(images, **{"rgb/1.png": "f"*64}))),
        ("rejects_sensor_snapshot_hash_mismatch", lambda: native_inputs_gate(execution, raw_lists, images, "f"*64)),
        ("rejects_unapplied_crop_intrinsics_shift", lambda: check_saved_intrinsics(incorrect_cropped, image_config, FREIBURG1)),
        ("rejects_saved_resolution_mismatch", lambda: check_saved_intrinsics(dict(intrinsic_params, org_width=640), image_config, FREIBURG1)),
    ):
        try:
            operation()
            controls[label] = False
        except ValueError:
            controls[label] = True
    controls = {key: bool(value) for key, value in controls.items()}
    result = dict(status="passed" if all(controls.values()) else "failed", synthetic_only=True,
                  controls=controls, checked=len(controls), evaluator_sha256=sha(__file__))
    output.mkdir(parents=True, exist_ok=True)
    save(output / "controls.json", result)
    print(json.dumps(result))
    if not all(controls.values()):
        raise ValueError("Independent evaluator controls failed")


def evaluate(run_dir, sequence, output, config_path=None):
    output.mkdir(parents=True, exist_ok=True)
    config_path = config_path or run_dir / "effective-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_gate(config)
    if config["data"]["sequence"] != sequence.name:
        raise ValueError("Independent data sequence differs from config")
    execution_path = run_dir / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    input_config_path = run_dir / "input_config.py"
    if sha(input_config_path) != execution.get("config_sha256"):
        raise ValueError("Actual input_config.py differs from execution config hash")
    if resolved_input_config(input_config_path) != config:
        raise ValueError("Effective JSON config differs from the actual hashed Python input config")
    if execution.get("child_exit_code") != 0 or execution.get("status") not in ("completed_pending_independent_evaluation", "completed"):
        raise ValueError("Real child process did not finish successfully")
    if execution.get("splatam_commit") != PIN or execution.get("source_unchanged_during_run") is not True:
        raise ValueError("Pinned unchanged SplaTAM source evidence absent")
    if not execution.get("source_sha256_before") or execution["source_sha256_before"] != execution.get("source_sha256_after"):
        raise ValueError("Source before/after hashes do not independently support unchanged-source flag")
    if execution.get("tracking_uses_gt_poses") is not False:
        raise ValueError("Execution receipt does not prove GT pose tracking was disabled")
    frames, absolute_gt, counts = associate(sequence)
    selected = select_frames(frames, config["data"])
    reference = relative_gt(selected, absolute_gt)
    raw_list_hashes = {name: sha(sequence / name) for name in ("rgb.txt", "depth.txt", counts["gt_name"])}
    selected_image_hashes = {name: sha(sequence / name) for row in selected for name in (row["rgb_path"], row["depth_path"])}
    sensor_snapshot = run_dir / "sensor-config.yaml"
    sensor_snapshot_hash = sha(sensor_snapshot) if sensor_snapshot.is_file() else None
    sensor_native_hash = native_inputs_gate(execution, raw_list_hashes, selected_image_hashes, sensor_snapshot_hash)
    official_sensor = LAB / "repos/SplaTAM/configs/data/TUM/freiburg1_desk.yaml"
    calibration = calibration_values(official_sensor)
    if sensor_snapshot_hash:
        calibration_values(sensor_snapshot)
    params_path = run_dir / "params.npz"
    params_hash = sha(params_path)
    if params_hash != execution.get("params_sha256"):
        raise ValueError("params.npz differs from completed execution receipt")
    with np.load(params_path, allow_pickle=False) as params:
        if not all(np.isfinite(params[key]).all() for key in params.files if np.issubdtype(params[key].dtype, np.number)):
            raise ValueError("Non-finite numeric model arrays")
        estimated = cameras_from_params(params)
        if len(estimated) != len(selected):
            raise ValueError(f"Configured input/camera mismatch: {len(selected)} vs {len(estimated)}")
        saved_gt_difference = check_saved_gt(params["gt_w2c_all_frames"], reference)
        intrinsics_check = check_saved_intrinsics(params, config, calibration)
        gaussian_count = int(params["means3D"].shape[0])
        if gaussian_count <= 0:
            raise ValueError("No mapped Gaussian primitives")
        keyframes = params["keyframe_time_indices"].tolist() if "keyframe_time_indices" in params else []
    metrics, aligned, errors = trajectory_errors(estimated, reference)
    stamps = np.asarray([row["rgb_timestamp"] for row in selected])
    input_hashes = dict(
        params=dict(path=str(params_path), bytes=params_path.stat().st_size, sha256=params_hash),
        effective_config=dict(path=str(config_path), sha256=sha(config_path)),
        input_config=dict(path=str(input_config_path), sha256=sha(input_config_path)),
        execution=dict(path=str(execution_path), sha256=sha(execution_path)),
        raw_lists=raw_list_hashes,
        selected_images=selected_image_hashes,
        sensor_config_native_sha256=sensor_native_hash,
        sensor_config_snapshot_sha256=sensor_snapshot_hash,
        independent_public_sensor_yaml_sha256=sha(official_sensor),
        evaluator_sha256=sha(__file__), association_auditor_sha256=sha(Path(__file__).with_name("audit_dataset.py")),
    )
    write_rows(output / "associations.csv", selected)
    for filename, poses in (("estimated.tum", estimated), ("reference-independent.tum", reference), ("estimated-se3-aligned.tum", aligned)):
        tum(output / filename, stamps, poses)
    error_rows = [dict(frame=i, timestamp=float(stamps[i]), ate_first_camera_gauge_m=float(errors[0][i]),
                       ate_se3_m=float(errors[1][i]), rpe_1frame_trans_m="" if i == 0 else float(errors[2][i-1]),
                       rpe_1frame_rot_deg="" if i == 0 else float(errors[3][i-1])) for i in range(len(stamps))]
    write_rows(output / "errors.csv", error_rows)
    save(output / "input-hashes.json", input_hashes)
    checks = dict(actual_child_exit_zero=True, unchanged_pinned_source=True, source_before_after_hashes_equal=True,
                  actual_input_config_hash_matches_receipt=True, effective_config_equals_hashed_input=True,
                  config_gt_tracking_disabled=True,
                  config_tracking_and_mapping_optimize=True, config_frame_count_matches_params=True,
                  all_numeric_model_arrays_finite=True, raw_groundtruth_reconstructed_independently=True,
                  saved_groundtruth_agrees_with_independent=True, params_hash_matches_completed_receipt=True,
                  gaussians_nonempty=True, alignment_rotation_is_proper=abs(metrics["alignment"]["determinant_rotation"]-1) < 1e-8,
                  alignment_scale_fixed_one=True, native_training_input_hashes_before_after_equal=True,
                  windows_raw_lists_exactly_match_native=True, windows_selected_images_exactly_match_native=True,
                  saved_intrinsics_match_public_calibration=True, saved_resolution_matches_config=True)
    if sensor_snapshot_hash:
        checks["sensor_snapshot_hash_matches_native"] = True
        checks["sensor_snapshot_values_match_public_freiburg1"] = True
    if not all(checks.values()):
        raise ValueError("An independent acceptance check failed")
    metrics.update(schema_version=1, status="independently_verified", run_name=run_dir.name,
                   sequence=sequence.name, frames=len(selected), available_loader_frames=counts["retained"],
                   full_associated_sequence=len(selected) == counts["retained"] and int(config["data"].get("start", 0)) == 0 and int(config["data"].get("stride", 1)) == 1,
                   resolution=[config["data"]["desired_image_width"], config["data"]["desired_image_height"]],
                   splatam_commit=PIN, gaussian_count=gaussian_count, keyframe_indices=keyframes,
                   trajectory_frame="c2w in first selected camera gauge", duration_s=float(stamps[-1]-stamps[0]),
                   saved_gt_max_matrix_difference=saved_gt_difference, saved_gt_tolerance=2e-5,
                   intrinsics_verification=intrinsics_check, sensor_config_native_sha256=sensor_native_hash,
                   checks=checks, check_count=len(checks), child_wall_seconds=execution.get("wall_seconds"),
                   tracking_uses_gt_poses=False,
                   gt_scope="Nearest-GT association, first-camera gauge, upstream finite-GT keyframe gate and evaluation. No per-frame GT tracking.",
                   input_hashes_sha256=sha(output / "input-hashes.json"),
                   limits=["320x240 runs are reduced-resolution local experiments, not paper-resolution results.",
                           "Metrics cover configured sequence only; short smoke runs do not establish full-sequence robustness.",
                           "RPE uses adjacent selected frames (delta=1 f), with no GT interpolation or scale fitting.",
                           "RGB-D model requires depth; this does not establish monocular, LiDAR or drone performance."])
    elapsed = stamps-stamps[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(reference[:, 0, 3], reference[:, 1, 3], label="Raw-GT independent reference", linewidth=2)
    axes[0, 0].plot(aligned[:, 0, 3], aligned[:, 1, 3], label="SplaTAM SE(3)-aligned", alpha=.85)
    axes[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="Camera trajectory; scale = 1")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(elapsed, errors[1], label="SE(3)-aligned ATE")
    axes[0, 1].plot(elapsed, errors[0], label="First-camera-gauge error", alpha=.65)
    axes[0, 1].set(xlabel="Time from first selected RGB (s)", ylabel="Translation error (m)", title=f"ATE RMSE {metrics['ate_se3_translation']['rmse']:.6f} m")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(elapsed[1:], errors[2])
    axes[1, 0].set(xlabel="Time (s)", ylabel="Translation RPE (m)", title="Relative pose error; delta = 1 selected frame")
    axes[1, 1].plot(elapsed[1:], errors[3], color="tab:orange")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Rotation RPE (degrees)", title="Rotation error; no scale alignment")
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.suptitle(f"Official SplaTAM | {sequence.name} | {len(selected)} frames | {metrics['resolution'][0]}x{metrics['resolution'][1]}")
    fig.savefig(output / "trajectory-errors.png", dpi=150)
    plt.close(fig)
    metrics["plot_sha256"] = sha(output / "trajectory-errors.png")
    save(output / "metrics.json", metrics)
    (output / "report.md").write_text("\n".join([
        f"# SplaTAM 独立评测：{run_dir.name}", "",
        f"真实子进程退出 0，模型哈希与运行回执一致，{len(checks)} 项独立验收通过。",
        f"{len(selected)} / {counts['retained']} 个官方关联帧；分辨率 {metrics['resolution'][0]}×{metrics['resolution'][1]}；{gaussian_count} 个高斯。",
        f"完整关联序列：{metrics['full_associated_sequence']}。", "",
        f"SE(3) 刚体对齐平移 ATE：RMSE **{metrics['ate_se3_translation']['rmse']:.6f} m**，mean {metrics['ate_se3_translation']['mean']:.6f} m，median {metrics['ate_se3_translation']['median']:.6f} m，max {metrics['ate_se3_translation']['max']:.6f} m。",
        f"相邻帧 RPE：平移 RMSE **{metrics['rpe_1frame_translation']['rmse']:.6f} m**，旋转 RMSE **{metrics['rpe_1frame_rotation']['rmse']:.6f}°**。", "",
        "GT 从原始 groundtruth.txt 按公开加载规则独立重建，再转为第一选中相机坐标；没有直接把 params 内 savedGT 当作唯一真值。",
        f"savedGT 与独立 GT 的最大矩阵元素差 {saved_gt_difference:.3e}（验收容差 2e-5）。",
        "WSL 实际训练的原始列表与选中 RGB/depth 哈希，逐项与 Windows 独立评测输入一致；训练前后输入哈希一致，缺证或任一差异均拒绝。",
        "模型 saved K 独立核对公开 Freiburg1 内参与目标分辨率缩放，深度尺度为 5000；没有把配置中的 crop_edge 字段误说成实际裁剪。",
        "固定 scale=1，只做 SE(3) 刚体对齐；RMSE 按 sqrt(mean(error²)) 计算，不采用上游标成 RMSE 的 mean ATE。",
        "use_gt_poses=false；GT 仍参与上游数据关联、第一相机坐标约定、有限 GT 的关键帧门控与评测，不能宣传完全不使用 GT。", "",
        "320×240 是本地降分辨率实验，不能与论文 640×480 结果直接对比。短序列通过只证明该输入区间完成；完整数据覆盖另行记录。",
        "逐帧误差 errors.csv；轨迹 estimated.tum、reference-independent.tum、estimated-se3-aligned.tum；原始输入绑定 input-hashes.json。", "",
        "![轨迹与误差](trajectory-errors.png)", ""
    ]), encoding="utf-8")
    print(json.dumps(dict(status=metrics["status"], frames=len(selected), full_sequence=metrics["full_associated_sequence"],
                          ate_rmse_m=metrics["ate_se3_translation"]["rmse"], rpe_trans_rmse_m=metrics["rpe_1frame_translation"]["rmse"],
                          rpe_rot_rmse_deg=metrics["rpe_1frame_rotation"]["rmse"], output=str(output))))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test((args.output or LAB / "outputs/splatam-evaluator-controls-001").resolve())
        return
    if not args.run_dir or not args.sequence:
        parser.error("Real evaluation requires --run-dir and independently available --sequence")
    output = (args.output or args.run_dir / "independent-evaluation").resolve()
    try:
        evaluate(args.run_dir.resolve(), args.sequence.resolve(), output, args.config)
    except (ValueError, OSError, KeyError, SyntaxError, TypeError) as error:
        output.mkdir(parents=True, exist_ok=True)
        save(output / "metrics.json", dict(status="rejected", run_name=args.run_dir.name,
                                           reason=str(error), evaluator_sha256=sha(__file__)))
        (output / "report.md").write_text(f"# SplaTAM 独立验收未通过\n\n{error}\n\n不把该运行登记为成功项目或精度结果。\n", encoding="utf-8")
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
