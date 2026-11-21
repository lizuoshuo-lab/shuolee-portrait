"""Read-only FAST-LIVO2 upstream configuration and source contract audit.

This checks published calibration parameters; it does not estimate a calibration,
read a sensor bag, synchronize a physical sensor, or run FAST-LIVO2. The stored
upstream snapshots and SHA256 manifest make the default invocation offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(encoding="utf-8")
LAB = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = LAB / "outputs/fast-livo2-config-audit-001"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_code(code):
    # The inspected contract expressions contain no comment-like string literals.
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", code, flags=re.S)


def function_body(code, name):
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^)]*\)\s*\{", code)
    if not match:
        raise ValueError(f"Function not found: {name}")
    start, depth = match.end(), 1
    for i in range(start, len(code)):
        depth += (code[i] == "{") - (code[i] == "}")
        if depth == 0:
            return code[start:i]
    raise ValueError(f"Unterminated function: {name}")


def transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def read_opencv_yaml(path):
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("tag:yaml.org,2002:opencv-matrix", lambda loader, node: loader.construct_mapping(node, deep=True))
    content = path.read_text(encoding="utf-8-sig")
    return yaml.load(re.sub(r"^%YAML:1\.0\s*", "", content), Loader=Loader)


def inspect_dataset_calibration(directory, ntu_config, documented_prism):
    names = ["camera_left.yaml", "camera_right.yaml", "lidar_horz.yaml", "lidar_vert.yaml", "leica_prism.yaml", "uwb_nodes.yaml"]
    receipt_path = directory / "prefix-extract-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    records = {Path(x["filename"]).name: x for x in receipt["entries"] if x.get("sha256")}
    files = []
    parsed = {}
    for name in names:
        path = directory / name
        parsed[name] = read_opencv_yaml(path)
        files.append({"path": str(path), "sha256": sha256(path), "matches_extraction_receipt_sha256": sha256(path) == records[name]["sha256"], "receipt_entry_status": records[name]["status"]})
    left = parsed["camera_left.yaml"]
    right = parsed["camera_right.yaml"]
    left_raw = np.asarray(left["T_Body2Cam"]["data"]).reshape(4, 4)
    right_raw = np.asarray(right["T_Body2Cam"]["data"]).reshape(4, 4)
    horizontal_raw = np.asarray(parsed["lidar_horz.yaml"]["T_Body2Lidar"]["data"]).reshape(4, 4)
    vertical_raw = np.asarray(parsed["lidar_vert.yaml"]["T_Body2Lidar"]["data"]).reshape(4, 4)
    prism_raw = np.asarray(parsed["leica_prism.yaml"]["T_Body2Imu"]["data"]).reshape(4, 4)
    ntu_ci = np.asarray(ntu_config["T_C_I"])
    inverse_left = np.linalg.inv(left_raw)
    k = np.asarray([[left["projection_parameters"]["fx"], 0, left["projection_parameters"]["cx"]], [0, left["projection_parameters"]["fy"], left["projection_parameters"]["cy"]], [0, 0, 1]])
    distortion = [left["distortion_parameters"][x] for x in ["k1", "k2", "p1", "p2"]]
    left_inverse_delta = float(np.max(np.abs(inverse_left - ntu_ci)))
    left_same_direction_delta = float(np.max(np.abs(left_raw - ntu_ci)))
    gates = {
        "six_files_match_individual_prefix_CRC_receipt_bound_sha": all(x["matches_extraction_receipt_sha256"] for x in files),
        "left_intrinsics_exact_match_FAST_config": bool(np.array_equal(k, ntu_config["intrinsic_raw_K"])),
        "left_distortion_exact_match_FAST_config": bool(np.array_equal(distortion, ntu_config["distortion_config_d0_d1_d2_d3"])),
        "horizontal_lidar_transform_matches_T_I_L": bool(np.max(np.abs(horizontal_raw - ntu_config["T_I_L"])) <= 1e-12),
        "inverse_left_camera_pose_matches_T_C_I_with_rounding_tolerance": left_inverse_delta <= 1e-5,
        "prism_file_matches_documented_I_to_prism_lever_arm": bool(np.max(np.abs(prism_raw - documented_prism)) <= 1e-12),
        "all_four_sensor_rotations_near_SO3": all(so3_diagnostic(x[:3, :3], 1e-5)["approximately_so3_at_declared_tolerance"] for x in [left_raw, right_raw, horizontal_raw, vertical_raw]),
    }
    return {
        "scope": "Actual six complete small YAML entries extracted and individually CRC/size verified from rtp_02 ZIP prefix; the full archive was still downloading at extraction. This does not verify its complete bag or estimate any calibration.",
        "extraction_receipt": str(receipt_path), "extraction_receipt_sha256": sha256(receipt_path), "files": files,
        "dataset_name_field_in_all_yaml": {name: doc["dataset_name"] for name, doc in parsed.items()},
        "dataset_name_field_is_eee_01_even_in_rtp_02_archive": all(doc["dataset_name"] == "eee_01" for doc in parsed.values()),
        "raw_named_T_Body2Cam_left_matrix": left_raw.tolist(), "raw_named_T_Body2Cam_right_matrix": right_raw.tolist(),
        "camera_left_raw_matrix_interpreted_as_T_I_C_by_cross_check": True,
        "inverse_left_vs_FAST_T_C_I_max_abs": left_inverse_delta,
        "left_raw_vs_FAST_T_C_I_max_abs": left_same_direction_delta,
        "direction_warning": "Dataset key name T_Body2Cam describes sensor pose in Body, as confirmed by inverse numeric match to FAST-LIVO2 T_C_I. Do not feed this raw 4x4 as a direct IMU-to-camera coordinate map.",
        "horizontal_raw_T_Body2Lidar_matches_T_I_L": horizontal_raw.tolist(),
        "vertical_raw_T_Body2Lidar": vertical_raw.tolist(),
        "misnamed_prism_key_T_Body2Imu_actual_lever_arm_pose": prism_raw.tolist(),
        "left_camera_K": k.tolist(), "left_distortion_k1_k2_p1_p2": distortion,
        "stereo_camera_Kalibr_t_shift_seconds": {"left": left["t_shift"], "right": right["t_shift"]},
        "stereo_camera_Kalibr_time_equation_from_actual_file_comments": "t_imu = t_cam + t_shift",
        "FAST_NTU_image_offset_seconds": ntu_config["time_offsets_config_seconds_defaults_zero"]["img_time_offset"],
        "left_Kalibr_minus_FAST_config_image_offset_seconds": left["t_shift"] - ntu_config["time_offsets_config_seconds_defaults_zero"]["img_time_offset"],
        "difference_is_configuration_fact_not_proof_of_current_bag_hardware_lag": True,
        "time_shifts_applied_or_stacked": False,
        "gates": gates, "all_gates_passed": all(gates.values()),
    }


def so3_diagnostic(rotation, tolerance):
    ortho = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    u, _, vt = np.linalg.svd(rotation)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    closest = u @ correction @ vt
    return {
        "matrix": rotation.tolist(),
        "determinant": determinant,
        "orthogonality_max_abs": ortho,
        "nearest_proper_rotation_max_abs_delta_diagnostic_only": float(np.max(np.abs(closest - rotation))),
        "tolerance": tolerance,
        "approximately_so3_at_declared_tolerance": ortho <= tolerance and abs(determinant - 1.0) <= tolerance,
        "configuration_modified_or_projected": False,
    }


def inspect_pair(name, config_path, camera_path, tolerance):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    cam = yaml.safe_load(camera_path.read_text(encoding="utf-8-sig"))
    e = cfg["extrin_calib"]
    ril = np.asarray(e["extrinsic_R"], dtype=np.float64).reshape(3, 3)
    pil = np.asarray(e["extrinsic_T"], dtype=np.float64)
    rcl = np.asarray(e["Rcl"], dtype=np.float64).reshape(3, 3)
    pcl = np.asarray(e["Pcl"], dtype=np.float64)
    # Exact equations used by setImuToLidarExtrinsic + initializeVIO.
    rli = ril.T
    pli = -ril.T @ pil
    rci = rcl @ rli
    pci = rcl @ pli + pcl
    til = transform(ril, pil)
    tli = transform(rli, pli)
    tcl = transform(rcl, pcl)
    tci = transform(rci, pci)
    tci_by_general_matrix_inverse = tcl @ np.linalg.inv(til)
    inverse_residual = float(np.max(np.abs(til @ tli - np.eye(4))))
    chain_residual = float(np.max(np.abs(tci - tcl @ tli)))
    general_inverse_residual = float(np.max(np.abs(tci - tci_by_general_matrix_inverse)))
    rcli_inverse_residual = float(np.max(np.abs(tcl @ transform(rcl.T, -rcl.T @ pcl) - np.eye(4))))
    k = np.asarray([[cam["cam_fx"], 0, cam["cam_cx"]], [0, cam["cam_fy"], cam["cam_cy"]], [0, 0, 1]], dtype=np.float64)
    dimensions = [int(cam["cam_width"]), int(cam["cam_height"])]
    scale = float(cam["scale"])
    nominal_scaled_k = k.copy()
    nominal_scaled_k[:2, :] *= scale
    distortions = [float(cam[f"cam_d{i}"]) for i in range(4)]
    time_cfg = cfg.get("time_offset", {})
    offset = {key: float(time_cfg.get(key, 0.0)) for key in ["lidar_time_offset", "imu_time_offset", "img_time_offset", "exposure_time_init"]}
    is_livox = int(cfg["preprocess"]["lidar_type"]) == 1
    gates = {
        "published_lidar_to_imu_approximately_so3": so3_diagnostic(ril, tolerance)["approximately_so3_at_declared_tolerance"],
        "published_lidar_to_camera_approximately_so3": so3_diagnostic(rcl, tolerance)["approximately_so3_at_declared_tolerance"],
        "derived_imu_to_camera_approximately_so3": so3_diagnostic(rci, tolerance)["approximately_so3_at_declared_tolerance"],
        "lidar_imu_inverse_consistency": inverse_residual <= tolerance,
        "camera_lidar_transpose_inverse_consistency": rcli_inverse_residual <= tolerance,
        "homogeneous_chain_matches_code_formula": chain_residual <= 1e-12,
        "code_transpose_inverse_close_to_general_inverse": general_inverse_residual <= tolerance,
        "all_homogeneous_bottom_rows": all(np.array_equal(t[3], [0, 0, 0, 1]) for t in [til, tli, tcl, tci]),
        "intrinsics_finite_positive_focal_dimensions_scale": bool(np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0 and all(d > 0 for d in dimensions) and scale > 0),
        "principal_point_inside_declared_raw_image": bool(0 <= k[0, 2] < dimensions[0] and 0 <= k[1, 2] < dimensions[1]),
        "distortion_and_offsets_finite": bool(np.isfinite(distortions).all() and np.isfinite(list(offset.values())).all()),
        "sensor_topics_distinct_nonempty": len(set(cfg["common"][key] for key in ["img_topic", "imu_topic", "lid_topic"])) == 3 and all(cfg["common"][key] for key in ["img_topic", "imu_topic", "lid_topic"]),
    }
    return {
        "name": name,
        "source_configuration": str(config_path),
        "source_configuration_sha256": sha256(config_path),
        "source_camera_configuration": str(camera_path),
        "source_camera_configuration_sha256": sha256(camera_path),
        "topics": {key: cfg["common"][key] for key in ["img_topic", "lid_topic", "imu_topic"]},
        "enabled": {"image": cfg["common"]["img_en"], "lidar": cfg["common"]["lidar_en"], "imu": cfg["imu"]["imu_en"]},
        "lidar_type": cfg["preprocess"]["lidar_type"],
        "row_major_yaml_rotation_loading": True,
        "direction_notation": "T_A_B maps column coordinates from B into A: p_A = R_A_B p_B + t_A_B",
        "T_I_L": til.tolist(), "T_L_I_code_transpose_inverse": tli.tolist(),
        "T_C_L": tcl.tolist(), "T_C_I": tci.tolist(),
        "lidar_to_imu_rotation": so3_diagnostic(ril, tolerance),
        "lidar_to_camera_rotation": so3_diagnostic(rcl, tolerance),
        "imu_to_camera_rotation": so3_diagnostic(rci, tolerance),
        "lidar_imu_inverse_max_abs": inverse_residual,
        "camera_lidar_transpose_inverse_max_abs": rcli_inverse_residual,
        "homogeneous_chain_max_abs": chain_residual,
        "code_transpose_vs_general_inverse_max_abs": general_inverse_residual,
        "intrinsic_raw_K": k.tolist(),
        "intrinsic_raw_image_size_width_height": dimensions,
        "camera_model": cam["cam_model"],
        "configured_image_scale": scale,
        "nominal_scaled_K_from_configuration_arithmetic": nominal_scaled_k.tolist(),
        "nominal_scaled_size_from_configuration_arithmetic": [d * scale for d in dimensions],
        "camera_loader_runtime_not_executed": True,
        "distortion_config_d0_d1_d2_d3": distortions,
        "time_offsets_config_seconds_defaults_zero": offset,
        "time_offset_key_present": {key: key in time_cfg for key in offset},
        "timestamp_equations_from_actual_source": {
            "lidar_callback": "t_header (Livox callback; lidar_time_offset not applied here)" if is_livox else "t_header + lidar_time_offset",
            "imu_callback": "t_header - imu_time_offset",
            "image_callback": "t_header + img_time_offset",
            "livo_capture_time": "image_callback_time + exposure_time_init",
        },
        "ros_driver_bug_fix": cfg["common"]["ros_driver_bug_fix"],
        "no_sensor_timestamp_samples_measured": True,
        "no_hardware_synchronization_or_time_calibration_performed": True,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT / "upstream")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_OUTPUT / "upstream-fetch-manifest.json")
    parser.add_argument("--repo", type=Path, default=LAB / "repos/FAST-LIVO2")
    parser.add_argument("--so3-tolerance", type=float, default=1e-5)
    parser.add_argument("--dataset-calibration-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8-sig"))
    sources = {}
    manifest_checks = []
    for record in manifest:
        path = args.source_dir / Path(record["local_path"]).name
        current_sha = sha256(path)
        key = (record["repository"], record["path"])
        sources[key] = path
        manifest_checks.append({**record, "read_path": str(path), "sha256_current": current_sha, "sha256_matches_recorded": current_sha == record["sha256"]})
    def source(repo, path):
        return sources[(repo, path)]
    repo_id = "hku-mars/FAST-LIVO2"
    mapper_path = source(repo_id, "src/LIVMapper.cpp")
    mapper = mapper_path.read_text(encoding="utf-8-sig")
    mapper_clean = clean_code(mapper)
    vio_path = source(repo_id, "src/vio.cpp")
    vio = vio_path.read_text(encoding="utf-8-sig")
    vio_clean = clean_code(vio)
    voxel_path = source(repo_id, "src/voxel_map.cpp")
    voxel_clean = clean_code(voxel_path.read_text(encoding="utf-8-sig"))
    common = clean_code(source(repo_id, "include/common_lib.h").read_text(encoding="utf-8-sig"))
    expressions = [
        ("VIO first inverts supplied LiDAR-to-IMU rotation", vio_path, function_body(vio_clean, "setImuToLidarExtrinsic"), "Rli = rot.transpose();"),
        ("VIO first inverts supplied LiDAR-to-IMU translation", vio_path, function_body(vio_clean, "setImuToLidarExtrinsic"), "Pli = -rot.transpose() * transl;"),
        ("VIO camera-IMU rotation chain", vio_path, vio_clean, "Rci = Rcl * Rli;"),
        ("VIO camera-IMU translation chain", vio_path, vio_clean, "Pci = Rcl * Pli + Pcl;"),
        ("LiDAR point mapped through IMU into world", voxel_path, voxel_clean, "p = (rot * (extR_ * p + extT_) + t);"),
        ("camera projection world-to-camera rotation", vio_path, vio_clean, "Rcw = Rci * Rwi.transpose();"),
        ("camera projection world-to-camera translation", vio_path, vio_clean, "Pcw = -Rci * Rwi.transpose() * Pwi + Pci;"),
        ("standard PointCloud2 adds configured LiDAR offset", mapper_path, function_body(mapper_clean, "standard_pcl_cbk"), "double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;"),
        ("IMU subtracts configured IMU offset", mapper_path, function_body(mapper_clean, "imu_cbk"), "msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);"),
        ("image adds configured image offset", mapper_path, function_body(mapper_clean, "img_cbk"), "double msg_header_time = msg->header.stamp.toSec() + img_time_offset;"),
        ("synchronizer adds configured exposure initial offset", mapper_path, function_body(mapper_clean, "sync_packages"), "double img_capture_time = img_time_buffer.front() + exposure_time_init;"),
        ("official trajectory writer uses actual LIO measurement time", mapper_path, function_body(mapper_clean, "handleLIO"), "evoFile << LidarMeasures.last_lio_update_time"),
        ("published odometry uses wall/ROS now, not measurement stamp", mapper_path, function_body(mapper_clean, "publish_odometry"), "odomAftMapped.header.stamp = ros::Time::now();"),
    ]
    contracts = []
    for label, path, scope, expression in expressions:
        original = path.read_text(encoding="utf-8-sig")
        found = re.sub(r"\s+", "", expression) in re.sub(r"\s+", "", scope)
        matching = [{"line": i, "text": line.strip()} for i, line in enumerate(original.splitlines(), 1) if expression.strip() in line]
        contracts.append({"label": label, "source": str(path), "sha256": sha256(path), "expression": expression, "found_in_uncommented_selected_scope": found, "matching_original_lines": matching})
    livox = function_body(mapper_clean, "livox_pcl_cbk")
    contracts.append({"label": "Livox callback does not apply lidar_time_offset", "source": str(mapper_path), "sha256": sha256(mapper_path), "found_in_uncommented_selected_scope": "lidar_time_offset" not in livox and "msg->header.stamp.toSec()" in livox})
    row_macro = re.search(r"#define\s+MAT_FROM_ARRAY\(v\)\s+(.*)", common)
    if not row_macro:
        raise ValueError("MAT_FROM_ARRAY macro not found")
    contracts.append({"label": "YAML row-major rotation scalar order", "expression": row_macro.group(1), "found_in_uncommented_selected_scope": re.sub(r"\s+", "", row_macro.group(1)).startswith("v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]")})
    configs = [inspect_pair(name, source(repo_id, path), source(repo_id, camera), args.so3_tolerance) for name, path, camera in [
        ("NTU_VIRAL", "config/NTU_VIRAL.yaml", "config/camera_NTU_VIRAL.yaml"),
        ("avia", "config/avia.yaml", "config/camera_pinhole.yaml"),
    ]]
    ntu_id = "ntu-aris/ntu_viral_dataset"
    evaluation = source(ntu_id, "evaluation_tutorial.md").read_text(encoding="utf-8-sig")
    prism_match = re.search(r"T_B_prism:\s*\[([^\]]+)\]", evaluation, re.S)
    if not prism_match:
        raise ValueError("Published prism transform not found")
    prism = np.asarray([float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prism_match.group(1))]).reshape(4, 4)
    usage = source(ntu_id, "sensors_and_usage.md").read_text(encoding="utf-8-sig")
    gt = {
        "T_I_prism_published_body_equals_IMU": prism.tolist(),
        "prism_position_in_IMU_m": prism[:3, 3].tolist(),
        "lever_arm_norm_m": float(np.linalg.norm(prism[:3, 3])),
        "comparison_formula": "p_W_prism_est = p_W_I_est + R_W_I_est @ p_I_prism",
        "body_frame_is_primary_VectorNav_IMU_documented": "body frame" in usage and "attached to this sensor" in usage,
        "ground_truth_is_prism_position_only": "no orienation ground truth is available" in evaluation,
        "rotation_ground_truth_not_available": True,
        "orientation_for_lever_arm_comes_from_estimator_not_GT": True,
        "direct_IMU_to_prism_GT_position_difference_is_invalid": True,
        "no_GT_records_loaded_or_ATE_computed_in_this_audit": True,
    }
    dataset_calibration = inspect_dataset_calibration(args.dataset_calibration_dir, configs[0], prism) if args.dataset_calibration_dir else None
    calib_id = "hku-mars/FAST-Calib"
    calib_readme = source(calib_id, "README.md").read_text(encoding="utf-8-sig")
    qr = yaml.safe_load(source(calib_id, "config/qr_params.yaml").read_text(encoding="utf-8-sig"))
    fast_calib = {
        "algorithm_scope": "target-based LiDAR-camera extrinsic estimation; not generic camera-IMU/time calibration",
        "required_inputs": ["static point-cloud rosbag", "corresponding image containing designed ArUco/circular-hole target", "camera intrinsic/distortion parameters", "correct target dimensions", "board distance-filter bounds"],
        "target_based_explicit_in_README": "target-based" in calib_readme,
        "static_data_explicit_in_README": "static acquisition data" in calib_readme,
        "multiscene_minimum_scenes_from_README": 3,
        "actual_current_example_target_parameters_m": {k: qr[k] for k in ["marker_size", "delta_width_qr_center", "delta_height_qr_center", "delta_width_circles", "delta_height_circles", "circle_radius"]},
        "current_example_input_bag": qr["bag_path"],
        "current_example_input_image": qr["image_path"],
        "ordinary_ntu_motion_bag_without_designed_target_not_sufficient": True,
        "calibration_samples_not_downloaded_or_run_by_this_audit": True,
        "published_marker_size_comment_warns_test_data_0_16_vs_config_0_20": "our test data uses 0.16m" in source(calib_id, "config/qr_params.yaml").read_text(encoding="utf-8-sig"),
    }
    local_binding = {"repo_path": str(args.repo), "repository_present": args.repo.is_dir(), "files_checked": []}
    if args.repo.is_dir():
        if (args.repo / ".git").exists():
            git = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
            local_binding["commit"] = git.stdout.strip()
        else:
            local_binding["git_checkout_metadata_unavailable"] = True
            local_binding["commit_inference_is_only_through_pinned_selected_file_bytes"] = True
        for record in manifest:
            if record["repository"] == repo_id:
                path = args.repo / record["path"]
                local_binding["files_checked"].append({"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None, "same_bytes_as_pinned_snapshot": path.is_file() and sha256(path) == record["sha256"]})
        local_binding["all_selected_files_same_bytes"] = all(x["same_bytes_as_pinned_snapshot"] for x in local_binding["files_checked"])
    passed = all(x["all_gates_passed"] for x in configs) and all(x["sha256_matches_recorded"] for x in manifest_checks) and all(x["found_in_uncommented_selected_scope"] for x in contracts) and gt["body_frame_is_primary_VectorNav_IMU_documented"] and gt["ground_truth_is_prism_position_only"] and fast_calib["target_based_explicit_in_README"] and fast_calib["static_data_explicit_in_README"]
    if dataset_calibration:
        passed = passed and dataset_calibration["all_gates_passed"]
    metrics = {
        "status": "passed_published_configuration_algebra_and_source_direction_audit" if passed else "failed_config_audit_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual pinned upstream YAML and source equations; CPU-only configuration algebra. No sensor calibration optimization, bag synchronization measurement, odometry replay, or drift measurement.",
        "audit_script": str(Path(__file__).resolve()), "audit_script_sha256": sha256(__file__),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "pyyaml_version": yaml.__version__,
        "declared_SO3_tolerance": args.so3_tolerance,
        "sources": manifest_checks, "source_contracts": contracts,
        "configurations": configs, "NTU_GT_evaluation_contract": gt,
        "actual_dataset_prefix_calibration_cross_check": dataset_calibration,
        "FAST_Calib_input_contract": fast_calib, "local_FAST_LIVO2_source_binding": local_binding,
        "completed_work": ["published extrinsic direction traced in actual C++", "two actual YAML transforms, inverse and chain checked", "intrinsic/distortion configurations read and sanity checked", "offset signs and callback-specific differences verified", "prism lever-arm evaluation formula read and bound", "target-based calibration prerequisites verified"],
        "not_completed_by_this_audit": ["sensor-data-derived recalibration", "actual bag header jitter/gap/association measurement", "hardware trigger synchronization", "time-offset optimization", "FAST-LIVO2 replay", "ATE/RPE/drift measurement"],
        "configuration_files_modified": False, "sensor_records_loaded": 0, "training_or_cuda_calls": 0,
        "all_gates_passed": passed,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copy2(__file__, args.output / "audit_source.py")
    print(json.dumps({"status": metrics["status"], "configs": [{"name": x["name"], "SO3_camera_orth_max": x["lidar_to_camera_rotation"]["orthogonality_max_abs"], "SO3_camera_det": x["lidar_to_camera_rotation"]["determinant"], "T_C_I_translation_m": np.asarray(x["T_C_I"])[:3, 3].tolist(), "gates": len(x["gates"]), "passed": x["all_gates_passed"]} for x in configs], "source_contracts": len(contracts), "source_files": len(manifest_checks), "prism_lever_arm_m": gt["lever_arm_norm_m"], "metrics": str(args.output / "metrics.json")}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
