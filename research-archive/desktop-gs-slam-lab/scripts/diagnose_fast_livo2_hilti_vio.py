"""Read-only exp14 VIO state transitions; no ROS, GT fitting or old-file edits.

Run with the existing LAB Python. Every execution creates a unique output.
The diagnostic parses the frozen mixed LIO/VIO text streams independently;
it never imports or executes the original runner/evaluator.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIO = "exp14_basement_2_lio_20261010T142104Z_c9bd3db8"
LIVO = "exp14_basement_2_livo_20261010T142554Z_ef89e887"
SOURCE_SPANS = {
    "src/LIVMapper.cpp": [[219, 227], [267, 286], [288, 334], [336, 405], [478, 481], [940, 943], [981, 987], [1045, 1056], [1271, 1285]],
    "src/IMU_Processing.cpp": [[237, 253], [433, 444]],
    "src/vio.cpp": [[162, 177], [775, 802], [1520, 1536], [1636, 1685], [1786, 1825], [1855, 1862]],
    "include/utils/so3_math.h": [[68, 87]],
    "include/common_lib.h": [[182, 205], [215, 222]],
    "src/voxel_map.cpp": [[372, 405]],
}


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p: Path, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(ok, label, checks):
    checks[label] = bool(ok)
    if not ok:
        raise ValueError(f"Diagnostic integrity check failed: {label}")


def rotation_rpy(a):
    """Inverse of pinned RotMtoEuler away from its singular branch.

    Text angles are radians * 57.3 (not exact degrees). Divide by the exact
    source multiplier before constructing Rz(yaw) Ry(pitch) Rx(roll).
    """
    x, y, z = np.asarray(a, float) / 57.3
    sx, cx, sy, cy, sz, cz = math.sin(x), math.cos(x), math.sin(y), math.cos(y), math.sin(z), math.cos(z)
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]])


def rotation_quat(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def angle_deg(a, b):
    r = a.T @ b
    sin_angle = np.linalg.norm([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]) / 2
    cos_angle = np.clip((np.trace(r)-1)/2, -1, 1)
    return float(np.degrees(np.arctan2(sin_angle, cos_angle)))


def print_bound(value):
    """Half one 6-significant-digit text quantum, plus fixed 6-decimal TUM."""
    v = abs(float(value))
    return (0 if v == 0 else 0.5 * 10**(math.floor(math.log10(v))-5)) + 0.75e-6


def stats(v):
    a = np.asarray(v, float)
    if len(a) == 0:
        return {"count": 0}
    return {"count": len(a), "nonzero_at_logged_precision": int(np.count_nonzero(a)),
            "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()),
            "min": float(a.min()), "rms": float(np.sqrt(np.mean(a*a)))}


def ranks(v):
    a = np.asarray(v, float); order = np.argsort(a, kind="stable"); r = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        r[order[i:j]] = (i+j-1)/2
        i = j
    return r


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {"count": len(a), "pearson": None, "spearman": None}
    return {"count": len(a), "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(np.corrcoef(ranks(a), ranks(b))[0, 1]),
            "scope": "Descriptive, autocorrelated single replay; no significance or causal claim."}


def read_states(p, width):
    tokens = [line.split() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(a) != width for a in tokens):
        raise ValueError(f"Unexpected width in {p}")
    a = np.asarray([[float(v) for v in row] for row in tokens])
    if not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) < 0):
        raise ValueError(f"Nonfinite or unordered states in {p}")
    groups = collections.OrderedDict()
    for i, row in enumerate(tokens):
        groups.setdefault(Decimal(row[0]), []).append(i)
    return tokens, a, groups


def read_poses(p):
    tokens = [s.split() for s in p.read_text(encoding="utf-8").splitlines() if s.strip() and not s.startswith("#")]
    if any(len(s) != 8 for s in tokens):
        raise ValueError(f"Invalid TUM pose in {p}")
    a = np.array([[float(v) for v in s] for s in tokens])
    ns = np.array([int(Decimal(s[0])*1_000_000_000) for s in tokens], dtype=np.int64)
    if not np.isfinite(a).all() or np.any(np.diff(ns) <= 0):
        raise ValueError(f"Invalid pose ordering in {p}")
    return tokens, a, ns


def parse_stdout(p):
    text = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(encoding="utf-8"))
    blocks, lio_blocks, current, iterations, latest = [], [], None, [], None
    for n, s in enumerate(text.splitlines(), 1):
        match = re.search(r"Get image, its header time: ([0-9.]+)", s)
        if match:
            latest = match.group(1)
        match = re.search(r"\[ VIO \] Raw feature num: (\d+)", s)
        if match:
            current = {"stdout_line": n, "image_header": latest, "raw_features": int(match.group(1)),
                       "retrieved_candidates": None, "jacobian_s": None, "ekf_s": None}
            blocks.append(current)
        match = re.search(r"\[ VIO \] Retrieve (\d+) points", s)
        if match and current is not None:
            current["retrieved_candidates"] = int(match.group(1))
        for key, label in [("jacobian_s", "computeJacobian"), ("ekf_s", "updateEKF")]:
            match = re.search(r"-> " + label + r"\s*\|\s*([0-9.]+)", s)
            if match and current is not None:
                current[key] = float(match.group(1))
        match = re.search(r"\[ LIO \] Raw feature num: (\d+), downsampled feature num:(\d+) effective feature num: (\d+) average residual: ([0-9.eE+-]+)", s)
        if match:
            iterations.append({"line": n, "raw": int(match.group(1)), "downsampled": int(match.group(2)),
                               "effective": int(match.group(3)), "mean_abs_plane_residual_m": float(match.group(4))})
        if "[ LIO ] Update Voxel Map" in s:
            lio_blocks.append(iterations); iterations = []
    if iterations:
        raise ValueError("Unterminated LIO iteration block")
    return blocks, lio_blocks, {"lio_no_point": text.count("[ LIO ]: No point!!!"),
                               "vio_no_point": text.count("[ VIO ] No point!!!"),
                               "visual_residual_debug_output_present": bool(re.search(r"last_error\s*=|new_error\s*=", text))}


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty CSV")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def analyse(lab, native, out):
    base = lab / "outputs/fast-livo2-hilti-gt-001"
    run = base / "runs" / LIVO
    run_lio = base / "runs" / LIO
    official = run / "official-log-after"
    old_docs = [lab/"experiments/30_splatam_tracking_ablation_repeat.md", lab/"experiments/31_fast_livo2_hilti_gt.md",
                lab/"outputs/splatam-ablation-repeat-001/report.md", base/"report.md"]
    protected = [base/"evaluation/metrics.json", base/"evaluation/execution.json", base/"evaluation-plan.json",
                 base/"evaluation/preregistration.json", base/"actual-paired-independent-review.json",
                 base/"completion-record-independent-review.json", base/"evaluation/paired-errors.csv",
                 official/"mat_pre.txt", official/"mat_out.txt", run/"roslaunch.log",
                 official/"result"/(LIVO+".txt"), official/"image/image_poses.txt",
                 run_lio/"official-log-after/result"/(LIO+".txt"), run_lio/"roslaunch.log",
                 run/"source-before.json", run/"source-after.json", run/"status.json",
                 run_lio/"status.json", run/"visual-log-evidence.json", run/"actual-parameters.yaml",
                 *old_docs, *[native/s for s in SOURCE_SPANS]]
    audit = lab / "outputs/fast-livo2-gt-input-recovery-003/offline-audit-runs/20261010T094728Z_ae2afedf/receipt.json"
    protected.append(audit)
    before = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in protected}
    checks = {}
    origin_ns = json.loads(audit.read_text(encoding="utf-8"))["observed_topics"]["/hesai/pandar"]["first_header_ns"]
    origin = Decimal(origin_ns) / Decimal(1_000_000_000)
    source_receipt = json.loads((run/"source-before.json").read_text(encoding="utf-8"))
    frozen_source = {v["path"]: v["sha256"] for v in source_receipt["files"]}
    for rel in SOURCE_SPANS:
        require(digest(native/rel) == frozen_source[rel], "pinned_source_"+rel, checks)
    snippets = []
    for rel, spans in SOURCE_SPANS.items():
        lines = (native/rel).read_text(encoding="utf-8").splitlines()
        for a, b in spans:
            snippets.append({"path": rel, "start": a, "end": b, "sha256": digest(native/rel),
                             "text": "\n".join(f"{i+1}: {lines[i]}" for i in range(a-1, min(b, len(lines))))})
    require(np.allclose(rotation_rpy([0, 0, 0]), np.eye(3)), "known_identity_rpy", checks)
    require(abs(angle_deg(np.eye(3), rotation_rpy([57.3*math.pi/2, 0, 0]))-90) < 1e-10, "known_90_degree_rpy", checks)
    require(np.allclose(rotation_quat([0, 0, 0, 1]), np.eye(3)), "known_identity_xyzw", checks)
    require(np.array_equal(ranks([2, 1, 2]), [1.5, 0, 1.5]), "rank_ties_average", checks)
    tpre, pre, pg = read_states(official/"mat_pre.txt", 19)
    tout, post, og = read_states(official/"mat_out.txt", 20)
    primary_tokens, primary, primary_ns = read_poses(official/"result"/(LIVO+".txt"))
    image_tokens, image, image_ns = read_poses(official/"image/image_poses.txt")
    blocks, lio_iterations, early = parse_stdout(run/"roslaunch.log")
    lio_visual, lio_only_iterations, lio_early = parse_stdout(run_lio/"roslaunch.log")
    require(len(og) == len(primary) == 738, "all_livo_primary_groups", checks)
    require(len(blocks) == 737 and len(lio_iterations) == 738, "all_livo_stage_stdout_blocks", checks)
    require(not lio_visual and len(lio_only_iterations) == 739, "pure_lio_visual_absent_and_lio_blocks", checks)
    require(np.all(pre[:, 17:19] == 0) and np.all(post[:, 17:19] == 0), "padding_not_gravity", checks)
    transitions, lidar, max_position_rounding_ratio, max_rot_error = [], [], 0.0, 0.0
    for j, (elapsed, indices) in enumerate(og.items()):
        require(len(indices) in [1, 2] and elapsed in pg and len(pg[elapsed]) == len(indices), f"group_shape_{j}", checks)
        a = post[indices[0]]; prior = pre[pg[elapsed][0]]
        require(a[19] > 0, f"lio_nonempty_output_{j}", checks)
        delta_time = abs(float((Decimal(primary_tokens[j][0])-origin)-elapsed))
        require(delta_time <= print_bound(elapsed), f"source_elapsed_sensor_time_{j}", checks)
        position_diff = np.abs(a[4:7]-primary[j, 1:4])
        bound = np.array([print_bound(v) for v in a[4:7]])
        max_position_rounding_ratio = max(max_position_rounding_ratio, float(np.max(position_diff/bound)))
        require(np.all(position_diff <= bound), f"lio_out_primary_position_{j}", checks)
        max_rot_error = max(max_rot_error, angle_deg(rotation_rpy(a[1:4]), rotation_quat(primary[j, 4:8])))
        it = lio_iterations[j]
        require(len(it) > 0 and all(v["raw"] == int(a[19]) for v in it), f"lio_iteration_identity_{j}", checks)
        lidar.append({"cycle": j, "timestamp_ns": int(primary_ns[j]), "iterations": len(it),
                      "first_mean_abs_plane_residual_m": it[0]["mean_abs_plane_residual_m"],
                      "last_mean_abs_plane_residual_m": it[-1]["mean_abs_plane_residual_m"],
                      "first_effective_points": it[0]["effective"], "last_effective_points": it[-1]["effective"],
                      "net_lio_position_change_m": float(np.linalg.norm(a[4:7]-prior[4:7]))})
        if len(indices) == 1:
            require(j == len(og)-1, "sole_unpaired_final_lio", checks); continue
        vpre = pre[pg[elapsed][1]]; vpost = post[indices[1]]; block = blocks[j]
        require(np.array_equal(a[:19], vpre), f"exact_lio_out_vio_pre_continuity_{j}", checks)
        require(vpost[19] == 0 and block["raw_features"] == int(a[19]), f"vio_point_stage_identity_{j}", checks)
        require(block["image_header"] == primary_tokens[j][0], f"exact_image_header_primary_stamp_{j}", checks)
        require(block["jacobian_s"] is not None and block["ekf_s"] is not None, f"stage_timing_present_{j}", checks)
        difference = vpost[:19]-vpre
        transitions.append({"cycle": j, "sensor_timestamp_ns": int(primary_ns[j]), "elapsed_printed_s": float(elapsed),
                            "pre_line": pg[elapsed][1]+1, "post_line": indices[1]+1, "stdout_line": block["stdout_line"],
                            "net_position_change_m": float(np.linalg.norm(difference[4:7])),
                            "net_rotation_geodesic_deg": angle_deg(rotation_rpy(vpre[1:4]), rotation_rpy(vpost[1:4])),
                            "net_velocity_change_m_s": float(np.linalg.norm(difference[7:10])),
                            "net_gyro_bias_change_rad_s": float(np.linalg.norm(difference[10:13])),
                            "net_accel_bias_change_m_s2": float(np.linalg.norm(difference[13:16])),
                            "net_inverse_exposure_change": float(difference[16]),
                            "raw_features": block["raw_features"], "retrieved_candidates": block["retrieved_candidates"],
                            "jacobian_s": block["jacobian_s"], "ekf_s": block["ekf_s"],
                            "logged_state_changed": bool(np.any(difference[1:17] != 0)),
                            "accepted_internal_iterations": None, "photometric_residual_before": None, "photometric_residual_after": None})
    require(max_rot_error < .002, "rpy_reconstruction_matches_primary_quaternion_with_text_precision", checks)
    unmatched_pre = [{"printed_elapsed_s": str(t), "rows": len(v), "line_numbers": [i+1 for i in v]} for t, v in pg.items() if t not in og]
    require(sum(v["rows"] for v in unmatched_pre) == 4 and early["lio_no_point"] == early["vio_no_point"] == 2, "four_initial_pre_only_rows_accounted", checks)
    require(len(image) == 36, "post_vio_sample_count", checks)
    image_checks = []
    for j, ns in enumerate(image_ns):
        indices = np.flatnonzero(primary_ns[:-1] == ns)
        require(len(indices) == 1, f"sample_sensor_stamp_{j}", checks)
        cycle = int(indices[0]); outrow = post[list(og.values())[cycle][1]]
        require(cycle == (j+1)*20-1, f"sample_interval20_cycle_{j}", checks)
        bounds = np.array([print_bound(v) for v in outrow[4:7]])
        pdiff = np.abs(outrow[4:7]-image[j, 1:4])
        rdiff = angle_deg(rotation_rpy(outrow[1:4]), rotation_quat(image[j, 4:8]))
        require(np.all(pdiff <= bounds) and rdiff < .002, f"sample_equals_actual_vio_out_{j}", checks)
        image_checks.append({"sample": j, "cycle": cycle, "timestamp_ns": int(ns),
                             "maximum_position_component_rounding_difference_m": float(pdiff.max()), "rotation_text_difference_deg": rdiff})
    with (base/"evaluation/paired-errors.csv").open(encoding="utf-8", newline="") as f:
        paired = list(csv.DictReader(f))
    queries = np.array([int(r["timestamp_ns"]) for r in paired], np.int64)
    query_indices = np.array([int(r["original_GT_index"]) for r in paired])
    metrics = json.loads((base/"evaluation/metrics.json").read_text(encoding="utf-8"))
    require(len(paired) == metrics["common_reference_rows"] == 687, "frozen_paired_common_count", checks)
    require(query_indices.tolist() == metrics["common_reference_indices"], "unchanged_original_gt_indices", checks)
    errors = {}
    for mode in ["lio", "livo"]:
        errors[mode] = np.array([float(r[mode+"_aligned_position_error_m"]) for r in paired])
        rmse = float(np.sqrt(np.mean(errors[mode]**2)))
        require(abs(rmse-metrics["metrics"][mode]["ate_position_se3"]["rmse"]) < 1e-14, mode+"_frozen_csv_rmse", checks)
    gap = errors["livo"]-errors["lio"]
    gauges = {mode: np.array([float(r[mode+"_first_matched_pose_error_m"]) for r in paired]) for mode in ["lio", "livo"]}
    vio_ns = primary_ns[:-1]
    windows = []
    for j in range(1, len(queries)):
        if query_indices[j] != query_indices[j-1]+1 or queries[j]-queries[j-1] > 200_000_000:
            continue
        selected = np.flatnonzero((vio_ns >= queries[j-1]) & (vio_ns < queries[j]))
        rows = [transitions[i] for i in selected]
        windows.append({"previous_GT_index": int(query_indices[j-1]), "GT_index": int(query_indices[j]),
                        "previous_timestamp_ns": int(queries[j-1]), "timestamp_ns": int(queries[j]),
                        "vio_calls_in_left_closed_right_open_interval": len(rows),
                        "sum_net_vio_position_norm_m": sum(v["net_position_change_m"] for v in rows),
                        "sum_net_vio_rotation_norm_deg": sum(v["net_rotation_geodesic_deg"] for v in rows),
                        "mean_retrieved_candidates_when_logged": float(np.mean([v["retrieved_candidates"] for v in rows if v["retrieved_candidates"] is not None])) if any(v["retrieved_candidates"] is not None for v in rows) else None,
                        "paired_ATE_error_difference_change_m": float(gap[j]-gap[j-1]),
                        "paired_first_gauge_error_difference_change_m": float((gauges["livo"][j]-gauges["lio"][j])-(gauges["livo"][j-1]-gauges["lio"][j-1]))})
    require(len(windows) == 676, "same_original_adjacent_reference_support_676", checks)
    position = [v["net_position_change_m"] for v in transitions]
    rotation = [v["net_rotation_geodesic_deg"] for v in transitions]
    summary = {"net_vio_position_change_m": stats(position), "net_vio_rotation_geodesic_deg": stats(rotation),
               "net_vio_velocity_change_m_s": stats([v["net_velocity_change_m_s"] for v in transitions]),
               "net_gyro_bias_change_rad_s": stats([v["net_gyro_bias_change_rad_s"] for v in transitions]),
               "net_accel_bias_change_m_s2": stats([v["net_accel_bias_change_m_s2"] for v in transitions]),
               "absolute_net_inverse_exposure_change": stats([abs(v["net_inverse_exposure_change"]) for v in transitions]),
               "logged_net_state_changed_calls": sum(v["logged_state_changed"] for v in transitions),
               "positive_jacobian_timing_calls": sum(v["jacobian_s"] > 0 for v in transitions),
               "positive_ekf_timing_calls": sum(v["ekf_s"] > 0 for v in transitions),
               "nonzero_retrieve_calls": sum(v["retrieved_candidates"] is not None and v["retrieved_candidates"] > 0 for v in transitions),
               "paired_error_difference_m": stats(gap), "paired_livo_lower_ATE_point_count": int(np.count_nonzero(gap < 0)),
               "window_position_vs_paired_error_change": correlation([v["sum_net_vio_position_norm_m"] for v in windows], [v["paired_ATE_error_difference_change_m"] for v in windows]),
               "window_rotation_vs_paired_error_change": correlation([v["sum_net_vio_rotation_norm_deg"] for v in windows], [v["paired_ATE_error_difference_change_m"] for v in windows]),
               "lio_residual_scope": "Per-iteration mean absolute LiDAR point-to-plane distance; correspondences/effective counts may change. Not visual photometric residual or fixed objective descent.",
               "lio_residuals": {}}
    for mode, its in [("lio", lio_only_iterations), ("livo", lio_iterations)]:
        first = [r[0]["mean_abs_plane_residual_m"] for r in its]
        last = [r[-1]["mean_abs_plane_residual_m"] for r in its]
        summary["lio_residuals"][mode] = {"cycle_count": len(its), "iteration_counts": dict(collections.Counter(len(r) for r in its)),
                                          "first_logged_mean_abs_plane_m": stats(first), "last_logged_mean_abs_plane_m": stats(last),
                                          "last_less_than_first_cycles": sum(b < a for a, b in zip(first, last)),
                                          "last_greater_than_first_cycles": sum(b > a for a, b in zip(first, last))}
    elapsed_vio = (vio_ns-origin_ns)/1e9
    elapsed_query = (queries-origin_ns)/1e9
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax[0,0].plot(elapsed_vio, np.array(position)*1000, lw=.8); ax[0,0].set(ylabel="Net VIO position change (mm)", xlabel="Sensor time since first LiDAR (s)")
    ax[0,1].plot(elapsed_vio, rotation, lw=.8); ax[0,1].set(ylabel="Net VIO rotation change (deg)", xlabel="Sensor time since first LiDAR (s)")
    ax[1,0].plot(elapsed_query, gap*100, label="SE(3) APE: LIVO - LIO", lw=1)
    ax[1,0].plot(elapsed_query, (gauges['livo']-gauges['lio'])*100, label="First pose gauge: LIVO - LIO", lw=1)
    ax[1,0].axhline(0, color="black", lw=.5); ax[1,0].legend(fontsize=8); ax[1,0].set(xlabel="Original common reference sensor time (s)", ylabel="Frozen paired position error difference (cm)")
    ax[1,1].scatter(np.array([v["sum_net_vio_position_norm_m"] for v in windows])*1000, np.array([v["paired_ATE_error_difference_change_m"] for v in windows])*1000, s=8, alpha=.5)
    ax[1,1].set(xlabel="Sum net VIO position-change norms in interval (mm)", ylabel="Change of paired APE difference (mm)", title="676 original adjacent reference intervals; association only")
    for a in ax.flat: a.grid(alpha=.2)
    fig.suptitle("Actual exp14 LIVO state transitions and frozen paired errors\nNo new alignment, GT-time fit, ROS replay or visual-residual score")
    fig.savefig(out/"vio-state-and-paired-errors.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(elapsed_vio, [v['retrieved_candidates'] if v['retrieved_candidates'] is not None else np.nan for v in transitions], lw=.8)
    axes[0].set(ylabel="Logged retrieved visual candidates", title="Missing first retrieve line stays unknown; candidate count is not accepted-step count")
    axes[1].plot(elapsed_vio, [v['net_inverse_exposure_change'] for v in transitions], lw=.8)
    axes[1].set(ylabel="Net inverse-exposure state change", xlabel="Sensor time since first LiDAR (s)")
    for a in axes: a.grid(alpha=.2)
    fig.savefig(out/"vio-candidates-and-exposure.png", dpi=160); plt.close(fig)
    csv_write(out/"vio-transitions.csv", transitions); csv_write(out/"lio-update-diagnostics.csv", lidar)
    csv_write(out/"common-reference-signal-windows.csv", windows); csv_write(out/"sampled-post-vio-checks.csv", image_checks)
    write_json(out/"source-schema-evidence.json", {"sources": snippets, "origin_header_ns": origin_ns,
             "pre_columns": ["elapsed_since_first_lidar", "roll_times57.3", "pitch_times57.3", "yaw_times57.3", "px", "py", "pz", "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz", "inverse_exposure", "padding0", "padding0"],
             "out_additional_column": "feats_undistort point count, not visual accepted/candidate count"})
    after = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in protected}
    require(before == after, "all_protected_inputs_source_and_30_31_reports_unchanged", checks)
    outputs = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name not in ["execution.json", "protocol.json"]}
    result = {"status": "actual_read_only_vio_mechanism_diagnostic_completed_pending_independent_review", "source_run": LIVO,
              "primary_lio_rows": 739, "primary_livo_rows": 738, "pre_rows": len(pre), "out_rows": len(post),
              "exact_vio_pairs": len(transitions), "unmatched_initial_pre_rows": unmatched_pre,
              "final_lio_without_vio": 1, "post_vio_sample_rows": len(image), "sensor_origin_ns": origin_ns,
              "text_precision_checks": {"max_position_rounding_ratio": max_position_rounding_ratio, "max_rpy_to_primary_quat_difference_deg": max_rot_error},
              "counts_and_summary": summary, "checks": checks, "check_count": len(checks), "early_stdout": early,
              "inputs_before": before, "inputs_after": after, "outputs": outputs,
              "limits": ["Net logged state changes prove retained inter-stage state effects, not every internal iteration accepted or true-pose improvement.",
                         "Visual photometric residuals and internal accepted/rollback counts are not logged; remain null.",
                         "Euler text uses source factor57.3 and finite decimal precision; reconstructed rotation is approximate, verified against quaternion output.",
                         "Four pre-only rows occur during initialization; never paired by naive row number.",
                         "Associations use frozen original-reference intervals [previous,current); query primary poses were interpolated in the original evaluator.",
                         "Observed signals and paired error changes are autocorrelated, confounded by motion/texture/map/history; no causal or significance claim.",
                         "No extra GT interpolation, time/scale/lever-arm fit, calibration, parameter search, ROS/GPU or model replay."]}
    write_json(out/"metrics.json", result)
    lines = ["# FAST-LIVO2 exp14：实际 VIO 状态修正的只读诊断", "",
             "本诊断只读取冻结输出与源码；没有重新回放、评分、拟合 GT 时间或修改旧知识页。独立复核尚待根代理完成。", "",
             f"混合日志 mat_pre={len(pre)} 行、mat_out={len(post)} 行。按同一 elapsed 组和源码阶段匹配得到 {len(transitions)} 对 VIO 前后状态；737 对 LIO-out 与紧接 VIO-pre 逐列完全一致。四行初始化 pre-only 保留，末尾一条 LIO 没有后续 VIO。", "",
             "19 个 pre 列依次为 elapsed、Euler×57.3、position、velocity、gyro bias、accel bias、inverse exposure、两个零占位。out 的额外列是未畸变点云数，不是视觉候选或接受计数。", "",
             f"{summary['logged_net_state_changed_calls']}/{len(transitions)} 次 VIO 在记录精度下有净状态改变。位置净修正 median={summary['net_vio_position_change_m']['median']:.9f} m、p95={summary['net_vio_position_change_m']['p95']:.9f} m、max={summary['net_vio_position_change_m']['max']:.9f} m；旋转 geodesic median={summary['net_vio_rotation_geodesic_deg']['median']:.9f}°、p95={summary['net_vio_rotation_geodesic_deg']['p95']:.9f}°、max={summary['net_vio_rotation_geodesic_deg']['max']:.9f}°。", "",
             "737 个视觉 stdout 块的 image header 与对应 primary TUM 时刻完全匹配，raw 点数与对应 LIO out 完全匹配。36 个 post-VIO 采样匹配每20次 VIO，并与实际 VIO-out 状态在文本量化误差内一致。", "",
             "因此已有证据从“调用/候选阶段执行”推进到“视觉阶段确实留下净状态修正”；不能推进为每次内部 EKF 迭代接受、photometric residual 下降或真实精度改善。残差 debug 语句被注释，回退/接受分支没有实际计数。", "",
             "LIO 日志另有平均绝对点到平面残差，它随匹配集合和有效点数改变，是 LiDAR 几何量，不是 VIO photometric loss。首尾迭代统计保留在 metrics，不将下降比例当作固定目标函数优化证明。", "",
             "原687个参考点、676对相邻支持与旧评分全保留。额外关联只使用 [前参考时刻,当前参考时刻) 的真实 VIO 修正范数之和，比较冻结 (LIVO APE−LIO APE) 的变化；不生成新 alignment 或挑选时段。", "",
             f"676 个支持区间的位置修正/paired-error-change Pearson={summary['window_position_vs_paired_error_change']['pearson']:.6f}、Spearman={summary['window_position_vs_paired_error_change']['spearman']:.6f}。这些是自相关单序列的描述性关联，不能把修正大小解释成误差增减原因。", "",
             "图：vio-state-and-paired-errors.png、vio-candidates-and-exposure.png。逐对 CSV 与源列定义／时间收据在同一目录，可独立复算。全部保护输入、官方源码和30/31/report前后 SHA 相同。"]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or args.lab/"outputs/fast-livo2-hilti-vio-diagnostic-001"/(stamp+"_"+uuid.uuid4().hex[:8])
    if out.exists():
        raise SystemExit(f"Refusing existing diagnostic output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out/"source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    execution = {"status": "diagnostic_started", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "argv": sys.argv, "script_sha256": digest(Path(__file__)), "output": str(out),
                 "scope": "New independent CPU readback only, no original runner/evaluator imports or calls."}
    write_json(out/"execution.json", execution)
    write_json(out/"protocol.json", {"registered_before_computation": True, "phase_rule": "Strict grouped pre/out stage order plus exact LIO-out/VIO-pre equality, source bindings and image header/raw identity.",
              "angle_rule": "ZYX reconstruction after dividing printed Euler by exact source factor57.3; geodesic in exact degrees.",
              "time_rule": "Official first LiDAR header is origin, primary exact six-decimal sensor stamps bind VIO; no GT fit.",
              "correlation_rule": "All frozen676 contiguous original reference intervals <=0.2s, signal times in [previous,current), paired APE difference changes, no new score.",
              "unknowns": "Internal accepted iteration/rollback counts and visual photometric residuals remain null."})
    start = time.perf_counter()
    try:
        metrics = analyse(args.lab.resolve(), args.native, out)
        execution.update(status="diagnostic_completed", wall_seconds=time.perf_counter()-start,
                         finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), expected_process_returncode=0,
                         metrics_sha256=digest(out/"metrics.json"), all_protected_files_unchanged=True)
        write_json(out/"execution.json", execution)
        print(json.dumps({"status": metrics["status"], "output": str(out), "checks": metrics["check_count"],
                          "VIO_pairs": metrics["exact_vio_pairs"], "net_changed": metrics["counts_and_summary"]["logged_net_state_changed_calls"]}))
        return 0
    except Exception:
        (out/"failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        execution.update(status="diagnostic_failed_preserved", wall_seconds=time.perf_counter()-start,
                         finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), expected_process_returncode=1)
        write_json(out/"execution.json", execution)
        raise


if __name__ == "__main__":
    sys.exit(main())
