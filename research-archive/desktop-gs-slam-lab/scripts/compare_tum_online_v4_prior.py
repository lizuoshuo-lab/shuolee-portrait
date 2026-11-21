"""CPU-only comparison of the two locked V4 prior runs; never invokes training.

All GT-derived quantities are post-hoc analysis. Warmup tensors are compared by
dtype, shape and raw bytes, not by seed or by serialization-container hashes.
The final maps have different insertions, so a full rowwise subtraction would
not have a correspondence interpretation. Only their persistent first-frame
cohort is compared rowwise; whole-map nearest distances are descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("tum-online-v4-prior0-transfer100-001", "tum-online-v4-prior001-transfer100-001")
V4_SHA = "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f"
PRIOR_SHA = "d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b"
METRICS = ("raw_anchored_translation_rmse_m", "ate_rmse_m_se3_no_scale",
           "adjacent_rpe_translation_rmse_m", "adjacent_rpe_rotation_rmse_degrees",
           "raw_anchored_rotation_rmse_degrees")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite comparison data")
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "rmse": float(np.sqrt(np.mean(a*a))), "min": float(a.min()),
            "max": float(a.max()), "p95": float(np.percentile(a, 95))}


def tensor_diff(x, y):
    a, b = np.ascontiguousarray(array(x)), np.ascontiguousarray(array(y))
    assert a.shape == b.shape and a.dtype == b.dtype
    d = a.astype(np.float64)-b.astype(np.float64)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "byte_identical": a.tobytes() == b.tobytes(),
            "sha256_a": hashlib.sha256(a.tobytes()).hexdigest(),
            "sha256_b": hashlib.sha256(b.tobytes()).hexdigest(),
            "unequal_value_count": int(np.count_nonzero(a != b)),
            "value_count": int(a.size), "max_abs_difference": float(np.abs(d).max()),
            "mean_abs_difference": float(np.abs(d).mean()),
            "rms_difference": float(np.sqrt(np.mean(d*d)))}


def rotation_difference_degrees(a, b):
    # Rotation.from_matrix projects tiny float32 orthogonality drift. It is used
    # for reporting only, never for changing the saved poses or selecting one.
    return np.degrees(Rotation.from_matrix(np.swapaxes(a, -1, -2) @ b).magnitude())


def metric_check(estimated, truth):
    gt = np.linalg.inv(truth[0]) @ truth
    p, q = estimated[:, :3, 3], gt[:, :3, 3]
    x, y = p-p.mean(0), q-q.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    s = np.eye(3)
    s[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ s @ u.T
    aligned = x @ r.T + q.mean(0)
    gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
    est_rel = np.linalg.inv(estimated[:-1]) @ estimated[1:]
    rel_error = np.linalg.inv(gt_rel) @ est_rel
    return {
        METRICS[0]: stats(np.linalg.norm(p-q, axis=1))["rmse"],
        METRICS[1]: stats(np.linalg.norm(aligned-q, axis=1))["rmse"],
        METRICS[2]: stats(np.linalg.norm(rel_error[:, :3, 3], axis=1))["rmse"],
        METRICS[3]: stats(np.degrees(Rotation.from_matrix(rel_error[:, :3, :3]).magnitude()))["rmse"],
        METRICS[4]: stats(rotation_difference_degrees(gt[:, :3, :3], estimated[:, :3, :3]))["rmse"],
    }


def validate_same_experiment(metrics, contexts):
    a, b = metrics
    for d in metrics:
        assert d["status"] == "completed" and d["implementation_version"] == 4
        assert d["frames_processed"] == 100
        assert d["selected_frame_ids"] == list(range(200, 300))
        assert d["script_sha256"] == V4_SHA
        assert d["pose_prior_helper_sha256"] == PRIOR_SHA
    changed = {k: [a["configuration"][k], b["configuration"][k]] for k in a["configuration"]
               if a["configuration"][k] != b["configuration"][k]}
    assert set(changed) == {"run", "pose_prior_weight"}
    assert changed["pose_prior_weight"] == [0.0, .01]
    for key in ("selected_sensor_metadata", "association", "camera", "loader_sha256", "pose_helper_sha256"):
        assert a[key] == b[key], key
    assert a["icp_initializer"]["configuration"] == b["icp_initializer"]["configuration"]
    assert a["icp_initializer"]["helper_sha256"] == b["icp_initializer"]["helper_sha256"]
    for d, context in zip(metrics, contexts):
        assert context["source"]["script_sha256"] == V4_SHA
        assert context["source"]["pose_prior_helper_sha256"] == PRIOR_SHA
        assert context["configuration"] == d["configuration"]
        assert not context["ground_truth_included"]
        assert not context["future_sensor_observations_included"]
        assert not context["future_maps_or_trajectories_included"]
        assert context["sensor_metadata"]["frame_id"] == 201
    return changed


def summarize_run(metrics, checkpoint):
    frames = metrics["frame_metrics"]
    gs = array(checkpoint["estimated_c2w"])
    prefix = array(checkpoint["icp_prefix_c2w_float64"])
    truth = array(checkpoint["evaluation_only_gt_c2w"])
    assert len(gs) == len(prefix) == len(frames) == 100
    recomputed, differences = {}, {}
    for key, poses in (("trajectory", gs), ("icp_prefix_trajectory_evaluation", prefix)):
        recomputed[key] = metric_check(poses, truth)
        differences[key] = {k: recomputed[key][k]-metrics[key][k] for k in METRICS}
        for k, d in differences[key].items():
            assert abs(d) < (1e-4 if "degrees" in k else 1e-7), (key, k, d)
    gt = np.linalg.inv(truth[0]) @ truth
    gs_error = np.linalg.norm(gs[:, :3, 3]-gt[:, :3, 3], axis=1)
    prefix_error = np.linalg.norm(prefix[:, :3, 3]-gt[:, :3, 3], axis=1)
    corr_m = np.linalg.norm(gs[:, :3, 3]-prefix[:, :3, 3], axis=1)
    corr_deg = rotation_difference_degrees(prefix[:, :3, :3], gs[:, :3, :3])
    rows, identities = [], []
    for i, f in enumerate(frames):
        row = {"local_index": i, "frame_id": f["frame_id"],
               "raw_gt_translation_error_m": float(gs_error[i]),
               "icp_prefix_raw_gt_translation_error_m": float(prefix_error[i]),
               "posthoc_gt_error_change_vs_prefix_m": float(gs_error[i]-prefix_error[i]),
               "prefix_to_gs_center_correction_m": float(corr_m[i]),
               "prefix_to_gs_rotation_correction_degrees": float(corr_deg[i]),
               "point_count": f["point_count"], "new_points_added": f["new_points_added"],
               "failure_flags": f["failure_flags"],
               "map_coverage_warning_flags": f.get("map_coverage_warning_flags", []),
               "icp_failure_flags": f["icp"].get("failure_flags", []),
               "pose_rejected": f.get("pose_rejected", False),
               "gs_pose_refinement_applied": f["gs_pose_refinement_applied"],
               "gs_refinement_steps_completed": f.get("gs_refinement_steps_completed", 0)}
        if i:
            initial, selected = f["initial_loss"], f["optimized_loss_before_mapping"]
            for label, losses in (("initial", initial), ("selected_before_mapping", selected)):
                for term in ("sensor_objective", "prior_objective", "objective", "coverage"):
                    row[label+"_"+term] = losses[term]
                identities.append(abs(losses["objective"]-losses["sensor_objective"]-losses["prior_objective"]))
            row["sensor_reduction_before_mapping"] = initial["sensor_objective"]-selected["sensor_objective"]
            row["combined_reduction_before_mapping"] = initial["objective"]-selected["objective"]
            row["accepted_sensor_objective_after_mapping"] = f["accepted_pose_loss_after_mapping"]["objective"]
            row["selected_rgb_mse_before_mapping"] = selected["rgb_mse"]
            row["selected_depth_mae_m_before_mapping"] = selected["depth_mae_m"]
            row["selected_alpha_loss_before_mapping"] = selected["alpha_loss"]
        rows.append(row)
    nonanchor = rows[1:]
    loss_fields = ("initial_sensor_objective", "selected_before_mapping_sensor_objective",
                   "selected_before_mapping_prior_objective", "selected_before_mapping_objective",
                   "sensor_reduction_before_mapping", "combined_reduction_before_mapping",
                   "accepted_sensor_objective_after_mapping", "selected_before_mapping_coverage")
    eps = 1e-12
    worsen = [r for r in nonanchor if r["posthoc_gt_error_change_vs_prefix_m"] > eps]
    better = [r for r in nonanchor if r["posthoc_gt_error_change_vs_prefix_m"] < -eps]
    assert max(identities) < 1e-7
    return {
        "run": metrics["experiment_id"], "lambda": metrics["configuration"]["pose_prior_weight"],
        "trajectory": {k: metrics["trajectory"][k] for k in METRICS},
        "icp_prefix_trajectory": {k: metrics["icp_prefix_trajectory_evaluation"][k] for k in METRICS},
        "cpu_recomputed_trajectory_metric_differences": differences,
        "nonanchor_frame_count": len(nonanchor),
        "prefix_to_gs_center_correction_m": stats(corr_m[1:]),
        "prefix_to_gs_rotation_correction_degrees": stats(corr_deg[1:]),
        "loss_statistics_over_nonanchor_frames": {key: stats([r[key] for r in nonanchor]) for key in loss_fields},
        "combined_minus_sensor_minus_weighted_prior_max_abs": max(identities),
        "posthoc_gt_comparison": {
            "definition": "raw anchored camera-center error after accepted GS pose minus independent ICP-prefix error; no per-trajectory alignment; excludes fixed anchor",
            "comparison_tolerance_m": eps, "worsened_frames": len(worsen), "improved_frames": len(better),
            "unchanged_frames": len(nonanchor)-len(worsen)-len(better),
            "worsened_frame_ids": [r["frame_id"] for r in worsen],
            "sensor_reduced_but_gt_worsened_frames": sum(r["sensor_reduction_before_mapping"] > eps for r in worsen),
            "combined_reduced_but_gt_worsened_frames": sum(r["combined_reduction_before_mapping"] > eps for r in worsen)},
        "flags": {"failure_frame_count": metrics["failure_frame_count"],
                  "icp_failed_frame_count": metrics["icp_failed_frame_count"],
                  "pose_rejected_count": sum(r["pose_rejected"] for r in rows),
                  "frame_flag_histogram": dict(Counter(x for r in rows for x in r["failure_flags"])),
                  "map_coverage_warning_histogram": dict(Counter(x for r in rows for x in r["map_coverage_warning_flags"])),
                  "mapping_event_failure_histogram": dict(Counter(x for e in metrics["mapping_events"] for x in e["failure_flags"]))},
        "initial_point_count": metrics["initial_point_count"], "final_point_count": metrics["final_point_count"],
        "inserted_point_count": sum(r["new_points_added"] for r in rows),
        "mapping_event_count": len(metrics["mapping_events"]),
        "tracking_steps_completed": sum(r["gs_refinement_steps_completed"] for r in rows),
        "frames": rows,
    }


def compare_maps(contexts, checkpoints):
    a, b = contexts
    warmup = {k: tensor_diff(a["parameters"][k], b["parameters"][k]) for k in a["parameters"]}
    inputs = {k: tensor_diff(a[k], b[k]) for k in ("image", "depth_m", "valid_depth_mask", "K", "initial_w2c", "prior_reference_w2c", "icp_prefix_c2w_float64")}
    n = len(a["parameters"]["means"])
    assert n == len(b["parameters"]["means"])
    assert a["source_frame_ids_per_gaussian"] == b["source_frame_ids_per_gaussian"] == [200]*n
    for c in checkpoints:
        assert c["source_frame_ids_per_gaussian"][:n] == [200]*n
    cohort = {k: tensor_diff(checkpoints[0]["parameters"][k][:n], checkpoints[1]["parameters"][k][:n]) for k in a["parameters"]}
    points = [array(c["parameters"]["means"]) for c in checkpoints]
    nearest = {"a_to_b_m": stats(cKDTree(points[1]).query(points[0], workers=1)[0]),
               "b_to_a_m": stats(cKDTree(points[0]).query(points[1], workers=1)[0]),
               "interpretation": "descriptive distances between learned centers, not accuracy, surface distance, or matched point displacement"}
    return {
        "first_warmup_parameters": warmup,
        "first_warmup_all_parameters_byte_identical": all(d["byte_identical"] for d in warmup.values()),
        "first_sensor_and_initialization_tensors": inputs,
        "first_sensor_and_initialization_all_byte_identical": all(d["byte_identical"] for d in inputs.values()),
        "first_warmup_same_source_gaussian_ids": True,
        "first_warmup_center_displacement_m": stats(np.linalg.norm(array(a["parameters"]["means"])-array(b["parameters"]["means"]), axis=1)),
        "final_persistent_first_frame_cohort": {"point_count": n, "parameters": cohort,
            "row_correspondence_basis": "locked V4 appends new points without deletion or reordering; original first-frame rows retained",
            "center_displacement_m": stats(np.linalg.norm(points[0][:n]-points[1][:n], axis=1))},
        "final_source_frame_gaussian_counts": [dict(sorted(Counter(c["source_frame_ids_per_gaussian"]).items())) for c in checkpoints],
        "final_whole_map_nearest_center_distances": nearest,
        "whole_final_map_rowwise_subtraction_performed": False,
    }


def make_figure(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.7))
    colors = ("#dc5c35", "#167a93")
    for run, color in zip(summary["runs"], colors):
        f = run["frames"]
        x = np.array([r["frame_id"] for r in f])
        label = f'lambda={run["lambda"]:g}'
        axes[0, 0].plot(x, [r["raw_gt_translation_error_m"]*1000 for r in f], label=label, color=color)
        axes[0, 1].plot(x[1:], [r["prefix_to_gs_center_correction_m"]*1000 for r in f[1:]], label=label, color=color)
        axes[1, 0].plot(x[1:], [r["sensor_reduction_before_mapping"] for r in f[1:]], label=label, color=color)
        axes[1, 1].plot(x[1:], [r["selected_before_mapping_prior_objective"] for r in f[1:]], label=label, color=color)
        axes[2, 0].step(x, [r["point_count"] for r in f], where="post", label=label, color=color)
    f = summary["runs"][0]["frames"]
    axes[0, 0].plot([r["frame_id"] for r in f], [r["icp_prefix_raw_gt_translation_error_m"]*1000 for r in f], color="#555555", linestyle="--", label="independent ICP prefix")
    metric_keys = METRICS[:3]
    x = np.arange(3)
    series = [("ICP prefix", summary["runs"][0]["icp_prefix_trajectory"], "#777777")]
    series.extend((f'lambda={r["lambda"]:g}', r["trajectory"], c) for r, c in zip(summary["runs"], colors))
    for j, (label, values, color) in enumerate(series):
        bars = axes[2, 1].bar(x+(j-1)*.25, [values[k]*1000 for k in metric_keys], width=.25, label=label, color=color)
        axes[2, 1].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[2, 1].set_xticks(x, ["Raw RMSE", "SE3 ATE", "Delta-1 RPE"])
    titles = ("Post-hoc raw camera-center error", "Accepted GS correction from ICP prefix",
              "Sensor loss reduction: initial minus selected", "Selected weighted pose-prior objective",
              "Gaussian map size", "Translation trajectory metrics")
    units = ("mm", "mm", "sensor objective units", "weighted normalized objective", "Gaussian count", "mm")
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        if ax is not axes[2, 1]:
            ax.set_xlabel("associated frame ID")
    axes[1, 0].axhline(0, color="#666666", linewidth=.7)
    fig.suptitle("Locked V4: lambda 0 vs 0.01 on TUM fr1_desk frames 200-299", fontsize=15, y=.995)
    fig.text(.02, .014, "Same code/configuration except lambda and run name; warmup maps are NOT byte-identical.\nSingle seed and one clip; post-hoc GT only. Loss reduction and zero flags do not prove pose accuracy.", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .978))
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tum-online-v4-prior-comparison-001")
    args = parser.parse_args()
    if Path(args.output).name != args.output or args.output in ("", ".", ".."):
        raise ValueError("Output must be one directory name")
    assert not torch.cuda.is_initialized(), "CPU-only analyzer"
    assert sha(ROOT/"scripts"/"train_tum_online_gs_v4.py") == V4_SHA
    assert sha(ROOT/"scripts"/"geometric_pose_prior.py") == PRIOR_SHA
    directories = [ROOT/"outputs"/name for name in RUNS]
    for p in directories:
        if not (p/"metrics.json").exists():
            raise SystemExit(f"Run has no completed metrics yet: {p.name}; no polling performed")
    metrics = [read_json(p/"metrics.json") for p in directories]
    context_json = [read_json(p/"first_tracking_context.json") for p in directories]
    contexts = [torch.load(p/"first_tracking_context.pt", map_location="cpu", weights_only=True) for p in directories]
    checkpoints = [torch.load(p/"online_map.pt", map_location="cpu", weights_only=True) for p in directories]
    changed = validate_same_experiment(metrics, contexts)
    for p, meta, c in zip(directories, context_json, contexts):
        assert sha(p/"first_tracking_context.pt") == meta["checkpoint_sha256"]
        assert meta["tracking_optimizer_steps_executed_before_save"] == 0
        assert meta["saved_before_first_pose_gradient_probe"]
        assert meta["source"] == c["source"]
    assert torch.equal(checkpoints[0]["icp_prefix_c2w_float64"], checkpoints[1]["icp_prefix_c2w_float64"])
    assert torch.equal(checkpoints[0]["evaluation_only_gt_c2w"], checkpoints[1]["evaluation_only_gt_c2w"])
    output = ROOT/"outputs"/args.output
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "study": args.output, "status": "completed_cpu_posthoc_comparison",
        "source": {"analysis_script_sha256": sha(Path(__file__)), "v4_script_sha256": V4_SHA, "pose_prior_helper_sha256": PRIOR_SHA,
                   "inputs": [{"run": p.name, "sha256": {name: sha(p/name) for name in ("metrics.json", "configuration.json", "first_tracking_context.json", "first_tracking_context.pt", "online_map.pt")}} for p in directories]},
        "configuration_differences": changed,
        "matched_sensor_metadata_and_association": True, "icp_prefix_tensor_exact_equal": True,
        "evaluation_only_gt_tensor_exact_equal": True,
        "map_comparison": compare_maps(contexts, checkpoints),
        "runs": [summarize_run(m, c) for m, c in zip(metrics, checkpoints)],
        "interpretation": [
            "Comparison contains only the same locked V4 source at lambda 0 and 0.01; no V3 result is an ablation control.",
            "Map tensor equality is measured, not inferred from seed. Different first warmup parameters confound strict fixed-state causal attribution to lambda.",
            "Each frame's initial and selected loss use the same current map within that run. Cross-run current maps diverge; lower cross-run sensor loss is not a common-map objective comparison.",
            "GT appears only in final/post-hoc evaluation. Worsening counts compare raw anchored center errors against the independent ICP prefix; 99 nonanchor frames, not independent repetitions.",
            "Selected prior objective already includes lambda; combined objective equals sensor plus this weighted prior. Mapping uses sensor only.",
            "Zero flags are protocol observations, not accuracy or convergence certificates. No speed, covariance calibration, significance, or cross-scene generalization claim.",
            "One seed, one 100-frame clip, no repeated runs; this analysis chooses no further lambda or training hyperparameters."
            ,"This fr1_desk start-200 clip was already used for diagnosis; it is not an unseen evaluation scene or held-out clip."
        ],
        "artifacts": {"figure": "comparison.png"},
        "cpu_only": True, "cuda_initialized": torch.cuda.is_initialized(),
    }
    assert not result["cuda_initialized"]
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    assert len(encoded) <= 1024*1024
    (output/"comparison.json").write_bytes(encoded)
    make_figure(result, output/"comparison.png")
    print(json.dumps({"output": str(output), "comparison_json_bytes": len(encoded),
                      "warmup_byte_identical": result["map_comparison"]["first_warmup_all_parameters_byte_identical"],
                      "runs": [{k: r[k] for k in ("run", "trajectory", "posthoc_gt_comparison", "final_point_count", "flags")} for r in result["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
