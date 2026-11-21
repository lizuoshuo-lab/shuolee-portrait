"""CPU post-hoc analysis of the preselected fr1_xyz V4 transfer pair.

Reuses read-only NumPy/SciPy analysis functions from the desk study. It never
invokes that study's main, alters its files, selects a new weight, or calls a
renderer. The standalone depth ICP run is checked against both GS prefixes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import compare_tum_online_v4_prior as common
from audit_tum_trajectory import parse_trajectory_text

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("tum-xyz-online-v4-prior0-transfer100-001", "tum-xyz-online-v4-prior001-transfer100-001")
DESK_RUNS = ("tum-online-v4-prior0-transfer100-001", "tum-online-v4-prior001-transfer100-001")
ICP_RUN = "tum-xyz-icp-transfer100-001"
OUTPUT = ROOT/"outputs/tum-xyz-online-v4-comparison-001"
ICP_SHA = "21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a"
LOADER_SHA = "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11"


def check_protocol(metrics, contexts, checkpoints, icp, icp_validation, icp_truth_text):
    config_diffs = []
    for d, context, checkpoint, desk_name, weight in zip(metrics, contexts, checkpoints, DESK_RUNS, (0., .01)):
        assert d["status"].startswith("completed") and d["implementation_version"] == 4
        assert d["frames_processed"] == 100 and d["selected_frame_ids"] == list(range(200, 300))
        assert d["configuration"]["pose_prior_weight"] == weight
        assert Path(d["configuration"]["dataset"]).resolve() == (ROOT/"data/tum/rgbd_dataset_freiburg1_xyz").resolve()
        assert d["script_sha256"] == common.V4_SHA and d["pose_prior_helper_sha256"] == common.PRIOR_SHA
        assert d["loader_sha256"] == LOADER_SHA and d["icp_initializer"]["helper_sha256"] == ICP_SHA
        desk = common.read_json(ROOT/"outputs"/desk_name/"configuration.json")
        differences = {k: [desk["configuration"][k], d["configuration"][k]]
                       for k in d["configuration"] if desk["configuration"][k] != d["configuration"][k]}
        assert set(differences) == {"run", "dataset"}
        config_diffs.append({"xyz_run": d["experiment_id"], "desk_reference_run": desk_name,
                             "changed_configuration_fields": sorted(differences),
                             "all_other_configuration_values_exact_equal": True})
        assert context["configuration"] == d["configuration"] == checkpoint["configuration"]
        assert context["source"]["script_sha256"] == common.V4_SHA
        assert context["source"]["pose_prior_helper_sha256"] == common.PRIOR_SHA
        assert not any(context[k] for k in ("ground_truth_included", "future_sensor_observations_included", "future_maps_or_trajectories_included"))
        assert d["selected_sensor_metadata"] == icp_validation["selected_sensor_metadata"]
        assert d["association"] == icp["association"]
        assert d["icp_initializer"]["configuration"] == icp["configuration"]["icp"]
        assert d["icp_initializer"]["helper_sha256"] == icp["configuration"]["source_sha256"]["track_rgbd_icp.py"]
    assert metrics[0]["selected_sensor_metadata"] == metrics[1]["selected_sensor_metadata"]
    pair_changes = [k for k in metrics[0]["configuration"] if metrics[0]["configuration"][k] != metrics[1]["configuration"][k]]
    assert sorted(pair_changes) == ["pose_prior_weight", "run"]
    assert metrics[0]["camera"] == metrics[1]["camera"]
    assert torch.equal(checkpoints[0]["evaluation_only_gt_c2w"], checkpoints[1]["evaluation_only_gt_c2w"])
    assert torch.equal(checkpoints[0]["icp_prefix_c2w_float64"], checkpoints[1]["icp_prefix_c2w_float64"])
    assert torch.equal(contexts[0]["K"], contexts[1]["K"])
    expected_K = np.array([[129.325, 0, 79.275], [0, 129.125, 63.45], [0, 0, 1.]])
    assert np.array_equal(np.asarray(metrics[0]["camera"]["K"]), expected_K)
    assert np.max(np.abs(common.array(contexts[0]["K"])-expected_K)) < 1e-5
    assert icp["configuration"]["clip"] == "long100"
    assert icp["configuration"]["start_frame"] == 200 and icp["configuration"]["frames"] == 100
    assert Path(icp["configuration"]["dataset"]).resolve() == (ROOT/"data/tum/rgbd_dataset_freiburg1_xyz").resolve()
    standalone_prefix = np.asarray([f["estimated_c2w"] for f in icp["frame_metrics"]])
    assert [f["frame_id"] for f in icp["frame_metrics"]] == list(range(200, 300))
    comparisons = []
    for d, checkpoint in zip(metrics, checkpoints):
        prefix = common.array(checkpoint["icp_prefix_c2w_float64"])
        difference = float(np.max(np.abs(prefix-standalone_prefix)))
        assert difference < 2e-12
        truth = common.array(checkpoint["evaluation_only_gt_c2w"])
        anchored = np.linalg.inv(truth[0]) @ truth
        truth_difference = float(np.max(np.abs(anchored-icp_truth_text["poses"])))
        assert truth_difference < 1e-8
        assert [float(x) for x in icp_truth_text["timestamps"]] == [s["rgb_timestamp"] for s in d["selected_sensor_metadata"]]
        comparisons.append({"run": d["experiment_id"], "standalone_ICP_prefix_max_abs_matrix_difference": difference,
                            "standalone_ICP_prefix_array_exact_equal": np.array_equal(prefix, standalone_prefix),
                            "anchored_GT_vs_standalone_text_max_abs_matrix_difference": truth_difference})
    return {"desk_to_xyz_configuration_checks": config_diffs,
            "pair_changed_configuration_fields": sorted(pair_changes),
            "dataset_frame_ids_sensor_metadata_association_and_GT_matched": True,
            "ICP_configuration_and_locked_source_matched": True,
            "full_prefix_and_evaluation_GT_tensors_exact_equal_between_GS_arms": True,
            "K_exact_equal_between_arms": True, "K_float64": expected_K.tolist(),
            "resolution": [160, 120], "original_resolution": [640, 480],
            "standalone_ICP_checks": comparisons}


def make_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.7))
    colors = ("#dc5c35", "#167a93")
    for run, color in zip(result["runs"], colors):
        rows = run["frames"]
        x = np.array([row["frame_id"] for row in rows])
        label = f'lambda={run["lambda"]:g}'
        axes[0, 0].plot(x, [f["raw_gt_translation_error_m"]*1000 for f in rows], color=color, label=label)
        axes[0, 1].plot(x[1:], [f["prefix_to_gs_center_correction_m"]*1000 for f in rows[1:]], color=color, label=label)
        axes[1, 0].plot(x[1:], [f["sensor_reduction_before_mapping"] for f in rows[1:]], color=color, label=label)
        axes[1, 1].plot(x[1:], [f["selected_before_mapping_prior_objective"] for f in rows[1:]], color=color, label=label)
        axes[2, 0].step(x, [f["point_count"] for f in rows], where="post", color=color, label=label)
    rows = result["runs"][0]["frames"]
    axes[0, 0].plot([r["frame_id"] for r in rows], [r["icp_prefix_raw_gt_translation_error_m"]*1000 for r in rows], color="#555555", linestyle="--", label="standalone ICP / prefix")
    series = [("ICP", result["standalone_ICP_reference"]["trajectory"], "#777777")]
    series += [(f'lambda={r["lambda"]:g}', r["trajectory"], c) for r, c in zip(result["runs"], colors)]
    xs = np.arange(3)
    for j, (label, values, color) in enumerate(series):
        bars = axes[2, 1].bar(xs+(j-1)*.25, [values[k]*1000 for k in common.METRICS[:3]], width=.25, color=color, label=label)
        axes[2, 1].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[2, 1].set_xticks(xs, ["Raw RMSE", "SE3 ATE", "Delta-1 RPE"])
    titles = ("Post-hoc raw camera-center error", "Accepted GS correction from ICP prefix",
              "Sensor reduction: initial minus selected", "Selected weighted prior objective",
              "Gaussian map size", "Translation trajectory metrics")
    units = ("mm", "mm", "sensor objective units", "weighted normalized objective", "Gaussian count", "mm")
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
        if ax is not axes[2, 1]:
            ax.set_xlabel("associated frame ID")
    axes[1, 0].axhline(0, color="#666666", linewidth=.7)
    equality = result["map_comparison"]["first_warmup_all_parameters_byte_identical"]
    fig.suptitle("Frozen desk settings transferred to fr1_xyz, frames 200-299", fontsize=15, y=.995)
    fig.text(.02, .014, f"First warmup maps byte-identical: {equality}. Same locked V4; no weights selected from xyz GT.\nOne clip / one seed; current maps can differ. GT is post-hoc only; zero flags do not certify pose accuracy.", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .978))
    fig.savefig(OUTPUT/"comparison.png", dpi=170)
    plt.close(fig)


def main():
    assert not torch.cuda.is_initialized()
    assert common.sha(ROOT/"scripts/train_tum_online_gs_v4.py") == common.V4_SHA
    assert common.sha(ROOT/"scripts/geometric_pose_prior.py") == common.PRIOR_SHA
    assert common.sha(ROOT/"scripts/track_rgbd_icp.py") == ICP_SHA
    assert common.sha(ROOT/"scripts/tum_rgbd.py") == LOADER_SHA
    paths = [ROOT/"outputs"/name for name in RUNS]
    if not all((p/"metrics.json").is_file() for p in paths):
        raise SystemExit("Both final metrics files required; no polling, partial comparison or optimization performed")
    metrics = [common.read_json(p/"metrics.json") for p in paths]
    if not all(d["status"].startswith("completed") for d in metrics):
        raise SystemExit("Both runs must be completed before this post-hoc comparison")
    contexts = [torch.load(p/"first_tracking_context.pt", map_location="cpu", weights_only=True) for p in paths]
    checkpoints = [torch.load(p/"online_map.pt", map_location="cpu", weights_only=True) for p in paths]
    for p in paths:
        meta = common.read_json(p/"first_tracking_context.json")
        assert common.sha(p/"first_tracking_context.pt") == meta["checkpoint_sha256"]
        assert meta["tracking_optimizer_steps_executed_before_save"] == 0 and meta["saved_before_first_pose_gradient_probe"]
    icp_path = ROOT/"outputs"/ICP_RUN
    icp = common.read_json(icp_path/"metrics.json")
    icp_validation = common.read_json(icp_path/"validation.json")
    icp_summary = common.read_json(icp_path/"summary.json")
    icp_truth = parse_trajectory_text((icp_path/"trajectory_groundtruth_anchored.txt").read_text(encoding="utf-8"))
    protocol = check_protocol(metrics, contexts, checkpoints, icp, icp_validation, icp_truth)
    normalized_icp = dict(zip(common.METRICS, (icp["trajectory"][k] for k in
        ("raw_anchored_translation_rmse_m", "se3_aligned_ate_translation_rmse_m", "adjacent_rpe_translation_rmse_m",
         "adjacent_rpe_rotation_rmse_deg", "raw_rotation_rmse_deg"))))
    runs = [common.summarize_run(m, c) for m, c in zip(metrics, checkpoints)]
    for run in runs:
        for key in common.METRICS:
            assert abs(run["icp_prefix_trajectory"][key]-normalized_icp[key]) < 1e-10
    files = ("metrics.json", "configuration.json", "first_tracking_context.json", "first_tracking_context.pt", "online_map.pt")
    result = {"study": OUTPUT.name, "status": "completed_cpu_fixed_parameter_transfer_comparison",
        "dataset": "TUM rgbd_dataset_freiburg1_xyz", "associated_frame_ids": list(range(200, 300)),
        "source": {"analysis_script_sha256": common.sha(Path(__file__)),
            "reused_readonly_analysis_module_sha256": common.sha(Path(common.__file__)),
            "v4_script_sha256": common.V4_SHA, "pose_prior_helper_sha256": common.PRIOR_SHA,
            "icp_script_sha256": ICP_SHA, "loader_sha256": LOADER_SHA,
            "inputs": [{"run": p.name, "sha256": {name: common.sha(p/name) for name in files}} for p in paths],
            "standalone_ICP_inputs": {name: common.sha(icp_path/name) for name in
                ("metrics.json", "configuration.json", "validation.json", "summary.json", "trajectory_text_audit.json", "trajectory_estimated.txt", "trajectory_groundtruth_anchored.txt")}},
        "protocol_checks": protocol,
        "standalone_ICP_reference": {"run": ICP_RUN, "trajectory": normalized_icp,
            "frames_processed": icp["frames_processed"], "failure_frame_count": icp["failure_frame_count"],
            "failed_frame_ids": [f["frame_id"] for f in icp["frame_metrics"] if f["failure_flags"]],
            "level_stop_reason_histogram": icp_summary["level_stop_reason_histogram"],
            "iteration_records": icp_summary["iteration_records"],
            "scope": "Standalone previous-frame depth geometry; both GS ICP prefixes are checked against these saved float64 poses"},
        "map_comparison": common.compare_maps(contexts, checkpoints), "runs": runs,
        "interpretation": [
            "Only dataset and run name changed from each matching desk V4 configuration; lambda values 0 and 0.01 and all scales/thresholds were fixed before this transfer.",
            "GT is used only for final/post-hoc error accounting; no xyz-based weight selection, rerun, restart or frame removal.",
            "This is a different sequence from the same Freiburg1 sensor and similar office setting, not evidence for broad cross-scene or cross-sensor generalization.",
            "Actual first-map tensor bytes are compared rather than inferred from seed. Independent warmups and subsequent map feedback limit strict single-variable attribution.",
            "Per-frame sensor reduction compares initial and selected candidate on that run's current map. Across-run losses are not a shared-map objective comparison.",
            "Post-hoc improved/worsened counts compare accepted raw anchored camera-center errors against the independent ICP prefix on 99 nonanchor frames; they are not independent samples.",
            "All failure and rejected-frame records remain; selected-candidate losses and accepted-pose error fields must be distinguished if rejection occurs.",
            "A weighted prior objective already includes lambda. The combined objective is sensor plus weighted prior; mapping still uses sensor only.",
            "Zero flags or a reduced objective do not certify convergence, accuracy, calibrated covariance or a speed advantage. No new optimizer, renderer or training call occurs in this script."
        ], "artifacts": {"figure": "comparison.png"}, "cpu_only": True, "cuda_initialized": torch.cuda.is_initialized()}
    assert not result["cuda_initialized"]
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    assert len(encoded) <= 1024*1024
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"comparison.json").write_bytes(encoded)
    make_figure(result)
    print(json.dumps({"output": str(OUTPUT), "comparison_json_bytes": len(encoded),
        "warmup_all_parameters_byte_identical": result["map_comparison"]["first_warmup_all_parameters_byte_identical"],
        "ICP_reference": normalized_icp,
        "runs": [{k: run[k] for k in ("run", "trajectory", "posthoc_gt_comparison", "flags", "final_point_count")} for run in runs]}, indent=2))


if __name__ == "__main__":
    main()
