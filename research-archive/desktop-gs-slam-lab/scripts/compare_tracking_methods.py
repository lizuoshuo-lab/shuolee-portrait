"""Centralized CPU trajectory comparison on identical sensor/time/gauge inputs.

This compares saved estimates. It neither trains nor uses GT to select poses.
The shared NumPy auditor is the evaluation implementation, not a second proof
of the tracking algorithm. CPU/GPU loop timings are explicitly incomparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_tum_trajectory import parse_trajectory_text, validate_time_pairing, audit_pose_arrays

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_trajectory(path):
    return parse_trajectory_text(path.read_text(encoding="utf-8-sig"), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["tum-online-v2-transfer-baseline-001",
                        "tum-online-v2-transfer-depth1-001", "tum-icp-transfer100-001"])
    parser.add_argument("--labels", nargs="+", default=["GS RGB-D weight=0.1", "GS RGB-D weight=1", "Depth projective ICP"])
    parser.add_argument("--output", default="tracking-methods-transfer100-001")
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        raise ValueError("One display label per run required")
    if any(Path(name).name != name or name in ("", ".", "..") for name in [*args.runs, args.output]):
        raise ValueError("Run/output names must be single directory names")
    destination = ROOT / "outputs" / args.output
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    entries = []
    reference = reference_truth = reference_times = None
    for run, label in zip(args.runs, args.labels):
        directory = ROOT / "outputs" / run
        metric_bytes = (directory / "metrics.json").read_bytes()
        metrics = json.loads(metric_bytes)
        if metrics["experiment_id"] != run or not metrics["status"].startswith("completed"):
            raise ValueError(f"Incomplete/mismatched run: {run}")
        truth = load_trajectory(directory / "trajectory_groundtruth_anchored.txt")
        estimate = load_trajectory(directory / "trajectory_estimated.txt")
        validate_time_pairing(estimate, truth)
        if reference is None:
            reference, reference_truth, reference_times = metrics, truth, estimate["timestamps"]
        else:
            validate_time_pairing(estimate, reference_truth)
            if np.max(np.abs(truth["poses"]-reference_truth["poses"])) > 1e-7:
                raise ValueError(f"GT gauge/poses differ beyond text precision: {run}")
            if metrics["association"] != reference["association"]:
                raise ValueError(f"Sensor association differs: {run}")
        evaluation = audit_pose_arrays(estimate["poses"], reference_truth["poses"])
        original = metrics["trajectory"]
        pairs = {"raw_anchored_translation_rmse_m": "raw_anchored_translation_rmse_m",
                 "se3_aligned_ate_translation_rmse_m": "ate_rmse_m_se3_no_scale" if "ate_rmse_m_se3_no_scale" in original else "se3_aligned_ate_translation_rmse_m",
                 "adjacent_rpe_translation_rmse_m": "adjacent_rpe_translation_rmse_m",
                 "adjacent_rpe_rotation_rmse_deg": "adjacent_rpe_rotation_rmse_degrees" if "adjacent_rpe_rotation_rmse_degrees" in original else "adjacent_rpe_rotation_rmse_deg"}
        differences = {key: abs(evaluation[key]-original[value]) for key, value in pairs.items()}
        if max(differences.values()) > 1e-6:
            raise ValueError(f"Saved trajectory and metrics disagree: {run}: {differences}")
        if not evaluation["first_pose_anchor_matches"] or metrics["frames_processed"] != len(estimate["poses"]):
            raise ValueError(f"Incomplete frame retention or invalid anchor: {run}")
        entries.append({"run": run, "label": label, "failure_frame_count": metrics["failure_frame_count"],
                        "centralized_evaluation": evaluation, "poses": estimate["poses"],
                        "in_memory_metric_difference": differences,
                        "metrics_sha256": hashlib.sha256(metric_bytes).hexdigest(),
                        "trajectory_sha256": hashlib.sha256((directory / "trajectory_estimated.txt").read_bytes()).hexdigest(),
                        "source": metrics.get("script_sha256", metrics["configuration"].get("source_sha256")),
                        "reported_loop_seconds": metrics.get("online_loop_seconds_excludes_warmup", metrics.get("cpu_loop_seconds")),
                        "timing_note": "different CPU/GPU pipelines; preparation, export and concurrency boundaries differ; no speed comparison"})
    times = np.array([float(t-reference_times[0]) for t in reference_times])
    gt = reference_truth["poses"][:, :3, 3]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(gt[:, 0], gt[:, 2], "k.-", linewidth=2, label="GT in first-camera gauge")
    for entry in entries:
        label, evaluation, positions = entry["label"], entry["centralized_evaluation"], entry["poses"][:, :3, 3]
        axes[0, 0].plot(positions[:, 0], positions[:, 2], ".-", markersize=3, label=label)
        axes[0, 1].plot(times, np.array(evaluation["per_frame_raw_translation_error_m"])*100, label=label)
        axes[1, 0].plot(times, np.array(evaluation["per_frame_se3_aligned_ate_translation_error_m"])*100, label=label)
        axes[1, 1].plot(times[1:], np.array(evaluation["per_interval_rpe_translation_error_m"])*100, label=label)
    axes[0, 0].set(xlabel="X (m)", ylabel="Z (m)", title="Unaligned X-Z camera-center projection")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel="Elapsed RGB time (s)", ylabel="Error (cm)", title="Raw first-camera anchored translation error")
    axes[1, 0].set(xlabel="Elapsed RGB time (s)", ylabel="Error (cm)", title="One SE3 alignment per complete trajectory, no scale")
    axes[1, 1].set(xlabel="Elapsed RGB time (s)", ylabel="Error (cm)", title="Adjacent-frame translation RPE")
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.suptitle(f"TUM fr1_desk | {len(times)} matched frames | {times[-1]:.4f}s | exploratory accuracy comparison")
    destination.mkdir(parents=True, exist_ok=False)
    fig.savefig(destination / "trajectory-comparison.png", dpi=180)
    plt.close(fig)
    for entry in entries:
        del entry["poses"]
    report = {"experiment_id": args.output, "status": "completed",
              "frames": len(times), "duration_s": float(times[-1]), "runs": entries,
              "association_equal": True, "timestamps_exactly_equal": True,
              "shared_first_camera_gt_gauge_within_text_precision": True,
              "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evaluation_note": "centralized use of the same NumPy auditor; not a second independent formula for ICP",
              "limitations": ["one scene and one segment; previously inspected validation now used for diagnosis",
                              "different tracking objectives and map usage; no universal method superiority claim",
                              "no controlled CPU/GPU speed comparison"]}
    (destination / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "frames": report["frames"],
                      "runs": [{"run": entry["run"], "ate_m": entry["centralized_evaluation"]["se3_aligned_ate_translation_rmse_m"],
                                "raw_m": entry["centralized_evaluation"]["raw_anchored_translation_rmse_m"]} for entry in entries]}, indent=2))


if __name__ == "__main__":
    main()
