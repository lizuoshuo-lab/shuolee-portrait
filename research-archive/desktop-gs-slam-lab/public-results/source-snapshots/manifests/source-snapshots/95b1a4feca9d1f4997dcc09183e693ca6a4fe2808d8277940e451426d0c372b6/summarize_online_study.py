"""Compare completed sequential Gaussian experiments; never train or select by GT.

This CPU report audits controlled configurations and plots saved measurements.
Independent trajectory audits must already exist for each input run.
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["tum-online-gsplat-001", "tum-online-previous-001",
                "tum-online-depth1-001", "tum-online-steps300-001"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output", default="tum-online-study-001")
    args = parser.parse_args()
    if any(Path(name).name != name or not name or name in (".", "..")
           for name in [*args.runs, args.output]):
        raise ValueError("Run/output names must be single directory names")
    out = ROOT / "outputs" / args.output
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    metrics = [read_json(ROOT / "outputs" / name / "metrics.json") for name in args.runs]
    self_tests = read_json(ROOT / "outputs" / "trajectory-audit-self-tests.json")
    if self_tests["independent_metric_self_tests"]["status"] != "passed":
        raise ValueError("Independent trajectory audit self-tests did not pass")
    for name, item in zip(args.runs, metrics):
        if item["experiment_id"] != name or not item["status"].startswith("completed"):
            raise ValueError(f"Incomplete or mismatched run: {name}")
        audit_path = ROOT / "outputs" / name / "independent_trajectory_audit.json"
        if not audit_path.is_file():
            raise ValueError(f"Independent trajectory audit missing: {name}")
        audit = read_json(audit_path)
        if audit.get("status") != "completed":
            raise ValueError(f"Independent audit did not complete: {name}")
        checks = {"raw_anchored_translation_rmse_m": "raw_anchored_translation_rmse_m",
                  "se3_aligned_ate_translation_rmse_m": "ate_rmse_m_se3_no_scale",
                  "adjacent_rpe_translation_rmse_m": "adjacent_rpe_translation_rmse_m",
                  "adjacent_rpe_rotation_rmse_deg": "adjacent_rpe_rotation_rmse_degrees"}
        if any(abs(audit["metrics"][a]-item["trajectory"][b]) > 1e-6 for a, b in checks.items()):
            raise ValueError(f"Independent metrics disagree beyond text/float precision: {name}")
        for filename, key in [("trajectory_estimated.txt", "estimated_sha256"),
                              ("trajectory_groundtruth_anchored.txt", "ground_truth_sha256")]:
            if hashlib.sha256((audit_path.parent / filename).read_bytes()).hexdigest() != audit[key]:
                raise ValueError(f"Independent audit stale for {name}/{filename}")
    reference = metrics[0]
    comparisons = []
    labels = []
    for name, item in zip(args.runs, metrics):
        differences = {key: {"baseline": reference["configuration"][key], "value": value}
                       for key, value in item["configuration"].items()
                       if key != "run" and value != reference["configuration"].get(key)}
        if item["selected_frame_ids"] != reference["selected_frame_ids"]:
            raise ValueError("This controlled comparison requires identical selected frames")
        if item["script_sha256"] != reference["script_sha256"]:
            raise ValueError("Controlled runs must have identical optimization source")
        if len(differences) > 1:
            raise ValueError(f"More than one changed factor: {name}: {differences}")
        label = "baseline" if not differences else ", ".join(
            f"{key}={value['value']}" for key, value in differences.items())
        labels.append(label)
        trajectory = item["trajectory"]
        rows = item["frame_metrics"]
        covered = [row.get("optimized_loss_before_mapping", row.get("optimized_loss"))["coverage"]
                   for row in rows]
        comparisons.append({"run": name, "changed_configuration": differences,
                            "ate_rmse_m_se3_no_scale": trajectory["ate_rmse_m_se3_no_scale"],
                            "raw_anchored_translation_rmse_m": trajectory["raw_anchored_translation_rmse_m"],
                            "adjacent_rpe_translation_rmse_m": trajectory["adjacent_rpe_translation_rmse_m"],
                            "adjacent_rpe_rotation_rmse_degrees": trajectory["adjacent_rpe_rotation_rmse_degrees"],
                            "failure_frame_count": item["failure_frame_count"],
                            "min_tracking_coverage": min(covered), "mean_tracking_coverage": float(np.mean(covered)),
                            "final_points": item["final_point_count"],
                            "online_loop_seconds_excludes_warmup": item["online_loop_seconds_excludes_warmup"],
                            "independent_audit": f"outputs/{name}/independent_trajectory_audit.json"})
    times = np.array([row["rgb_timestamp"] for row in reference["frame_metrics"]])
    times -= times[0]
    truth = np.array(reference["trajectory"]["anchored_truth_c2w"])[:, :3, 3]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(truth[:, 0], truth[:, 2], "k.-", lw=2, label="GT in first-camera gauge")
    for label, item in zip(labels, metrics):
        estimated = np.array([row["estimated_c2w"] for row in item["frame_metrics"]])[:, :3, 3]
        axes[0, 0].plot(estimated[:, 0], estimated[:, 2], ".-", label=label)
        raw = [row["raw_anchored_translation_error_m"] for row in item["trajectory"]["per_view"]]
        axes[0, 1].plot(times, np.array(raw)*100, ".-", label=label)
        depth = [row.get("optimized_loss_before_mapping", row.get("optimized_loss"))["depth_mae_m"]
                 for row in item["frame_metrics"]]
        axes[1, 0].plot(times, np.array(depth)*100, ".-", label=label)
        coverage = [row.get("optimized_loss_before_mapping", row.get("optimized_loss"))["coverage"]
                    for row in item["frame_metrics"]]
        axes[1, 1].plot(times, np.array(coverage)*100, ".-", label=label)
    axes[0, 0].set(xlabel="X (m)", ylabel="Z (m)", title="Unaligned camera centers: X-Z projection")
    axes[0, 0].axis("equal")
    axes[0, 1].set(xlabel="Elapsed RGB time (s)", ylabel="Translation error (cm)", title="Raw first-camera anchored error")
    axes[1, 0].set(xlabel="Elapsed RGB time (s)", ylabel="Covered depth MAE (cm)", title="Sensor loss at selected pose, before mapping")
    axes[1, 1].set(xlabel="Elapsed RGB time (s)", ylabel="Valid-depth coverage (%)", title="Coverage alone does not measure tracking accuracy")
    for ax in axes.flat:
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"TUM fr1_desk | {len(times)} frames | {times[-1]:.4f}s | seed 0 | exploratory controls")
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out / "tracking-controls.png", dpi=180)
    plt.close(fig)
    # Exact target/render PNGs, same last sensor frame in every experiment.
    width, height = reference["camera"]["resolution"]
    preview = Image.new("RGB", (width*2, (height+28)*len(metrics)), "#ffffff")
    draw = ImageDraw.Draw(preview)
    index = len(times)-1
    for row, (name, label) in enumerate(zip(args.runs, labels)):
        target = Image.open(ROOT / "outputs" / name / "views" / f"frame{index:03d}_target.png").convert("RGB")
        render = Image.open(ROOT / "outputs" / name / "views" / f"frame{index:03d}_estimated_map.png").convert("RGB")
        y = row*(height+28)
        draw.text((4, y+4), label, fill="#111111")
        preview.paste(target, (0, y+28)); preview.paste(render, (width, y+28))
    preview.save(out / "last-frame-comparison.png")
    report = {"experiment_id": args.output, "status": "completed",
              "scope": "controlled exploratory comparison on one short segment and one seed; no benchmark claim",
              "runs": comparisons, "frame_ids": reference["selected_frame_ids"],
              "duration_seconds": float(times[-1]), "script_sha256": reference["script_sha256"],
              "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["depth_weight changes the depth term in both tracking and mapping",
                         "GT errors are analysis only; no GT pose enters optimization",
                         "This segment has been inspected for parameter decisions and is development data",
                         "Independent audits validate metric consistency, not physical ground truth accuracy",
                         "Timing includes online exports/probes and excludes first-frame warmup; no SLAM FPS claim"]}
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
