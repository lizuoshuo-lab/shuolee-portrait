"""Plot independently accepted full results; never fill missing runs with zeros."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "outputs/splatam-ablation-repeat-001"
PROFILES = ("rgbd", "rgb_only", "depth_only")
LABELS = ("RGB-D tracking", "RGB residual", "Depth residual")
COLORS = ("#3463b3", "#da7540", "#208773")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    source = OUT / "metrics.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    assert record["status"] == "full_ablation_repeats_and_room_independently_verified", "Wait for all real full-run seals."
    runs = record["verified_runs"]
    assert len(runs) == 10 and all(r["status"] == "completed_and_independently_verified" for r in runs)
    groups = [sorted((r for r in runs if r["sequence"] == "freiburg1_desk" and r["profile"] == p), key=lambda r: r["seed"]) for p in PROFILES]
    assert all([r["seed"] for r in group] == [0, 1, 2] for group in groups)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    panels = (("ate_se3_rmse_m", 100., "SE(3)-aligned ATE RMSE (cm)"),
              ("rpe_translation_rmse_m", 100., "Adjacent translation RPE RMSE (cm)"),
              ("rpe_rotation_rmse_deg", 1., "Adjacent rotation RPE RMSE (deg)"),
              ("gaussian_count", 1e-3, "Final Gaussians (thousands)"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    offsets = (-.12, 0, .12)
    plot_values = {}
    for ax, (field, factor, label) in zip(axes.flat, panels):
        values = np.array([[r[field] for r in group] for group in groups], dtype=float) * factor
        assert np.isfinite(values).all()
        for index, color in enumerate(COLORS):
            ax.scatter(index + np.array(offsets), values[index], c=color, s=48, zorder=3)
            ax.errorbar(index, values[index].mean(), yerr=values[index].std(ddof=1), fmt="_", markersize=24,
                        color=color, capsize=5, elinewidth=1.5, zorder=4)
            for seed, val in enumerate(values[index]):
                ax.annotate(f"s{seed}", (index + offsets[seed], val), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xticks(range(3), LABELS)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_xlim(-.45, 2.45)
        plot_values[field] = {"display_scale_factor": factor, "display_axis_label": label,
                              "raw_values": {p: [r[field] for r in groups[i]] for i, p in enumerate(PROFILES)},
                              "display_values": {p: list(values[i]) for i, p in enumerate(PROFILES)}}
    fig.suptitle("Same 592 desk frames, 320 x 240: tracking residual ablation\nThree seeds; marks = actual runs, bars = mean +/- sample SD (ddof=1), not confidence intervals", fontsize=13)
    comparison = OUT / "tracking-ablation-three-seeds.png"
    fig.savefig(comparison, dpi=160)
    plt.close(fig)

    room = next(r for r in runs if r["sequence"] == "freiburg1_room")
    evaluation = LAB / room["evaluation_path"] / "trajectory"
    seal = json.loads((evaluation.parent / "independent-receipt.json").read_text(encoding="utf-8"))
    assert seal["status"] == "independently_verified" and seal["run_name"] == room["run_name"]
    assert seal["execution_sha256"] == room["execution_sha256"] and seal["final_model_sha256"] == room["model_sha256"]
    for name in ("errors.csv", "reference-independent.tum", "estimated-se3-aligned.tum"):
        assert sha(evaluation / name) == seal["frozen_result_sha256"]["trajectory/" + name]
    with (evaluation / "errors.csv").open(encoding="utf-8", newline="") as stream:
        errors = list(csv.DictReader(stream))
    assert len(errors) == room["frames"] == 1315
    elapsed = np.array([float(r["timestamp"]) for r in errors])
    elapsed -= elapsed[0]
    aligned_errors = np.array([float(r["ate_se3_m"]) for r in errors])
    gauge_errors = np.array([float(r["ate_first_camera_gauge_m"]) for r in errors])
    reference = np.loadtxt(evaluation / "reference-independent.tum")
    estimated = np.loadtxt(evaluation / "estimated-se3-aligned.tum")
    assert len(reference) == len(estimated) == len(errors)
    assert np.isfinite(reference).all() and np.isfinite(estimated).all()
    assert abs(np.sqrt(np.mean(aligned_errors ** 2)) - room["ate_se3_rmse_m"]) < 1e-9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].plot(reference[:, 1], reference[:, 2], label="Independent raw GT", color="#333333", lw=1.8)
    axes[0].plot(estimated[:, 1], estimated[:, 2], label="Estimated, rigid aligned", color=COLORS[0], lw=1)
    axes[0].scatter(reference[0, 1], reference[0, 2], marker="o", c="#333333", s=30, label="GT start")
    axes[0].scatter(reference[-1, 1], reference[-1, 2], marker="x", c="#333333", s=45, label="GT end")
    axes[0].set(xlabel="Reference x (m)", ylabel="Reference y (m)", title="XY projection; scale fixed to 1")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].plot(elapsed, gauge_errors * 100, label="First-camera gauge", color=COLORS[1], lw=1)
    axes[1].plot(elapsed, aligned_errors * 100, label="SE(3)-aligned", color=COLORS[0], lw=1)
    axes[1].set(xlabel="Elapsed selected RGB time (s)", ylabel="Position error (cm)", title="No loop correction or global BA")
    axes[1].legend(fontsize=8)
    fig.suptitle("Freiburg1 room: 1,315 frames, broader coverage and a loop trajectory in the same office", fontsize=12)
    room_figure = OUT / "room-trajectory-and-error.png"
    fig.savefig(room_figure, dpi=160)
    plt.close(fig)
    quartiles = []
    for start, stop in zip(np.linspace(0, len(errors), 5, dtype=int)[:-1], np.linspace(0, len(errors), 5, dtype=int)[1:]):
        quartiles.append({"start_frame": int(start), "end_frame": int(stop - 1),
                          "aligned_position_rmse_m": float(np.sqrt(np.mean(aligned_errors[start:stop] ** 2))),
                          "first_camera_gauge_rmse_m": float(np.sqrt(np.mean(gauge_errors[start:stop] ** 2)))})
    receipt = {"status": "plotted_only_complete_independently_verified_results", "source_metrics_sha256": sha(source),
               "plot_data": plot_values, "seed_count": 3, "error_bar": "sample standard deviation ddof=1, not confidence interval",
               "room_run_name": room["run_name"], "room_errors_sha256": sha(evaluation / "errors.csv"),
               "room_reference_sha256": sha(evaluation / "reference-independent.tum"),
               "room_aligned_estimated_sha256": sha(evaluation / "estimated-se3-aligned.tum"),
               "room_path_length_gt_m": float(np.linalg.norm(np.diff(reference[:, 1:4], axis=0), axis=1).sum()),
               "room_gt_start_end_distance_m": float(np.linalg.norm(reference[-1, 1:4] - reference[0, 1:4])),
               "room_estimated_start_end_distance_m": float(np.linalg.norm(estimated[-1, 1:4] - estimated[0, 1:4])),
               "room_endpoint_aligned_error_m": float(aligned_errors[-1]), "room_endpoint_first_camera_gauge_error_m": float(gauge_errors[-1]),
               "room_equal_frame_quarters": quartiles,
               "figures": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in (comparison, room_figure)},
               "limits": ["N=3 is exploratory; no statistical significance or broad robustness claim.",
                          "Mapping keeps its configuration, but maps evolve under each estimated trajectory.",
                          "Room revisits the same office; input loop does not imply SplaTAM loop closure.",
                          "Start-end distance is not endpoint accuracy; report the reference and both error gauges."]}
    (OUT / "plot-receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "figures": list(receipt["figures"])}))


if __name__ == "__main__":
    main()
