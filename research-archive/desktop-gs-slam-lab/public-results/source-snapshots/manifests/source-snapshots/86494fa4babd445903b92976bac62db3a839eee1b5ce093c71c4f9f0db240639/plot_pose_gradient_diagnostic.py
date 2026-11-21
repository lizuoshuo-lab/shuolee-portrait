"""Plot recorded read-only FD and coverage controls; no new renderer calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/tum-pose-gradient-diagnostic-001"


def main():
    raw = (DIRECTORY / "pose_gradient_diagnostic.json").read_bytes()
    recorded = json.loads(raw)
    if recorded["status"] != "diagnostic_completed" or recorded["online_run_completed"]:
        raise ValueError("This plot requires a stopped, completed diagnostic")
    sweeps = recorded["sweep"]["epsilon_sweeps"]
    epsilon = np.array([s["epsilon"] for s in sweeps])
    dynamic_error = np.array([s["relative_l2_error"] for s in sweeps])
    fixed_error = np.array([s["fixed_zero_coverage_mask_diagnostic"]["relative_l2_error"] for s in sweeps])
    dynamic_cosine = np.array([s["cosine_similarity"] for s in sweeps])
    fixed_cosine = np.array([s["fixed_zero_coverage_mask_diagnostic"]["cosine_similarity"] for s in sweeps])
    xor = np.array([[a["plus_minus_covered_mask_changed_pixels"] for a in s["axes"]] for s in sweeps])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].semilogx(epsilon, dynamic_error, "o-", label="Original dynamic coverage mask")
    axes[0].semilogx(epsilon, fixed_error, "o-", label="Fixed zero-pose mask (diagnostic only)")
    axes[0].axhline(.5, color="gray", linestyle="--", label="Original maximum relative L2")
    axes[0].set(ylabel="Relative L2 difference", title="Analytic vs centered finite difference")
    axes[0].legend(fontsize=7)
    axes[1].semilogx(epsilon, dynamic_cosine, "o-")
    axes[1].semilogx(epsilon, fixed_cosine, "o-")
    axes[1].axhline(.8, color="gray", linestyle="--")
    axes[1].set(ylabel="Direction cosine", ylim=(.75, 1.01), title="Original minimum cosine = 0.8")
    for dimension, label in enumerate(("rho_x", "rho_y", "rho_z", "omega_x", "omega_y", "omega_z")):
        axes[2].semilogx(epsilon, xor[:, dimension], "o-", label=label)
    axes[2].set(ylabel="Pixels whose mask differs (+epsilon vs -epsilon)", title="Hard coverage support changes")
    axes[2].legend(fontsize=7, ncol=2)
    for axis in axes:
        axis.set_xlabel("SE3 twist coordinate step (m or rad)")
        axis.grid(alpha=.25)
    fig.suptitle("Recomputed warmup | one ICP-initialized frame | no tracking steps | historical failure not reproduced", fontsize=11)
    fig.savefig(DIRECTORY / "gradient-mask-controls.png", dpi=180)
    plt.close(fig)
    summary = {"status": "completed_read_only_plot", "input_sha256": hashlib.sha256(raw).hexdigest(),
               "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "epsilons": epsilon.tolist(), "dynamic_relative_l2": dynamic_error.tolist(),
               "fixed_mask_relative_l2": fixed_error.tolist(), "mask_xor_pixels_by_axis": xor.tolist(),
               "scope": "Current diagnostic only; not historical-state reproduction, a training change or a trajectory benchmark"}
    (DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"image": "gradient-mask-controls.png", "epsilon_count": len(sweeps)}))


if __name__ == "__main__":
    main()
