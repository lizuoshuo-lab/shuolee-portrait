"""Read completed frozen-context replay; evaluate GT only after optimization.

No GPU, optimization, pose selection, data download or training occurs here.
A one-frame diagnostic has pose errors, not trajectory ATE/RPE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def c2w(w2c):
    w = np.asarray(w2c, dtype=np.float64)
    out = np.eye(4)
    out[:3,:3] = w[:3,:3].T
    out[:3,3] = -w[:3,:3].T @ w[:3,3]
    return out

def error(camera, truth):
    return {"translation_m": float(np.linalg.norm(camera[:3,3]-truth[:3,3])),
            "rotation_degrees": float(np.degrees(Rotation.from_matrix(camera[:3,:3].T @ truth[:3,:3]).magnitude()))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tum-pose-prior-context-replay-001")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ("", ".", ".."):
        raise ValueError("Single run basename required")
    folder = ROOT/"outputs"/args.run
    source = folder/"metrics.json"
    d = read(source)
    if not d["status"].startswith("completed") or len(d["arms"]) != 2:
        raise ValueError("Both arms must have completed before any posthoc GT evaluation")
    if d["ground_truth_read"] is not False or not d["same_fixed_input_bytes_for_both_arms"]:
        raise ValueError("Expected GT-free shared-input replay")
    # This is the sole GT consumer, after the completed replay file was read.
    original_path = ROOT/"outputs/tum-online-v4-prior0-transfer100-001/metrics.json"
    original = read(original_path)
    if original["selected_frame_ids"][1] != 201:
        raise ValueError("Expected source target frame201")
    truth = np.asarray(original["trajectory"]["anchored_truth_c2w"][1], dtype=np.float64)
    initial = c2w(d["arms"][0]["initial_w2c"])
    initial_error = error(initial, truth)
    summaries = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for name, arm, color in zip(("prior0", "prior001"), d["arms"], ("#df5e36", "#16859b")):
        if arm["fixed_tensor_hashes_before"] != d["fixed_tensor_hashes_before"] or arm["fixed_tensor_hashes_after"] != d["fixed_tensor_hashes_after"]:
            raise ValueError("Arm input bytes did not stay shared and frozen")
        if arm["initial_w2c"] != d["arms"][0]["initial_w2c"]:
            raise ValueError("Initial camera mismatch")
        steps_path = folder/name/"steps.jsonl"
        rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8-sig").splitlines() if line]
        series = []
        for row in rows:
            camera = c2w(row.get("selected_canonical_w2c", arm["initial_w2c"]))
            series.append({"candidate_step": row["candidate_step"],
                           "sensor_objective": row["loss"]["sensor_objective"],
                           "prior_objective": row["loss"]["prior_objective"],
                           "combined_objective": row["loss"]["objective"],
                           "best_combined_objective": row.get("best_combined_objective"),
                           "selected_center_correction_m": float(np.linalg.norm(camera[:3,3]-initial[:3,3])),
                           "selected_posthoc_pose_error": error(camera, truth),
                           "step_status": row.get("status", "optimizer_update_recorded")})
        final = c2w(arm["final_selected_w2c"])
        final_error = error(final, truth)
        native_first = c2w(original["frame_metrics"][1]["candidate_optimized_w2c"])
        summary = {"arm": name, "weight": arm["weight"], "status": arm["status"],
                   "failure_flags": arm["failure_flags"], "gradient_probe": arm["pose_gradient_probe"],
                   "steps_requested": arm["steps_requested"], "steps_completed": arm["steps_completed"],
                   "initial_loss": arm["initial_loss"], "selected_loss": arm["selected_loss"],
                   "center_correction_m": float(np.linalg.norm(final[:3,3]-initial[:3,3])),
                   "initial_pose_error": initial_error, "final_selected_candidate_pose_error": final_error,
                   "posthoc_translation_error_change_m": final_error["translation_m"]-initial_error["translation_m"],
                   "comparison_with_original_prior0_first_candidate": {
                       "center_difference_m": float(np.linalg.norm(final[:3,3]-native_first[:3,3])),
                       "only_prior0_is_same_objective": arm["weight"] == 0,
                       "not_claimed_bitwise_reproduction": True},
                   "source_steps_sha256": sha(steps_path), "all_recorded_steps": series}
        summaries.append(summary)
        label = "lambda="+str(arm["weight"])
        xs = [x["candidate_step"] for x in series]
        axes[0].plot(xs, [x["sensor_objective"] for x in series], color=color, label=label)
        axes[0].scatter([arm["steps_completed"]], [arm["selected_loss"]["sensor_objective"]], color=color, marker="x")
        axes[1].plot(xs, [x["selected_center_correction_m"]*1000 for x in series], color=color, label=label)
        axes[1].scatter([arm["steps_completed"]], [summary["center_correction_m"]*1000], color=color, marker="x")
    axes[0].set(title="Same frozen-map sensor objective", xlabel="Candidate / optimizer step", ylabel="Sensor objective")
    axes[1].set(title="Best-candidate center displacement", xlabel="Candidate / optimizer step", ylabel="mm from fixed ICP initializer")
    axes[2].bar(["ICP initial", "lambda=0", "lambda=.01"], [initial_error["translation_m"]*1000]+[x["final_selected_candidate_pose_error"]["translation_m"]*1000 for x in summaries], color=["#777777", "#df5e36", "#16859b"])
    axes[2].set(title="Posthoc single-frame position error", ylabel="mm; no fitted alignment")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(); axes[1].legend()
    fig.suptitle("Frame201 replay: one warmup map, one RGB-D input, two prior weights")
    fig.savefig(folder/"replay-comparison.png", dpi=150)
    plt.close(fig)
    result = {"status": "completed_readonly_posthoc_analysis", "run": args.run,
              "source_metrics_sha256": sha(source), "source_metrics_status": d["status"],
              "analysis_script_sha256": sha(Path(__file__)),
              "ground_truth_usage": "After completed replay only; never candidate selection, initialization or optimization",
              "ground_truth_source_metrics_sha256": sha(original_path), "target_frame_id": 201,
              "source_context_sha256": d["source_context_sha256"],
              "same_fixed_input_bytes_for_both_arms": True, "initial_pose_error": initial_error,
              "arms": summaries,
              "boundaries": ["One frame, one map and one GPU run: no trajectory ATE/RPE or cross-sequence claim",
                             "Prior-only perturbation on identical frozen tensor bytes; CUDA reductions remain potentially nondeterministic",
                             "A selected candidate is not an accepted online pose; failures remain in every arm",
                             "The lower sensor loss or smaller displacement need not mean lower GT pose error"]}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if len(text.encode("utf-8")) > 1024*1024:
        raise ValueError("Analysis JSON exceeds website review limit")
    (folder/"analysis.json").write_text(text, encoding="utf-8")
    print(json.dumps({"run": args.run, "initial_error": initial_error,
                      "arms": [{k:x[k] for k in ("weight", "center_correction_m", "final_selected_candidate_pose_error", "failure_flags")} for x in summaries]}, indent=2))

if __name__ == "__main__":
    main()
