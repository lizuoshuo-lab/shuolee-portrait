"""CPU-only NPZ metric reaggregation and plots for the one frozen map evaluation.

No renderer, torch, optimization, registration, or source-artifact writes. This
checks saved predictions against the frozen loader's sensor values; it is not
an independent calibration or rasterizer correctness proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tum-v4-final-map-future10-001"
EVALUATOR_SHA = "0e6d8d976c924bf9ce492d2a405735d7cb0f4288b7da68a6518bf6589f970896"
LOADER_SHA = "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11"
PROTOCOL_SHA = "264ba4481cac19a1a2336481634a59b96a3bb123fc597dcbaafede05cf1b48be"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message: str) -> None:
    if not value:
        raise ValueError(message)


def write_json(path: Path, value) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require(len(data.encode("utf-8")) <= 1024 * 1024, "Public summary exceeds 1 MiB")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def compare_values(actual, recorded, path: str, audit: dict) -> None:
    if isinstance(actual, dict):
        for key, value in actual.items():
            require(key in recorded, "Missing recorded metric: " + path + "/" + key)
            compare_values(value, recorded[key], path + "/" + key, audit)
    elif actual is None or isinstance(actual, bool):
        require(actual == recorded, "Null/bool differs: " + path)
    else:
        require(isinstance(recorded, (int, float)) and math.isfinite(recorded), "Invalid scalar: " + path)
        error = abs(float(actual) - float(recorded))
        audit["numeric_leaves_checked"] += 1
        audit["max_absolute_scalar_difference"] = max(audit["max_absolute_scalar_difference"], error)
        require(error <= 1e-10 * (1. + abs(float(recorded))), "Metric reaggregation mismatch: " + path)


def depth_stats(prediction, truth, mask) -> dict:
    difference = prediction.astype(np.float64)[mask] - truth.astype(np.float64)[mask]
    count = int(mask.sum())
    absolute = float(np.sum(np.abs(difference), dtype=np.float64))
    squared = float(np.sum(difference*difference, dtype=np.float64))
    return {"pixels": count, "absolute_error_sum_m": absolute, "squared_error_sum_m2": squared,
            "mae_m": absolute/count if count else None, "rmse_m": math.sqrt(squared/count) if count else None}


def rgb_stats(prediction, truth) -> dict:
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    sse = float(np.sum(difference*difference, dtype=np.float64))
    count = difference.size
    mse = sse/count
    return {"squared_error_sum": sse, "channel_values": count, "mse": mse,
            "psnr_db": 10.*math.log10(1./mse) if mse else None, "psnr_positive_infinity": mse == 0}


def covered_mask(depth, alpha, valid):
    return valid & np.isfinite(depth) & (depth > 0) & (alpha > .5)


def sum_depth(rows):
    n = sum(r["pixels"] for r in rows)
    absolute = sum(r["absolute_error_sum_m"] for r in rows)
    squared = sum(r["squared_error_sum_m2"] for r in rows)
    return {"pixels": n, "absolute_error_sum_m": absolute, "squared_error_sum_m2": squared,
            "mae_m": absolute/n if n else None, "rmse_m": math.sqrt(squared/n) if n else None}


def stats(values: list[float]) -> dict:
    return {"n": len(values), "values_in_seed_order_0_1_2": values,
            "mean": float(np.mean(values)), "sample_standard_deviation_ddof1": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
            "minimum": min(values), "maximum": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tum-v4-final-map-future10-summary-001")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", args.run), "Invalid output directory name")
    output = ROOT / "outputs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    try:
        build(output)
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "exception_type": type(error).__name__,
                   "reason": str(error), "traceback": traceback.format_exc()})
        raise


def build(output: Path) -> None:
    require(digest(ROOT / "scripts/evaluate_tum_v4_final_maps.py") == EVALUATOR_SHA, "Frozen evaluator changed")
    require(digest(ROOT / "scripts/tum_rgbd.py") == LOADER_SHA, "Frozen loader changed")
    require(digest(SOURCE / "protocol_used.json") == PROTOCOL_SHA, "Executed protocol copy changed")
    metrics_sha = digest(SOURCE / "metrics.json")
    metrics = json.loads((SOURCE / "metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((SOURCE / "protocol_used.json").read_text(encoding="utf-8"))
    require(metrics["status"] == "completed" and metrics["completed_map_count"] == 12 and
            metrics["completed_render_count"] == 120, "Only the completed all-twelve-map experiment can supply this descriptive summary")
    require(metrics["protocol_sha256"] == PROTOCOL_SHA and metrics["optimization_steps"] == 0 and
            metrics["gt_used_for_fitting_or_map_updates"] is False, "Evaluation boundary differs")
    require(len(metrics["runs"]) == 12 and len(metrics["pairs"]) == 6, "Missing map or pair")
    from tum_rgbd import associate_frames, load_frames
    observations = {}
    for scene in ("desk", "xyz"):
        dataset = ROOT / f"data/tum/rgbd_dataset_freiburg1_{scene}"
        frames, _ = associate_frames(dataset, .02, .02, 32.)
        selected = [frames[i] for i in range(300, 310)]
        arrays = load_frames(dataset, selected, 160, 120, .2, 5.)
        for key, value in zip(("rgb", "depth", "valid_depth_mask"), arrays):
            record = protocol["datasets"][scene]["tensors"][key]
            require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record["sha256"], "Sensor array changed")
        observations[scene] = arrays
    audit = {"status": "passed", "numeric_leaves_checked": 0, "max_absolute_scalar_difference": 0.,
             "prediction_npz_files_checked": 0, "source_checkpoint_hashes_checked": 0,
             "scope": "CPU NumPy sums from all saved NPZ predictions and exact frozen-loader sensor arrays; not an independent renderer, calibration or physical-depth proof"}
    predictions, diagnostics = {}, {}
    compact_runs = []
    for row, frozen in zip(metrics["runs"], protocol["runs"]):
        require(row["run"] == frozen["run"] and row["status"] == "completed" and
                row["parameter_bytes_unchanged"] and row["original_checkpoint_bytes_unchanged"], "Map identity/invariance failure")
        require(digest(ROOT / frozen["checkpoint"]) == frozen["checkpoint_sha256"] == row["checkpoint_sha256"], "Source checkpoint changed")
        audit["source_checkpoint_hashes_checked"] += 1
        require([v["frame_id"] for v in row["views"]] == list(range(300, 310)), "Missing future view")
        sensor_rgb, sensor_depth, valid = observations[row["scene"]]
        calculated = []
        per_frame = []
        for i, view in enumerate(row["views"]):
            filename = (SOURCE / view["prediction"]).resolve()
            require(filename.is_relative_to(SOURCE.resolve()) and digest(filename) == view["prediction_sha256"], "Prediction path/hash mismatch")
            with np.load(filename, allow_pickle=False) as source:
                rgb, depth, alpha = source["rgb"].copy(), source["depth"].copy(), source["alpha"].copy()
            require(rgb.shape == (120, 160, 3) and depth.shape == alpha.shape == (120, 160) and
                    all(np.isfinite(x).all() for x in (rgb, depth, alpha)), "Invalid saved prediction")
            mask = covered_mask(depth, alpha, valid[i])
            actual = {"rgb": rgb_stats(rgb, sensor_rgb[i]), "valid_depth_pixels": int(valid[i].sum()),
                      "covered_valid_pixels": int(mask.sum()), "sensor_valid_coverage": float(mask.sum()/valid[i].sum()),
                      "covered_depth": depth_stats(depth, sensor_depth[i], mask),
                      "all_valid_holes_as_zero_depth": depth_stats(np.where(mask, depth, 0.), sensor_depth[i], valid[i])}
            compare_values(actual, view["metrics"], row["run"]+f"/{view['frame_id']}", audit)
            predictions[(row["run"], view["frame_id"])] = (depth, alpha)
            if row["seed"] == 0 and view["frame_id"] == 300:
                diagnostics[(row["scene"], row["prior_weight"])] = (rgb, depth, alpha)
            calculated.append(actual)
            per_frame.append({"frame_id": view["frame_id"], **actual})
            audit["prediction_npz_files_checked"] += 1
        sse = sum(x["rgb"]["squared_error_sum"] for x in calculated)
        count = sum(x["rgb"]["channel_values"] for x in calculated)
        valid_count = sum(x["valid_depth_pixels"] for x in calculated)
        covered_count = sum(x["covered_valid_pixels"] for x in calculated)
        mse = sse/count
        pooled = {"rgb": {"squared_error_sum": sse, "channel_values": count, "mse": mse,
                           "psnr_db": 10.*math.log10(1./mse) if mse else None, "psnr_positive_infinity": mse == 0},
                  "sensor_valid_pixels": valid_count, "covered_valid_pixels": covered_count,
                  "sensor_valid_coverage": covered_count/valid_count,
                  "covered_depth": sum_depth([x["covered_depth"] for x in calculated]),
                  "all_valid_holes_as_zero_depth": sum_depth([x["all_valid_holes_as_zero_depth"] for x in calculated])}
        compare_values(pooled, row["aggregate"]["pooled"], row["run"]+"/pooled", audit)
        compact_runs.append({key: row[key] for key in ("run", "scene", "seed", "prior_weight", "checkpoint_sha256", "status")}
                            | {"aggregate": row["aggregate"], "per_frame": per_frame})
    for pair in metrics["pairs"]:
        _, depth_truth, valid = observations[pair["scene"]]
        values_a, values_b = [], []
        common_pixels = sensor_pixels = 0
        for i, view in enumerate(pair["views"]):
            frame_id = view["frame_id"]
            depth_a, alpha_a = predictions[(pair["prior0_run"], frame_id)]
            depth_b, alpha_b = predictions[(pair["prior001_run"], frame_id)]
            common = covered_mask(depth_a, alpha_a, valid[i]) & covered_mask(depth_b, alpha_b, valid[i])
            a, b = depth_stats(depth_a, depth_truth[i], common), depth_stats(depth_b, depth_truth[i], common)
            actual = {"sensor_valid_pixels": int(valid[i].sum()), "intersection_pixels": int(common.sum()),
                      "intersection_sensor_valid_coverage": float(common.sum()/valid[i].sum()), "prior0_depth": a, "prior001_depth": b,
                      "mae_difference_prior001_minus_prior0_m": b["mae_m"]-a["mae_m"] if a["pixels"] else None,
                      "rmse_difference_prior001_minus_prior0_m": b["rmse_m"]-a["rmse_m"] if a["pixels"] else None}
            compare_values(actual, view, f"{pair['scene']}/seed{pair['seed']}/common/{frame_id}", audit)
            values_a.append(a)
            values_b.append(b)
            common_pixels += int(common.sum())
            sensor_pixels += int(valid[i].sum())
        compare_values({"intersection_pixels": common_pixels, "sensor_valid_pixels": sensor_pixels,
                        "intersection_sensor_valid_coverage": common_pixels/sensor_pixels,
                        "prior0_depth": sum_depth(values_a), "prior001_depth": sum_depth(values_b)},
                       pair["pooled"], f"{pair['scene']}/seed{pair['seed']}/common/pooled", audit)
    scalar_paths = {"pooled_rgb_psnr_db": ("rgb", "psnr_db"), "pooled_sensor_valid_coverage": ("sensor_valid_coverage",),
                    "pooled_covered_depth_mae_m": ("covered_depth", "mae_m"),
                    "pooled_covered_depth_rmse_m": ("covered_depth", "rmse_m"),
                    "pooled_all_valid_holes_as_zero_rmse_m": ("all_valid_holes_as_zero_depth", "rmse_m")}
    def get_scalar(row, key):
        value = row["aggregate"]["pooled"]
        for part in scalar_paths[key]:
            value = value[part]
        return value
    groups, paired_differences = [], []
    for scene in ("desk", "xyz"):
        arms = {}
        for weight in (0., .01):
            arm = sorted([r for r in compact_runs if r["scene"] == scene and r["prior_weight"] == weight], key=lambda r: r["seed"])
            require([r["seed"] for r in arm] == [0, 1, 2], "Seed dropped from summary")
            arms[weight] = arm
            groups.append({"scene": scene, "prior_weight": weight,
                           "statistics_over_three_per_map_pooled_metrics": {key: stats([get_scalar(r, key) for r in arm]) for key in scalar_paths}})
        paired_differences.append({"scene": scene, "difference_direction": "prior001 minus prior0 at same seed",
                                   "statistics": {key: stats([get_scalar(b, key)-get_scalar(a, key) for a, b in zip(arms[0.], arms[.01])]) for key in scalar_paths}})
    figures = make_plots(compact_runs, metrics["pairs"], output)
    figures += make_diagnostics(observations, diagnostics, output)
    require(digest(SOURCE / "metrics.json") == metrics_sha, "Source metrics changed during summary")
    require("torch" not in sys.modules and "gsplat" not in sys.modules, "CPU summarizer imported GPU framework")
    audit.update(torch_imported=False, gsplat_imported=False, original_metrics_sha256_unchanged=True)
    result = {"status": "completed_cpu_summary", "source_run": SOURCE.relative_to(ROOT).as_posix(),
              "source_metrics_sha256": metrics_sha, "executed_protocol_sha256": PROTOCOL_SHA,
              "summary_script": "scripts/summarize_tum_v4_final_maps.py", "summary_script_sha256": digest(Path(__file__)),
              "frozen_evaluator_sha256": EVALUATOR_SHA, "frozen_loader_sha256": LOADER_SHA,
              "map_count": 12, "render_count": 120, "failed_map_count": 0, "failed_render_count": 0,
              "settings": metrics["settings"], "limitations": metrics["limitations"],
              "statistical_scope": "Descriptive mean and ddof=1 sample SD of three per-map ten-view pooled metrics; not IID evidence, significance, confidence intervals or generalized superiority",
              "groups": groups, "paired_differences": paired_differences, "runs": compact_runs,
              "common_coverage_pairs": metrics["pairs"], "figures": figures,
              "diagnostic_frame_selection": {"frame_id": 300, "seed": 0, "rule": "Fixed in advance by reviewer, not chosen by measured quality",
                                              "ED_display_limits_m": [0., 5.], "absolute_depth_residual_display_limits_m": [0., .5],
                                              "black_in_mask_or_residual": "Sensor-valid pixel without rendered coverage, not a low error",
                                              "gray_in_mask_or_residual": "Sensor depth outside the valid mask",
                                              "no_geometric_alignment_or_image_enhancement": True,
                                              "values_above_color_limit": "Saturated display only; numerical metrics and NPZ retain uncapped values"},
              "validation": "validation.json", "raw_npz_policy": "120 prediction NPZ files remain in original local evaluation output; not copied to website"}
    write_json(output / "summary.json", result)
    write_json(output / "validation.json", audit)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "status": result["status"],
                      "maps": 12, "renders": 120, "cpu_reaggregation": audit, "groups": groups}, indent=2))


def make_plots(runs, pairs, output):
    colors = ("#2563eb", "#ef7d32")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    main, axes = plt.subplots(2, 3, figsize=(13.3, 7.7), constrained_layout=True)
    detail, d_axes = plt.subplots(2, 3, figsize=(13.3, 7.7), constrained_layout=True)
    x, width = np.arange(3), .34
    for row_index, scene in enumerate(("desk", "xyz")):
        scene_pairs = sorted([p for p in pairs if p["scene"] == scene], key=lambda p: p["seed"])
        depth_limits = []
        for arm_index, weight in enumerate((0., .01)):
            arm = sorted([r for r in runs if r["scene"] == scene and r["prior_weight"] == weight], key=lambda r: r["seed"])
            pooled = [r["aggregate"]["pooled"] for r in arm]
            label = f"lambda = {weight:g}"
            xx = x + (arm_index-.5)*width
            panels = ([p["rgb"]["psnr_db"] for p in pooled],
                      [100*p["sensor_valid_coverage"] for p in pooled],
                      [p["all_valid_holes_as_zero_depth"]["rmse_m"] for p in pooled])
            for column, values in enumerate(panels):
                axes[row_index, column].bar(xx, values, width=width, color=colors[arm_index], label=label)
            own = [p["covered_depth"]["rmse_m"] for p in pooled]
            shared = [p["pooled"]["prior0_depth" if weight == 0 else "prior001_depth"]["rmse_m"] for p in scene_pairs]
            depth_limits.extend(own+shared)
            for column, values in enumerate((own, shared)):
                d_axes[row_index, column].bar(xx, values, width=width, color=colors[arm_index], label=label)
        joint_coverage = [100*p["pooled"]["intersection_sensor_valid_coverage"] for p in scene_pairs]
        d_axes[row_index, 2].bar(x, joint_coverage, width=.56, color="#687a88")
        for column in (0, 1):
            d_axes[row_index, column].set_ylim(0, max(depth_limits)*1.13)
        axes[row_index, 1].set_ylim(0, 100)
        d_axes[row_index, 2].set_ylim(0, 100)
        for array in (axes, d_axes):
            for column in range(3):
                axis = array[row_index, column]
                axis.set_xticks(x, ["seed 0", "seed 1", "seed 2"])
                axis.set_xlabel(f"TUM fr1_{scene}")
                axis.grid(axis="y", alpha=.2)
                axis.set_axisbelow(True)
        for column, label in enumerate(("RGB PSNR, pooled (dB; higher better)", "Sensor-valid coverage (%)", "Holes-as-zero depth RMSE (m; lower better)")):
            axes[row_index, column].set_ylabel(label)
        for column, label in enumerate(("Own-covered depth RMSE (m)", "Common-covered depth RMSE (m)", "Common sensor-valid coverage (%)")):
            d_axes[row_index, column].set_ylabel(label)
    axes[0, 0].legend(loc="upper left", fontsize=9)
    d_axes[0, 0].legend(loc="upper left", fontsize=9)
    main.suptitle("Fixed final maps, known cameras, future frames 300-309\nAll 12 maps / 120 views; no map fitting or pose estimation", fontsize=14)
    detail.suptitle("Depth denominators: each map's coverage versus pairwise intersection\nStrict alpha > 0.5 and valid positive depth; pooled over 10 views", fontsize=14)
    records = []
    for figure, name in ((main, "comparison.png"), (detail, "depth-comparison.png")):
        filename = output / name
        figure.savefig(filename, dpi=170)
        plt.close(figure)
        records.append({"path": name, "sha256": digest(filename)})
    return records


def make_diagnostics(observations, predictions, output):
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch
    # Both sequences/methods use these fixed limits, independent of observations.
    depth_norm, residual_norm, alpha_norm = Normalize(0., 5., clip=True), Normalize(0., .5, clip=True), Normalize(0., 1., clip=True)
    depth_cmap, residual_cmap = plt.get_cmap("turbo").copy(), plt.get_cmap("viridis").copy()
    depth_cmap.set_bad("#a0a0a0")
    mask_cmap = ListedColormap(["#a0a0a0", "#000000", "#64d6ac"])
    records = []
    for scene in ("desk", "xyz"):
        rgb_observed, depths_observed, valid = observations[scene]
        rgb_observed, depth_observed, mask_observed = rgb_observed[0], depths_observed[0], valid[0]
        figure, axes = plt.subplots(3, 5, figsize=(17.8, 9.7), constrained_layout=True)
        axes[0, 0].imshow(rgb_observed)
        axes[0, 1].imshow(np.ma.masked_where(~mask_observed, depth_observed), cmap=depth_cmap, norm=depth_norm)
        axes[0, 3].imshow(np.where(mask_observed, 2, 0), cmap=mask_cmap, vmin=0, vmax=2)
        for column, message in ((2, "No observed alpha"), (4, "Reference depth\nNo residual defined")):
            axes[0, column].text(.5, .5, message, transform=axes[0, column].transAxes, ha="center", va="center", color="#777777")
        for row_index, weight in enumerate((0., .01), 1):
            rgb, depth, alpha = predictions[(scene, weight)]
            covered = covered_mask(depth, alpha, mask_observed)
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 1].imshow(depth, cmap=depth_cmap, norm=depth_norm)
            axes[row_index, 2].imshow(alpha, cmap="gray", norm=alpha_norm)
            classification = np.where(~mask_observed, 0, np.where(covered, 2, 1))
            axes[row_index, 3].imshow(classification, cmap=mask_cmap, vmin=0, vmax=2)
            residual_rgba = residual_cmap(residual_norm(np.abs(depth.astype(np.float64)-depth_observed.astype(np.float64))))
            residual_rgba[mask_observed & ~covered] = [0., 0., 0., 1.]
            residual_rgba[~mask_observed] = [160/255, 160/255, 160/255, 1.]
            axes[row_index, 4].imshow(residual_rgba)
        for column, title in enumerate(("RGB", "Sensor depth / rendered ED", "Rendered alpha", "Sensor validity / coverage", "Absolute depth residual")):
            axes[0, column].set_title(title, fontsize=11)
        for row_index, title in enumerate(("Observed RGB-D", "lambda = 0 | seed 0", "lambda = 0.01 | seed 0")):
            axes[row_index, 0].set_ylabel(title, fontsize=11)
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        figure.colorbar(ScalarMappable(norm=depth_norm, cmap=depth_cmap), ax=axes[:, 1], shrink=.65, label="z / ED (m), fixed 0..5")
        figure.colorbar(ScalarMappable(norm=alpha_norm, cmap="gray"), ax=axes[1:, 2], shrink=.7, label="alpha, fixed 0..1")
        figure.colorbar(ScalarMappable(norm=residual_norm, cmap=residual_cmap), ax=axes[:, 4], shrink=.65, label="Absolute residual (m), fixed 0..0.5")
        figure.suptitle(f"TUM fr1_{scene} | preselected frame 300 | known camera, anchored at frame 200\n"
                       "Black in coverage/residual = uncovered valid sensor pixel, NOT low error; gray = invalid sensor depth.\n"
                       "Residual color saturates at 0.5 m; ED at 5 m. Raw values retained; no alignment or image enhancement.", fontsize=12)
        # The same green identifies valid observed depth or covered valid pixels.
        axes[0, 3].legend(handles=[Patch(color="#64d6ac", label="Valid sensor depth"),
                                  Patch(color="#a0a0a0", label="Invalid sensor depth")],
                          loc="lower center", bbox_to_anchor=(.5, -.21), fontsize=8, frameon=False)
        axes[2, 3].legend(handles=[Patch(color="#64d6ac", label="Covered valid depth"),
                                  Patch(color="#000000", label="Uncovered valid depth")],
                          loc="lower center", bbox_to_anchor=(.5, -.21), fontsize=8, frameon=False)
        filename = output / f"diagnostic-{scene}-frame300-seed0.png"
        figure.savefig(filename, dpi=160)
        plt.close(figure)
        records.append({"path": filename.name, "sha256": digest(filename), "scene": scene, "frame_id": 300, "seed": 0})
    return records


if __name__ == "__main__":
    main()
