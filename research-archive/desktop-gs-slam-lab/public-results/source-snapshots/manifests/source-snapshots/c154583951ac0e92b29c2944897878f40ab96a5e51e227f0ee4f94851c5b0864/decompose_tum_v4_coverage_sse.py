"""CPU-only exact coverage/SSE decomposition of the completed fixed-map study.

Compare lambda=.01 (on) minus lambda=0 (off) on the same sensor-valid pixels.
No rendering, optimization, alignment, mask tuning, or writes to source runs.
The depth reference is observed RGB-D depth, not a claim of true surface geometry.
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
PROTOCOL_SHA = "264ba4481cac19a1a2336481634a59b96a3bb123fc597dcbaafede05cf1b48be"
LOADER_SHA = "471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11"
GROUPS = ("common_covered", "newly_covered_on_only", "lost_covered_off_only", "never_covered")
FORMULAS = {
    "common_covered": "sum((z_on-z_sensor)^2-(z_off-z_sensor)^2)",
    "newly_covered_on_only": "sum((z_on-z_sensor)^2-z_sensor^2)",
    "lost_covered_off_only": "sum(z_sensor^2-(z_off-z_sensor)^2)",
    "never_covered": "0: both holes-as-zero predictions equal zero",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require(len(data.encode("utf-8")) <= 1024*1024, "JSON output exceeds 1 MiB")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def cover(depth, alpha, valid):
    return valid & (alpha > .5) & np.isfinite(depth) & (depth > 0)


def close(actual, expected, label, checks):
    difference = abs(float(actual)-float(expected))
    checks["scalar_comparisons"] += 1
    if difference > checks["max_absolute_scalar_difference"]:
        checks["max_absolute_scalar_difference"] = difference
        checks["largest_difference_field"] = label
    require(difference <= 1e-8 + 1e-12*abs(float(expected)), "SSE identity mismatch: " + label)


def group_record(mask, sensor_depth, off_error2, on_error2, off_covered, on_covered):
    reference = sensor_depth[mask]
    count = int(mask.sum())
    target_sum = float(reference.sum(dtype=np.float64))
    target_squared_sum = float(np.square(reference).sum(dtype=np.float64))
    off_render_error = float(off_error2[mask].sum(dtype=np.float64)) if off_covered else None
    on_render_error = float(on_error2[mask].sum(dtype=np.float64)) if on_covered else None
    off_sse = off_render_error if off_covered else target_squared_sum
    on_sse = on_render_error if on_covered else target_squared_sum
    return {"pixels": count, "target_depth_sum_m": target_sum,
            "target_depth_mean_m": target_sum/count if count else None,
            "target_depth_squared_sum_m2": target_squared_sum,
            "covered_render_error_squared_sum_off_m2": off_render_error,
            "covered_render_error_squared_sum_on_m2": on_render_error,
            "holes_as_zero_squared_error_sum_off_m2": off_sse,
            "holes_as_zero_squared_error_sum_on_m2": on_sse,
            "contribution_delta_sse_on_minus_off_m2": on_sse-off_sse,
            "mask": array_hash(mask)}


def decompose(off_depth, off_alpha, on_depth, on_alpha, target, valid):
    require(off_depth.shape == off_alpha.shape == on_depth.shape == on_alpha.shape == target.shape == valid.shape,
            "Prediction/observation shape mismatch")
    require(all(np.isfinite(x).all() for x in (off_depth, off_alpha, on_depth, on_alpha, target)), "Nonfinite source values")
    target = target.astype(np.float64)
    off_depth, on_depth = off_depth.astype(np.float64), on_depth.astype(np.float64)
    off, on = cover(off_depth, off_alpha, valid), cover(on_depth, on_alpha, valid)
    masks = {"common_covered": off & on,
             "newly_covered_on_only": valid & on & ~off,
             "lost_covered_off_only": valid & off & ~on,
             "never_covered": valid & ~off & ~on}
    membership = sum(mask.astype(np.int8) for mask in masks.values())
    require(np.array_equal(membership, valid.astype(np.int8)), "Four coverage groups are not a disjoint sensor-valid partition")
    off_error2, on_error2 = np.square(off_depth-target), np.square(on_depth-target)
    groups = {name: group_record(mask, target, off_error2, on_error2,
                                 name in ("common_covered", "lost_covered_off_only"),
                                 name in ("common_covered", "newly_covered_on_only")) for name, mask in masks.items()}
    n = int(valid.sum())
    require(n > 0, "Empty sensor validity denominator")
    off_sse = float(np.square(np.where(off, off_depth, 0.)[valid]-target[valid]).sum(dtype=np.float64))
    on_sse = float(np.square(np.where(on, on_depth, 0.)[valid]-target[valid]).sum(dtype=np.float64))
    delta = on_sse-off_sse
    decomposed = math.fsum(group["contribution_delta_sse_on_minus_off_m2"] for group in groups.values())
    for group in groups.values():
        group["contribution_delta_mse_on_minus_off_m2"] = group["contribution_delta_sse_on_minus_off_m2"]/n
    record = {"sensor_valid_pixels": n, "covered_pixels_off": int(off.sum()), "covered_pixels_on": int(on.sum()),
              "coverage_off": float(off.sum()/n), "coverage_on": float(on.sum()/n),
              "covered_pixel_change_on_minus_off": int(on.sum())-int(off.sum()),
              "sensor_valid_mask": array_hash(valid), "coverage_mask_off": array_hash(off), "coverage_mask_on": array_hash(on),
              "groups": groups, "direct_holes_as_zero_sse_off_m2": off_sse, "direct_holes_as_zero_sse_on_m2": on_sse,
              "direct_delta_sse_on_minus_off_m2": delta, "sum_group_delta_sse_m2": decomposed,
              "decomposition_residual_m2": decomposed-delta,
              "direct_holes_as_zero_mse_off_m2": off_sse/n, "direct_holes_as_zero_mse_on_m2": on_sse/n,
              "direct_delta_mse_on_minus_off_m2": delta/n,
              "holes_as_zero_rmse_off_m": math.sqrt(off_sse/n), "holes_as_zero_rmse_on_m": math.sqrt(on_sse/n),
              "rmse_difference_on_minus_off_m": math.sqrt(on_sse/n)-math.sqrt(off_sse/n),
              "rmse_difference_is_not_a_sum_of_group_rmse_differences": True}
    return record, masks


def pooled_record(frames, all_masks):
    n = sum(row["sensor_valid_pixels"] for row in frames)
    result = {"sensor_valid_pixels": n,
              "covered_pixels_off": sum(row["covered_pixels_off"] for row in frames),
              "covered_pixels_on": sum(row["covered_pixels_on"] for row in frames), "groups": {}}
    for name in GROUPS:
        rows = [frame["groups"][name] for frame in frames]
        group = {}
        for key in ("pixels", "target_depth_sum_m", "target_depth_squared_sum_m2",
                    "holes_as_zero_squared_error_sum_off_m2", "holes_as_zero_squared_error_sum_on_m2",
                    "contribution_delta_sse_on_minus_off_m2"):
            group[key] = sum(row[key] for row in rows)
        for key in ("covered_render_error_squared_sum_off_m2", "covered_render_error_squared_sum_on_m2"):
            group[key] = sum(row[key] for row in rows) if all(row[key] is not None for row in rows) else None
        group["target_depth_mean_m"] = group["target_depth_sum_m"]/group["pixels"] if group["pixels"] else None
        group["contribution_delta_mse_on_minus_off_m2"] = group["contribution_delta_sse_on_minus_off_m2"]/n
        group["mask_stack_in_frame_order_300_309"] = array_hash(np.stack([masks[name] for masks in all_masks]))
        result["groups"][name] = group
    off = math.fsum(frame["direct_holes_as_zero_sse_off_m2"] for frame in frames)
    on = math.fsum(frame["direct_holes_as_zero_sse_on_m2"] for frame in frames)
    result.update(coverage_off=result["covered_pixels_off"]/n, coverage_on=result["covered_pixels_on"]/n,
                  covered_pixel_change_on_minus_off=result["covered_pixels_on"]-result["covered_pixels_off"],
                  direct_holes_as_zero_sse_off_m2=off, direct_holes_as_zero_sse_on_m2=on,
                  direct_delta_sse_on_minus_off_m2=on-off,
                  sum_group_delta_sse_m2=math.fsum(group["contribution_delta_sse_on_minus_off_m2"] for group in result["groups"].values()),
                  direct_holes_as_zero_mse_off_m2=off/n, direct_holes_as_zero_mse_on_m2=on/n,
                  direct_delta_mse_on_minus_off_m2=(on-off)/n,
                  holes_as_zero_rmse_off_m=math.sqrt(off/n), holes_as_zero_rmse_on_m=math.sqrt(on/n),
                  rmse_difference_on_minus_off_m=math.sqrt(on/n)-math.sqrt(off/n),
                  rmse_difference_is_not_additive=True)
    result["decomposition_residual_m2"] = result["sum_group_delta_sse_m2"]-(on-off)
    return result


def synthetic_control():
    target = np.array([[1., 2., 3., 4., 5.]])
    off = np.array([[1.2, 9., 3.3, 9., 2.]])
    on = np.array([[1.1, 2.2, 9., 9., 2.]])
    off_alpha = np.array([[1., .5, 1., 0., 1.]])
    on_alpha = np.array([[1., 1., 0., 0., 1.]])
    valid = np.array([[True, True, True, True, False]])
    row, _ = decompose(off, off_alpha, on, on_alpha, target, valid)
    expected = {"common_covered": -.03, "newly_covered_on_only": -3.96,
                "lost_covered_off_only": 8.91, "never_covered": 0.}
    for name, value in expected.items():
        require(row["groups"][name]["pixels"] == 1 and abs(row["groups"][name]["contribution_delta_sse_on_minus_off_m2"]-value) < 1e-12,
                "Synthetic group arithmetic failed")
    require(abs(row["direct_delta_sse_on_minus_off_m2"]-4.92) < 1e-12 and row["sensor_valid_pixels"] == 4,
            "Synthetic total or invalid-sensor exclusion failed")
    return {"status": "passed", "each_group_has_one_valid_pixel": True, "invalid_sensor_pixel_excluded": True,
            "alpha_exactly_half_is_uncovered": True, "expected_group_delta_sse_m2": expected,
            "expected_total_delta_sse_m2": 4.92, "expected_delta_mse_m2": 1.23}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tum-v4-coverage-sse-decomposition-001")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", args.run), "Invalid unique output name")
    output = ROOT / "outputs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    try:
        build(output)
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "exception_type": type(error).__name__,
                   "reason": str(error), "traceback": traceback.format_exc()})
        raise


def build(output):
    require(sha(ROOT / "scripts/tum_rgbd.py") == LOADER_SHA, "Frozen loader changed")
    require(sha(SOURCE / "protocol_used.json") == PROTOCOL_SHA, "Executed protocol changed")
    source_metrics_sha = sha(SOURCE / "metrics.json")
    metrics, protocol = read_json(SOURCE / "metrics.json"), read_json(SOURCE / "protocol_used.json")
    require(metrics["status"] == "completed" and metrics["completed_map_count"] == 12 and metrics["completed_render_count"] == 120,
            "This artifact requires the complete predefined all-map experiment")
    require(metrics["settings"]["alpha_threshold_strictly_greater"] == .5 and
            metrics["settings"]["sensor_valid_depth_m"] == [.2, 5.], "Fixed mask thresholds changed")
    controls = synthetic_control()
    from tum_rgbd import associate_frames, load_frames
    sensors = {}
    for scene in ("desk", "xyz"):
        dataset = ROOT / f"data/tum/rgbd_dataset_freiburg1_{scene}"
        frames, _ = associate_frames(dataset, .02, .02, 32.)
        _, depths, masks = load_frames(dataset, [frames[i] for i in range(300, 310)], 160, 120, .2, 5.)
        for key, value in (("depth", depths), ("valid_depth_mask", masks)):
            require(array_hash(value)["sha256"] == protocol["datasets"][scene]["tensors"][key]["sha256"], "Sensor reference changed")
        sensors[scene] = depths, masks
    by_run = {row["run"]: row for row in metrics["runs"]}
    require(len(by_run) == 12, "Missing or duplicate source map")
    checks = {"status": "passed", "scalar_comparisons": 0, "max_absolute_scalar_difference": 0.,
              "largest_difference_field": None, "predictions_checked": 0, "paired_frames_checked": 0,
              "mask_partition_controls_passed": True, "synthetic_control": controls}
    pairs = []
    for pair in metrics["pairs"]:
        require(pair["completed_views"] == 10, "Missing paired view")
        off_row, on_row = by_run[pair["prior0_run"]], by_run[pair["prior001_run"]]
        require(off_row["prior_weight"] == 0. and on_row["prior_weight"] == .01 and
                off_row["scene"] == on_row["scene"] == pair["scene"] and off_row["seed"] == on_row["seed"] == pair["seed"],
                "Pair does not use same scene/seed and prescribed on/off weights")
        frames, all_masks = [], []
        sensor_depths, valid_masks = sensors[pair["scene"]]
        require([row["frame_id"] for row in off_row["views"]] == [row["frame_id"] for row in on_row["views"]] == list(range(300, 310)),
                "Paired frame IDs/order differ")
        for index, (off_view, on_view) in enumerate(zip(off_row["views"], on_row["views"])):
            arrays, provenance = [], []
            for view in (off_view, on_view):
                filename = (SOURCE / view["prediction"]).resolve()
                require(filename.is_relative_to(SOURCE.resolve()) and sha(filename) == view["prediction_sha256"], "Unsafe or changed NPZ")
                with np.load(filename, allow_pickle=False) as source:
                    arrays.append((source["depth"].copy(), source["alpha"].copy()))
                provenance.append({"prediction": view["prediction"], "sha256": view["prediction_sha256"]})
                checks["predictions_checked"] += 1
            record, masks = decompose(*arrays[0], *arrays[1], sensor_depths[index], valid_masks[index])
            record.update(frame_id=off_view["frame_id"], source_off=provenance[0], source_on=provenance[1])
            close(record["sum_group_delta_sse_m2"], record["direct_delta_sse_on_minus_off_m2"], "frame/group sum", checks)
            for arm, view in (("off", off_view), ("on", on_view)):
                reference = view["metrics"]["all_valid_holes_as_zero_depth"]
                require(record["sensor_valid_pixels"] == reference["pixels"], "Sensor-valid N differs")
                close(record[f"direct_holes_as_zero_sse_{arm}_m2"], reference["squared_error_sum_m2"], "frame/original holes SSE", checks)
                close(record[f"direct_holes_as_zero_mse_{arm}_m2"]*record["sensor_valid_pixels"], reference["rmse_m"]**2*reference["pixels"], "frame/MSE times N", checks)
            frames.append(record)
            all_masks.append(masks)
            checks["paired_frames_checked"] += 1
        pooled = pooled_record(frames, all_masks)
        close(pooled["sum_group_delta_sse_m2"], pooled["direct_delta_sse_on_minus_off_m2"], "pooled/group sum", checks)
        for arm, row in (("off", off_row), ("on", on_row)):
            reference = row["aggregate"]["pooled"]["all_valid_holes_as_zero_depth"]
            require(pooled["sensor_valid_pixels"] == reference["pixels"], "Pooled N differs")
            close(pooled[f"direct_holes_as_zero_sse_{arm}_m2"], reference["squared_error_sum_m2"], "pooled/original holes SSE", checks)
            close(pooled[f"direct_holes_as_zero_mse_{arm}_m2"]*pooled["sensor_valid_pixels"], reference["rmse_m"]**2*reference["pixels"], "pooled/MSE times N", checks)
        original_delta = on_row["aggregate"]["pooled"]["all_valid_holes_as_zero_depth"]["squared_error_sum_m2"] - off_row["aggregate"]["pooled"]["all_valid_holes_as_zero_depth"]["squared_error_sum_m2"]
        close(pooled["sum_group_delta_sse_m2"], original_delta, "pooled/original delta SSE", checks)
        pairs.append({"scene": pair["scene"], "seed": pair["seed"], "off_run": off_row["run"], "on_run": on_row["run"],
                      "off_checkpoint_sha256": off_row["checkpoint_sha256"], "on_checkpoint_sha256": on_row["checkpoint_sha256"],
                      "pooled": pooled, "frames": frames})
    require(len(pairs) == 6 and {(p["scene"], p["seed"]) for p in pairs} == {(s, i) for s in ("desk", "xyz") for i in (0, 1, 2)}, "Omitted seed/scene")
    require(checks["paired_frames_checked"] == 60 and checks["predictions_checked"] == 120, "Missing prediction or paired frame")
    require(sha(SOURCE / "metrics.json") == source_metrics_sha, "Original source metrics changed")
    require("torch" not in sys.modules and "gsplat" not in sys.modules, "CPU decomposition imported GPU framework")
    checks.update(torch_imported=False, gsplat_imported=False, original_metrics_unchanged=True)
    make_plot(pairs, output / "decomposition.png")
    result = {"status": "completed_cpu_decomposition", "source_metrics": (SOURCE / "metrics.json").relative_to(ROOT).as_posix(),
              "source_metrics_sha256": source_metrics_sha, "executed_protocol_sha256": PROTOCOL_SHA,
              "source_script": "scripts/decompose_tum_v4_coverage_sse.py", "source_script_sha256": sha(Path(__file__)),
              "frozen_loader_sha256": LOADER_SHA, "paired_map_count": 6, "source_map_count": 12,
              "paired_frame_count": 60, "source_prediction_count": 120,
              "reference": "Same observed TUM RGB-D sensor depth and fixed sensor-valid mask; not claimed true surface depth",
              "direction": "on=lambda .01 minus off=lambda 0, same seed and scene",
              "definitions": {"sensor_valid": "finite observed depth in inclusive [0.2,5] m, frozen loader",
                              "covered": "sensor_valid & alpha>0.5 & finite rendered ED & rendered ED>0",
                              "group_contributions": FORMULAS,
                              "total_identity": "delta SSE = common + newly covered + lost covered + never(0)",
                              "delta_MSE": "each delta SSE term divided by the same sensor-valid N within that frame or pooled ten-view pair",
                              "delta_RMSE": "sqrt(SSE_on/N)-sqrt(SSE_off/N); not additive over groups"},
              "limitations": ["Exact metric accounting, not a unique physical or causal explanation of map errors",
                              "No mask threshold selection or adjustment from GT; no new GPU render, model update or camera alignment",
                              "Coverage count alone ignores which pixels/depths become covered or lost",
                              "Own-mask and common-mask depth errors have different supports; descriptive conditional metrics are not interchangeable",
                              "All three seeds and both same-domain sequences retained; no best-seed selection or broad superiority claim"],
              "pairs": pairs, "plot": {"path": "decomposition.png", "sha256": sha(output / "decomposition.png"),
                                      "same_y_scale_across_scenes": True, "black_diamond": "total delta MSE, not uncertainty"},
              "validation": "validation.json"}
    write_json(output / "decomposition.json", result)
    write_json(output / "validation.json", checks)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "validation": checks,
                      "pooled": [{"scene": p["scene"], "seed": p["seed"], "delta_covered_pixels": p["pooled"]["covered_pixel_change_on_minus_off"],
                                  "delta_mse_m2": p["pooled"]["direct_delta_mse_on_minus_off_m2"],
                                  "terms_mse_m2": {name: p["pooled"]["groups"][name]["contribution_delta_mse_on_minus_off_m2"] for name in GROUPS}}
                                 for p in pairs]}, indent=2))


def make_plot(pairs, filename):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True, constrained_layout=True)
    colors = ("#3b82f6", "#16a085", "#ed8936")
    labels = ("Common covered", "Newly covered (on only)", "Lost covered (off only)")
    ymin, ymax = 0., 0.
    for axis, scene in zip(axes, ("desk", "xyz")):
        rows = sorted([p for p in pairs if p["scene"] == scene], key=lambda p: p["seed"])
        x = np.arange(3)
        positive, negative = np.zeros(3), np.zeros(3)
        for name, color, label in zip(GROUPS[:3], colors, labels):
            values = np.array([p["pooled"]["groups"][name]["contribution_delta_mse_on_minus_off_m2"] for p in rows])
            bottom = np.where(values >= 0, positive, negative)
            axis.bar(x, values, bottom=bottom, color=color, width=.55, label=label)
            positive += np.maximum(values, 0.)
            negative += np.minimum(values, 0.)
        totals = [p["pooled"]["direct_delta_mse_on_minus_off_m2"] for p in rows]
        axis.plot(x, totals, "D", color="#111111", markersize=6, label="Total delta MSE")
        axis.axhline(0, color="#111111", linewidth=.9)
        axis.set_xticks(x, ["seed 0", "seed 1", "seed 2"])
        axis.set_title(f"TUM fr1_{scene}")
        axis.set_xlabel("Ten future frames 300-309 pooled; same sensor-valid N per pair")
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        ymin, ymax = min(ymin, float(negative.min())), max(ymax, float(positive.max()))
    margin = max(ymax-ymin, .01)*.15
    axes[0].set_ylim(ymin-margin, ymax+margin)
    axes[0].set_ylabel("Contribution to holes-as-zero delta MSE (m²)\non (lambda .01) minus off (lambda 0)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("Exact coverage decomposition on the same observed depth reference\n"
                    "Positive raises error; negative lowers error. Never-covered contribution = 0.\n"
                    "Black diamond = algebraic total, not uncertainty; RMSE differences are not additive.", fontsize=12)
    figure.savefig(filename, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
