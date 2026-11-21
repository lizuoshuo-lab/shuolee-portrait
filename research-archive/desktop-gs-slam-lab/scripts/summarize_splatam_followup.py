"""Summarize independently verified motion-sequence and resolution runs locally.

The first project's receipts and metrics remain unchanged. New results require
actual completed execution, independent trajectory evaluation and runtime audit.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

LAB = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("run_name")
    value["wandb"].pop("name")
    for key in ("sequence", "gradslam_data_cfg", "desired_image_width", "desired_image_height"):
        value["data"].pop(key)
    return value


def summarize(name: str, label: str, sequence: str, resolution: list[int], out: Path) -> tuple[dict, dict, dict]:
    if Path(name).name != name or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise SystemExit("Run name must be one local directory name")
    run = LAB / "outputs/splatam-official-runs-001" / name
    eval_dir = LAB / f"outputs/splatam-independent-eval-{label}-001"
    audit_path = eval_dir / name / "metrics.json"
    receipt = read(run / "execution.json")
    metric = read(eval_dir / "metrics.json")
    audit = read(audit_path)
    if not (receipt["status"] == "completed_pending_independent_evaluation"
            and receipt["child_exit_code"] == 0
            and metric["status"] == "independently_verified" and all(metric["checks"].values())
            and metric["run_name"] == name and metric["full_associated_sequence"]
            and metric["sequence"] == sequence and metric["resolution"] == resolution
            and audit["status"] == "passed" and all(audit["checks"].values())
            and audit["run_name"] == name and audit["input_execution_sha256"] == sha(run / "execution.json")
            and receipt["params_sha256"] == sha(run / "params.npz")):
        raise SystemExit(f"Completed execution or independent evidence mismatch: {name}")
    config = read(run / "effective-config.json")
    sampled = [i for i in range(metric["frames"]) if i == 0 or (i + 1) % config["eval_every"] == 0]
    plot_indices = sorted(int(path.stem) for path in (run / "upstream-eval/plots").glob("*.png"))
    if plot_indices != sampled:
        raise SystemExit(f"Final render plots do not cover the actual fixed sample schedule: {name}")
    rendering = {"sampled_frame_indices": sampled, "sampled_frame_count": len(sampled)}
    for filename, key in (("psnr", "psnr_db"), ("ssim", "ms_ssim"), ("lpips", "lpips"), ("l1", "depth_mae_m")):
        values = np.atleast_1d(np.loadtxt(run / "upstream-eval" / f"{filename}.txt"))
        if len(values) != len(sampled) or not np.isfinite(values).all():
            raise SystemExit(f"Invalid actual render metric: {name}/{filename}")
        rendering[key] = float(values.mean())
        if key == "psnr_db":
            rendering.update(psnr_db_min=float(values.min()), psnr_db_max=float(values.max()),
                             lowest_psnr_frame_index=sampled[int(values.argmin())],
                             highest_psnr_frame_index=sampled[int(values.argmax())])
    if not np.array_equal(np.atleast_1d(np.loadtxt(run / "upstream-eval/l1.txt")),
                          np.atleast_1d(np.loadtxt(run / "upstream-eval/rmse.txt"))):
        raise SystemExit("Upstream depth metric semantics changed; review before summarizing")
    rendering["scope"] = ("Final-map seen training views; RGB zero-masked by valid GT depth then whole-image metrics; "
                          "depth MAE on valid GT pixels. No held-out or novel-view evaluation.")
    copies = {}
    for selection, frame in (("last_sample", sampled[-1]), ("lowest_psnr", rendering["lowest_psnr_frame_index"])):
        source = run / "upstream-eval/plots" / f"{frame:04d}.png"
        destination = out / f"{label}-{selection}.png"
        shutil.copyfile(source, destination)
        copies[selection] = {"frame_index": frame, "source": source.relative_to(LAB).as_posix(),
                             "copy": destination.relative_to(LAB).as_posix(), "sha256": sha(source),
                             "copy_sha256": sha(destination)}
    summary = {
        "run_name": name, "run_path": run.relative_to(LAB).as_posix(),
        "evaluation_path": eval_dir.relative_to(LAB).as_posix(),
        "actual_source_audit_path": audit_path.relative_to(LAB).as_posix(),
        "sequence": sequence, "frames": metric["frames"], "resolution": metric["resolution"],
        "full_associated_sequence": True, "sequence_duration_seconds": metric["duration_s"],
        "gaussian_count": metric["gaussian_count"], "keyframe_count": len(metric["keyframe_indices"]),
        "actual_child_wall_seconds": receipt["wall_seconds"],
        "wall_seconds_per_frame": receipt["wall_seconds"] / metric["frames"],
        "ate_se3_rmse_m": metric["ate_se3_translation"]["rmse"],
        "ate_se3_mean_m": metric["ate_se3_translation"]["mean"],
        "ate_se3_max_m": metric["ate_se3_translation"]["max"],
        "ate_first_camera_gauge_rmse_m": metric["ate_first_camera_gauge_translation"]["rmse"],
        "rpe_1frame_translation_rmse_m": metric["rpe_1frame_translation"]["rmse"],
        "rpe_1frame_rotation_rmse_deg": metric["rpe_1frame_rotation"]["rmse"],
        "alignment": "SE3 rigid, scale fixed to 1; C2W camera centers, all configured frames",
        "trajectory_check_count": metric["check_count"], "actual_source_check_count": audit["check_count"],
        "upstream_rendering": rendering, "renders": copies,
        "params_sha256": receipt["params_sha256"], "params_bytes": receipt["params_bytes"],
        "execution_sha256": sha(run / "execution.json"), "independent_metrics_sha256": sha(eval_dir / "metrics.json"),
        "actual_source_review_sha256": sha(audit_path),
    }
    return summary, receipt, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-run", required=True)
    parser.add_argument("--desk640-run", required=True)
    args = parser.parse_args()
    out = LAB / "outputs/splatam-followup-001"
    out.mkdir(parents=True, exist_ok=True)
    baseline_path = LAB / "outputs/splatam-project-001/metrics.json"
    baseline_hash = sha(baseline_path)
    baseline_project = read(baseline_path)
    if baseline_project["status"] != "official_smoke_and_full_sequence_independently_verified":
        raise SystemExit("The first full sequence is not independently verified")
    baseline = baseline_project["runs"]["full"]
    baseline_run = LAB / baseline["run_path"]
    baseline_receipt = read(baseline_run / "execution.json")
    baseline_config = read(baseline_run / "effective-config.json")
    if not (baseline["execution_sha256"] == sha(baseline_run / "execution.json")
            and baseline["params_sha256"] == sha(baseline_run / "params.npz")
            and baseline["independent_metrics_sha256"] == sha(LAB / baseline["evaluation_path"] / "metrics.json")
            and baseline["actual_source_review_sha256"] == sha(LAB / baseline["actual_source_audit_path"])):
        raise SystemExit("Frozen baseline execution, model or independent evidence changed")
    xyz, xyz_receipt, xyz_config = summarize(args.xyz_run, "xyz-full", "rgbd_dataset_freiburg1_xyz", [320, 240], out)
    desk640, desk640_receipt, desk640_config = summarize(args.desk640_run, "desk-640-full", "rgbd_dataset_freiburg1_desk", [640, 480], out)
    same_inputs = baseline_receipt["native_input_sha256_before"] == desk640_receipt["native_input_sha256_before"]
    base_times = np.loadtxt(LAB / baseline["evaluation_path"] / "estimated.tum")[:, 0]
    high_times = np.loadtxt(LAB / desk640["evaluation_path"] / "estimated.tum")[:, 0]
    same_times = np.array_equal(base_times, high_times)
    same_protocol = normalized_config(baseline_config) == normalized_config(xyz_config) == normalized_config(desk640_config)
    same_sources = (baseline_receipt["source_sha256_before"] == xyz_receipt["source_sha256_before"]
                    == desk640_receipt["source_sha256_before"])
    same_binary = (baseline_receipt["renderer_provenance"]["extension_sha256_before"]
                   == xyz_receipt["renderer_provenance"]["extension_sha256_before"]
                   == desk640_receipt["renderer_provenance"]["extension_sha256_before"])
    checks = {"desk_original_inputs_identical": same_inputs, "desk_selected_timestamps_identical": same_times,
              "protocol_except_sequence_and_resolution_identical": same_protocol,
              "official_algorithm_sources_identical": same_sources, "renderer_binary_identical": same_binary,
              "frozen_baseline_summary_unchanged": baseline_hash == sha(baseline_path)}
    if not all(checks.values()):
        raise SystemExit(f"Comparison protocol mismatch: {checks}")
    result = {
        "schema_version": 1, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "official_two_sequences_and_resolution_independently_verified", "priority": 1, "local_only": True,
        "baseline_summary": baseline_path.relative_to(LAB).as_posix(), "baseline_summary_sha256": baseline_hash,
        "runs": {"desk320": baseline, "xyz320": xyz, "desk640": desk640}, "comparison_checks": checks,
        "desk_resolution_comparison": {
            "ate_rmse_640_over_320": desk640["ate_se3_rmse_m"] / baseline["ate_se3_rmse_m"],
            "wall_time_640_over_320": desk640["actual_child_wall_seconds"] / baseline["actual_child_wall_seconds"],
            "gaussian_count_640_over_320": desk640["gaussian_count"] / baseline["gaussian_count"],
            "native_grid_psnr_change_db": desk640["upstream_rendering"]["psnr_db"] - baseline["upstream_rendering"]["psnr_db"],
            "psnr_comparison_scope": "Different native pixel grids and masks; descriptive values, not common-grid pixel improvement.",
        },
        "limits": ["Two Freiburg1 motion sequences share camera calibration; no cross-sensor or outdoor generalization.",
                   "One seed per configuration; no statistical robustness conclusion or paper-table replication.",
                   "Resolution changes pixel sampling, initial point count, Gaussian scales and summed tracking losses together.",
                   "Actual child wall time includes final render evaluation/save; serial local runs, no isolated kernel benchmark.",
                   "Rendering metrics use observed training views; no held-out test. Depth metric is MAE, not RMSE.",
                   "GT association/gauge/keyframe gate/evaluation remain; per-frame GT tracking disabled.",
                   "No loop closure, relocalization, global BA, LiDAR/IMU fusion or physical synchronization/recalibration established."],
    }
    save(out / "metrics.json", result)
    save(LAB / "manifests/splatam-followup.json", {
        "updated": "2026-10-10", "project": "SplaTAM", "priority": 1, "status": result["status"], "local_only": True,
        "summary": "outputs/splatam-followup-001/metrics.json", "baseline": "manifests/splatam-local-project.json",
        "knowledge": "knowledge/24_splatam_official_rgbd_system.md", "experiment": "experiments/29_splatam_sequences_resolution.md",
        "launch": "scripts/splatam_local/run_wsl.py", "evaluate": "scripts/splatam_local/evaluate_run.py",
        "summarize": "scripts/summarize_splatam_followup.py",
        "runs": {key: value["run_path"] for key, value in result["runs"].items()}, "limits": result["limits"],
    })
    progress = read(LAB / "manifests/progress.json")
    items = []
    for label, entry in (("xyz-full", xyz), ("desk-640-full", desk640)):
        items.append({"id": f"splatam-official-local-{label}-001",
                      "title": "官方 SplaTAM 完整 xyz 运动序列" if label == "xyz-full" else "官方 SplaTAM desk 640×480 分辨率对照",
                      "status": "completed_and_independently_verified", "metrics": entry,
                      "artifacts": ["experiments/29_splatam_sequences_resolution.md", "outputs/splatam-followup-001/metrics.json",
                                    "outputs/splatam-followup-001/report.md", "manifests/splatam-followup.json",
                                    f"{entry['evaluation_path']}/metrics.json", f"{entry['evaluation_path']}/report.md",
                                    f"{entry['evaluation_path']}/trajectory-errors.png",
                                    *[image["copy"] for image in entry["renders"].values()]],
                      "scope": "Full associated sequence, single seed; C2W SE3 fixed scale1 trajectory evaluation. Seen-view render metrics; local offline experiment."})
    owned_ids = {item["id"] for item in items}
    progress["results"] = items + [item for item in progress["results"] if item["id"] not in owned_ids]
    progress["updated"] = "2026-10-10"
    progress["current_stage"] = "SplaTAM_two_sequences_and_resolution_independently_verified"
    progress["next_steps"] = ["固定同序列、同分辨率的颜色／深度跟踪消融", "关键帧窗口及累计误差分析", "多 seed 对照；FAST-LIVO2 有 GT 输入完整回放仍待完成"]
    save(LAB / "manifests/progress.json", progress)
    print(json.dumps({"status": result["status"], "comparison_checks": checks,
                      "xyz_ate_rmse_m": xyz["ate_se3_rmse_m"], "desk640_ate_rmse_m": desk640["ate_se3_rmse_m"]}))


if __name__ == "__main__":
    main()
