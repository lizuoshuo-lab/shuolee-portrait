"""Aggregate completed official runs; never substitutes preparation for a real result."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--smoke-run", required=True)
    parser.add_argument("--full-run", required=True)
    args = parser.parse_args()
    lab = args.lab_root.resolve()
    out = lab / "outputs/splatam-project-001"
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for label, name in (("smoke", args.smoke_run), ("full", args.full_run)):
        if Path(name).name != name or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise SystemExit("Run name must be a single local directory name")
        relative = f"outputs/splatam-official-runs-001/{name}"
        run = lab / relative
        evaluation = lab / f"outputs/splatam-independent-eval-{label}-001"
        audit = lab / f"outputs/splatam-project-independent-review-001/{name}/metrics.json"
        receipt = read(run / "execution.json")
        metrics = read(evaluation / "metrics.json")
        review = read(audit)
        if not (receipt["status"] == "completed_pending_independent_evaluation"
                and receipt["child_exit_code"] == 0
                and metrics["status"] == "independently_verified"
                and metrics["run_name"] == name and all(metrics["checks"].values())
                and review["status"] == "passed" and review["run_name"] == name
                and all(review["checks"].values())
                and review["input_execution_sha256"] == digest(run / "execution.json")
                and receipt["params_sha256"] == digest(run / "params.npz")):
            raise SystemExit(f"Actual execution / independent evidence mismatch: {name}")
        if label == "full" and not metrics["full_associated_sequence"]:
            raise SystemExit("Full result must cover the complete official associated sequence")
        config = read(run / "effective-config.json")
        stride = config["eval_every"]
        sampled = [i for i in range(metrics["frames"]) if i == 0 or (i + 1) % stride == 0]
        rendering = {}
        for filename, key in (("psnr", "psnr_db"), ("ssim", "ms_ssim"),
                              ("lpips", "lpips"), ("l1", "depth_mae_m")):
            values = np.atleast_1d(np.loadtxt(run / "upstream-eval" / f"{filename}.txt"))
            if len(values) != len(sampled) or not np.isfinite(values).all():
                raise SystemExit(f"Invalid actual rendering metrics: {name}/{filename}")
            rendering[key] = float(values.mean())
            if key == "psnr_db":
                rendering["psnr_db_min"] = float(values.min())
                rendering["psnr_db_max"] = float(values.max())
                rendering["lowest_psnr_frame_index"] = sampled[int(values.argmin())]
        depth_l1 = np.atleast_1d(np.loadtxt(run / "upstream-eval/l1.txt"))
        depth_misnamed = np.atleast_1d(np.loadtxt(run / "upstream-eval/rmse.txt"))
        if not np.array_equal(depth_l1, depth_misnamed):
            raise SystemExit("Upstream depth metric semantics changed; review before summarizing")
        rendering.update(sampled_frame_indices=sampled, sampled_frame_count=len(sampled),
                         scope="Final-map seen training views; RGB zero-masked by valid GT depth then whole-image metrics; depth MAE on valid GT pixels. Not held-out or novel-view performance.")
        log = (run / "stdout.log").read_text(encoding="utf-8", errors="replace")
        timings = {}
        for printed, key in (("Tracking/Frame", "tracking_frame_seconds"),
                             ("Mapping/Frame", "mapping_frame_seconds")):
            match = re.search(rf"Average {printed} Time: ([0-9.e+-]+) s", log)
            if match:
                timings[key] = float(match.group(1))
        timings["scope"] = "Original time.time wall-clock averages without explicit CUDA synchronize; first tracking frame contributes its initialization branch. Not kernel benchmarks."
        summaries[label] = {
            "run_name": name, "run_path": relative, "evaluation_path": str(evaluation.relative_to(lab)).replace("\\", "/"),
            "actual_source_audit_path": str(audit.relative_to(lab)).replace("\\", "/"),
            "frames": metrics["frames"], "available_loader_frames": metrics["available_loader_frames"],
            "full_associated_sequence": metrics["full_associated_sequence"],
            "sequence_duration_seconds": metrics["duration_s"], "resolution": metrics["resolution"],
            "gaussian_count": metrics["gaussian_count"], "keyframe_count": len(metrics["keyframe_indices"]),
            "actual_child_wall_seconds": receipt["wall_seconds"],
            "wall_seconds_per_frame": receipt["wall_seconds"] / metrics["frames"],
            "wall_time_scope": "Actual official child, including loading/initialization, tracking/mapping, final sampled rendering evaluation and final save; excludes environment setup, wrapper copy, independent evaluation.",
            "ate_se3_rmse_m": metrics["ate_se3_translation"]["rmse"],
            "ate_se3_mean_m": metrics["ate_se3_translation"]["mean"],
            "ate_se3_max_m": metrics["ate_se3_translation"]["max"],
            "ate_first_camera_gauge_rmse_m": metrics["ate_first_camera_gauge_translation"]["rmse"],
            "rpe_1frame_translation_rmse_m": metrics["rpe_1frame_translation"]["rmse"],
            "rpe_1frame_rotation_rmse_deg": metrics["rpe_1frame_rotation"]["rmse"],
            "alignment": "SE3, fixed scale 1; C2W camera centers; all configured frames",
            "trajectory_check_count": metrics["check_count"], "actual_source_check_count": review["check_count"],
            "upstream_rendering": rendering, "upstream_timing": timings,
            "params_bytes": receipt["params_bytes"], "params_sha256": receipt["params_sha256"],
            "execution_sha256": digest(run / "execution.json"), "independent_metrics_sha256": digest(evaluation / "metrics.json"),
            "actual_source_review_sha256": digest(audit),
        }
        if label == "full":
            last = run / "upstream-eval/plots" / f"{sampled[-1]:04d}.png"
            shutil.copyfile(last, out / "rendered-view-comparison.png")
            summaries[label]["representative_render"] = {"frame_index": sampled[-1], "source": f"{relative}/upstream-eval/plots/{last.name}", "sha256": digest(last), "copy_sha256": digest(out / "rendered-view-comparison.png")}
            worst = run / "upstream-eval/plots" / f"{rendering['lowest_psnr_frame_index']:04d}.png"
            shutil.copyfile(worst, out / "lowest-psnr-view-comparison.png")
            summaries[label]["lowest_psnr_render"] = {"frame_index": rendering["lowest_psnr_frame_index"], "selection": "Minimum actual PSNR among the 30 fixed seen-frame samples", "source": f"{relative}/upstream-eval/plots/{worst.name}", "sha256": digest(worst), "copy_sha256": digest(out / "lowest-psnr-view-comparison.png")}
    result = {
        "schema_version": 1, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "official_smoke_and_full_sequence_independently_verified",
        "priority": 1, "local_only": True, "splatam_commit": "da6bbcd24c248dc884ac7f49d62e91b841b26ccc",
        "renderer_commit": "cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110", "algorithm_or_renderer_source_patches": False,
        "bootstrap_metrics_sha256": digest(lab / "outputs/splatam-linux-bootstrap-001/metrics.json"),
        "runs": summaries,
        "limits": ["320x240 local reduced-resolution configuration, not a paper-resolution replication.",
                   "Single full indoor RGB-D sequence and one seed; no robustness or cross-scene conclusion.",
                   "GT used for association/gauge/finite keyframe gate/evaluation, disabled for per-frame tracking.",
                   "No loop closure, relocalization, global BA, LiDAR/IMU fusion, hardware synchronization or physical recalibration established.",
                   "Checkpoints omit complete optimizer state; no arbitrary-interruption exact resume claim."],
    }
    write(out / "metrics.json", result)
    write(lab / "manifests/splatam-local-project.json", {
        "updated": "2026-10-10", "project": "SplaTAM", "priority": 1, "status": result["status"], "local_only": True,
        "summary": "outputs/splatam-project-001/metrics.json", "knowledge": "knowledge/24_splatam_official_rgbd_system.md",
        "experiment": "experiments/28_splatam_official_local_project.md",
        "launch": "scripts/splatam_local/run_wsl.py", "evaluate": "scripts/splatam_local/evaluate_run.py",
        "runs": {key: value["run_path"] for key, value in summaries.items()}, "limits": result["limits"],
    })
    print(json.dumps({"status": result["status"], "full": summaries["full"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
