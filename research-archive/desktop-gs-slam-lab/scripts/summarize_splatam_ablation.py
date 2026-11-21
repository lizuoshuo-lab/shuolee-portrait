"""Bind actual loss ablations and seed repeats to independent local evidence.

Pending and failed jobs remain visible; statistics require all three seeds.
Never edits original runs, the frozen baseline or independent evaluations.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

LAB = Path(__file__).resolve().parent.parent
OUT = LAB / "outputs/splatam-ablation-repeat-001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def normalized(config):
    value = copy.deepcopy(config)
    value.pop("run_name")
    value.pop("seed")
    value["wandb"].pop("name")
    value["tracking"]["loss_weights"] = {"im": .5, "depth": 1.0}
    for key in ("sequence", "gradslam_data_cfg"):
        value["data"].pop(key)
    return value


def summarize_job(job, plan, baseline_receipt, baseline_config, baseline_times):
    name = job["run_name"]
    run = LAB / "outputs/splatam-official-runs-001" / name
    group = "ablation" if job["phase"] == "tracking_weight_ablation" else "repeat"
    evaluation = LAB / f"outputs/splatam-{group}-evaluations-001" / name
    trajectory = evaluation / "trajectory"
    actual = read(run / "execution.json")
    sealed = read(evaluation / "independent-receipt.json")
    assert sealed["status"] == "independently_verified" and sealed["run_name"] == name
    assert sealed["execution_sha256"] == sha(run / "execution.json")
    assert sealed["final_model_sha256"] == actual["params_sha256"]
    assert sealed["plan_sha256"] == sha(OUT / "plan.json")
    assert sealed["profile"] == job["tracking_loss"] and sealed["seed"] == job["seed"]
    assert sealed["sequence"] == job["sequence"] and sealed["index"] == job["index"]
    assert sealed["runtime_call"]["exit_code"] == sealed["trajectory_call"]["exit_code"] == 0
    for relative, expected in sealed["frozen_result_sha256"].items():
        frozen_path = (evaluation / relative).resolve()
        assert frozen_path.is_relative_to(evaluation.resolve()) and sha(frozen_path) == expected
    config = read(run / "effective-config.json")
    metric = read(trajectory / "metrics.json")
    audit = read(evaluation / "metrics.json")
    protocol = read(evaluation / "protocol-review.json")
    assert actual["status"] == "completed_pending_independent_evaluation" and actual["child_exit_code"] == 0
    assert actual["params_sha256"] == sha(run / "params.npz")
    assert job["execution_sha256"] == sha(run / "execution.json")
    assert metric["status"] == "independently_verified" and all(metric["checks"].values())
    assert metric["run_name"] == name and metric["full_associated_sequence"]
    assert metric["sequence"] == "rgbd_dataset_" + job["sequence"] and metric["resolution"] == [320, 240]
    assert audit["status"] == "passed" and all(audit["checks"].values())
    assert audit["run_name"] == name and audit["input_execution_sha256"] == sha(run / "execution.json")
    assert protocol["status"] == "verified_preregistered_full_protocol"
    assert protocol["run_name"] == name and protocol["mapping_config_identical"]
    assert protocol["unexpected_protocol_differences"] == [] and protocol["plan_sha256"] == sealed["plan_sha256"]
    assert protocol["actual_stdout_tracking_depth_threshold_enabled"] is False
    assert config["seed"] == job["seed"] and config["tracking"]["loss_weights"] == plan["tracking_weights"][job["tracking_loss"]]
    checks = {
        "protocol_except_seed_tracking_weight_and_sequence_identical": normalized(config) == normalized(baseline_config),
        "mapping_configuration_identical": config["mapping"] == baseline_config["mapping"],
        "official_algorithm_sources_identical": actual["source_sha256_before"] == baseline_receipt["source_sha256_before"],
        "renderer_binary_identical": actual["renderer_provenance"]["extension_sha256_before"] == baseline_receipt["renderer_provenance"]["extension_sha256_before"],
        "selected_tracking_weight_exact": config["tracking"]["loss_weights"] == plan["tracking_weights"][job["tracking_loss"]],
    }
    if job["sequence"] == "freiburg1_desk":
        checks["desk_raw_inputs_identical"] = actual["native_input_sha256_before"] == baseline_receipt["native_input_sha256_before"]
        checks["desk_selected_timestamps_identical"] = np.array_equal(np.loadtxt(trajectory / "estimated.tum")[:, 0], baseline_times)
    assert all(checks.values()), (name, checks)
    sampled = [i for i in range(metric["frames"]) if i == 0 or (i + 1) % config["eval_every"] == 0]
    assert sorted(int(p.stem) for p in (run / "upstream-eval/plots").glob("*.png")) == sampled
    render = {"sampled_frame_count": len(sampled), "sampled_frame_indices": sampled}
    for file, key in (("psnr", "psnr_db"), ("ssim", "ms_ssim"), ("lpips", "lpips"), ("l1", "depth_mae_m")):
        values = np.atleast_1d(np.loadtxt(run / f"upstream-eval/{file}.txt"))
        assert len(values) == len(sampled) and np.isfinite(values).all()
        render[key] = float(values.mean())
        if file == "psnr":
            render["lowest_psnr_frame_index"] = sampled[int(values.argmin())]
            render["lowest_psnr_db"] = float(values.min())
    assert np.array_equal(np.loadtxt(run / "upstream-eval/l1.txt"), np.loadtxt(run / "upstream-eval/rmse.txt"))
    render["scope"] = "Seen training views under each run estimated cameras. Valid GT depth mask applied. Depth MAE, not RMSE. No fixed-map or held-out quality claim."
    label = f"{job['sequence']}-{job['tracking_loss']}-s{job['seed']}"
    copies = {}
    for kind, frame in (("last_sample", sampled[-1]), ("lowest_psnr", render["lowest_psnr_frame_index"])):
        source = run / "upstream-eval/plots" / f"{frame:04d}.png"
        destination = OUT / f"{label}-{kind}.png"
        shutil.copyfile(source, destination)
        copies[kind] = {"frame_index": frame, "source": source.relative_to(LAB).as_posix(), "copy": destination.relative_to(LAB).as_posix(), "sha256": sha(source), "copy_sha256": sha(destination)}
    return {
        "run_name": name, "profile": job["tracking_loss"], "seed": job["seed"], "sequence": job["sequence"],
        "status": "completed_and_independently_verified", "frames": metric["frames"], "full_associated_sequence": True,
        "resolution": metric["resolution"], "duration_seconds": metric["duration_s"],
        "ate_se3_rmse_m": metric["ate_se3_translation"]["rmse"],
        "ate_first_camera_gauge_rmse_m": metric["ate_first_camera_gauge_translation"]["rmse"],
        "rpe_translation_rmse_m": metric["rpe_1frame_translation"]["rmse"], "rpe_rotation_rmse_deg": metric["rpe_1frame_rotation"]["rmse"],
        "gaussian_count": metric["gaussian_count"], "keyframe_count": len(metric["keyframe_indices"]), "actual_child_wall_seconds": actual["wall_seconds"],
        "trajectory_checks": metric["check_count"], "runtime_checks": audit["check_count"], "comparison_checks": checks,
        "rendering": render, "renders": copies, "run_path": run.relative_to(LAB).as_posix(), "evaluation_path": evaluation.relative_to(LAB).as_posix(),
        "execution_sha256": sha(run / "execution.json"), "model_sha256": sha(run / "params.npz"),
        "trajectory_metrics_sha256": sha(trajectory / "metrics.json"), "runtime_metrics_sha256": sha(evaluation / "metrics.json"), "protocol_review_sha256": sha(evaluation / "protocol-review.json"),
    }


def main():
    plan = read(OUT / "plan.json")
    batch = read(OUT / "batch-progress.json")
    assert batch["plan_sha256"] == sha(OUT / "plan.json")
    baseline_project = read(LAB / "outputs/splatam-project-001/metrics.json")
    assert sha(LAB / "outputs/splatam-project-001/metrics.json") == plan["baseline"]["summary_sha256"]
    base = baseline_project["runs"]["full"]
    base_path = LAB / base["run_path"]
    base_receipt = read(base_path / "execution.json")
    base_config = read(base_path / "effective-config.json")
    assert sha(base_path / "execution.json") == plan["baseline"]["execution_sha256"]
    assert sha(base_path / "params.npz") == plan["baseline"]["params_sha256"]
    base_trajectory = LAB / base["evaluation_path"]
    assert sha(base_trajectory / "metrics.json") == base["independent_metrics_sha256"]
    assert sha(LAB / base["actual_source_audit_path"]) == base["actual_source_review_sha256"]
    base_times = np.loadtxt(base_trajectory / "estimated.tum")[:, 0]
    baseline = {
        "run_name": base["run_name"], "profile": "rgbd", "seed": 0, "sequence": "freiburg1_desk", "status": "completed_and_independently_verified",
        "frames": base["frames"], "full_associated_sequence": True, "resolution": base["resolution"],
        "ate_se3_rmse_m": base["ate_se3_rmse_m"], "ate_first_camera_gauge_rmse_m": base["ate_first_camera_gauge_rmse_m"],
        "rpe_translation_rmse_m": base["rpe_1frame_translation_rmse_m"], "rpe_rotation_rmse_deg": base["rpe_1frame_rotation_rmse_deg"],
        "gaussian_count": base["gaussian_count"], "actual_child_wall_seconds": base["actual_child_wall_seconds"], "source": "Frozen first project, reused without rerun or modification",
    }
    verified, jobs = [baseline], []
    watcher_path = LAB / "outputs/splatam-ablation-evaluations-001/evaluation-progress.json"
    watcher = read(watcher_path) if watcher_path.is_file() else {"failures": {}}
    if "plan_sha256" in watcher:
        assert watcher["plan_sha256"] == batch["plan_sha256"]
    assert len(batch["jobs"]) <= len(plan["full_queue"])
    for index, job in enumerate(batch["jobs"]):
        assert job["index"] == index
        assert all(job[field] == plan["full_queue"][index][field] for field in ("phase", "sequence", "tracking_loss", "seed"))
        entry = {key: job.get(key) for key in ("index", "phase", "sequence", "tracking_loss", "seed", "status", "run_name", "windows_wrapper_exit_code", "actual_child_exit_code", "error")}
        if job["status"] == "completed_pending_independent_evaluation":
            group = "ablation" if job["phase"] == "tracking_weight_ablation" else "repeat"
            seal_path = LAB / f"outputs/splatam-{group}-evaluations-001" / job["run_name"] / "independent-receipt.json"
            failure = watcher["failures"].get(job["run_name"])
            if failure:
                entry.update(status="independent_rejected", rejection_reason=failure.get("reason", failure["status"]))
            elif not seal_path.is_file():
                entry["status"] = "awaiting_independent_evaluation"
            else:
                try:
                    sealed = read(seal_path)
                except json.JSONDecodeError:
                    # The seal is written last; a concurrent partial write is not acceptance.
                    entry["status"] = "awaiting_independent_evaluation"
                else:
                    if sealed.get("status") != "independently_verified":
                        entry.update(status="independent_rejected", rejection_reason=sealed.get("status", "Missing independent status"))
                    else:
                        try:
                            result = summarize_job(job, plan, base_receipt, base_config, base_times)
                            verified.append(result)
                            entry["status"] = result["status"]
                        except (AssertionError, ValueError, KeyError, OSError) as error:
                            entry.update(status="comparison_or_evidence_rejected", rejection_reason=repr(error))
        jobs.append(entry)
    stats = {}
    for profile in plan["tracking_weights"]:
        values = sorted((r for r in verified if r["sequence"] == "freiburg1_desk" and r["profile"] == profile), key=lambda r: r["seed"])
        item = {"verified_seeds": [r["seed"] for r in values], "expected_seeds": [0, 1, 2], "status": "incomplete_three_seed_group"}
        if item["verified_seeds"] == [0, 1, 2]:
            item["status"] = "complete_exploratory_three_seed_group"
            for field in ("ate_se3_rmse_m", "ate_first_camera_gauge_rmse_m", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg", "gaussian_count", "actual_child_wall_seconds"):
                raw = [r[field] for r in values]
                item[field] = {"per_seed_values": raw, "mean": statistics.mean(raw), "sample_std_ddof1": statistics.stdev(raw), "min": min(raw), "max": max(raw)}
        stats[profile] = item
    complete = (len(jobs) == len(plan["full_queue"]) == 9
                and batch["status"] == "training_queue_completed_pending_independent_evaluation"
                and all(j["status"] == "completed_and_independently_verified" for j in jobs)
                and all(item["status"] == "complete_exploratory_three_seed_group" for item in stats.values()))
    result = {"schema_version": 1, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "local_only": True,
              "status": "full_ablation_repeats_and_room_independently_verified" if complete else "partial_actual_results_not_complete",
              "plan_sha256": sha(OUT / "plan.json"), "batch_receipt_path": batch["execution_path"] + "/receipt.json", "batch_status": batch["status"],
              "verified_runs": verified, "jobs": jobs, "three_seed_statistics": stats,
              "limits": plan["scope"] + ["Room expands coverage and includes a loop trajectory in the same office; it is not an independent environment or cross-sensor benchmark."]}
    save(OUT / "metrics.json", result)
    save(LAB / "manifests/splatam-ablation-repeat.json", {"updated": dt.datetime.now().date().isoformat(), "priority": 1, "project": "SplaTAM", "status": result["status"], "summary": "outputs/splatam-ablation-repeat-001/metrics.json", "plan": "outputs/splatam-ablation-repeat-001/plan.json", "experiment": "experiments/30_splatam_tracking_ablation_repeat.md", "local_only": True})
    print(json.dumps({"status": result["status"], "verified_full_runs_including_baseline": len(verified), "launched_jobs": len(jobs), "three_seed_groups": {k: v["verified_seeds"] for k, v in stats.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
