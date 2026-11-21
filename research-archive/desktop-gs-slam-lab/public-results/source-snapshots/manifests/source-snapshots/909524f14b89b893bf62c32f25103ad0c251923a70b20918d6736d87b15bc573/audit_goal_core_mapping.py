"""Read-only completion audit; no rendering, training or metric recomputation."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "outputs/goal-completion-core-mapping-audit-002"
CONSUMED = {}
MISSING = []
ISSUES = []


def consume(relative):
    path = (LAB / relative).resolve()
    key = path.relative_to(LAB).as_posix()
    if not path.is_file():
        if key not in MISSING:
            MISSING.append(key)
        return None
    if key not in CONSUMED:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        CONSUMED[key] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return path


def read_json(relative):
    path = consume(relative)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, label):
    if not condition:
        ISSUES.append(label)


def check_run(run, points, steps, tiny):
    base = "outputs/" + run + "/"
    metrics = read_json(base + "metrics.json")
    config = metrics["config" if tiny else "configuration"]
    require(metrics["status"] == "completed", run + ": non-completed metrics")
    require(config["steps"] == steps and config["points"] == points, run + ": config count mismatch")
    checkpoint_path = consume(base + "gaussians.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["config" if tiny else "configuration"] == config, run + ": checkpoint config mismatch")
    shapes = {key: list(value.shape) for key, value in checkpoint["parameters"].items()}
    expected = {"means": [points, 3], "log_scales": [points, 3], "quats": [points, 4],
                "color_logits": [points, 3], "opacity_logits": [points]}
    require(shapes == expected, run + ": checkpoint parameter shape mismatch")
    loss_path = consume(base + "loss.csv")
    with loss_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    saved_steps = [int(row["step"]) for row in rows]
    require(saved_steps[0] == 1 and saved_steps[-1] == steps and
            all(a < b for a, b in zip(saved_steps, saved_steps[1:])), run + ": loss completion mismatch")
    log_path = consume("logs/" + run + ".log")
    log = log_path.read_text(encoding="utf-8")
    require("step=" + str(steps) + "/" + str(steps) in log, run + ": no terminal step in log")
    require('"status": "completed"' in log and "Traceback" not in log, run + ": log terminal mismatch")
    if tiny:
        train, heldout = metrics["train_views"], metrics["test_views"]
        require(train == list(range(100)) and heldout == list(range(100, 106)), run + ": wrong split")
        require(checkpoint["training_views"] == train and checkpoint["test_views"] == heldout,
                run + ": checkpoint split mismatch")
        pngs = ["initial_test101.png", "target_test101.png", "comparison.png"]
        pngs += ["predicted_test" + str(i) + ".png" for i in heldout]
        saved_result = {"test_mean_psnr_db": metrics["test_mean_psnr_db"],
                        "training_seconds_excludes_compile_and_final_evaluation":
                        metrics["training_seconds_excludes_compile_and_final_evaluation"]}
        split = {"train_count": len(train), "heldout_count": len(heldout), "disjoint": not set(train) & set(heldout)}
    else:
        configuration = read_json(base + "configuration.json")
        require(configuration["configuration"] == config, run + ": configuration file mismatch")
        split = metrics["split"]
        train, heldout = split["train_frame_ids"], split["heldout_frame_ids"]
        require(len(train) == 10 and len(heldout) == 5 and not set(train) & set(heldout), run + ": wrong split")
        source_frame_ids = checkpoint["initialization_source_frame_ids"].tolist()
        require(checkpoint["train_frame_ids"] == train and checkpoint["heldout_frame_ids"] == heldout and
                len(source_frame_ids) == points and sorted(set(source_frame_ids)) == sorted(train) and
                [source_frame_ids.count(fid) for fid in train] == metrics["initialization"]["selected_counts_by_training_frame"],
                run + ": initialization split mismatch")
        require(list(checkpoint["K"].shape) == [1, 3, 3] and list(checkpoint["w2c"].shape) == [15, 4, 4],
                run + ": camera shape mismatch")
        require(len(metrics["initial_per_view"]) == len(metrics["optimized_per_view"]) == 15,
                run + ": incomplete per-view metrics")
        pngs = ["comparison.png"]
        for frame in checkpoint["frame_ids"]:
            prefix = "views/frame" + str(frame).zfill(4) + "_"
            pngs += [prefix + "target_" + suffix + ".png" for suffix in
                     ["rgb", "depth_u16", "depth_preview"]]
            pngs += [prefix + stage + "_" + suffix + ".png" for stage in ["initial", "optimized"]
                     for suffix in ["rgb", "depth_u16", "depth_preview", "alpha"]]
        consume(base + "tracking_data.npz")
        consume(base + "README.md")
        saved_result = {"initial_summary": metrics["initial_summary"], "optimized_summary": metrics["optimized_summary"]}
    for png in pngs:
        consume(base + png)
    result_config = dict(config)
    if "dataset" in result_config:
        result_config["dataset"] = Path(config["dataset"]).relative_to(LAB).as_posix()
    return {"run": run, "terminal_status": metrics["status"], "metrics_path": base + "metrics.json",
            "configuration": result_config, "checkpoint": base + "gaussians.pt", "checkpoint_shapes": shapes,
            "loss": {"path": base + "loss.csv", "saved_rows": len(rows), "last_step": saved_steps[-1]},
            "log": "logs/" + run + ".log", "required_png_count": len(pngs), "split": split,
            "reported_results_not_recomputed": saved_result}, metrics


def main():
    if OUT.exists():
        raise FileExistsError("Unique audit output already exists")
    consume("scripts/audit_goal_core_mapping.py")
    consume("outputs/goal-completion-core-mapping-audit-001/metrics.json")
    consume("outputs/goal-completion-core-mapping-audit-001/source_script.py")
    study = read_json("outputs/gaussian-count-study-001/metrics.json")
    validation = read_json("outputs/gaussian-count-study-001/validation.json")
    require(study["status"] == "completed" and validation["status"] == "passed", "Tiny study terminal validation")
    require(len(study["records"]) == len(validation["runs"]) == 6, "Tiny study expected six runs")
    validation_by_run = {item["run"]: item for item in validation["runs"]}
    tiny_runs = []
    for item in study["records"]:
        row, metrics = check_run(item["run"], item["points"], 1500, True)
        require(metrics["config"]["seed"] == item["seed"] and metrics["test_mean_psnr_db"] == item["test_mean_psnr_db"],
                item["run"] + ": aggregate binding mismatch")
        v = validation_by_run[item["run"]]
        require(CONSUMED[row["checkpoint"]]["sha256"] == v["checkpoint_sha256"], item["run"] + ": validation model SHA mismatch")
        row["existing_validation"] = v
        tiny_runs.append(row)
    require({(x["configuration"]["points"], x["configuration"]["seed"]) for x in tiny_runs} ==
            {(p, s) for p in [4000, 8000] for s in [0, 1, 2]}, "Tiny count/seed coverage")
    for source in ["scripts/train_tiny_gs.py", "scripts/train_tum_gs.py", "scripts/tum_rgbd.py"]:
        consume(source)
    require(CONSUMED["scripts/train_tiny_gs.py"]["sha256"] == study["trainer_sha256"], "Tiny frozen source mismatch")
    consume("manifests/source-snapshots/" + study["trainer_sha256"] + "/train_tiny_gs.py")
    tiny_data = read_json("manifests/datasets/tiny_nerf.json")
    consume(tiny_data["relative_path"])
    require(CONSUMED[tiny_data["relative_path"]]["sha256"] == tiny_data["sha256"], "Tiny data SHA mismatch")
    tum_runs, tum_metrics = [], []
    for run in ["tum-rgbd-gsplat-001", "tum-rgbd-rgbonly-001"]:
        row, metrics = check_run(run, 30000, 1000, False)
        require(CONSUMED["scripts/train_tum_gs.py"]["sha256"] == metrics["script_sha256"] and
                CONSUMED["scripts/tum_rgbd.py"]["sha256"] == metrics["loader_sha256"], run + ": frozen source mismatch")
        tum_runs.append(row)
        tum_metrics.append(metrics)
    differences = [k for k in tum_metrics[0]["configuration"] if
                   tum_metrics[0]["configuration"][k] != tum_metrics[1]["configuration"][k]]
    require(set(differences) == {"run", "depth_weight"}, "TUM ablation has extra changed configuration")
    require(tum_metrics[0]["initial_summary"] == tum_metrics[1]["initial_summary"], "TUM recorded initialization metrics differ")
    require(tum_metrics[0]["selected_frames"] == tum_metrics[1]["selected_frames"], "TUM sensor selection differs")
    tum_data = read_json("manifests/datasets/tum_freiburg1_desk.json")
    files = read_json("manifests/datasets/tum_freiburg1_desk_files.json")
    file_manifest = {entry["path"]: entry for entry in files}
    consume(tum_data["archive_relative_path"])
    require(CONSUMED[tum_data["archive_relative_path"]]["sha256"] == tum_data["archive_sha256"], "TUM archive SHA mismatch")
    sensor_paths = [name for frame in tum_metrics[0]["selected_frames"] for name in
                    [frame["rgb_path"], frame["depth_path"]]] + ["rgb.txt", "depth.txt", "groundtruth.txt"]
    for name in sensor_paths:
        relative = tum_data["extracted_relative_path"] + "/" + name
        consume(relative)
        expected = file_manifest["rgbd_dataset_freiburg1_desk/" + name]
        require(CONSUMED[relative]["sha256"] == expected["sha256"], "TUM sensor SHA mismatch: " + name)
    early_cpu = read_json("logs/tum-rgbd-gsplat-cpu-check.json")
    result = {"status": "passed_core_mapping_artifact_requirements" if not ISSUES and not MISSING else "incomplete",
              "scope": "Read-only requirement/evidence audit; reported numeric results copied, not recalculated.",
              "audit_revision": {"previous_audit": "outputs/goal-completion-core-mapping-audit-001/metrics.json",
                 "reason": "001 assumed unbatched K and one initialization source ID per frame. Frozen trainer actually saves singleton-batch K and one source ID per Gaussian. These four 001 findings are audit schema false positives, not missing training artifacts. Original 001/source bytes are preserved."},
              "created_utc": datetime.now(timezone.utc).isoformat(), "tiny_nerf_six_runs": tiny_runs,
              "tum_known_pose_depth_loss_ablation": tum_runs,
              "tum_configuration_differences": differences,
              "existing_tiny_validation": "outputs/gaussian-count-study-001/validation.json",
              "early_tum_cpu_validation": {"path": "logs/tum-rgbd-gsplat-cpu-check.json",
                 "status": early_cpu["status"], "script_sha256": early_cpu["script_sha256"],
                 "final_trainer_sha256": tum_metrics[0]["script_sha256"],
                 "same_trainer_version": early_cpu["script_sha256"] == tum_metrics[0]["script_sha256"]},
              "missing_explicit_model_log_image_or_other_required_files": MISSING, "issues": ISSUES,
              "limitations": ["All cameras are known and fixed; these runs do not establish tracking or SLAM.",
                 "TinyNeRF is one 100x100 synthetic scene, three seeds per count; 8000 seed0 reuses its original run.",
                 "Tiny per-run metrics omit trainer SHA; the original study source declaration is checked against current and frozen snapshot bytes.",
                 "TUM is one 1.97-second nearby-view segment, one seed, 10 training and 5 interleaved heldout views.",
                 "RGB-only removes depth loss but still initializes from training depth and keeps valid-depth alpha loss.",
                 "Covered-valid-depth RMSE excludes holes and uses differing covered masks; coverage and separate hole-penalized scores are retained.",
                 "TUM early CPU check predates final trainer; no separate final standalone validation.json was found. Terminal artifacts and source bindings are checked here, without rerendering or recalculating geometric metrics.",
                 "Logged timing excludes compile/final evaluation; no process-exit ledger is claimed for these early runs."],
              "training_calls": 0, "render_calls": 0, "metric_recomputations": 0,
              "cuda_initialized": torch.cuda.is_initialized(), "consumed_file_sha256": CONSUMED}
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    for relative, old in CONSUMED.items():
        path = LAB / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != old["sha256"]:
            raise RuntimeError("Input changed during audit: " + relative)
    OUT.mkdir()
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (OUT / "metrics.json").write_bytes(payload)
    print(json.dumps({"status": result["status"], "output": (OUT / "metrics.json").relative_to(LAB).as_posix(),
                      "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                      "consumed_files": len(CONSUMED), "missing": MISSING, "issues": ISSUES}, ensure_ascii=True))


if __name__ == "__main__":
    main()
