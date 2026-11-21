"""Frozen SplaTAM maps on common 320x240 GT camera views, using official eval.

GPU work requires --execute, finished full runs and independent evaluation/audit.
Only in-memory evaluation camera arrays are replaced; saved models stay intact.
This is seen-view map rendering with GT poses, not estimator trajectory accuracy.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True  # Keep official source directories read-only during imports.

BASE = "freiburg1_desk_full_320x240_s0_20261010T062636Z_c516706e"
HIGH = "freiburg1_desk_full_640x480_s0_20261010T080134Z_823ff9d6"
PIN = "da6bbcd24c248dc884ac7f49d62e91b841b26ccc"
RENDERER_PIN = "cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"
SAMPLE = [0] + list(range(19, 592, 20))
SEMANTICS = {
    "map": "Frozen final Gaussian parameters in the common first-camera reference frame; no optimization or parameter updates.",
    "cameras": "Same official TUM loader GT C2W, relative to the first selected camera, converted to W2C/wxyz in evaluation memory only.",
    "grid": "320x240 for both maps; K from actual common loader __getitem__, not the saved 640x480 camera intrinsics.",
    "rgb": "Official eval: rendered and reference RGB zero-masked by valid GT depth, then whole-image PSNR/MS-SSIM/LPIPS.",
    "depth": "Official eval depth MAE on valid GT pixels; upstream rmse.txt is numerically the same MAE as l1.txt.",
    "images": "Official save_frames outputs 8-bit RGB PNGs and clipped color-coded depth visualization PNGs; not unquantized depth tensors.",
    "metrics": "All 30 per-frame raw metric values retained, with unrounded means; same sample indices and RGB/depth inputs.",
    "scope": "Fixed-map seen GT-view rerendering. No novel-view/held-out claim, no training or tuning, no estimated-trajectory accuracy claim.",
    "trajectory": "Original estimated C2W ATE/RPE remain in the frozen independent trajectory evaluations. Disregard upstream GT-substituted ATE prints/plot title here.",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def current_inputs(receipt: dict, config: dict) -> dict:
    sequence = Path(config["data"]["basedir"]) / config["data"]["sequence"]
    reference = receipt["native_input_sha256_before"]
    return {"raw_lists": {name: digest(sequence / name) for name in reference["raw_lists"]},
            "selected_images": {name: digest(sequence / name) for name in reference["selected_images"]},
            "sensor_config_sha256": digest(Path(config["data"]["gradslam_data_cfg"]))}


def current_sources(repo: Path, receipt: dict) -> dict:
    return {name: digest(repo / name) for name in receipt["source_sha256_before"]}


def normalized(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("run_name")
    value["wandb"].pop("name")
    for name in ("desired_image_width", "desired_image_height"):
        value["data"].pop(name)
    return value


def input_run(root: Path, lab: Path, name: str, label: str) -> dict:
    require(re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None, "Run must be one directory name")
    run = root / "outputs/runs" / name
    receipt_path = run / "execution.json"
    receipt, config = read(receipt_path), read(run / "effective-config.json")
    evaluation = lab / ("outputs/splatam-independent-eval-full-001" if label == "desk320" else
                        "outputs/splatam-independent-eval-desk-640-full-001")
    audit = (lab / "outputs/splatam-project-independent-review-001" / name / "metrics.json" if label == "desk320" else
             evaluation / name / "metrics.json")
    metric, review = read(evaluation / "metrics.json"), read(audit)
    require(receipt["status"] == "completed_pending_independent_evaluation" and receipt["child_exit_code"] == 0,
            "Full child is not finished with exit 0: " + name)
    require(metric["status"] == "independently_verified" and all(metric["checks"].values()) and
            metric["run_name"] == name and metric["full_associated_sequence"] and metric["frames"] == 592,
            "Independent full trajectory evaluation has not passed: " + name)
    require(review["status"] == "passed" and all(review["checks"].values()) and review["run_name"] == name and
            review["input_execution_sha256"] == digest(receipt_path), "Actual source/runtime audit has not passed: " + name)
    audited_files = {x["filename"]: x["sha256"] for x in review["copies"]}
    require(all(audited_files[filename] == digest(run / filename) for filename in
                ("effective-config.json", "input_config.py", "params.npz")), "Actual audited config/model bytes changed")
    require(receipt["splatam_commit"] == PIN and receipt["renderer_provenance"]["commit"] == RENDERER_PIN and
            receipt["tracking_uses_gt_poses"] is False and config["tracking"]["use_gt_poses"] is False,
            "Expected pinned source and estimated training camera poses")
    require(receipt["params_sha256"] == digest(run / "params.npz"), "Actual model hash mismatch")
    require(receipt["config_sha256"] == digest(run / "input_config.py"), "Actual training config hash mismatch")
    require(receipt["source_sha256_before"] == receipt["source_sha256_after"] and
            receipt["native_input_sha256_before"] == receipt["native_input_sha256_after"], "Training inputs/source changed")
    data = config["data"]
    require(data["sequence"] == "rgbd_dataset_freiburg1_desk" and data["start"] == 0 and data["end"] == -1 and
            data["stride"] == 1 and data["num_frames"] == -1 and config["eval_every"] == 20,
            "Expected entire identical desk association and 30-frame sample schedule")
    require([data["desired_image_width"], data["desired_image_height"]] ==
            ([320, 240] if label == "desk320" else [640, 480]), "Training resolution does not match its label")
    return {"label": label, "run_name": name, "run_path": str(run), "model_path": str(run / "params.npz"),
            "model_sha256": receipt["params_sha256"], "execution_path": str(receipt_path),
            "execution_sha256": digest(receipt_path), "config_path": str(run / "effective-config.json"),
            "effective_config_sha256": digest(run / "effective-config.json"), "input_config_sha256": receipt["config_sha256"],
            "independent_metrics_path": str(evaluation / "metrics.json"),
            "independent_metrics_sha256": digest(evaluation / "metrics.json"),
            "runtime_audit_path": str(audit), "runtime_audit_sha256": digest(audit),
            "receipt": receipt, "config": config}


def worker(plan_path: Path, label: str) -> None:
    # This function is reached only in a real child launched after all controller gates.
    plan = read(plan_path)
    out, repo = Path(plan["output"]) / label, Path(plan["repo"])
    out.mkdir(exist_ok=False)
    entry = plan["runs"][label]
    require(digest(Path(entry["model_path"])) == entry["model_sha256"], "Model changed before GPU load")
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from datasets.gradslam_datasets import TUMDataset, load_dataset_config
    from utils.eval_helpers import eval as official_eval
    from utils.slam_helpers import matrix_to_quaternion
    from utils.slam_external import build_rotation
    import diff_gaussian_rasterization as renderer
    require(Path(renderer._C.__file__).resolve() == Path(plan["renderer_extension_path"]).resolve() and
            digest(Path(renderer._C.__file__)) == plan["renderer_extension_sha256"], "Unexpected loaded renderer")
    require(Path(renderer.__file__).resolve() == Path(plan["renderer_module_path"]).resolve() and
            digest(Path(renderer.__file__)) == plan["renderer_module_sha256"], "Unexpected loaded renderer Python module")
    data = plan["common_data"]
    dataset = TUMDataset(load_dataset_config(data["gradslam_data_cfg"]), basedir=data["basedir"], sequence=data["sequence"],
                         start=0, end=-1, stride=1, desired_width=320, desired_height=240,
                         device=torch.device("cuda:0"), relative_pose=True, ignore_bad=False, use_train_split=True)
    require(len(dataset) == 592, "Common loader does not retain the expected 592 frames")
    selected = {str(Path(p).relative_to(Path(data["basedir"]) / data["sequence"])) for p in
                list(dataset.color_paths) + list(dataset.depth_paths)}
    require(selected == set(plan["inputs_before"]["selected_images"]), "Common loader image membership differs from training")
    rgb, depth, K4, first_c2w = dataset[0]
    common_K = K4[:3, :3].detach().cpu().numpy()
    expected_K = np.array([[258.65, 0.0, 159.3], [0.0, 258.25, 127.65], [0.0, 0.0, 1.0]])
    require(rgb.shape == (240, 320, 3) and depth.shape == (240, 320, 1) and
            np.allclose(common_K, expected_K, atol=2e-5, rtol=0), "Actual common loader resolution/K mismatch")
    common_c2w = dataset.transformed_poses.detach().cpu().numpy()
    require(np.allclose(common_c2w[0], np.eye(4), atol=2e-5, rtol=0), "Unexpected first-camera GT reference frame")
    trajectory_path = Path(plan["output"]) / "common-gt-c2w.npy"
    intrinsics_path = Path(plan["output"]) / "common-loader-intrinsics.npy"
    if label == "desk320":
        np.save(trajectory_path, common_c2w)
        np.save(intrinsics_path, common_K)
    else:
        require(np.array_equal(np.load(trajectory_path, allow_pickle=False), common_c2w) and
                np.array_equal(np.load(intrinsics_path, allow_pickle=False), common_K), "Common cameras or K differ between workers")
    with np.load(entry["model_path"], allow_pickle=False) as saved:
        all_arrays = {key: saved[key] for key in saved.files}
    require(all(np.isfinite(v).all() for v in all_arrays.values()), "Nonfinite final model arrays")
    require(all_arrays["cam_unnorm_rots"].shape == (1, 4, 592) and all_arrays["cam_trans"].shape == (1, 3, 592),
            "Expected final camera count")
    require(np.allclose(all_arrays["w2c"], torch.linalg.inv(first_c2w).detach().cpu().numpy(), atol=2e-5, rtol=0) and
            np.allclose(all_arrays["gt_w2c_all_frames"], np.linalg.inv(common_c2w), atol=2e-5, rtol=0),
            "Saved model reference frame differs from common GT reference frame")
    rendering_keys = ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales")
    params = {key: torch.as_tensor(all_arrays[key], dtype=torch.float32, device="cuda:0") for key in rendering_keys}
    common_w2c = torch.linalg.inv(torch.as_tensor(common_c2w, dtype=torch.float32, device="cuda:0"))
    rotations = matrix_to_quaternion(common_w2c[:, :3, :3])
    require(torch.max(torch.abs(build_rotation(rotations) - common_w2c[:, :3, :3])).item() < 2e-5,
            "GT W2C quaternion reconstruction mismatch")
    params["cam_unnorm_rots"] = rotations.transpose(0, 1).unsqueeze(0)
    params["cam_trans"] = common_w2c[:, :3, 3].transpose(0, 1).unsqueeze(0)
    np.savez(out / "evaluation-camera-arrays.npz", cam_unnorm_rots=params["cam_unnorm_rots"].cpu().numpy(),
             cam_trans=params["cam_trans"].cpu().numpy())
    state_hash_before = {key: hashlib.sha256(params[key].cpu().numpy().tobytes()).hexdigest() for key in rendering_keys}
    camera = {"resolution": [320, 240], "loader_K": common_K.tolist(), "sampled_frame_indices": SAMPLE,
              "common_gt_c2w_sha256": digest(trajectory_path), "common_loader_K_sha256": digest(intrinsics_path),
              "model_training_K": all_arrays["intrinsics"].tolist(),
              "model_training_resolution": [int(all_arrays["org_width"]), int(all_arrays["org_height"])],
              "gaussian_count": len(all_arrays["means3D"]), "semantics": SEMANTICS}
    save(out / "camera-and-map-binding.json", camera)
    mapping = entry["config"]["mapping"]
    with torch.no_grad():
        official_eval(dataset, params, num_frames=592, eval_dir=str(out / "official-eval"),
                      sil_thres=mapping["sil_thres"], mapping_iters=mapping["num_iters"],
                      add_new_gaussians=mapping["add_new_gaussians"], wandb_run=None, eval_every=20, save_frames=True)
    state_hash_after = {key: hashlib.sha256(params[key].cpu().numpy().tobytes()).hexdigest() for key in rendering_keys}
    require(state_hash_before == state_hash_after, "Gaussian tensors changed during fixed-map evaluation")
    require(digest(Path(entry["model_path"])) == entry["model_sha256"], "Saved model changed during evaluation")
    result = {"status": "common_gt_view_fixed_map_rendered", "run_name": entry["run_name"], "label": label,
              **camera, "map_tensors_unchanged": True, "map_tensor_sha256_before": state_hash_before,
              "map_tensor_sha256_after": state_hash_after, "original_model_sha256": entry["model_sha256"], "metrics": {}}
    directory = out / "official-eval"
    for filename, key in (("psnr", "psnr_db"), ("ssim", "ms_ssim"), ("lpips", "lpips"), ("l1", "depth_mae_m")):
        values = np.atleast_1d(np.loadtxt(directory / (filename + ".txt")))
        require(values.shape == (30,) and np.isfinite(values).all(), "Actual metric count/finiteness mismatch: " + filename)
        result["metrics"][key] = {"mean": float(values.mean()), "per_frame_values": values.tolist(),
                                  "source_sha256": digest(directory / (filename + ".txt"))}
    require(np.array_equal(np.loadtxt(directory / "l1.txt"), np.loadtxt(directory / "rmse.txt")), "Depth MAE semantics changed")
    for folder, pattern in (("plots", "*.png"), ("rendered_rgb", "gs_*.png"), ("rendered_depth", "gs_*.png"),
                            ("rgb", "gt_*.png"), ("depth", "gt_*.png")):
        indices = sorted(int(p.stem.split("_")[-1]) for p in (directory / folder).glob(pattern))
        require(indices == SAMPLE, "Saved render/reference views do not match fixed schedule: " + folder)
    result["artifact_sha256"] = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
    save(out / "metrics.json", result)


def execute(args: argparse.Namespace) -> None:
    require(sys.platform == "linux", "Execute using the isolated native Linux venv")
    root, lab = args.root.resolve(), args.lab_root.resolve()
    require(Path(sys.prefix).resolve() == (root / ".venv").resolve(), "Use the accepted isolated SplaTAM Python venv")
    repo = root / "repos/SplaTAM"
    runs = {label: input_run(root, lab, name, label) for label, name in (("desk320", args.baseline_run), ("desk640", args.high_run))}
    low, high = runs["desk320"], runs["desk640"]
    require(normalized(low["config"]) == normalized(high["config"]), "Training configs differ beyond resolution/run names")
    require(low["receipt"]["native_input_sha256_before"] == high["receipt"]["native_input_sha256_before"], "Training raw inputs differ")
    require(low["receipt"]["source_sha256_before"] == high["receipt"]["source_sha256_before"], "Training algorithm sources differ")
    sources = current_sources(repo, low["receipt"])
    require(sources == low["receipt"]["source_sha256_before"], "Current official sources differ from accepted training sources")
    require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == PIN,
            "Current source commit differs")
    inputs = current_inputs(low["receipt"], low["config"])
    require(inputs == low["receipt"]["native_input_sha256_before"], "Current raw training input bytes differ")
    renderer = low["receipt"]["renderer_provenance"]
    require(renderer["extension_sha256_before"] == high["receipt"]["renderer_provenance"]["extension_sha256_before"] and
            digest(Path(renderer["extension_file"])) == renderer["extension_sha256_before"], "Renderer binary differs")
    require(renderer["module_sha256"] == high["receipt"]["renderer_provenance"]["module_sha256"] and
            digest(Path(renderer["module_file"])) == renderer["module_sha256"], "Renderer Python module differs")
    require(os.environ.get("TORCH_HOME") == str(root / "torch-cache"), "Source the accepted env.sh so LPIPS uses its existing weights")
    require(digest(root / "torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth") ==
            "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02", "Cached accepted LPIPS backbone differs")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out = lab / "outputs/splatam-common-grid-001/runs" / stamp
    out.mkdir(parents=True, exist_ok=False)
    common_data = copy.deepcopy(low["config"]["data"])
    common_data.update(desired_image_width=320, desired_image_height=240)
    plan = {"schema_version": 1, "status": "prepared_for_actual_execution", "output": str(out), "repo": str(repo),
            "runs": runs, "common_data": common_data, "inputs_before": inputs, "source_sha256_before": sources,
            "renderer_extension_path": renderer["extension_file"], "renderer_extension_sha256": renderer["extension_sha256_before"],
            "renderer_module_path": renderer["module_file"], "renderer_module_sha256": renderer["module_sha256"],
            "entrypoint_sha256": digest(Path(__file__)), "sampled_frame_indices": SAMPLE, "semantics": SEMANTICS}
    save(out / "plan.json", plan)
    shutil.copyfile(__file__, out / "entrypoint-source.py")
    save(out / "receipt.json", {"status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "entrypoint_sha256": plan["entrypoint_sha256"], "semantics": SEMANTICS})
    state = {"status": "failed", "workers": {}, "semantics": SEMANTICS, "output": str(out)}
    try:
        for label in ("desk320", "desk640"):
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-plan", str(out / "plan.json"), "--worker-label", label]
            began = time.monotonic()
            with (out / (label + "-stdout.log")).open("wb") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo)
                code = child.wait()
            state["workers"][label] = {"command": command, "pid": child.pid, "actual_child_exit_code": code,
                                       "actual_child_wall_seconds": time.monotonic() - began,
                                       "stdout_sha256": digest(out / (label + "-stdout.log"))}
            require(code == 0, "Actual common-grid GPU worker failed: " + label)
        state["sources_unchanged"] = current_sources(repo, low["receipt"]) == sources
        state["inputs_unchanged"] = current_inputs(low["receipt"], low["config"]) == inputs
        state["renderer_unchanged"] = digest(Path(renderer["extension_file"])) == plan["renderer_extension_sha256"]
        state["renderer_module_unchanged"] = digest(Path(renderer["module_file"])) == plan["renderer_module_sha256"]
        state["entrypoint_unchanged"] = digest(Path(__file__)) == plan["entrypoint_sha256"]
        state["saved_models_unchanged"] = all(digest(Path(x["model_path"])) == x["model_sha256"] for x in runs.values())
        require(all(state[k] for k in ("sources_unchanged", "inputs_unchanged", "renderer_unchanged", "renderer_module_unchanged", "entrypoint_unchanged", "saved_models_unchanged")),
                "A fixed source/input/model binding changed during actual evaluation")
        low_metric, high_metric = read(out / "desk320/metrics.json"), read(out / "desk640/metrics.json")
        for folder in ("rgb", "depth"):
            low_files = {p.name: digest(p) for p in (out / "desk320/official-eval" / folder).glob("*.png")}
            high_files = {p.name: digest(p) for p in (out / "desk640/official-eval" / folder).glob("*.png")}
            require(low_files == high_files and len(low_files) == 30, "Common reference render inputs differ")
        require(low_metric["common_gt_c2w_sha256"] == high_metric["common_gt_c2w_sha256"] and
                low_metric["common_loader_K_sha256"] == high_metric["common_loader_K_sha256"], "Camera grid/view binding differs")
        state.update(status="completed_common_gt_view_fixed_map_rendering", models={"desk320": low_metric, "desk640": high_metric},
                     common_resolution=[320, 240], sampled_frame_indices=SAMPLE,
                     comparison_640_minus_320={key: high_metric["metrics"][key]["mean"] - low_metric["metrics"][key]["mean"]
                                                for key in ("psnr_db", "ms_ssim", "lpips", "depth_mae_m")})
        save(out / "metrics.json", state)
    except Exception as error:
        state.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        state["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(out / "receipt.json", state)
        print(json.dumps({"status": state["status"], "output": str(out), "workers": state["workers"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/gs-slam-lab/splatam"))
    parser.add_argument("--lab-root", type=Path, default=Path("/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab"))
    parser.add_argument("--baseline-run", default=BASE)
    parser.add_argument("--high-run", default=HIGH)
    parser.add_argument("--execute", action="store_true", help="Actually run GPU evaluation after all completed-run/audit gates")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", choices=("desk320", "desk640"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        require(args.worker_label is not None, "Internal worker label required")
        worker(args.worker_plan, args.worker_label)
    elif args.execute:
        execute(args)
    else:
        print(json.dumps({"status": "prepared_only_GPU_not_started", "baseline_run": args.baseline_run,
                          "high_run": args.high_run, "sampled_frame_indices": SAMPLE, "semantics": SEMANTICS}))


if __name__ == "__main__":
    main()
