"""One real native replay of saved estimated cameras to validate map loading.

Uses the frozen desk320 model and original official eval, then compares all 30
raw metrics to the frozen original final eval. Never substitutes GT cameras.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
import evaluate_common_grid as common

COMMON_SHA = "feb14a015424eb6a9b72cbe5d129cf4bfd6f7844e3f571ec0e7c91885b81ec4c"
FILES = ("psnr.txt", "ssim.txt", "lpips.txt", "l1.txt", "rmse.txt")


def worker(plan_path: Path) -> None:
    plan = common.read(plan_path)
    out = Path(plan["output"])
    entry = plan["run"]
    repo = Path(plan["repo"])
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import diff_gaussian_rasterization as renderer
    from datasets.gradslam_datasets import TUMDataset, load_dataset_config
    from utils.eval_helpers import eval as official_eval
    common.require(common.digest(Path(renderer._C.__file__)) == plan["renderer_extension_sha256"] and
                   common.digest(Path(renderer.__file__)) == plan["renderer_module_sha256"], "Unexpected actual renderer")
    data = entry["config"]["data"]
    dataset = TUMDataset(load_dataset_config(data["gradslam_data_cfg"]), basedir=data["basedir"], sequence=data["sequence"],
                         start=0, end=-1, stride=1, desired_width=320, desired_height=240,
                         device=torch.device("cuda:0"), relative_pose=True, ignore_bad=False, use_train_split=True)
    common.require(len(dataset) == 592, "Expected full original desk loader")
    with np.load(entry["model_path"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    keys = ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales", "cam_unnorm_rots", "cam_trans")
    params = {key: torch.as_tensor(arrays[key], dtype=torch.float32, device="cuda:0") for key in keys}
    before = {key: hashlib.sha256(params[key].cpu().numpy().tobytes()).hexdigest() for key in keys}
    _, _, K, first_c2w = dataset[0]
    common.require(np.allclose(K[:3, :3].cpu().numpy(), arrays["intrinsics"], rtol=0, atol=2e-5), "Loader K differs from saved desk320 K")
    common.require(np.allclose(torch.linalg.inv(first_c2w).cpu().numpy(), arrays["w2c"], rtol=0, atol=2e-5), "First camera reference differs")
    common.save(out / "camera-binding.json", {"resolution": [320, 240], "actual_loader_K": K[:3, :3].cpu().numpy().tolist(),
                                              "camera_source": "Original saved estimated cam_unnorm_rots/cam_trans; no GT replacement.",
                                              "tensor_sha256_before": before, "model_sha256": entry["model_sha256"]})
    mapping = entry["config"]["mapping"]
    with torch.no_grad():
        official_eval(dataset, params, num_frames=592, eval_dir=str(out / "official-eval"), sil_thres=mapping["sil_thres"],
                      mapping_iters=mapping["num_iters"], add_new_gaussians=mapping["add_new_gaussians"],
                      wandb_run=None, eval_every=20, save_frames=False)
    after = {key: hashlib.sha256(params[key].cpu().numpy().tobytes()).hexdigest() for key in keys}
    common.require(before == after, "Frozen map or original estimated camera tensors changed")
    results = {}
    for filename in FILES:
        actual = np.atleast_1d(np.loadtxt(out / "official-eval" / filename))
        original = np.atleast_1d(np.loadtxt(Path(plan["original_eval"]) / filename))
        common.require(actual.shape == original.shape == (30,) and np.isfinite(actual).all() and np.isfinite(original).all(),
                       "Raw metric count/finiteness mismatch: " + filename)
        difference = np.abs(actual - original)
        results[filename] = {"passed_absolute_tolerance": bool(np.all(difference <= plan["absolute_tolerance"])),
                             "absolute_tolerance": plan["absolute_tolerance"], "maximum_absolute_difference": float(difference.max()),
                             "mean_absolute_difference": float(difference.mean()), "actual_mean": float(actual.mean()),
                             "original_mean": float(original.mean()), "actual_per_frame_values": actual.tolist(),
                             "original_per_frame_values": original.tolist(),
                             "actual_metric_sha256": common.digest(out / "official-eval" / filename),
                             "original_metric_sha256": common.digest(Path(plan["original_eval"]) / filename)}
    common.save(out / "worker-metrics.json", {"status": "passed" if all(x["passed_absolute_tolerance"] for x in results.values()) else "failed",
                                              "raw_metric_comparisons": results, "sampled_frame_indices": common.SAMPLE,
                                              "original_estimated_cameras_unchanged": True, "tensor_sha256_before": before,
                                              "tensor_sha256_after": after, "map_camera_tensors_unchanged": True,
                                              "scope": "Necessary saved-map/loading positive control, not new training or GT-view accuracy."})
    common.require(all(x["passed_absolute_tolerance"] for x in results.values()), "Raw render metric positive control did not match original eval")


def execute(args: argparse.Namespace) -> None:
    root, lab = args.root.resolve(), args.lab_root.resolve()
    common.require(sys.platform == "linux" and Path(sys.prefix).resolve() == (root / ".venv").resolve(), "Use accepted native venv")
    common.require(common.digest(Path(common.__file__)) == COMMON_SHA, "Frozen common-grid helper source changed")
    entry = common.input_run(root, lab, common.BASE, "desk320")
    repo = root / "repos/SplaTAM"
    sources = common.current_sources(repo, entry["receipt"])
    inputs = common.current_inputs(entry["receipt"], entry["config"])
    common.require(sources == entry["receipt"]["source_sha256_before"] and inputs == entry["receipt"]["native_input_sha256_before"],
                   "Current original source/input bytes differ")
    original_eval = Path(entry["run_path"]) / "eval"
    audit = common.read(Path(entry["runtime_audit_path"]))
    expected = {name: common.digest(original_eval / name) for name in FILES}
    common.require(all(expected[name] == audit["official_metric_files"][name]["sha256"] for name in FILES), "Original eval raw metric bytes changed")
    renderer = entry["receipt"]["renderer_provenance"]
    common.require(common.digest(Path(renderer["extension_file"])) == renderer["extension_sha256_before"] and
                   common.digest(Path(renderer["module_file"])) == renderer["module_sha256"], "Expected original renderer")
    common.require(os.environ.get("TORCH_HOME") == str(root / "torch-cache"), "Source the accepted env.sh")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out = lab / "outputs/splatam-common-grid-001/positive-control-runs" / stamp
    out.mkdir(parents=True, exist_ok=False)
    plan = {"output": str(out), "run": entry, "repo": str(repo), "original_eval": str(original_eval),
            "original_metric_sha256": expected, "source_sha256_before": sources, "input_sha256_before": inputs,
            "renderer_extension_sha256": renderer["extension_sha256_before"], "renderer_module_sha256": renderer["module_sha256"],
            "entrypoint_sha256": common.digest(Path(__file__)), "common_helper_sha256": COMMON_SHA,
            "absolute_tolerance": args.absolute_tolerance, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    common.save(out / "plan.json", plan)
    shutil.copyfile(__file__, out / "entrypoint-source.py")
    state = {"status": "running", "unique_output": str(out), "plan_sha256": common.digest(out / "plan.json")}
    common.save(out / "receipt.json", state)
    try:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-plan", str(out / "plan.json")]
        started = time.monotonic()
        with (out / "stdout.log").open("wb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo)
            code = child.wait()
        state.update(command=command, pid=child.pid, actual_child_exit_code=code,
                     actual_child_wall_seconds=time.monotonic() - started, stdout_sha256=common.digest(out / "stdout.log"))
        common.require(code == 0, "Actual positive-control child failed; preserve its worker metrics and log")
        checks = {"saved_model_unchanged": common.digest(Path(entry["model_path"])) == entry["model_sha256"],
                  "official_sources_unchanged": common.current_sources(repo, entry["receipt"]) == sources,
                  "raw_inputs_unchanged": common.current_inputs(entry["receipt"], entry["config"]) == inputs,
                  "original_raw_eval_metrics_unchanged": all(common.digest(original_eval / name) == expected[name] for name in FILES),
                  "renderer_extension_unchanged": common.digest(Path(renderer["extension_file"])) == renderer["extension_sha256_before"],
                  "renderer_module_unchanged": common.digest(Path(renderer["module_file"])) == renderer["module_sha256"],
                  "entrypoint_unchanged": common.digest(Path(__file__)) == plan["entrypoint_sha256"],
                  "frozen_common_grid_helper_unchanged": common.digest(Path(common.__file__)) == COMMON_SHA}
        common.require(all(checks.values()), "Positive-control fixed source/model/input gate changed")
        state.update(status="passed_original_estimated_camera_raw_render_positive_control", checks=checks,
                     worker_metrics=common.read(out / "worker-metrics.json"), original_model_sha256=entry["model_sha256"])
        common.save(out / "metrics.json", state)
    except Exception as error:
        state.update(status="failed_preserved_positive_control", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        state["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        common.save(out / "receipt.json", state)
        print(json.dumps({"status": state["status"], "output": str(out), "actual_child_exit_code": state.get("actual_child_exit_code")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/gs-slam-lab/splatam"))
    parser.add_argument("--lab-root", type=Path, default=Path("/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    common.require(0 < args.absolute_tolerance <= 1e-4, "Keep an explicit absolute metric tolerance no larger than 1e-4")
    if args.worker_plan:
        worker(args.worker_plan)
    elif args.execute:
        execute(args)
    else:
        print(json.dumps({"status": "prepared_only_GPU_not_started", "absolute_tolerance": args.absolute_tolerance}))


if __name__ == "__main__":
    main()
