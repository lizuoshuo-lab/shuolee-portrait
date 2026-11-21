"""Run the pinned SplaTAM program with its original CUDA renderer in WSL.

Creates a unique, self-contained configuration and records the real child exit.
Only the selected tracking residual weight changes; official source stays intact.
Depth validity/coverage masks, RGB-D mapping, initialization and eval remain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import uuid

PIN = "da6bbcd24c248dc884ac7f49d62e91b841b26ccc"
SOURCE_FILES = (
    "scripts/splatam.py", "utils/slam_helpers.py", "utils/slam_external.py",
    "utils/recon_helpers.py", "utils/eval_helpers.py", "utils/keyframe_selection.py",
    "utils/common_utils.py", "datasets/gradslam_datasets/tum.py",
    "datasets/gradslam_datasets/basedataset.py", "configs/tum/splatam.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/gs-slam-lab/splatam"))
    parser.add_argument("--sequence", choices=("freiburg1_desk", "freiburg1_xyz", "freiburg1_desk2", "freiburg1_room"), default="freiburg1_desk")
    parser.add_argument("--frames", type=int, default=20, help="-1 means the whole upstream-associated sequence")
    parser.add_argument("--width", type=int, choices=(320, 640), default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracking-loss", choices=("rgbd", "rgb_only", "depth_only"), default="rgbd")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        raise SystemExit("Seed must be nonnegative")
    root = args.root.resolve()
    repo = root / "repos/SplaTAM"
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise SystemExit(f"Pinned SplaTAM mismatch: {actual}")
    if args.frames != -1 and args.frames < 2:
        raise SystemExit("Need at least two frames, or -1 for a complete sequence")
    data_root = root / "data/tum"
    seqdir = data_root / f"rgbd_dataset_{args.sequence}"
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "rgb", "depth"):
        if not (seqdir / name).exists():
            raise SystemExit(f"Missing input: {seqdir / name}")
    config = runpy.run_path(str(repo / "configs/tum/splatam.py"))["config"]
    token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    height = args.width * 3 // 4
    label = "full" if args.frames == -1 else f"{args.frames}f"
    run_name = f"{args.sequence}_{label}_{args.width}x{height}_{args.tracking_loss}_s{args.seed}_{token}"
    runs = root / "outputs/runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / run_name
    run_dir.mkdir(exist_ok=False)
    shutil.copy2(__file__, run_dir / "launcher-source.py")
    config.update(
        workdir=str(runs), run_name=run_name, seed=args.seed, use_wandb=False,
        report_global_progress_every=100, eval_every=5 if args.frames != -1 else 20,
        save_checkpoints=True, checkpoint_interval=20 if args.frames != -1 else 100,
    )
    config["data"].update(
        basedir=str(data_root), sequence=f"rgbd_dataset_{args.sequence}",
        gradslam_data_cfg=str(repo / f"configs/data/TUM/{args.sequence}.yaml"),
        desired_image_width=args.width, desired_image_height=height,
        start=0, end=args.frames, stride=1, num_frames=-1,
    )
    if not Path(config["data"]["gradslam_data_cfg"]).is_file():
        # Missing YAML must be explicitly supplied for this Freiburg1 sequence.
        config["data"]["gradslam_data_cfg"] = str(root / f"configs/{args.sequence}.yaml")
    if not Path(config["data"]["gradslam_data_cfg"]).is_file():
        raise SystemExit("Missing sequence sensor calibration")
    assert config["tracking"]["use_gt_poses"] is False
    assert config["tracking"]["num_iters"] == 200
    assert config["mapping"]["num_iters"] == 30
    original_tracking_weights = dict(config["tracking"]["loss_weights"])
    assert original_tracking_weights == {"im": 0.5, "depth": 1.0}
    mapping_weights = dict(config["mapping"]["loss_weights"])
    if args.tracking_loss == "rgb_only":
        config["tracking"]["loss_weights"]["depth"] = 0.0
    elif args.tracking_loss == "depth_only":
        config["tracking"]["loss_weights"]["im"] = 0.0
    assert config["mapping"]["loss_weights"] == mapping_weights == original_tracking_weights
    config["tracking"]["visualize_tracking_loss"] = False
    config["viz"]["enter_interactive_post_online"] = False
    config["wandb"].update(save_qual=False, eval_save_qual=False)
    config_path = run_dir / "input_config.py"
    config_path.write_text("# Fully resolved local configuration; selected tracking loss weight ablation only.\nconfig = " + repr(config) + "\n", encoding="utf-8")
    save(run_dir / "effective-config.json", config)
    difference = subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--"], capture_output=True)
    if difference.returncode:
        raise SystemExit("Pinned official SplaTAM tracked content differs from HEAD")
    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0")
    tracked = sorted(name for name in tracked if name and (repo / name).is_file())
    files = {name: sha(repo / name) for name in tracked}
    for name, actual_hash in files.items():
        official_blob = subprocess.check_output(["git", "-C", str(repo), "show", f"{PIN}:{name}"])
        if hashlib.sha256(official_blob).hexdigest() != actual_hash:
            raise SystemExit(f"Official core source differs from pinned Git blob: {name}")
    import numpy as np
    rgb = np.loadtxt(seqdir / "rgb.txt", dtype=str, comments="#", ndmin=2)
    depth = np.loadtxt(seqdir / "depth.txt", dtype=str, comments="#", ndmin=2)
    gt = np.loadtxt(seqdir / "groundtruth.txt", dtype=str, comments="#", skiprows=1, ndmin=2)
    td, tg = depth[:, 0].astype(float), gt[:, 0].astype(float)
    retained, input_image_names = [], []
    for index, stamp in enumerate(rgb[:, 0].astype(float)):
        depth_index, gt_index = int(np.argmin(np.abs(td-stamp))), int(np.argmin(np.abs(tg-stamp)))
        if abs(td[depth_index]-stamp) < 0.08 and abs(tg[gt_index]-stamp) < 0.08:
            if not retained or stamp-retained[-1] > 1/32:
                retained.append(float(stamp))
                input_image_names.append((rgb[index, 1], depth[depth_index, 1]))
    expected_frames = len(retained) if args.frames == -1 else min(args.frames, len(retained))
    selected_names = sorted({str(name) for pair in input_image_names[:expected_frames] for name in pair})
    sensor = Path(config["data"]["gradslam_data_cfg"])
    shutil.copy2(sensor, run_dir / "sensor-config.yaml")
    def input_hashes():
        return {
            "raw_lists": {name: sha(seqdir / name) for name in ("rgb.txt", "depth.txt", "groundtruth.txt")},
            "selected_images": {name: sha(seqdir / name) for name in selected_names},
            "sensor_config_sha256": sha(sensor),
        }
    native_inputs = input_hashes()
    renderer = root / "repos/diff-gaussian-rasterization-w-depth"
    renderer_pin = "cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"
    source_receipt = json.loads((root / "source-receipt.json").read_text(encoding="utf-8"))
    if source_receipt.get("renderer_commit") != renderer_pin:
        raise SystemExit("Original depth renderer fixed commit receipt mismatch")
    renderer_tree = root / "downloads/renderer-tree.json"
    renderer_entries = json.loads(renderer_tree.read_text(encoding="utf-8"))["tree"]
    renderer_tracked = {}
    for entry in renderer_entries:
        if entry["type"] != "blob":
            continue
        payload = (renderer / entry["path"]).read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if git_blob != entry["sha"]:
            raise SystemExit(f"Renderer source differs from official fixed tree blob: {entry['path']}")
        renderer_tracked[entry["path"]] = hashlib.sha256(payload).hexdigest()
    import diff_gaussian_rasterization as rasterizer
    import torch
    module_path, binary_path = Path(rasterizer.__file__).resolve(), Path(rasterizer._C.__file__).resolve()
    for path in (module_path, binary_path):
        if not path.is_relative_to(root):
            raise SystemExit(f"Renderer was imported outside the isolated project: {path}")
    if sha(module_path) != sha(renderer / "diff_gaussian_rasterization/__init__.py"):
        raise SystemExit("Actually imported renderer Python source differs from pinned source")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("Expected local RTX 5080 CUDA device is unavailable")
    gpu_environment = dict(torch=torch.__version__, torch_cuda=torch.version.cuda,
                           numpy=np.__version__, gpu=torch.cuda.get_device_name(),
                           compute_capability=list(torch.cuda.get_device_capability()),
                           renderer_module=str(module_path), renderer_module_sha256=sha(module_path),
                           renderer_binary=str(binary_path), renderer_binary_sha256=sha(binary_path),
                           renderer_commit=renderer_pin)
    build_receipt = root / "bootstrap-metrics.json"
    if not build_receipt.is_file():
        raise SystemExit("Final native GPU environment acceptance receipt is missing")
    bootstrap = json.loads(build_receipt.read_text(encoding="utf-8"))
    if (bootstrap.get("status") != "isolated_official_renderer_cuda_acceptance_passed"
            or bootstrap.get("smoke", {}).get("extension_sha256") != sha(binary_path)
            or bootstrap.get("loaded_module_sha256") != sha(module_path)):
        raise SystemExit("Actually loaded renderer differs from the passed native GPU acceptance receipt")
    renderer_provenance = dict(
        commit=renderer_pin, working_tree_clean=True, expected_paths_verified=True,
        acquisition="official fixed-commit archive and GitHub tree; no native renderer .git directory",
        official_tree_blob_match=True, tracked_blob_count=len(renderer_tracked),
        tree_metadata_sha256=sha(renderer_tree),
        module_file=str(module_path), module_sha256=sha(module_path),
        extension_file=str(binary_path), extension_sha256_before=sha(binary_path),
        build_receipt_file=str(build_receipt), build_receipt_sha256=sha(build_receipt),
    )
    shutil.copy2(build_receipt, run_dir / "environment-acceptance.json")
    renderer_files = renderer_tracked
    receipt = dict(
        schema_version=1, status="prepared", run_name=run_name, root=str(root),
        launcher_sha256=sha(__file__), launcher_snapshot="launcher-source.py",
        splatam_commit=PIN, python=sys.executable, platform=sys.platform,
        renderer="original diff-gaussian-rasterization-w-depth; no gsplat substitution",
        requested_frames=args.frames, expected_frames=expected_frames, resolution=[args.width, height], config_sha256=sha(config_path),
        source_sha256_before=files, renderer_source_sha256=renderer_files,
        native_input_sha256_before=native_inputs, sensor_config_path=str(sensor),
        sensor_config_snapshot="sensor-config.yaml",
        environment=gpu_environment,
        renderer_provenance=renderer_provenance, official_git_blob_match=True,
        tracked_source_file_count=len(files),
        tracking_uses_gt_poses=False, upstream_tracking_iters=200, upstream_mapping_iters=30,
        experiment_tracking_loss=args.tracking_loss,
        tracking_loss_weights_before=original_tracking_weights,
        tracking_loss_weights_effective=dict(config["tracking"]["loss_weights"]),
        mapping_loss_weights_effective=mapping_weights,
        ablation_scope="Tracking weighted residual only; depth validity and silhouette masks, RGB-D initialization and mapping retained. Not monocular or depth-free SLAM.",
        scope="local reduced-resolution run; not paper-resolution benchmark" if args.width == 320 else "local upstream-resolution run",
        gt_scope="upstream dataset association, first-camera gauge, evaluation and finite-GT keyframe gate; no per-frame GT pose tracking",
    )
    target = run_dir / "execution.json"
    save(target, receipt)
    print(json.dumps({"run_dir": str(run_dir), "status": "prepared"}), flush=True)
    if args.check_only:
        return
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", WANDB_MODE="disabled", WANDB_DISABLED="true", PYTHONHASHSEED=str(args.seed), PYTHONUNBUFFERED="1")
    command = [sys.executable, "scripts/splatam.py", str(config_path)]
    receipt.update(status="running", command=command, started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(target, receipt)
    start = time.monotonic()
    with (run_dir / "stdout.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["pid"] = child.pid
        save(target, receipt)
        code = child.wait()
    receipt.update(child_exit_code=code, wall_seconds=time.monotonic()-start, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    receipt["source_sha256_after"] = {name: sha(repo / name) for name in tracked}
    receipt["source_unchanged_during_run"] = files == receipt["source_sha256_after"]
    receipt["launcher_unchanged_during_run"] = receipt["launcher_sha256"] == sha(__file__)
    receipt["native_input_sha256_after"] = input_hashes()
    receipt["inputs_unchanged_during_run"] = native_inputs == receipt["native_input_sha256_after"]
    receipt["renderer_binary_unchanged_during_run"] = gpu_environment["renderer_binary_sha256"] == sha(binary_path)
    receipt["renderer_source_sha256_after"] = {name: sha(renderer / name) for name in renderer_tracked}
    receipt["renderer_source_unchanged_during_run"] = renderer_files == receipt["renderer_source_sha256_after"]
    receipt["renderer_provenance"]["extension_sha256_after"] = sha(binary_path)
    params = run_dir / "params.npz"
    receipt["params_exists"] = params.is_file()
    if params.is_file():
        with np.load(params, allow_pickle=False) as p:
            receipt["estimated_camera_count"] = int(p["cam_trans"].shape[-1])
            receipt["saved_gt_count"] = int(len(p["gt_w2c_all_frames"]))
            receipt["gaussian_count"] = int(p["means3D"].shape[0])
            receipt["all_numeric_arrays_finite"] = all(np.isfinite(p[k]).all() for k in p.files if np.issubdtype(p[k].dtype, np.number))
            receipt["keyframe_indices"] = p["keyframe_time_indices"].tolist()
        receipt["params_sha256"] = sha(params)
        receipt["params_bytes"] = params.stat().st_size
    success = (code == 0 and receipt["params_exists"] and receipt["source_unchanged_during_run"]
               and receipt["launcher_unchanged_during_run"]
               and receipt["inputs_unchanged_during_run"] and receipt["renderer_binary_unchanged_during_run"]
               and receipt["renderer_source_unchanged_during_run"]
               and receipt.get("all_numeric_arrays_finite", False)
               and receipt.get("estimated_camera_count") == expected_frames
               and receipt.get("saved_gt_count") == expected_frames)
    receipt["status"] = "completed_pending_independent_evaluation" if success else "failed"
    save(target, receipt)
    print(json.dumps({"run_dir": str(run_dir), "status": receipt["status"], "child_exit_code": code}), flush=True)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
