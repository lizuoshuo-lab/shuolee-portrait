"""Build the COLMAP scene and its masked static Gaussian model from one video."""

from argparse import ArgumentParser
from bisect import bisect_left
from hashlib import sha256 as hash_sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np


def sha256(path):
    digest = hash_sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_image(path, image, params=()):
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write {path}")


def fixed_time_samples(timestamps, sample_fps):
    """Pair each grid time through the last decoded timestamp with its closest frame."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("Sampling FPS must be positive and finite.")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Fixed-FPS sampling needs strictly increasing source timestamps.")
    intervals = (timestamps[-1] - timestamps[0]) * sample_fps
    if not math.isfinite(intervals) or intervals >= len(timestamps):
        raise ValueError("The requested FPS would repeat decoded source frames.")
    count = math.floor(intervals) + 1
    if count < 3:
        raise ValueError("The requested FPS produces fewer than three scene frames.")

    samples = []
    for tick in range(count):
        target = timestamps[0] + tick / sample_fps
        right = min(bisect_left(timestamps, target), len(timestamps) - 1)
        left = max(0, right - 1)
        nearest = min((left, right), key=lambda index: (abs(timestamps[index] - target), index))
        if samples and nearest <= samples[-1][0]:
            raise ValueError("The requested FPS would repeat decoded source frames.")
        samples.append((nearest, target))
    return samples


def sample_frames(video, image_dir, count, width, *, sample_fps=None):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    timestamps = [float(record["best_effort_timestamp_time"])
                  for record in json.loads(probe.stdout)["frames"]]
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or total < 3 or total != len(timestamps):
            raise ValueError("Decoded frame count does not agree with the video timestamps.")
        if not math.isfinite(fps) or fps <= 0 or not all(math.isfinite(t) for t in timestamps):
            raise ValueError("Video contains invalid timestamps.")
        digits = max(5, len(str(total - 1)))
        if sample_fps is None:
            count = min(total, count)
            samples = [(index * (total - 1) // (count - 1), None) for index in range(count)]
        else:
            samples = fixed_time_samples(timestamps, sample_fps)
        frames = []
        image_dir.mkdir()
        for index, target in samples:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise OSError(f"Could not seek to source frame {index}.")
            ok, image = cap.read()
            if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise OSError(f"Could not decode source frame {index}.")
            source_height, source_width = image.shape[:2]
            height = int(source_height * width / source_width)
            frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            path = image_dir / f"{index:0{digits}d}.jpg"
            write_image(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            record = {"name": path.name, "source_index": index,
                      "timestamp_s": timestamps[index], "raw_sha256": sha256(path)}
            if target is not None:
                record["sample_timestamp_s"] = target
            frames.append(record)
        return fps, frames
    finally:
        cap.release()


def matte_and_inpaint(scene, frames, home):
    sys.path.insert(0, str(home / "VHAP"))
    import torch
    from vhap.preprocess_video import robust_video_matting

    hub = Path(torch.hub.get_dir())
    rvm_repo = hub / "PeterL1n_RobustVideoMatting_master"
    rvm_weights = hub / "checkpoints/rvm_resnet50.pth"
    if not (rvm_repo / "hubconf.py").is_file() or not rvm_weights.is_file():
        raise FileNotFoundError(f"Cached VHAP RVM ResNet50 code and weights are required in {hub}.")
    hub_load = torch.hub.load

    def cached_rvm(repo, model):
        if (repo, model) != ("PeterL1n/RobustVideoMatting", "resnet50"):
            raise ValueError(f"Unexpected VHAP matting model: {repo}/{model}")
        return hub_load(str(rvm_repo), model, source="local")

    # The upstream helper derives alpha paths by replacing the relative 'images' directory.
    original_dir = Path.cwd()
    try:
        os.chdir(scene)
        with patch.object(torch.hub, "load", cached_rvm):
            robust_video_matting(Path("images"))
    finally:
        os.chdir(original_dir)
    person_masks = scene / "person_masks"
    inpainted_images = scene / "images_inpaint"
    person_masks.mkdir()
    inpainted_images.mkdir()
    dilation = np.ones((9, 9), dtype=np.uint8)
    for frame in frames:
        name = frame["name"]
        image = cv2.imread(str(scene / "images" / name))
        alpha = cv2.imread(str(scene / "alpha_maps" / name), cv2.IMREAD_GRAYSCALE)
        if image is None or alpha is None or image.shape[:2] != alpha.shape:
            raise ValueError(f"Original RGB and matte do not correspond: {name}")
        mask = cv2.dilate((alpha > 40).astype(np.uint8), dilation, iterations=2)
        if not mask.any():
            raise ValueError(f"Matting did not find a person in {name}.")
        write_image(person_masks / name, mask * 255)
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        inpainted = cv2.inpaint(small, small_mask, 7, cv2.INPAINT_TELEA)
        inpainted = cv2.resize(inpainted, image.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        write_image(inpainted_images / name, inpainted, [cv2.IMWRITE_JPEG_QUALITY, 95])


def run_logged(command, log_path, cwd=None, env=None):
    print(f"Running {log_path.stem}, log: {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run([str(part) for part in command], stdout=log, stderr=subprocess.STDOUT,
                       cwd=cwd, env=env, check=True)


def build_colmap(scene, home, logs):
    colmap = "/usr/local/bin/colmap"
    work = scene / "colmap_bg"
    sparse = work / "sparse"
    sparse.mkdir(parents=True)
    database = work / "db.db"
    inpaint = scene / "images_inpaint"
    run_logged(["xvfb-run", "-a", colmap, "feature_extractor", "--database_path", database,
                "--image_path", inpaint, "--ImageReader.single_camera", 1,
                "--ImageReader.camera_model", "OPENCV", "--SiftExtraction.max_num_features", 8192],
               logs / "colmap-features.log")
    run_logged(["xvfb-run", "-a", colmap, "sequential_matcher", "--database_path", database,
                "--SequentialMatching.overlap", 20, "--SequentialMatching.loop_detection", 0],
               logs / "colmap-matches.log")
    run_logged([colmap, "mapper", "--database_path", database, "--image_path", inpaint,
                "--output_path", sparse, "--Mapper.ba_global_function_tolerance", "1e-5"],
               logs / "colmap-mapper.log")
    # Use the same COLMAP reader as the static Gaussian trainer.
    sys.path.insert(0, str(home / "gaussian-splatting/scene"))
    from colmap_loader import read_extrinsics_binary, read_intrinsics_binary

    models = [(path, len(read_extrinsics_binary(path / "images.bin")))
              for path in sorted(sparse.iterdir()) if (path / "images.bin").is_file()]
    if not models or max(count for _, count in models) < 3:
        raise RuntimeError("COLMAP did not register three images in one model.")
    model, count = max(models, key=lambda item: item[1])
    print(f"COLMAP model {model.name} registered {count} frames.", flush=True)
    for name, src in (("orig_undist", "images"), ("mask_undist", "person_masks")):
        run_logged([colmap, "image_undistorter", "--image_path", scene / src,
                    "--input_path", model, "--output_path", scene / name, "--output_type", "COLMAP"],
                   logs / f"colmap-{name}.log")

    native_sparse = scene / "orig_undist/sparse"
    cameras = read_intrinsics_binary(native_sparse / "cameras.bin")
    if len(cameras) != 1 or next(iter(cameras.values())).model != "PINHOLE":
        raise ValueError("Expected one registered PINHOLE after COLMAP undistortion.")
    camera = next(iter(cameras.values()))
    fx, fy, cx, cy = camera.params
    if (min(fx, fy) <= 0 or not np.isfinite(camera.params).all()
            or abs(cx - camera.width / 2) > 1e-6 or abs(cy - camera.height / 2) > 1e-6):
        raise ValueError("The reused static Gaussian trainer needs a centered, native PINHOLE camera.")
    for binary in ("cameras.bin", "images.bin"):
        if sha256(native_sparse / binary) != sha256(scene / "mask_undist/sparse" / binary):
            raise ValueError("RGB and mask undistortion projection records disagree.")
    training_sparse = native_sparse / "0"
    training_sparse.mkdir()
    for binary in ("cameras.bin", "images.bin", "points3D.bin"):
        os.link(native_sparse / binary, training_sparse / binary)
    registered = {record.name for record in read_extrinsics_binary(native_sparse / "images.bin").values()}
    if len(registered) != count:
        raise ValueError("The native COLMAP model does not retain the registered image set.")
    return registered, (camera.width, camera.height)


def prepare_head_masks(scene, frames, native_size):
    out = scene / "headmask_undist/images"
    out.mkdir(parents=True)
    for frame in frames:
        name = frame["name"]
        native = scene / "orig_undist/images" / name
        image = cv2.imread(str(native), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(scene / "mask_undist/images" / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[1::-1] != native_size or mask.shape[::-1] != native_size:
            raise ValueError(f"Undistorted RGB and mask do not agree with COLMAP: {name}")
        binary = (mask > 60).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count < 2:
            raise ValueError(f"Missing foreground after mask undistortion: {name}")
        component = labels == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        rows = np.flatnonzero(component.any(axis=1))
        top = rows[0]
        bottom = top + max(1, int(0.2 * (rows[-1] - top + 1)))
        head = np.zeros_like(binary)
        head[top:bottom] = component[top:bottom]
        head = cv2.dilate(head, np.ones((11, 11), np.uint8))
        write_image(out / f"{native.stem}.png", head * 255)
        frame["native_sha256"] = sha256(native)


def train_static_scene(scene, background_steps=18000, holdout=()):
    scene = Path(scene).resolve(strict=True)
    if background_steps < 1:
        raise ValueError("Static training requires positive iterations.")
    manifest = json.loads((scene / "source-frames.json").read_text(encoding="utf-8"))
    names = {frame["name"] for frame in manifest["frames"]}
    if not set(holdout) <= names:
        raise ValueError(f"Unknown static RGB holdout: {sorted(set(holdout) - names)}")
    for frame in manifest["frames"]:
        native = scene / "orig_undist/images" / frame["name"]
        if sha256(native) != frame["native_sha256"]:
            raise ValueError(f"Native scene input changed: {native}")
    home = Path.home()
    background = scene / "scene_out/bg_body2"
    background.parent.mkdir(exist_ok=True)
    if background.exists():
        raise FileExistsError(f"Static scene output already exists: {background}")
    staged = Path(tempfile.mkdtemp(prefix=".bg-body2-", dir=background.parent))
    training_env = os.environ.copy()
    training_env["PYTHONPATH"] = str(home / "ga_pkgs")
    trainer = home / "tools/pipeline/train_masked.py"
    # Reuse its selected camera objects and leave the complete camera export intact.
    entry = """
import json
from pathlib import Path
import runpy
import sys

from scene import Scene

excluded = set(json.loads(sys.argv.pop(1)))
get_training_cameras = Scene.getTrainCameras

def selected_training_cameras(self, *args, **kwargs):
    current = get_training_cameras(self, *args, **kwargs)
    selected = [camera for camera in current if camera.image_name not in excluded]
    if len(selected) < 2:
        raise ValueError("The static RGB split leaves fewer than two training views.")
    return selected

Scene.getTrainCameras = selected_training_cameras
sys.argv.pop(0)
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    excluded = sorted({Path(name).stem for name in holdout})
    run_logged(
        [home / "miniconda3/envs/fastavatar/bin/python", "-u", "-B", "-c", entry,
         json.dumps(excluded), trainer, "--source", scene / "orig_undist",
         "--masks", scene / "headmask_undist/images", "--model", staged,
         "--iterations", background_steps],
        staged / "static-background.log", cwd=home / "gaussian-splatting", env=training_env,
    )
    ply = staged / "point_cloud" / f"iteration_{background_steps}" / "point_cloud.ply"
    cameras = json.loads((staged / "cameras.json").read_text(encoding="utf-8"))
    if not ply.is_file() or {record["img_name"] for record in cameras} != {Path(name).stem for name in names}:
        raise RuntimeError("Static training did not export the PLY and all registered cameras.")
    (staged / "rgb-split.json").write_text(json.dumps({
        "excluded_after_repo_split": sorted(holdout), "iterations": background_steps,
        "source_manifest_sha256": sha256(scene / "source-frames.json"),
    }, indent=2) + "\n", encoding="utf-8")
    staged.rename(background)
    return background


def prepare_video_scene(video, scene, frame_count=80, width=960, background_steps=18000, *, sample_fps=None):
    video = Path(video).resolve(strict=True)
    scene = Path(scene).resolve()
    if not video.is_file() or frame_count < 3 or width < 32 or background_steps < 0:
        raise ValueError("Expected a video, at least three frames, width >= 32, and nonnegative steps.")
    if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
        raise ValueError("Sampling FPS must be positive and finite.")
    home = Path.home()
    scene.mkdir(parents=True, exist_ok=False)
    logs = scene / "logs"
    logs.mkdir()
    print("Sampling original video frames.", flush=True)
    fps, frames = sample_frames(video, scene / "images", frame_count, width, sample_fps=sample_fps)
    print(f"Running VHAP RVM matting on {len(frames)} sampled frames.", flush=True)
    matte_and_inpaint(scene, frames, home)
    registered, native_size = build_colmap(scene, home, logs)
    selected = [frame for frame in frames if frame["name"] in registered]
    if len(selected) != len(registered):
        raise ValueError("COLMAP contains names outside the original sampling manifest.")
    missing = [frame for frame in frames if frame["name"] not in registered]
    manifest = {
        "video": str(video), "video_sha256": sha256(video), "fps": fps,
        "native_size": native_size, "frames": selected,
        "unregistered": missing,
    }
    if sample_fps is not None:
        manifest["sample_fps"] = sample_fps
        if missing:
            diagnostic = scene / "source-frames-incomplete.json"
            diagnostic.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Fixed-FPS scene registered {len(selected)}/{len(frames)} sampled frames. "
                               f"Missing frame names and timestamps: {diagnostic}")
    prepare_head_masks(scene, selected, native_size)
    (scene / "source-frames.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if background_steps:
        train_static_scene(scene, background_steps)
    return manifest


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--frame-count", type=int, default=80)
    sampling.add_argument("--sample-fps", type=float, help="Sample the complete clip on a fixed time grid.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--background-steps", type=int, default=18000)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--train-only", action="store_true")
    parser.add_argument("--holdout", nargs="+", default=[], help="Source JPG names excluded from static RGB training.")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.train_only:
        train_static_scene(args.scene, args.background_steps, args.holdout)
    else:
        if args.video is None or args.holdout:
            parser.error("Specify --video to prepare, and use --holdout with --train-only.")
        prepare_video_scene(args.video, args.scene, args.frame_count, args.width,
                            0 if args.prepare_only else args.background_steps, sample_fps=args.sample_fps)
