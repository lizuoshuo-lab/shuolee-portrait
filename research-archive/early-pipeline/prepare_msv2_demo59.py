"""Prepare the original 59 full RGB demo frames without temporal resampling.

Run on Windows with --config msv2_demo59.json. --validate-only reads the
inputs and decodes checksums, but creates no files and runs no COLMAP work.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import traceback

import numpy as np
from PIL import Image


WSL_ROOT = Path(r"\\wsl.localhost\Ubuntu-20.04")
FFMPEG = Path("C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe")
FFPROBE = Path("C:/Users/Administrator/ffmpeg/bin/ffprobe.exe")
COLMAP = Path("C:/Users/Administrator/colmap-x64-windows-cuda/bin/colmap.exe")
OFFICIAL_IO = Path("C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/mini-splatting2/scene/colmap_loader.py")
IDENTITY_KEYS = ("frame", "name", "source_frame_index", "source_sha256", "wh")
TIME_KEYS = ("time_seconds", "target_time_seconds", "sampling_error_seconds")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value):
    """Read Linux-owned raw data through WSL; use native drive paths for /mnt."""
    value = str(value)
    if value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
        return Path(value[5].upper() + ":/" + value[7:])
    if value.startswith("/"):
        parts = PurePosixPath(value).parts[1:]
        if ".." in parts:
            raise ValueError("Parent traversal is not allowed in an input path")
        return WSL_ROOT.joinpath(*parts)
    return Path(value)


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    if config["expected_frames"] != 59 or len(config["frames"]) != 59:
        raise ValueError("This experiment must retain the original 59 frames")
    if config["sample_fps"] != 3 or list(config["source_wh"]) != [2704, 1520]:
        raise ValueError("Expected the original 3-fps selection at 2704x1520")
    if config["run_root"] == config.get("reuse_superresolution_root"):
        raise ValueError("The new run must be independent of the 158-frame run")
    if Path(config["prep_root"]).name != PurePosixPath(config["run_root"]).name:
        raise ValueError("Preparation and training roots must identify the same new experiment")
    manifest_path = local_path(config["source_manifest"])
    if sha256(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("Source manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    if manifest["video_sha256"] != config["video_sha256"]:
        raise ValueError("Source manifest belongs to a different video")
    old = manifest["records"]
    if len(old) != 59:
        raise ValueError("The source manifest is not the original 59-frame manifest")
    by_name = {row["name"]: row for row in old}
    if len(by_name) != 59:
        raise ValueError("Duplicate source frame names")
    selected = []
    for number, row in enumerate(config["frames"], 1):
        expected_name = f"frame_{number:06d}.png"
        if row["name"] != expected_name or row["frame"] != Path(expected_name).stem:
            raise ValueError("The original frame names and order must be retained")
        reference = by_name[row["name"]]
        for key in IDENTITY_KEYS:
            if row[key] != reference[key]:
                raise ValueError(f"Whitelist differs from source manifest: {row['name']} / {key}")
        if row["source_image"] != reference["image"]:
            raise ValueError("Whitelist must point to the preserved full-frame raw PNG")
        for key in TIME_KEYS:
            if not math.isfinite(row[key]) or not math.isclose(row[key], reference[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Source timing mismatch: {row['name']} / {key}")
        # Copy only frame identity and timing; old K, w2c and mask are not consumed.
        item = {key: reference[key] for key in IDENTITY_KEYS + TIME_KEYS}
        item["source_image"] = row["source_image"]
        selected.append(item)
    if any(a["source_frame_index"] >= b["source_frame_index"] or a["time_seconds"] >= b["time_seconds"]
           for a, b in zip(selected, selected[1:])):
        raise ValueError("Source indices and real presentation times must increase strictly")
    return config, path, selected


def decode_checksums(video):
    # There is no -r, spatial filter, hardware decoder or image extraction.
    # Encoder timestamps are not used as evidence of original presentation time.
    command = [str(FFMPEG), "-nostdin", "-v", "error", "-threads", "4", "-i", str(video),
               "-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base", "-1",
               "-pix_fmt", "rgb24", "-f", "framemd5", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", check=True)
    time_base = None
    frames = []
    for line in result.stdout.splitlines():
        if line.startswith("#tb 0:"):
            time_base = Fraction(line.split(":", 1)[1].strip())
        elif line and not line.startswith("#"):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6 or int(fields[0]) != 0:
                raise ValueError("Unexpected decoded framemd5 record")
            frames.append({"source_index": len(frames), "framemd5_pts": int(fields[2]),
                           "decoded_rgb_bytes": int(fields[4]), "rgb_md5": fields[5]})
    if time_base is None or not frames:
        raise ValueError("The video decoder returned no frame time base or checksums")
    probe_command = [str(FFPROBE), "-v", "error", "-threads", "4", "-select_streams", "v:0",
                     "-show_frames", "-show_entries",
                     "stream=width,height,time_base,nb_frames:frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time",
                     "-of", "json", str(video)]
    probe = json.loads(subprocess.check_output(probe_command, text=True, encoding="utf-8"))
    streams, presentations = probe["streams"], probe["frames"]
    if len(streams) != 1 or len(presentations) != len(frames):
        raise ValueError("Decoded RGB and source presentation-time frame counts differ")
    stream = streams[0]
    if str(stream.get("nb_frames", "")).isdigit() and int(stream["nb_frames"]) != len(frames):
        raise ValueError("The original video metadata and decoded frame counts differ")
    source_time_base = Fraction(stream["time_base"])
    for frame, presentation in zip(frames, presentations):
        stamp = presentation.get("best_effort_timestamp")
        if stamp is None:
            stamp = presentation.get("pts")
        if stamp is None:
            raise ValueError("The original video has a frame without a presentation timestamp")
        frame.update(source_pts=int(stamp),
                     source_time_base=[source_time_base.numerator, source_time_base.denominator],
                     decoded_pts_seconds=float(int(stamp) * source_time_base),
                     decoded_wh=[stream["width"], stream["height"]])
    return frames, result.stdout, command, probe_command


def verify_inputs(config, config_path, selected):
    if read_json(config_path) != config:
        raise ValueError("Configuration changed after loading")
    config_digest = sha256(config_path)
    video = local_path(config["source_video"])
    if sha256(video) != config["video_sha256"]:
        raise ValueError("The original video SHA256 differs from the experiment configuration")
    decoded, checksum_text, command, probe_command = decode_checksums(video)
    records = []
    width, height = config["source_wh"]
    for row in selected:
        source = local_path(row["source_image"])
        if sha256(source) != row["source_sha256"]:
            raise ValueError(f"Raw PNG SHA256 mismatch: {source}")
        with Image.open(source) as image:
            if image.mode != "RGB" or image.size != (width, height):
                raise ValueError(f"Raw image is not full native RGB: {source}")
            digest = hashlib.md5(image.tobytes()).hexdigest()
        index = row["source_frame_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(decoded):
            raise ValueError("Source frame index outside decoded video")
        original = decoded[index]
        if (original["decoded_wh"] != [width, height] or
                original["decoded_rgb_bytes"] != width * height * 3 or original["rgb_md5"] != digest):
            raise ValueError(f"Raw pixels differ from the original video at decoded index {index}")
        if not math.isclose(original["decoded_pts_seconds"], row["time_seconds"], rel_tol=0, abs_tol=1.1e-6):
            raise ValueError(f"Original video PTS differs from the preserved manifest at index {index}")
        match = {key: original[key] for key in ("source_index", "source_pts", "source_time_base", "decoded_pts_seconds")}
        match.update({key: row[key] for key in TIME_KEYS})
        records.append(dict(image=row["name"], sha256=row["source_sha256"], rgb_md5=digest,
                            matching_original_frames=[match], wh=list(config["source_wh"]),
                            source_image=row["source_image"], source_frame_index=index,
                            **{key: row[key] for key in TIME_KEYS}))
    if sha256(config_path) != config_digest:
        raise ValueError("Configuration changed during input verification")
    audit = dict(passed=True, output_frames=len(records), source_frames=len(decoded),
                 no_temporal_resampling=True, no_spatial_resizing=True, no_mask=True,
                 video_sha256=config["video_sha256"], source_manifest=config["source_manifest"],
                 source_manifest_sha256=config["source_manifest_sha256"], config=str(config_path),
                 config_sha256=config_digest, verification_command=command,
                 pts_verification_command=probe_command,
                 source_pts_origin="ffprobe original best_effort_timestamp and stream time_base", records=records)
    return audit, checksum_text


def official_reader():
    spec = importlib.util.spec_from_file_location("demo59_official_colmap_io", OFFICIAL_IO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def colmap_commands(root):
    database, images, sparse = root / "database.db", root / "dataset/images", root / "raw-sparse"
    return [
        ("features", [COLMAP, "feature_extractor", "--database_path", database, "--image_path", images,
                      "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
                      "--SiftExtraction.use_gpu", "0", "--SiftExtraction.num_threads", "4",
                      "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]),
        ("matching", [COLMAP, "sequential_matcher", "--database_path", database,
                      "--SiftMatching.guided_matching", "1", "--SiftMatching.use_gpu", "0",
                      "--SiftMatching.num_threads", "4", "--SequentialMatching.overlap", "10"]),
        ("mapping", [COLMAP, "mapper", "--database_path", database, "--image_path", images,
                     "--output_path", sparse, "--Mapper.num_threads", "4", "--Mapper.ba_refine_principal_point", "1"]),
    ]


def run_command(root, state, stage, command, log_name=None):
    args = list(map(str, command))
    state.update(stage=stage, updated=now())
    state["commands"].append(args)
    save_json(root / "state.json", state)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with (root / (log_name or stage + ".log")).open("w", encoding="utf-8") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)


def finalize(config, config_path, root, state):
    if sha256(config_path) != state["config_sha256"]:
        raise ValueError("Configuration changed during preparation")
    audit = read_json(root / "pixel-lineage.json")
    if not audit["passed"] or audit["config_sha256"] != state["config_sha256"]:
        raise ValueError("Missing matching full-frame pixel verification")
    expected = {row["image"]: row for row in audit["records"]}
    images = root / "dataset/images"
    if {path.name for path in images.iterdir()} != set(expected):
        raise ValueError("Prepared image directory differs from the exact whitelist")
    for name, row in expected.items():
        if sha256(images / name) != row["sha256"]:
            raise ValueError(f"Prepared input changed: {name}")
    io = official_reader()
    models = []
    for model in sorted((root / "raw-sparse").iterdir()):
        if model.is_dir() and all((model / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
            extrinsics = io.read_extrinsics_binary(model / "images.bin")
            models.append((len(extrinsics), model, extrinsics))
    if not models:
        raise ValueError("COLMAP produced no complete binary model")
    registered, selected_model, extrinsics = max(models, key=lambda item: item[0])
    names = [image.name for image in extrinsics.values()]
    if registered != 59 or len(set(names)) != 59 or set(names) != set(expected):
        raise ValueError(f"COLMAP must register all 59 whitelist frames; got {registered}, missing={sorted(set(expected)-set(names))}")
    intrinsics = io.read_intrinsics_binary(selected_model / "cameras.bin")
    if len(intrinsics) != 1:
        raise ValueError("Expected a single shared PINHOLE camera")
    xyz, _, _ = io.read_points3D_binary(selected_model / "points3D.bin")
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("COLMAP sparse geometry is empty or nonfinite")
    records = []
    for image in sorted(extrinsics.values(), key=lambda value: value.name):
        camera = intrinsics[image.camera_id]
        if camera.model != "PINHOLE" or [camera.width, camera.height] != config["source_wh"]:
            raise ValueError("Recovered camera model or image dimensions differ from full RGB inputs")
        fx, fy, cx, cy = map(float, camera.params)
        if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
            raise ValueError("Invalid recovered PINHOLE intrinsics")
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = image.qvec2rotmat()
        world_to_camera[:3, 3] = image.tvec
        rotation = world_to_camera[:3, :3]
        if (not np.isfinite(world_to_camera).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6) or
                abs(np.linalg.det(rotation) - 1) > 1e-6):
            raise ValueError("Invalid recovered camera pose")
        original = expected[image.name]
        records.append(dict(frame=Path(image.name).stem, name=image.name, image=str(images / image.name),
                            wh=list(config["source_wh"]), K=[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                            w2c=world_to_camera.tolist(), source_sha256=original["sha256"],
                            source_frame_matches=original["matching_original_frames"],
                            source_frame_index=original["source_frame_index"],
                            **{key: original[key] for key in TIME_KEYS}))
    destination = root / "dataset/sparse/0"
    destination.mkdir(parents=True, exist_ok=True)
    binary_hashes = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(selected_model / name, destination / name)
        binary_hashes[name] = sha256(destination / name)
        if binary_hashes[name] != sha256(selected_model / name):
            raise ValueError("COLMAP binary copy mismatch")
    run_command(root, state, "model_analysis", [COLMAP, "model_analyzer", "--path", destination], "model-info.log")
    save_json(root / "camera-inputs.json", dict(records=records, registered=59, extracted=59,
              coordinate_contract="COLMAP pixel-corner coordinates", mask_used=False, crop_used=False,
              old_camera_parameters_used=False, video_sha256=config["video_sha256"],
              source_manifest_sha256=config["source_manifest_sha256"], config_sha256=state["config_sha256"],
              sparse_binary_sha256=binary_hashes))
    recipe = read_json(root / "recipe.json")
    recipe.update(frames_registered=59, sparse_points=len(xyz), selected_model=str(selected_model),
                  sparse_binary_sha256=binary_hashes, completed=now())
    save_json(root / "recipe.json", recipe)
    state.update(state="completed", stage="camera_inputs", finished=now(), frames_extracted=59,
                 frames_registered=59, source_frames=audit["source_frames"], selected_model=str(selected_model))
    save_json(root / "state.json", state)
    print("DEMO59_DATASET_READY 59 59", flush=True)


def prepare(config, config_path, selected):
    started = now()
    audit, checksum_text = verify_inputs(config, config_path, selected)
    root = local_path(config["prep_root"])
    root.mkdir(parents=True, exist_ok=False)
    state = dict(state="running", stage="copying_whitelist_images", started=started,
                 config=str(config_path), config_sha256=audit["config_sha256"],
                 source_manifest=config["source_manifest"], source_manifest_sha256=config["source_manifest_sha256"],
                 video=config["source_video"], video_sha256=config["video_sha256"], expected_frames=59,
                 commands=[audit["verification_command"], audit["pts_verification_command"]],
                 training_started=False, quality_accepted=False)
    save_json(root / "state.json", state)
    try:
        images = root / "dataset/images"
        images.mkdir(parents=True)
        (root / "raw-sparse").mkdir()
        for row in audit["records"]:
            target = images / row["image"]
            shutil.copy2(local_path(row["source_image"]), target)
            if sha256(target) != row["sha256"]:
                raise ValueError(f"Byte-preserving input copy failed: {target}")
        (root / "original-video-framemd5.txt").write_text(checksum_text, encoding="utf-8")
        save_json(root / "pixel-lineage.json", audit)
        save_json(root / "recipe.json", dict(created=now(), sample_fps=3, fps=3,
                  selection="Original 59 exact source indices and real PTS; no new -r3 extraction",
                  source_wh=config["source_wh"], frames=59, video_sha256=config["video_sha256"],
                  config=str(config_path), config_sha256=audit["config_sha256"],
                  source_manifest=config["source_manifest"], source_manifest_sha256=config["source_manifest_sha256"],
                  camera_model="PINHOLE", single_camera=True, cpu_threads=4,
                  estimate_affine_shape=True, domain_size_pooling=True, guided_matching=True,
                  sequential_overlap=10, ba_refine_principal_point=True,
                  camera_reader=str(OFFICIAL_IO), camera_reader_sha256=sha256(OFFICIAL_IO),
                  colmap=str(COLMAP), colmap_sha256=sha256(COLMAP), run_root=config["run_root"],
                  record_root=config["record_root"], masks_used=False, crop_used=False,
                  no_spatial_resizing=True, old_camera_parameters_used=False,
                  original_video_changed=False, raw_image_bytes_changed=False,
                  fitted_gaussians_loaded=False, training_started=False, quality_accepted=False))
        for stage, command in colmap_commands(root):
            run_command(root, state, stage, command)
        finalize(config, config_path, root, state)
    except BaseException:
        state.update(state="failed", finished=now(), error=traceback.format_exc())
        save_json(root / "state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Verify inputs read-only; no files or COLMAP")
    mode.add_argument("--finalize", action="store_true", help="Read completed COLMAP outputs without rerunning recovery")
    args = parser.parse_args()
    config, config_path, selected = load_config(args.config)
    if args.validate_only:
        audit, _ = verify_inputs(config, config_path, selected)
        print("DEMO59_INPUTS_VERIFIED_READ_ONLY " + json.dumps(dict(
            frames=audit["output_frames"], source_frames=audit["source_frames"], source_wh=config["source_wh"],
            video_sha256=audit["video_sha256"], config_sha256=audit["config_sha256"],
            source_manifest_sha256=audit["source_manifest_sha256"], all_pixels_match_original_video=True)), flush=True)
    elif args.finalize:
        root = local_path(config["prep_root"])
        state = read_json(root / "state.json")
        try:
            finalize(config, config_path, root, state)
        except BaseException:
            state.update(state="failed", finished=now(), error=traceback.format_exc())
            save_json(root / "state.json", state)
            raise
    else:
        prepare(config, config_path, selected)


if __name__ == "__main__":
    main()
