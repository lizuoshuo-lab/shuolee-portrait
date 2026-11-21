"""Run the VHAP + body/scene Gaussian pipeline from one video."""

from argparse import ArgumentParser, SUPPRESS
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import monotonic


PROJECT = Path(__file__).resolve().parent
VHAP = Path.home() / "VHAP"
GAUSSIANS = Path.home() / "GaussianAvatars"
VHAP_PYTHON = Path.home() / "venvs/vhap/bin/python"
GAUSSIANS_PYTHON = Path.home() / "miniconda3/envs/fastavatar/bin/python"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def input_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(root, state, name, script, arguments, required, *, gaussian=False, stage_output=None,
             working_dir=None, pythonpath=(), interpreter=None):
    existing = state["stages"].get(name, {})
    if existing.get("status") == "complete":
        if not all(path.is_file() and path.stat().st_size for path in required):
            raise RuntimeError(f"Previously completed {name} output is missing.")
        return
    interpreter = interpreter or (GAUSSIANS_PYTHON if gaussian else VHAP_PYTHON)
    cwd = working_dir or (GAUSSIANS if gaussian else VHAP)
    paths = [str(Path.home() / "ga_pkgs"), str(cwd)] if gaussian else [str(cwd)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([*paths, *map(str, pythonpath)])}
    if gaussian:
        # The installed Gaussian renderer loads FLAME through chumpy and its user-site six dependency.
        env.pop("PYTHONNOUSERSITE", None)
    else:
        env["PYTHONNOUSERSITE"] = "1"
    command = [str(interpreter), "-u", "-B", str(PROJECT / script), *map(str, arguments)]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    attempt = existing.get("attempt", 0) + 1
    if existing and stage_output is not None and stage_output.exists():
        archive = root / "attempts"
        archive.mkdir(exist_ok=True)
        stage_output.rename(archive / f"{name}-{attempt - 1}")
    log = log_dir / f"{name}-{attempt}.log"
    stage = {"status": "running", "attempt": attempt, "log": str(log),
             "command": command, "cwd": str(cwd), "script_sha256": input_hash(Path(command[3]))}
    state["stages"][name] = stage
    write_json(root / "run.json", state)
    start = monotonic()
    print(f"{name} -> {log}", flush=True)
    try:
        with log.open("x", encoding="utf-8") as stream:
            subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not all(path.is_file() and path.stat().st_size for path in required):
            raise RuntimeError(f"Expected outputs are missing for {name}")
        stage["status"] = "complete"
        stage["outputs_sha256"] = {str(path): input_hash(path) for path in required}
    except BaseException:
        stage["status"] = "failed"
        print(f"{name} failed. See {log}.", flush=True)
        raise
    finally:
        stage["elapsed_s"] = round(monotonic() - start, 2)
        write_json(root / "run.json", state)


def run_pipeline(args, root, state):
    scene = root / "scene"
    selections = root / "selection"
    avatar = root / "avatar"
    output = root / "model"
    source_manifest = scene / "source-frames.json"
    run_step(root, state, "prepare_scene", "prepare_video_scene.py",
             ["--prepare-only", "--video", args.video, "--scene", scene,
              "--frame-count", args.frame_count, "--width", args.width],
             [source_manifest, scene / "orig_undist/sparse/cameras.bin"], stage_output=scene)
    source = json.loads(source_manifest.read_text())
    if source["video_sha256"] != state["configuration"]["video_sha256"]:
        raise RuntimeError("Prepared COLMAP input belongs to a different video.")
    selection = selections / "selection.json"
    observations = selections / "selected-landmarks.npz"
    run_step(root, state, "select_head_frames", "measure_video_frames.py",
             ["--scene", scene, "--output", selections, "--flame", VHAP / "asset/flame",
              "--train-count", args.train_count, "--holdout-count", args.holdout_count],
             [selection, observations], stage_output=selections)
    choice = json.loads(selection.read_text())
    background = scene / "scene_out/bg_body2"
    background_ply = background / f"point_cloud/iteration_{args.background_steps}/point_cloud.ply"
    reserved = [row["name"] for row in source["frames"]
                if any(block["start_s"] <= row["timestamp_s"] <= block["end_s"]
                       for block in choice["holdout_blocks"])]
    run_step(root, state, "train_scene", "prepare_video_scene.py",
             ["--train-only", "--scene", scene, "--background-steps", args.background_steps,
              "--holdout", *reserved],
             [background / "cameras.json", background_ply], stage_output=background)
    avatar_file = avatar / "avatar.json"
    avatar_stage = "train_avatar" if "train_avatar" in state["stages"] else "initialize_head"
    run_step(root, state, avatar_stage, "prepare_video_avatar.py",
             ["--scene", scene, "--selection", selection, "--landmarks", observations,
              "--output", avatar], [avatar_file], stage_output=avatar)
    paths = json.loads(avatar_file.read_text())
    if paths.get("gaussian_steps") != 0:
        raise RuntimeError("The head initializer must come directly from VHAP without Gaussian pretraining.")
    train = choice["train"]
    reference = max(train, key=lambda name: (
        choice["decisions"][name]["quality_score"] - abs(choice["decisions"][name]["yaw_deg"]) / 120,
        name,
    ))
    run_step(root, state, "train_joint", "train_joint.py",
             ["--scene", scene, "--background", background, "--background-ply", background_ply,
              "--bound", paths["bound"], "--tracker", paths["tracker"], "--landmarks", paths["landmarks"],
              "--output", output, "--export-frame", reference, "--holdout", *choice["holdout"],
              "--steps", args.joint_steps, "--warmup", args.joint_warmup,
              "--flame-boundary-mask", GAUSSIANS / "flame_model/assets/flame/FLAME_masks.pkl"],
             [output / name for name in ("point_cloud.ply", "cameras.json", "metrics.json",
                                         "training_state.pt", "selected-frame-preview.mp4")],
             gaussian=True, stage_output=output)
    if args.side_poses:
        side = root / "side"
        run_step(root, state, "fit_side_frames", "fit_side_frames.py",
                 ["--run", root, "--output", side, "--dense"],
                 [side / "side-poses.json"], gaussian=True, stage_output=side)
    if args.side_pose_weights:
        image_side = root / "side-image"
        run_step(root, state, "fit_image_profile_pose", "image_profile_pose.py",
                 ["--run", root, "--baseline", side / "side-poses.json", "--weights", args.side_pose_weights,
                  "--output", image_side], [image_side / "side-poses.json"], gaussian=True, stage_output=image_side)
    state["delivery"] = {"reference_frame": reference,
                         "ply": str(output / "point_cloud.ply"), "cameras": str(output / "cameras.json"),
                         "metrics": str(output / "metrics.json"), "frame_selection": str(selection),
                         "preview": str(output / "selected-frame-preview.mp4"),
                         "head_gaussian_pretraining_steps": paths["gaussian_steps"]}
    if state["stages"].get("fit_side_frames", {}).get("status") == "complete":
        state["delivery"]["side_poses"] = str(root / "side/side-poses.json")
    if state["stages"].get("fit_image_profile_pose", {}).get("status") == "complete":
        state["delivery"]["side_poses"] = str(root / "side-image/side-poses.json")
    write_json(root / "run.json", state)
    print(f"Complete: {output}", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue the identical run from a completed stage.")
    parser.add_argument("--side-poses", action="store_true", help="Fit additional registered views without a face landmark requirement.")
    parser.add_argument("--side-pose-weights", type=Path, help="6DRepNet360 Full Rotation weights for the side RGB fallback.")
    parser.add_argument("--frame-count", type=int, default=80)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument("--background-steps", type=int, default=18000)
    parser.add_argument("--avatar-steps", type=int, help=SUPPRESS, default=None)
    parser.add_argument("--joint-steps", type=int, default=520)
    parser.add_argument("--joint-warmup", type=int, default=104)
    args = parser.parse_args()
    if args.side_pose_weights:
        args.side_poses = True
        args.side_pose_weights = args.side_pose_weights.resolve(strict=True)
    args.video = args.video.resolve(strict=True)
    if not args.video.is_file() or args.video.suffix.lower() != ".mp4":
        parser.error("Supply a readable MP4 video.")
    if args.background_steps < 1 or not 0 <= args.joint_warmup < args.joint_steps:
        parser.error("Choose positive training budgets and 0 <= joint warmup < joint steps.")
    if args.train_count < 6 or args.holdout_count < 2:
        parser.error("Require at least six training frames and two holdout frames.")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output or PROJECT / "runs" / f"{args.video.stem}-video-{tag}").resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    configuration = {key: value for key, value in vars(args).items()
                     if key not in ("output", "resume", "video", "avatar_steps", "side_poses", "side_pose_weights")}
    configuration.update(video=str(args.video), video_sha256=input_hash(args.video))
    if args.side_pose_weights:
        configuration["side_pose_weights_sha256"] = input_hash(args.side_pose_weights)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            state = json.loads((root / "run.json").read_text())
            previous = {k: v for k, v in state["configuration"].items() if k != "avatar_steps"}
            if previous != configuration:
                raise RuntimeError("The input video or settings changed. Use a fresh output directory.")
        else:
            state = {"configuration": configuration, "stages": {}}
            write_json(root / "run.json", state)
        run_pipeline(args, root, state)


if __name__ == "__main__":
    main()
