"""Rebuild the observed-view portrait baseline from source pixels in this run."""

import json
import os
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from run_video import GAUSSIANS, PROJECT, VHAP, input_hash, run_step, write_json


def read(path):
    return json.loads(path.read_bytes())


def artifact(root, report, key):
    path = Path(read(report)[key]).resolve(strict=True)
    path.relative_to(root)
    return path


def run(args, root, state):
    stock = PROJECT / "upstream/VHAP-6e1e770"
    models = Path.home() / "smpl-x/models"
    body_python = Path(os.environ.get("BODY_POSE_PYTHON", str(Path.home() / ".local/share/shuolee-portrait/env-cu128/bin/python")))
    masks = stock / "asset/flame/FLAME_masks.pkl"
    correspondence = PROJECT / "assets/smplx-correspondence/SMPL-X__FLAME_vertex_ids.npy"
    body_model = models / "smplx/SMPLX_NEUTRAL.npz"
    parse = ["--parser-repo", PROJECT/"assets/face-parsing/code", "--weights", PROJECT/"assets/face-parsing/resnet18.pt"]

    def stage(name, script, flags, required, cwd=VHAP, python=None):
        run_step(root, state, name, script, flags, required, working_dir=cwd, gaussian=cwd == GAUSSIANS,
                 interpreter=python, pythonpath=(PROJECT,))

    def prepare(name, sampling):
        scene = root/name
        source = scene/"source-frames.json"
        stage("prepare_"+name, "prepare_video_scene.py",
              ["--prepare-only", "--video", args.video, "--scene", scene, "--width", args.width, *sampling],
              [source, scene/"orig_undist/sparse/cameras.bin", scene/"orig_undist/sparse/images.bin"])
        if read(source)["video_sha256"] != state["configuration"]["video_sha256"] or read(source)["unregistered"]:
            raise ValueError(f"Source or camera coverage failed in {name}")
        directory = root/(name+"-anchors")
        choice, points = directory/"selection.json", directory/"selected-landmarks.npz"
        stage("select_"+name, "measure_video_frames.py",
              ["--scene", scene, "--output", directory, "--flame", VHAP/"asset/flame",
               "--train-count", 16, "--holdout-count", 4], [choice, points])
        return scene, choice, points

    anchor_scene, anchor_choice, anchor_points = prepare("identity", ["--frame-count", 80])
    identity = root/"frontal-head"
    stage("frontal_identity", "prepare_video_avatar.py", ["--scene", anchor_scene, "--selection", anchor_choice,
          "--landmarks", anchor_points, "--output", identity], [identity/"avatar.json"])
    shared = artifact(root, identity/"avatar.json", "tracker")
    scene, choice, points = prepare("scene", ["--sample-fps", args.sample_fps])
    guide = root/"guidance"
    narrow = guide/"vhap-input"
    stage("guide_sequence", "guided_vhap_input.py", ["--scene", scene, "--selection", choice, "--landmarks", points,
          "--weights", PROJECT/"assets/6drepnet360/full-rotation.pth", "--output", guide],
          [narrow/"initialization.json"], GAUSSIANS)
    body = root/"body-observations.npz"
    stage("measure_body", "measure_body_landmarks.py", ["--scene", scene, "--assets", models/"rtmw",
          "--output", body], [body], python=body_python)
    sequence = root/"head-neck"
    init = sequence/"initialization.json"
    stage("frame_head_neck", "prepare_head_neck_frames.py", ["--scene", scene, "--sequence", narrow,
          "--body-observations", body, "--output", sequence], [init, sequence/"landmark2d/STAR.npz"], GAUSSIANS)
    frames = read(init)["frames"]
    names = [row["name"] for row in frames]
    # Evaluation windows belong to the capture. Other videos use their measured view coverage.
    if args.split:
        split = read(args.split)
        if split["video_sha256"] != state["configuration"]["video_sha256"] or split["source_names"] != names:
            raise ValueError("The replay split does not match the input video and its complete frame order")
    else:
        from run_fresh_vhap_scene import build_split
        split = build_split(init, choice)
        split["neck_blocks"] = [f"{names.index(n)}:{names.index(n)}" for n in split["held_frames"]]
    write_json(root/"split.json", split)
    reference, held, blocks = split["reference"], split["held_frames"], split["neck_blocks"]
    if reference not in names or reference in held or len(set(held)) != len(held) or not set(held) <= set(names):
        raise ValueError("Invalid evaluation camera split")
    head = root/"registered-fit"
    baseline = head/"guided-vhap-report.json"
    stage("track_registered", "fit_guided_vhap.py", ["--sequence", sequence, "--shared-checkpoint", shared,
          "--output", head, "--baseline-only", "--epochs", 6], [baseline])
    world = root/"tracked-world.npz"
    stage("project_to_world", "export_vhap_world.py", ["--sequence", sequence, "--report", baseline,
          "--output", world], [world])
    skeleton = root/"anatomical-prior"
    stage("fit_anatomical_prior", "fit_smplx_head_neck.py", ["--initialization", init, "--body-observations", body,
          "--vhap-world", world, "--model", body_model, "--correspondence", correspondence, "--output", skeleton],
          [skeleton/"smplx-frames.npz"], python=body_python)
    archive = root/"registered-crops.zip"
    with ZipFile(archive, "x", compression=ZIP_STORED) as out:
        for path in sorted(sequence.rglob("*")):
            if path.is_file():
                out.write(path, path.relative_to(sequence))
    surface = root/"anatomical-surface"
    prior = surface/"head-neck-frames.npz"
    stage("extract_head_neck", "export_head_neck_frames.py", ["--smplx-frames", skeleton/"smplx-frames.npz",
          "--model", body_model, "--correspondence", correspondence, "--masks", masks, "--frames", archive,
          "--output", surface], [prior], python=body_python)
    outline = root/"outline-targets"
    stage("observe_outline", "prepare_vhap_outline_targets.py", ["--frames", archive, *parse, "--output", outline],
          [outline/"outline-targets.npz"], GAUSSIANS)
    scalp = root/"scalp"
    anatomy = ["--sequence", sequence, "--prior", prior, "--correspondence", correspondence,
               "--outline-targets", outline/"outline-targets.npz"]
    stage("fit_scalp", "fit_vhap_head_neck.py", [*anatomy, "--baseline", baseline, "--epochs", 8,
          "--output", scalp], [scalp/"fit.json", scalp/"shared-scalp-offset.npz"])
    hair = root/"hair"
    stage("joint_head_hair", "refine_vhap_original.py", [*anatomy, "--seed", artifact(root, scalp/"fit.json", "checkpoint"),
          "--scalp-offset", scalp/"shared-scalp-offset.npz", "--output", hair], [hair/"result.json"])
    rigid = root/"vhap-rigid"
    neutral = (["--upper-torso", skeleton/"smplx-frames.npz", "--neutral-body-template", body_model,
                "--flame-correspondence", correspondence, "--flame-masks", masks]
               if args.initial_neck_posture=="template-neutral" else [])
    stage("native_vhap", "run_vhap_cold_head_pose.py", ["--sequence", sequence, "--pose", artifact(root, hair/"result.json", "checkpoint"),
          "--output", rigid, *neutral], [rigid/"result.json"], stock)
    cold = artifact(root, rigid/"result.json", "checkpoint")

    def bind(label, tracker, report):
        dest = root/label
        stage(label, "prepare_registered_head.py", ["--sequence", sequence, "--checkpoint", tracker,
              "--training-result", report, "--reference", Path(reference).stem, "--output", dest],
              [dest/"registered-head.npz", dest/"avatar.json", dest/"bound-head/point_cloud.ply"], stock)
        return dest

    registration = bind("geometry-binding", cold, rigid/"result.json")
    def observe(seq, label):
        dest = root/(label+".json")
        stage(label, "detect_vhap_neck.py", ["--scene", scene, "--sequence", seq, "--initialization",
              seq/"initialization.json", *parse, "--output", dest], [dest], GAUSSIANS)
        return dest
    targets = observe(sequence, "neck-expanded")
    fallback = observe(narrow, "neck-narrow")
    native = root/"native-regions"
    stage("observe_native_skin", "prepare_native_skin_profile.py", ["--scene", scene, "--sequence", sequence,
          *parse, "--output", native, "--device", "cuda"], [native/"source.json"], GAUSSIANS)
    added = root/"front-contours.json"
    stage("observe_front", "supplement_native_neck_front.py", ["--scene", scene, "--initialization", init,
          "--neck-points", targets, "--native-crops", native, "--output", added], [added], GAUSSIANS)
    if args.native_only:
        state["delivery"] = {"checkpoint": str(cold), "registered": str(registration/"registered-head.npz"),
                             "native_grids": str(artifact(root, rigid/"result.json", "image_grid")),
                             "frames": len(names)}
        return
    previous = None
    for label, options in [("neck-pose", []), ("neck-surface", ["--pitch-limit", 8, "--local-mesh"]),
                           ("neck-pitch", ["--pitch-limit", 12, "--frontal-pitch-limit", 3,
                                           "--pitch-anchor", .10, "--freeze-previous-mesh"])]:
        directory = root/label
        report = directory/"neck-joint-fit.json"
        if previous:
            options += ["--supplement", added, "--prior-report", previous,
                        "--prior-checkpoint", artifact(root, previous, "checkpoint"), "--holdout-blocks", *blocks]
        stage(label, "fit_vhap_neck_sequence.py", ["--scene", scene, "--sequence", sequence,
              "--checkpoint", cold, "--registered", registration/"registered-head.npz", "--targets", targets,
              "--native-fallback", fallback, "--flame-masks", masks, *options, "--output", directory], [report], stock)
        previous = report
    geometry = artifact(root, previous, "checkpoint")
    if args.monocular_neck_only:
        state["delivery"] = {"checkpoint": str(geometry), "neck_report": str(previous),
                             "native_grids": str(artifact(root, previous, "image_grid")),
                             "pose_status": read(previous)["status"], "frames": len(names)}
        return
    support = root/"color-support.npz"
    stage("observe_color", "prepare_vhap_color_support.py", ["--frames", archive, *parse, "--output", support],
          [support], GAUSSIANS)
    appearance = ["--sequence", sequence, "--geometry", geometry, "--pose-result", previous, "--support", support]
    base_rgb, rear_rgb = root/"visible-rgb", root/"rear-rgb"
    stage("fit_visible_rgb", "train_vhap_appearance_from_zero.py", [*appearance, "--output", base_rgb],
          [base_rgb/"result.json"], stock)
    stage("fit_rear_rgb", "train_vhap_appearance_from_zero.py", [*appearance, "--refine-result", base_rgb/"result.json",
          "--output", rear_rgb], [rear_rgb/"result.json"], stock)
    tracker = artifact(root, rear_rgb/"result.json", "checkpoint")
    grid = artifact(root, rear_rgb/"result.json", "image_grid")
    if set(p.name for p in grid.glob("*.jpg")) != {f"frame_{i:05d}.jpg" for i in range(len(names))}:
        raise RuntimeError("Incomplete native VHAP grids")
    head_seed = bind("registered-head", tracker, rear_rgb/"result.json")
    if args.torso_neck_grid:
        torso = root/"torso-texture"
        stage("track_torso_texture", "track_garment_frame.py",
              ["--run", root, "--output", torso], [torso/"poses.npz", torso/"tracks.json"], stock)
        dest = root/"torso-neck"
        stage("fit_shared_torso_neck", "fit_torso_neck.py",
              ["--run", root, "--output", dest, "--flame-masks", masks],
              [dest/"result.json", dest/"tracked_flame_params.npz", dest/"grid/both-sides.png"], stock)
        state["delivery"] = {"native_baseline_grids": str(grid), "torso_neck": str(dest/"result.json"),
                             "comparison": str(dest/"grid/both-sides.png"), "reference": reference,
                             "frames": len(names)}
        return
    background = root/"scene-initial"
    stage("scene_seed", "initialize_msv2_scene.py", ["--scene", scene, "--output", background, "--holdout", *held],
          [background/"initialization.json", background/"point_cloud.ply"], GAUSSIANS)
    model = root/"gaussians"
    stage("gaussian_training", "train_msv2_vhap.py", ["--scene", scene, "--background", background,
          "--bound", head_seed/"bound-head/point_cloud.ply", "--tracker", tracker, "--landmarks", init,
          "--registered-vhap", head_seed/"registered-head.npz", "--vendor", Path.home()/"msv2/vendor/mini-splatting2",
          "--flame-mask", masks, "--output", model, "--export-frame", reference, "--holdout", *held,
          "--steps", args.steps, "--mesh-offset", "--freeze-mesh-offset"],
          [model/"metrics.json", model/"cameras.json", model/"point_cloud.ply"], GAUSSIANS)
    stats = read(model/"metrics.json")
    if stats["resumed_step"] != 0 or stats["steps"] != args.steps or stats["scene_initialization"]["trained_steps"] != 0:
        raise RuntimeError("Gaussian initialization or training budget differs from the requested run")
    with ZipFile(root/"point-cloud-and-cameras.zip", "x", compression=ZIP_STORED) as out:
        for name in ("point_cloud.ply", "cameras.json"):
            out.write(model/name, name)
    with ZipFile(root/"point-cloud-and-cameras.zip") as out:
        if out.testzip() or len(json.loads(out.read("cameras.json"))) != len(names):
            raise RuntimeError("Exported camera count or ZIP integrity mismatch")
    state["delivery"] = {"grids": str(grid), "neck_report": str(previous), "metrics": str(model/"metrics.json"),
                         "package": str(root/"point-cloud-and-cameras.zip"), "reference": reference,
                         "steps": stats["steps"], "frames": len(names)}
