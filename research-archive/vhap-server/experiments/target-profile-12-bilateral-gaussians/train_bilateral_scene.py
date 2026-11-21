"""Train a fresh MSv2 scene on the output of run_target_profile.sh."""

from argparse import ArgumentParser
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from run_video import GAUSSIANS, PROJECT, input_hash, run_step, write_json


def run(source, output, steps):
    report = source / "target-profile-bilateral/result.json"
    pose = json.loads(report.read_bytes())
    split = json.loads((source / "split.json").read_bytes())
    checkpoint = Path(pose["checkpoint"]).resolve(strict=True)
    checkpoint.relative_to(source)
    reference, held = split["reference"], split["held_frames"]
    if (reference != pose["reference"] or reference in held or steps < 1
            or input_hash(checkpoint) != pose["checkpoint_sha256"]
            or not pose.get("requested_both_sides") or pose["status"] != "completed"):
        raise ValueError("Target geometry, export camera or split mismatch")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", source_run=str(source), target_checkpoint=str(checkpoint),
                 target_sha256=pose["checkpoint_sha256"], source_pose_accepted=pose["accepted"],
                 reference=reference, rgb_holdout=held, steps=steps, stages={})
    write_json(output / "run.json",state)
    native = PROJECT / "upstream/VHAP-6e1e770"
    scene = source / "scene"
    head = output / "registered-head"
    background = output / "scene-initial"
    joint = output / "joint"
    try:
        run_step(output,state,"head_seed","prepare_registered_head.py",
                 ["--sequence",source/"head-neck","--checkpoint",checkpoint,"--training-result",source/"vhap-rigid/result.json",
                  "--pose-result",report,"--reference",Path(reference).stem,"--output",head],
                 [head/"avatar.json",head/"registered-head.npz",head/"bound-head/point_cloud.ply"],working_dir=native)
        run_step(output,state,"scene_seed","initialize_msv2_scene.py",
                 ["--scene",scene,"--output",background,"--holdout",*held],
                 [background/"point_cloud.ply",background/"initialization.json"],gaussian=True)
        run_step(output,state,"gaussian_training","train_msv2_vhap.py",
                 ["--scene",scene,"--background",background,"--tracker",checkpoint,"--bound",head/"bound-head/point_cloud.ply",
                  "--landmarks",source/"head-neck/initialization.json","--registered-vhap",head/"registered-head.npz",
                  "--vendor",Path.home()/"msv2/vendor/mini-splatting2","--flame-mask",native/"asset/flame/FLAME_masks.pkl",
                  "--output",joint,"--export-frame",reference,"--holdout",*held,"--steps",steps,"--mesh-offset","--freeze-mesh-offset"],
                 [joint/"metrics.json",joint/"cameras.json",joint/"point_cloud.ply"],gaussian=True)
        stats = json.loads((joint/"metrics.json").read_bytes())
        if (stats["resumed_step"] != 0 or stats["steps"] != steps or stats["reference"] != reference
                or stats["source_signature"]["tracker_sha256"] != state["target_sha256"]
                or stats["scene_initialization"]["trained_steps"] != 0):
            raise ValueError("Trained model does not match the requested geometry or fresh Gaussian initialization")
        package = output / "point-cloud-and-cameras.zip"
        with ZipFile(package,"x",compression=ZIP_STORED) as z:
            for name in ("point_cloud.ply","cameras.json"):
                z.write(joint/name,name)
        with ZipFile(package) as z:
            if z.testzip() or len(json.loads(z.read("cameras.json"))) != len(split["source_names"]):
                raise ValueError("Inconsistent point cloud and camera package")
        state.update(status="complete",package=str(package))
    except BaseException:
        state["status"]="failed"
        raise
    finally:
        write_json(output / "run.json",state)


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("source_run",type=Path)
    p.add_argument("new_output",type=Path)
    p.add_argument("--steps",type=int,default=10000)
    args=p.parse_args()
    run(args.source_run.resolve(strict=True),args.new_output.resolve(),args.steps)
