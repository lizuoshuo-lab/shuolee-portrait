"""Train a fresh MSv2 scene on the output of run_target_profile.sh."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from run_video import PROJECT, input_hash, run_step, write_json


def run(source, output, steps, freeze_mesh_offset=False, surface_samples=1, prepared_seeds=None, fit_footprints=False,
        no_head_scale_growth=False, late_prune=False):
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
                 reference=reference, rgb_holdout=held, steps=steps, freeze_mesh_offset=freeze_mesh_offset,
                 surface_samples=surface_samples, fit_footprints=fit_footprints, no_head_scale_growth=no_head_scale_growth, stages={})
    if late_prune:
        state["late_prune"] = True
    write_json(output / "run.json",state)
    native = PROJECT / "upstream/VHAP-6e1e770"
    scene = source / "scene"
    head = output / "registered-head"
    background = output / "scene-initial"
    joint = output / "joint"
    try:
        if prepared_seeds is None:
            run_step(output,state,"head_seed","prepare_registered_head.py",
                     ["--sequence",source/"head-neck","--checkpoint",checkpoint,"--training-result",source/"vhap-rigid/result.json",
                      "--pose-result",report,"--reference",Path(reference).stem,"--output",head],
                     [head/"avatar.json",head/"registered-head.npz",head/"bound-head/point_cloud.ply"],working_dir=native)
            run_step(output,state,"scene_seed","initialize_msv2_scene.py",
                     ["--scene",scene,"--output",background,"--holdout",*held],
                     [background/"point_cloud.ply",background/"initialization.json"],gaussian=True)
        else:
            control=json.loads((prepared_seeds/"run.json").read_bytes())
            initialization=json.loads((prepared_seeds/"scene-initial/initialization.json").read_bytes())
            if (Path(control["source_run"]).resolve()!=source or control["target_sha256"]!=state["target_sha256"]
                    or initialization["trained_steps"]!=0 or sorted(initialization["holdout"])!=sorted(held)):
                raise ValueError("Paired controls require identical untrained scene and registered head inputs")
            for directory in (head,background):
                directory.symlink_to((prepared_seeds/directory.name).resolve(strict=True),target_is_directory=True)
            state["untrained_inputs_from"]=str(prepared_seeds)
            write_json(output/"run.json",state)
        run_step(output,state,"gaussian_training","train_msv2_vhap.py",
                 ["--scene",scene,"--background",background,"--tracker",checkpoint,"--bound",head/"bound-head/point_cloud.ply",
                  "--landmarks",source/"head-neck/initialization.json","--registered-vhap",head/"registered-head.npz",
                  "--vendor",Path.home()/"msv2/vendor/mini-splatting2","--flame-mask",native/"asset/flame/FLAME_masks.pkl",
                  "--output",joint,"--export-frame",reference,"--holdout",*held,"--steps",steps,"--mesh-offset",
                  "--surface-samples",surface_samples,
                  *(["--fit-footprints"] if fit_footprints else []),
                  *(["--no-head-scale-growth"] if no_head_scale_growth else []),
                  *(["--late-prune"] if late_prune else []),
                  *(["--freeze-mesh-offset"] if freeze_mesh_offset else [])],
                 [joint/"metrics.json",joint/"cameras.json",joint/"point_cloud.ply"],gaussian=True)
        stats = json.loads((joint/"metrics.json").read_bytes())
        if (stats["resumed_step"] != 0 or stats["steps"] != steps or stats["reference"] != reference
                or stats["source_signature"]["tracker_sha256"] != state["target_sha256"]
                or stats["scene_initialization"]["trained_steps"] != 0):
            raise ValueError("Trained model does not match the requested geometry or fresh Gaussian initialization")
        ref_mesh=stats["mesh_offset"]["frames"][reference]
        shifts=[v["vertex_displacement_world"]["max"] for v in ref_mesh["regions"].values()
                if v["editable_vertices"]]
        if (not shifts or not all(math.isfinite(v) for v in shifts)
                or (max(shifts)>0)==freeze_mesh_offset or ref_mesh["frozen_vertex_max_displacement_world"]!=0):
            raise ValueError("Exported vertex displacements disagree with the requested trainable/protected regions")
        state["measured_max_vertex_shift_world"]=max(shifts)
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
    p.add_argument("--freeze-mesh-offset",action="store_true",help="Reproduce the fixed-mesh control.")
    p.add_argument("--surface-samples",type=int,choices=(1,4),default=1)
    p.add_argument("--fit-footprints",action="store_true")
    p.add_argument("--no-head-scale-growth",action="store_true")
    p.add_argument("--late-prune",action="store_true")
    p.add_argument("--prepared-seeds",type=Path,help="Use only the zero-step inputs of a paired control.")
    args=p.parse_args()
    run(args.source_run.resolve(strict=True),args.new_output.resolve(),args.steps,args.freeze_mesh_offset,
        args.surface_samples,args.prepared_seeds.resolve(strict=True) if args.prepared_seeds else None,
        args.fit_footprints,args.no_head_scale_growth,args.late_prune)
