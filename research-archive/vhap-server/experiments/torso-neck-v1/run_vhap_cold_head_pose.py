"""Run native VHAP cold-start tracking with registered rigid head seeds."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.lbs import batch_rodrigues
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from prepare_video_avatar import setup_vhap_cameras


def calibrated_seeds(context, locations, virtual_to_tracker):
    """Return neutral FLAME rigid seeds, an observed up direction, and profile confidence."""
    records = context["frames"]
    conditioning = context["conditioning"]
    names = {record["name"] for record in records}
    anchors = set(conditioning["frontal_anchors"])
    if (len(names) != len(records) or not anchors <= names or len(anchors) < 3
            or not all(r["has_landmarks"] for r in records if r["name"] in anchors)):
        raise ValueError("The calibrated input requires real frontal landmark anchors")
    base = virtual_to_tracker @ np.asarray([r["baseline"]["rotation"] for r in records])
    prior_up = base[:, :, 1]
    yaw = np.abs([r["baseline_yaw_deg"] for r in records])
    profile = np.clip((yaw - 35.) / 30., 0., 1.)
    profile = profile ** 2 * (3. - 2. * profile)
    profile *= [not r["has_landmarks"] for r in records]

    trusted = conditioning["calibration_passed"]
    use_heading = np.asarray([trusted and r["guided_from_network"] for r in records])
    guided = virtual_to_tracker @ np.asarray([r["guided"]["rotation"] for r in records])
    forward = base[:, :, 2] + (profile * use_heading)[:, None] * (guided[:, :, 2] - base[:, :, 2])
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.cross(prior_up, forward)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    pose = np.stack((right, np.cross(forward, right), forward), axis=-1)
    translation = np.asarray([r["baseline"]["translation"] for r in records])
    root = np.asarray(context["root_flame_units"])
    root_position = locations + (virtual_to_tracker @ translation[..., None])[..., 0] + (base @ root)
    return {"rotation": Rotation.from_matrix(pose).as_rotvec(), "translation": root_position - root}, prior_up, profile


def add_cross_view_loss(tracker, seeds, prior_up, optical_axis, profile):
    """Constrain the weakly observed roll and optical-axis depth in the common 3D gauge."""
    native_energy = tracker.compute_energy
    prior_up, optical_axis, profile = (
        torch.as_tensor(value, device=tracker.rotation.device, dtype=tracker.rotation.dtype)
        for value in (prior_up, optical_axis, profile)
    )

    def energy(sample, step_i=None, stage=None):
        total, log, *rendering = native_energy(sample, step_i=step_i, stage=stage)
        if stage is not None and "pose" in tracker.cfg.pipeline[stage].optimizable_params:
            index = sample["timestep_index"]
            posed = batch_rodrigues(tracker.rotation[index]) @ batch_rodrigues(tracker.neck_pose[index])
            up = prior_up[index]
            along_forward = (posed[:, :, 2] * up).sum(dim=-1)
            roll_sine = (posed[:, :, 0] * up).sum(dim=-1)
            roll = roll_sine.square() / (1. - along_forward.square()).clamp_min(1e-4)
            depth = ((tracker.translation[index] - seeds["translation"][index]) * optical_axis[index]).sum(dim=-1)
            log["cross_view_roll"] = 5. * (profile[index] * roll).mean()
            log["cross_view_depth"] = 10. * (profile[index] * depth.square()).mean()
            total = total + log["cross_view_roll"] + log["cross_view_depth"]
            log["total"] = total
        return total, log, *rendering

    tracker.compute_energy = energy


def add_neck_observations(tracker, args, records):
    from fit_vhap_neck_sequence import load_observations
    from vhap_neck_loss import configure

    observations = {}
    for index, side, x, y, split, _, _ in load_observations(args, records):
        warp = np.asarray(records[index]["warp"])
        xy = cv2.perspectiveTransform(np.array([[[x, y]]]), warp)[0, 0]
        observations.setdefault(index, []).append(
            {"xy": xy.tolist(), "side": "left" if side < 0 else "right", "split": split}
        )
    loss = configure(tracker, observations, weight=.6)
    loss.enabled = True
    return loss


def run(args):
    context = json.loads((args.sequence / "initialization.json").read_text())
    records = context["frames"]
    if [frame["timestep"] for frame in records] != list(range(len(records))):
        raise ValueError("Source frame order differs from the registered camera sequence")
    locations, virtual_to_tracker, extrinsics, _, _ = camera_gauge(records, context["scene_scale"])
    setup_vhap_cameras(extrinsics)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent),
        "--data.sequence", args.sequence.name,
        "--data.calibrated",
        "--data.use-alpha-map",
        "--exp.output-folder", str(args.output / "track"),
        "--no-async-func",
    ])
    torch.manual_seed(20261117)
    np.random.seed(20261117)
    tracker = GlobalTracker(cfg)
    if tracker.n_timesteps != len(records):
        raise ValueError("Native VHAP did not load every registered frame")
    if not (cfg.model.tex_painted and cfg.model.tex_extra and cfg.model.residual_tex):
        raise ValueError("Expected the original painted base and residual texture model")
    if torch.count_nonzero(tracker.tex_extra).item() != 0:
        raise ValueError("Color residual was already nonzero before training")

    if args.calibrated_start:
        source, prior_up, profile = calibrated_seeds(context, locations, virtual_to_tracker)
    else:
        with np.load(args.pose, allow_pickle=False) as archive:
            if (archive["timestep_id"].tolist() != list(tracker.dataset.timestep_ids)
                    or int(archive["n_processed_frames"]) != len(records)):
                raise ValueError("Rigid pose seeds and dataset frame order differ")
            source = {key: archive[key] for key in ("rotation", "translation")}

    seeds = {}
    for key, array in source.items():
        target = getattr(tracker, key)
        values = torch.from_numpy(array).to(target)
        if values.shape != target.shape or not torch.isfinite(values).all():
            raise ValueError(f"Invalid rigid pose seed for {key}")
        seeds[key] = values

    neck = add_neck_observations(tracker, args, records) if args.targets else None
    if args.calibrated_start:
        add_cross_view_loss(tracker, seeds, prior_up, virtual_to_tracker[:, :, 2], profile)

    seeded_frames = set()
    native_stage = tracker.optimize_stage

    def stage_with_rigid_seed(stage, sample=None, dataloader=None, lr_scale=1.):
        if sample is not None and stage in ("lmk_init_rigid", "rgb_sequential_tracking"):
            indices = sample["timestep_index"].tolist()
            new_indices = [index for index in indices if index not in seeded_frames]
            if new_indices:
                if new_indices != indices:
                    raise RuntimeError("Unexpected partial overlap in a native tracking batch")
                with torch.no_grad():
                    for key, values in seeds.items():
                        getattr(tracker, key)[new_indices] = values[new_indices]
                seeded_frames.update(new_indices)
                print(f"Rigid head seed at {stage}: {indices[0]}–{indices[-1]}", flush=True)
        return native_stage(stage, sample=sample, dataloader=dataloader, lr_scale=lr_scale)

    tracker.optimize_stage = stage_with_rigid_seed
    initial = {
        "pose_source": str(args.pose or args.sequence / "initialization.json"), "injected_fields": list(seeds),
        "frame_names": [frame["name"] for frame in records],
        "color_init": "native painted mean with zero texture residual and native lighting initialization",
        "initial_texture_residual_nonzero": 0, "batch_size": cfg.batch_size,
        "native_global_epochs": cfg.pipeline.rgb_global_tracking.num_epochs,
        "output": str(tracker.out_dir),
    }
    if args.calibrated_start:
        initial.update(profile_prior_frames=int(np.count_nonzero(profile)),
                       roll_weight=5., depth_weight=10.,
                       shape_init_nonzero=int(torch.count_nonzero(tracker.shape)),
                       mesh_offset_init_nonzero=int(torch.count_nonzero(tracker.static_offset)),
                       neck_pose_init_nonzero=int(torch.count_nonzero(tracker.neck_pose)))
        if any(initial[key] for key in ("shape_init_nonzero", "mesh_offset_init_nonzero", "neck_pose_init_nonzero")):
            raise ValueError("The calibrated cold start requires neutral shape, offset, and neck")
    (args.output / "initialization-proof.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(f"Starting native VHAP cold start on {len(records)} frames with batch size {cfg.batch_size}", flush=True)
    tracker.optimize()

    epochs = cfg.pipeline.rgb_global_tracking.num_epochs
    checkpoint = tracker.out_dir / f"tracked_flame_params_{epochs}.npz"
    grid = tracker.out_dir / f"eval_{epochs}" / "image_grid"
    expected = {f"frame_{index:05d}.jpg" for index in range(len(records))}
    if (seeded_frames != set(range(len(records))) or not checkpoint.is_file()
            or {frame.name for frame in grid.glob("*.jpg")} != expected or not torch.isfinite(tracker.tex_extra).all()):
        raise RuntimeError("The final native checkpoint or original image_grid is incomplete")
    tracker.tb_writer.close()
    result = dict(initial, trained_steps=tracker.global_step, seeded_count=len(seeded_frames),
                  checkpoint=str(checkpoint), image_grid=str(grid))
    if neck is not None:
        result["neck_training_point_uses"] = {
            records[index]["name"]: count for index, count in sorted(neck.training_counts.items())
        }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed {tracker.global_step} native VHAP updates; {len(expected)} original grids: {grid}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pose", type=Path, help="Registered rigid seed archive")
    inputs.add_argument("--calibrated-start", action="store_true", help="Estimate fresh rigid seeds from calibrated observations")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--targets", type=Path, help="Native neck contour observations")
    parser.add_argument("--native-fallback", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any((args.scene, args.targets, args.native_fallback, args.supplement)) and not all(
        (args.scene, args.targets, args.native_fallback)
    ):
        parser.error("Neck observations require --scene, --targets and --native-fallback")
    run(args)
