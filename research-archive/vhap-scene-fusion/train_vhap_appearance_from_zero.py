"""Fit native VHAP albedo and light from their defaults on a verified fixed mesh."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from fit_vhap_head_neck import STATE_KEYS
from prepare_registered_head import check_checkpoint_frames
from prepare_video_avatar import setup_vhap_cameras


def use_visible_color_support(tracker, args, names, records):
    from refit_vhap_texture import read_observations

    raw, matte, hair, skin = read_observations(args, names)
    if not hair.any() or not skin.any():
        raise ValueError("Expected registered hair and skin color support")
    posterior = torch.zeros_like(hair)
    if args.refine_result is not None:
        for i, frame in enumerate(records):
            yaw = frame["baseline_yaw_deg"]
            rows, columns = torch.where(hair[i])
            if abs(yaw) < 60 or len(rows) < 96:
                continue
            fraction = .45 if yaw < 0 else .55
            edge = torch.quantile(columns.float(), fraction)
            keep = columns <= edge if yaw < 0 else columns >= edge
            posterior[i, rows[keep], columns[keep]] = True

    def photometric(sample, verts, faces, albedos, rast_dict, step_i=None, stage=None):
        index = sample["timestep_index"].cpu()
        target = raw[index].to(verts) / 255.
        foreground, h, s = (labels[index].to(verts.device) for labels in (matte, hair, skin))
        background = tracker.get_background_color(target, sample["alpha_map"], stage)
        out = tracker.render_rgba(rast_dict, verts, faces, albedos, tracker.lights[None],
                                  background_color=background, enable_disturbance=False)
        rgb, alpha = out["rgba"][:, :3], out["rgba"][:, 3:]
        interior = foreground & (alpha[:, 0].detach() >= .995)
        view = sample["extrinsic"][:, 2, :3].to(verts)
        cosine = (out["normal"].detach() * view[:, :, None, None]).sum(1).abs().clamp(0, 1)
        weight = interior * (.20 + .80 * cosine) * (1. + 2. * h + .5 * s)
        area = weight.sum(dim=(1, 2))
        if (interior.sum(dim=(1, 2)) < 256).any():
            raise RuntimeError("No robust RGB observations on a registered VHAP frame")
        error = F.smooth_l1_loss(rgb, target, beta=.025, reduction="none").mean(dim=1)
        loss = ((error * weight).sum(dim=(1, 2)) / area).mean()
        if args.refine_result is not None:
            rear = posterior[index].to(verts.device) & interior
            count = rear.sum(dim=(1, 2))
            useful = count >= 96
            loss = loss + 6. * (((error * rear).sum(dim=(1, 2)) / count.clamp_min(1)) * useful).sum() / useful.sum().clamp_min(1)
        out.update(gt_rgb=target, pred_rgb=rgb, error_rgb=target - rgb, pred_alpha=alpha)
        return loss, out

    tracker.compute_photometric_energy = photometric


def run(args):
    init = (args.sequence / "initialization.json").read_bytes()
    context = json.loads(init)
    records = context["frames"]
    frame_names = [row["name"] for row in records]
    if [row["timestep"] for row in records] != list(range(len(records))):
        raise ValueError("Expected the exact registered frame order")
    setup_vhap_cameras(camera_gauge(records, context["scene_scale"])[2])
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "track"),
        "--batch-size", "4", "--log.interval-media", "1000000", "--log.interval-scalar", "50",
        "--no-async-func",
    ])
    stage = cfg.pipeline.rgb_global_tracking
    stage.optimizable_params = ("texture", "lights") if args.refine_result is None else ("texture",)
    if args.refine_result is not None:
        stage.num_epochs = 10
    torch.manual_seed(20261117)
    np.random.seed(20261117)
    tracker = GlobalTracker(cfg)
    if (tracker.n_timesteps != len(records) or cfg.model.flame_params_path is not None
            or not (cfg.model.tex_painted and cfg.model.tex_extra and cfg.model.residual_tex)
            or torch.count_nonzero(tracker.tex_extra).item()
            or not torch.equal(tracker.lights, tracker.lights_uniform)):
        raise RuntimeError("Expected the untouched VHAP painted prior, zero UV residual and uniform lighting")

    geometry = tuple(k for k in STATE_KEYS if k not in ("tex_extra", "lights"))
    with np.load(args.geometry, allow_pickle=False) as state, torch.no_grad():
        check_checkpoint_frames(state, list(tracker.dataset.timestep_ids), frame_names,
                                args.geometry, None, args.pose_result, init)
        for key in geometry:
            tensor = getattr(tracker, key)
            fixed = torch.from_numpy(state[key]).to(tensor)
            if fixed.shape != tensor.shape or not torch.isfinite(fixed).all():
                raise ValueError(f"Invalid accepted geometry field {key}")
            tensor.copy_(fixed)
            tensor.requires_grad_(False)
    inherited = None
    if args.refine_result is not None:
        inherited = json.loads(args.refine_result.read_bytes())
        if (inherited["geometry_sha256"] != sha256(args.geometry.read_bytes()).hexdigest()
                or inherited["initialization_sha256"] != sha256(init).hexdigest()
                or inherited["frame_names"] != frame_names or inherited["initial_uv_residual_nonzero"] != 0
                or inherited["visible_color_support_sha256"] != sha256(args.support.read_bytes()).hexdigest()):
            raise ValueError("Rear color refinement must continue the verified cold RGB fit")
        with np.load(inherited["checkpoint"], allow_pickle=False) as prior, torch.no_grad():
            for name in ("tex_extra", "lights"):
                torch_value = getattr(tracker, name)
                new = torch.from_numpy(prior[name]).to(torch_value)
                if new.shape != torch_value.shape or not torch.isfinite(new).all():
                    raise ValueError(f"Invalid cold appearance parameter {name}")
                torch_value.copy_(new)
        tracker.lights.requires_grad_(False)
    lights_start = tracker.lights.detach().clone()
    if args.support is not None:
        use_visible_color_support(tracker, args, frame_names, records)
    receipt = {
        "geometry_source": str(args.geometry), "geometry_sha256": sha256(args.geometry.read_bytes()).hexdigest(),
        "pose_result": str(args.pose_result), "initialization_sha256": sha256(init).hexdigest(),
        "frame_names": frame_names, "frozen_geometry_fields": list(geometry), "epochs": stage.num_epochs,
        "batch_size": cfg.batch_size, "lr_scale": 1.0,
        "initial_uv_residual_nonzero": int(torch.count_nonzero(tracker.tex_extra)),
        "initial_lights": lights_start.cpu().tolist(),
        "appearance_initialization": "stock painted mean and uniform SH" if inherited is None else "fresh visible RGB fit",
        "zero_start_result": str(args.refine_result) if inherited is not None else None,
        "visible_color_support_sha256": sha256(args.support.read_bytes()).hexdigest() if args.support is not None else None,
    }
    (args.output / "initialization-proof.json").write_text(json.dumps(receipt, indent=2) + "\n")
    visits = np.zeros(len(frame_names), dtype=int)
    tracker.global_step = 0
    native_iteration = tracker.optimize_iter

    def observe(sample, optimizer, stage_name):
        native_iteration(sample, optimizer, stage_name)
        visits[sample["timestep_index"].tolist()] += 1

    tracker.optimize_iter = observe
    loader = DataLoader(tracker.dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                        generator=torch.Generator().manual_seed(20261117))
    print(f"Native VHAP appearance: {len(frame_names)} frames, {stage.num_epochs} epochs, "
          f"{receipt['appearance_initialization']}", flush=True)
    tracker.optimize_stage("rgb_global_tracking", dataloader=loader, lr_scale=1.)
    with np.load(args.geometry, allow_pickle=False) as source:
        if any(not np.array_equal(getattr(tracker, key).detach().cpu().numpy(), source[key]) for key in geometry):
            raise RuntimeError("The accepted 12-degree geometry changed during appearance fitting")
    grid = tracker.out_dir / f"eval_{stage.num_epochs}/image_grid"
    if (not (visits == stage.num_epochs).all() or tracker.global_step != len(loader) * stage.num_epochs
            or not torch.isfinite(tracker.tex_extra).all() or not torch.isfinite(tracker.lights).all()
            or not torch.count_nonzero(tracker.tex_extra)
            or torch.equal(tracker.lights, lights_start) != (inherited is not None)
            or {path.name for path in grid.glob("*.jpg")} != {f"frame_{i:05d}.jpg" for i in range(len(records))}):
        raise RuntimeError("Fresh appearance fit has incomplete updates or native rendering")
    tracker.timestep = len(frame_names)
    tracker.save_result(epoch=stage.num_epochs)
    checkpoint = tracker.out_dir / f"tracked_flame_params_{stage.num_epochs}.npz"
    tracker.tb_writer.close()
    receipt.update(trained_steps=tracker.global_step, seeded_count=len(frame_names), visits=visits.tolist(),
                   checkpoint=str(checkpoint), image_grid=str(grid), frozen_geometry_verified=True)
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Finished fresh VHAP RGB fit: {tracker.global_step} updates, {len(frame_names)} grids", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "geometry", "pose-result", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--refine-result", type=Path, help="Emphasize rear hair using the completed zero-start fit.")
    args = parser.parse_args()
    if args.refine_result is not None and args.support is None:
        parser.error("Rear color refinement requires the same verified support.")
    args.geometry = args.geometry.resolve(strict=True)
    run(args)
