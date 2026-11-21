"""Train an untrained COLMAP/SMPL-X scene and a registered VHAP head in one rasterizer."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from utils.loss_utils import ssim

from fuse_scene import check_principal_point, render
from review_layer_visibility import review_layers
from train_joint import (HeadParameters, combine, evaluate, export, prepare_frames,
                         rgb_error, visible_head)


def scene_tensors(model):
    features = model.get_features
    count = (model.active_sh_degree + 1) ** 2
    if count < features.shape[1]:
        features = torch.cat((features[:, :count], torch.zeros_like(features[:, count:])), dim=1)
    return model.get_xyz, features, model.get_opacity, model.get_scaling, model.get_rotation


def joint_step(frame, model, head_parameters, measure_visibility):
    scene = scene_tensors(model)
    head = head_parameters.head(frame, geometry=True)
    xyz, features, opacity, scales, rotations = combine(scene, head)
    projected = torch.zeros_like(scene[0], requires_grad=True)
    means2d = torch.cat((projected, torch.zeros_like(head[0])))
    rgb, radii = frame.rasterizer(
        means3D=xyz, means2D=means2d, shs=features, colors_precomp=None,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    # Balance the small head region against the full-frame signal.
    sy, sx = frame.crop
    reference = frame.reference
    inside = frame.masks["head_rgb"]
    rgb_loss = (0.75 * (rgb - reference).abs().mean() + 0.25 * rgb_error(rgb, reference, inside))
    crop = torch.where(inside[sy, sx], rgb[:, sy, sx], reference[:, sy, sx])
    loss = 0.8 * rgb_loss + 0.2 * (1 - ssim(crop, reference[:, sy, sx])) + head_parameters.prior()
    if measure_visibility:
        # In the observed head matte, scene Gaussians must not absorb the
        # contribution of already opaque head surfaces. Keep the head fixed
        # for this loss so it cannot win by moving forward or inflating.
        fixed_head = tuple(value.detach() for value in head)
        with torch.no_grad():
            alone = render(frame.rasterizer, fixed_head, alpha_only=True)
            support = inside & (alone >= 0.8)
        visible = visible_head(frame.rasterizer, combine(scene, fixed_head), len(scene[0]))
        occlusion = ((alone - visible).relu() * support).sum() / support.sum().clamp_min(1)
        loss = loss + 0.1 * occlusion
    loss.backward()
    return loss.detach(), radii[:len(scene[0])], projected


def scene_optimizer_config(steps):
    # Vanilla 3DGS world-space learning rates. The registered head uses local
    # triangle offsets instead of these world-space values.
    return Namespace(percent_dense=0.01, position_lr_init=0.00016,
                     position_lr_final=0.0000016, position_lr_delay_mult=0.01,
                     position_lr_max_steps=steps, feature_lr=0.0025,
                     opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001)


def save_checkpoint(path, step, model, head, head_optimizer, run_signature, losses):
    temporary = path.with_suffix(".partial")
    torch.save({"step": step, "scene": model.capture(), "head": head.state_dict(),
                "head_optimizer": head_optimizer.state_dict(), "run_signature": run_signature,
                "losses": losses, "cuda_rng": torch.cuda.get_rng_state()}, temporary)
    temporary.replace(path)


def run(args):
    initial = json.loads((args.background / "initialization.json").read_text())
    extent = float(initial["scene_extent"])
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid initialization scene extent.")
    if sorted(initial["holdout"]) != sorted(args.holdout):
        raise ValueError("Scene initialization and training must use the same RGB holdout.")
    if not args.resume and args.output.exists():
        raise FileExistsError(args.output)
    if args.resume and (args.output / "metrics.json").exists():
        raise FileExistsError("This run already has its completed export and metrics.")
    args.output.mkdir(parents=True, exist_ok=True)
    args.background_ply = args.background / "point_cloud.ply"
    with torch.no_grad():
        principal = check_principal_point()
        K, size, scale, seed_path, source_scene, frames = prepare_frames(args)
    train = [frame for frame in frames if frame.name not in args.holdout]
    train_names = [frame.name for frame in train]
    print(f"{len(train)} training frames, {len(args.holdout)} RGB holdouts, "
          f"{len(source_scene[0])} untrained scene Gaussians, {len(train[0].head[0])} bound head Gaussians",
          flush=True)
    model = GaussianModel(3)
    model.load_ply(str(seed_path))
    model.active_sh_degree = 0
    model.spatial_lr_scale = extent
    model.max_radii2D = torch.zeros(len(model.get_xyz), device="cuda")
    config = scene_optimizer_config(args.steps)
    head = HeadParameters(train[0].head)
    head_optimizer = torch.optim.Adam([
        {"params": [head.head_dc], "lr": 0.025},
        {"params": [head.head_offset, head.head_log_scale, head.head_opacity], "lr": 0.018},
    ])
    first = 1
    losses = []
    run_signature = {
        "steps": args.steps, "seed": args.seed, "export_frame": args.export_frame,
        "holdout": sorted(args.holdout), "initialization": initial,
        "registered_vhap": str(args.registered_vhap.resolve()),
    }
    checkpoint = args.output / "latest.pt"
    if args.resume:
        # The upstream exponential schedule stores the last LR as a NumPy float64.
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
            saved = torch.load(checkpoint, weights_only=True)
        if saved["run_signature"] != run_signature:
            raise ValueError("Resume requires the same inputs, RGB split and training budget.")
        model.restore(saved["scene"], config)
        head.load_state_dict(saved["head"])
        head_optimizer.load_state_dict(saved["head_optimizer"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        losses = saved["losses"]
        first = saved["step"] + 1
        del saved
    else:
        model.training_setup(config)
    started = monotonic()
    generator_epoch = -1
    for step in range(first, args.steps + 1):
        epoch, index = divmod(step - 1, len(train))
        if epoch != generator_epoch:
            order = np.random.default_rng(args.seed + epoch).permutation(len(train))
            generator_epoch = epoch
        frame = train[order[index]]
        model.update_learning_rate(step)
        if step % 1000 == 0:
            model.oneupSHdegree()
        model.optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss, radii, projected = joint_step(frame, model, head, measure_visibility=step % 4 == 0)
        with torch.no_grad():
            if step == first or step % 100 == 0 or step == args.steps:
                value = float(loss)
                if not np.isfinite(value):
                    raise RuntimeError(f"Nonfinite loss at step {step}, {frame.name}")
                losses.append({"step": step, "loss": value, "scene_points": len(model.get_xyz)})
                print(f"step {step}/{args.steps} loss={value:.5f} scene_points={len(model.get_xyz)} "
                      f"elapsed={monotonic() - started:.1f}s", flush=True)
            if step == 1:
                gradients = [model._xyz.grad, model._opacity.grad, head.head_dc.grad]
                magnitudes = [float(gradient.abs().max()) if gradient is not None else 0.
                              for gradient in gradients]
                if not all(np.isfinite(value) and value > 0 for value in magnitudes):
                    raise RuntimeError(f"Scene or head receives no finite gradient at initialization: {magnitudes}")
                print(f"Initial gradient maxima (scene xyz, scene opacity, head color): {magnitudes}", flush=True)
            if step < 15000:
                valid = radii > 0
                model.max_radii2D[valid] = torch.maximum(model.max_radii2D[valid], radii[valid].float())
                model.add_densification_stats(projected, valid)
            model.optimizer.step()
            head_optimizer.step()
            if 500 < step < 15000 and step % 100 == 0:
                # Bound growth on the shared GPU. Pruning remains enabled at the cap.
                threshold = 0.0003 if len(model.get_xyz) < 400000 else float("inf")
                model.densify_and_prune(threshold, 0.005, extent, 20 if step > 3000 else None)
            if step < 15000 and step % 3000 == 0:
                model.reset_opacity()
            if step == 100 or step % 5000 == 0 or step == args.steps:
                save_checkpoint(checkpoint, step, model, head, head_optimizer, run_signature, losses)
            if step == 1000:
                reviews = [f for f in frames if f.name in {*args.holdout, args.export_frame}]
                review_layers(args.output / "review-01000", reviews, scene_tensors(model),
                              lambda f: head.head(f, geometry=True), render=render, visible_head=visible_head)
    with torch.no_grad():
        fitted_scene = scene_tensors(model)
        per_frame = evaluate(args.output, frames, fitted_scene, source_scene, head,
                             train_names, True, args.export_frame)
        frame = next(f for f in frames if f.name == args.export_frame)
        review_frames = [f for f in frames if f.name in {*args.holdout, args.export_frame}]
        review_layers(args.output, review_frames, fitted_scene, lambda f: head.head(f, geometry=True),
                      render=render, visible_head=visible_head)
        trained_path = args.output / "scene-final.ply"
        model.save_ply(str(trained_path))
        args.readback_frames = [frames[0], frame, frames[-1]]
        report = export(args, K, trained_path, fitted_scene, frame, head.head(frame, geometry=True))
    report.update(steps=args.steps, seed=args.seed, scale=scale, image_size=size,
                  K_integer_pixel_centers=K.tolist(), cuda_principal_point_px=principal,
                  frames=per_frame, training_frames=train_names, holdout_frames=args.holdout,
                  scene_initial_points=len(source_scene[0]), scene_initialization=initial, loss_history=losses,
                  scene_geometry_trainable_from_step=1, head_sh_degree=0,
                  scene_sh_degree=model.active_sh_degree, seed_ply=str(seed_path),
                  registered_head=str(args.registered_vhap), tracker=str(args.tracker))
    metrics = args.output / "metrics.json"
    temporary = metrics.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(metrics)
    print(f"Finished {args.output}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("scene", "background", "bound", "tracker", "landmarks", "registered-vhap", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--export-frame", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive.")
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run(args)
