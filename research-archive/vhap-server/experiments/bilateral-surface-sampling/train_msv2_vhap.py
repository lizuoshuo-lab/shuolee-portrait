"""Train a template-free MSv2 scene and an R15-style VHAP head in the same cameras."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
import pickle
from time import monotonic

import cv2
import numpy as np
import torch
from utils.loss_utils import ssim

from fuse_scene import check_principal_point, render
from msv2_scene_engine import SceneController, load_scene_model, make_rasterizer, scene_tensors
from r15_bound_head import R15BoundHead
from r15_head_loss import head_loss, prepare_head_targets
from review_layer_visibility import review_layers
from prepare_video_scene import sha256
from train_joint import combine, evaluate, export, prepare_frames, rgb_error, visible_head
from train_joint_scene import scene_optimizer_config


HEAD_WEIGHT = 0.25


def surface_quarters(frame, mesh, deformed):
    """Place one color sample at the centroid of each of four equal-area subtriangles."""
    xyz, sh, opacity, scales, rotations = frame.head
    triangles = mesh.vertices_world(frame, deformed=deformed)[mesh.faces]
    offsets = .5 * (triangles - triangles.mean(dim=1, keepdim=True))
    centers = torch.cat((xyz[:, None], xyz[:, None] + offsets), dim=1).flatten(0, 1)
    repeat = lambda x: x.repeat_interleave(4, dim=0)
    return replace(frame, head=(centers, repeat(sh), repeat(opacity), repeat(scales * .5),
                                repeat(rotations)), axes=repeat(frame.axes))


class MeshBoundHead(R15BoundHead):
    def __init__(self, reference, protected, mesh, surface_samples=1):
        if surface_samples == 4:
            reference = surface_quarters(reference, mesh, False)
            protected = np.repeat(protected, 4)
        super().__init__(reference, protected_triangles=protected)
        self.mesh = mesh
        self.surface_samples = surface_samples
        self.head_offset.requires_grad_(False)

    def initial_head(self, frame):
        return surface_quarters(frame, self.mesh, False).head if self.surface_samples == 4 else frame.head

    def head(self, frame, geometry=True):
        if geometry:
            frame = self.mesh.deform(frame)
        if self.surface_samples == 4:
            frame = surface_quarters(frame, self.mesh, geometry)
        return super().head(frame, geometry)

    def prior(self):
        return super().prior(regularize_opacity=False) + self.mesh.regularizer()


@torch.no_grad()
def head_targets(scene, frame):
    image = cv2.imread(str(scene / "mask_undist/images" / frame.name), cv2.IMREAD_GRAYSCALE)
    if image is None or tuple(image.shape) != tuple(frame.reference.shape[-2:]):
        raise ValueError(f"Missing native matte for {frame.name}")
    person = torch.as_tensor(image, dtype=torch.float32, device="cuda") / 255
    allowed = frame.masks["head_allowed"]
    target = torch.where(allowed, person, 0)
    # The lower head/body boundary is approximate. A free collar band keeps
    # head-only alpha supervision from treating its uncertain cut as real air.
    outside_support = ~frame.masks["head_exterior"]
    return prepare_head_targets(frame.reference, target, allowed=outside_support, face=frame.masks["face"])


def raster_arguments(tensors, projected, colors=None):
    xyz, features, opacity, scales, rotations = tensors
    return dict(means3D=xyz, means2D=projected, shs=features if colors is None else None,
                colors_precomp=colors, opacities=opacity, scales=scales,
                rotations=rotations, cov3D_precomp=None)


def train_step(frame, native, model, actor, targets, controller, step, *, photo_only=False):
    scene = scene_tensors(model)
    head = actor.head(native, geometry=True)
    projected_scene = torch.zeros_like(scene[0], requires_grad=True)
    projected_head = torch.zeros_like(head[0], requires_grad=True)
    joined = combine(scene, head)
    rgb, radii, areas = frame.rasterizer.training(
        **raster_arguments(joined, torch.cat((projected_scene, torch.zeros_like(head[0])))),
        culling=controller.culling(frame, len(head[0])),
    )
    scene_rgb = 0.8 * (rgb - frame.reference).abs().mean() + 0.2 * (1 - ssim(rgb, frame.reference))
    head_rgb, radii_head = native.rasterizer(**raster_arguments(head, projected_head))
    alpha = native.rasterizer(**raster_arguments(head, torch.zeros_like(head[0]),
                                                 torch.ones_like(head[0])))[0][0]
    ramp = min(max((step - 2400) / 600, 0), 1)
    fit, _ = head_loss(head_rgb, alpha, targets, constraint_ramp=ramp, photo_only=photo_only)
    loss = (1 - HEAD_WEIGHT) * scene_rgb + HEAD_WEIGHT * fit + actor.prior()
    if not photo_only and step % 4 == 0:
        fixed = tuple(value.detach() for value in head)
        if frame is native:
            isolated = alpha.detach()
        else:
            with torch.no_grad():
                isolated = render(frame.rasterizer, fixed, alpha_only=True)
        shown = visible_head(frame.rasterizer, combine(scene, fixed), len(scene[0]))
        supported = frame.masks["head_rgb"] & (isolated >= 0.8)
        loss += 0.1 * ((isolated - shown).relu() * supported).sum() / supported.sum().clamp_min(1)
    loss.backward()
    if step == 1:
        gradient_max = [float(g.abs().max()) if g is not None else 0. for g in
                        (model._xyz.grad, model._opacity.grad, actor.head_dc.grad, projected_head.grad)]
        if not all(np.isfinite(g) and g > 0 for g in gradient_max):
            raise RuntimeError(f"Missing initial scene or head gradient: {gradient_max}")
        print(f"Initial scene xyz, opacity, head color and head screen gradients: {gradient_max}", flush=True)
        if photo_only and actor.mesh.offset.requires_grad:
            mesh_gradient = actor.mesh.offset.grad
            if mesh_gradient is None or not torch.isfinite(mesh_gradient).all() or not mesh_gradient.abs().max() > 0:
                raise RuntimeError("RGB loss did not backpropagate to the shared VHAP mesh.")
            print(f"Initial shared mesh RGB gradient max: {float(mesh_gradient.abs().max()):.8g}", flush=True)
    count = len(scene[0])
    return loss.detach(), projected_scene, projected_head, radii[:count], areas[:count], radii_head


def protect_topology(frames):
    """Check existing head coverage in fixed training views before committing a split."""
    @torch.no_grad()
    def check(before, after, event):
        for frame in frames:
            inside = frame.masks["head_rgb"]
            outside = frame.masks["head_exterior"]
            old = render(frame.rasterizer, before(frame), alpha_only=True)
            new = render(frame.rasterizer, after(frame), alpha_only=True)
            if ((old - new).relu()[inside].mean() > 0.012
                    or (new - old).relu()[outside].mean() > 0.006):
                return False
        return True
    return check


def protected_triangles(registration, boundary_mask):
    with boundary_mask.open("rb") as file:
        regions = pickle.load(file, encoding="latin1")
    with np.load(registration, allow_pickle=False) as data:
        faces = data["faces"]
    landmarks = np.concatenate([regions[key] for key in
                                ("eye_region", "left_eyeball", "right_eyeball", "lips", "left_ear", "right_ear")])
    return np.isin(faces, landmarks).any(axis=1)


def refit_head_colors(actor, model, train, targets, steps, seed):
    """Adapt SH0 colors to the corrected mesh while holding rendered geometry fixed."""
    for param in actor.parameters():
        param.requires_grad_(param is actor.head_dc)
    optimizer = torch.optim.Adam([actor.head_dc], lr=0.005)
    with torch.no_grad():
        scene = tuple(value.detach() for value in scene_tensors(model))
    log = []
    for step in range(1, steps + 1):
        epoch, index = divmod(step - 1, len(train))
        if index == 0:
            order = np.random.default_rng(seed + epoch).permutation(len(train))
        frame = train[order[index]]
        optimizer.zero_grad(set_to_none=True)
        head = actor.head(frame)
        joint_rgb = render(frame.rasterizer, combine(scene, head))
        head_rgb = render(frame.rasterizer, head)
        with torch.no_grad():
            alpha = render(frame.rasterizer, head, alpha_only=True)
        photo, _ = head_loss(head_rgb, alpha, targets[frame.name], photo_only=True)
        loss = 0.75 * rgb_error(joint_rgb, frame.reference, frame.masks["head_rgb"]) + 0.25 * photo
        loss.backward()
        optimizer.step()
        actor.project_color(optimizer)
        if step == 1 or step % 300 == 0 or step == steps:
            value = float(loss.detach())
            if not np.isfinite(value):
                raise RuntimeError(f"Nonfinite head-color loss at iteration {step}")
            log.append({"iteration": step, "loss": value})
            print(f"Head color iteration {step}/{steps}: {value:.5f}", flush=True)
    return log


def run(args):
    args.background_ply = args.background / "point_cloud.ply"
    initial = json.loads((args.background / "initialization.json").read_text())
    if sorted(args.holdout) != sorted(initial["holdout"]) or initial["trained_steps"] != 0:
        raise ValueError("Expected untrained scene seeds with the exact training RGB split.")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (args.output / "metrics.json").exists():
        raise FileExistsError("This training run already has its final export.")
    with torch.no_grad():
        principal_error = check_principal_point(make_rasterizer)
        K, size, scale, seed_path, source, frames = prepare_frames(args, make_rasterizer)
    train = [frame for frame in frames if frame.name not in args.holdout]
    export_frame = next(frame for frame in train if frame.name == args.export_frame)
    targets = {frame.name: head_targets(args.scene, frame) for frame in train}
    head_area = {name: float((target.alpha > .99).float().mean()) for name, target in targets.items()}
    protected = protected_triangles(args.registered_vhap, args.flame_mask)
    mesh = None
    posed_seed = None
    if args.mesh_offset:
        from vhap_mesh_offset import VhapMeshOffset

        mesh = VhapMeshOffset(args.registered_vhap, frames, args.flame_mask, posed_mesh=args.posed_mesh)
        if args.freeze_mesh_offset:
            mesh.offset.requires_grad_(False)
        if args.posed_mesh:
            posed_seed = {"sha256": sha256(args.posed_mesh), **mesh.validate_posed_seed(frames)}
            print(f"Posed mesh handoff: {posed_seed}", flush=True)
        else:
            print(f"Zero-offset geometry parity: {mesh.validate_zero(frames)}", flush=True)
        actor = MeshBoundHead(export_frame, protected, mesh, args.surface_samples)
    else:
        actor = R15BoundHead(export_frame, protected_triangles=protected)
    groups = [
        {"params": [actor.head_dc], "lr": 0.0025},
        {"params": [actor.head_offset], "lr": 0.00016},
        {"params": [actor.head_log_scale], "lr": 0.005},
        {"params": [actor.head_opacity], "lr": 0.05},
    ]
    if mesh is not None and mesh.offset.requires_grad:
        groups.append({"params": [mesh.offset], "lr": 0.002})
    head_optimizer = torch.optim.Adam(groups, lr=0., eps=1e-15)
    opt = scene_optimizer_config(args.steps)
    opt.iterations = args.steps
    opt.densification_interval = 100
    opt.densify_grad_threshold = 0.0002
    opt.opacity_lr = 0.025
    extent = float(initial["scene_extent"])
    model = load_scene_model(args.vendor, seed_path, extent, opt, len(train))
    controller = SceneController(model, train, extent, lambda f: actor.head(f, geometry=True), opt)
    designated = [train[i] for i in np.linspace(0, len(train) - 1, 4).round().astype(int)] + [export_frame]
    guard = protect_topology(designated)
    print(f"{len(train)} training native views, {len(source[0])} scene seeds, "
          f"{len(export_frame.head[0])} FLAME face seeds", flush=True)
    review = [f for f in frames if f.name in {*args.holdout, args.export_frame}]
    start = monotonic()
    head_events = []
    history = []
    first = 1
    signature = dict(steps=args.steps, seed=args.seed, export_frame=args.export_frame,
                     holdout=sorted(args.holdout), seed_sha256=sha256(seed_path),
                     tracker_sha256=sha256(args.tracker), bound_sha256=sha256(args.bound),
                     vendor_source_sha256=sha256(args.vendor / ".msv2-source.json"))
    if mesh is not None:
        signature["mesh_offset"] = {"registered": sha256(args.registered_vhap), "regions": sha256(args.flame_mask),
                                   "landmarks": sha256(args.flame_mask.with_name("landmark_embedding_with_eyes.npy")),
                                   "offset_bound_tracker_units": mesh.max_offset,
                                   "alpha_supervision": False, "head_densification": False}
        if args.freeze_mesh_offset:
            signature["mesh_offset"]["frozen"] = True
        if args.surface_samples != 1:
            signature["mesh_offset"]["surface_samples_per_triangle"] = args.surface_samples
        if posed_seed is not None:
            signature["mesh_offset"]["posed_seed"] = posed_seed["sha256"]
    checkpoint = args.output / "training-state.pt"
    fit_from = args.hair_outline_from or args.head_color_from
    outline_fit = None
    color_fit = None
    if args.resume or fit_from is not None:
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
            saved = torch.load(fit_from or checkpoint, weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint and frozen MSv2/R15 inputs differ.")
        if fit_from is not None and saved["step"] != args.steps:
            raise ValueError("Refinement requires the completed RGB mesh checkpoint.")
        model.restore(saved["scene"], opt)
        model._culling, model.factor_culling = saved["culling"]
        controller.blur = saved["blur"]
        controller.events = saved["scene_events"]
        actor.prepare_restore(saved["head"], head_optimizer)
        actor.load_state_dict(saved["head"])
        if mesh is not None:
            actor.head_offset.requires_grad_(False)
        head_optimizer.load_state_dict(saved["head_optimizer"])
        head_events, history = saved["head_events"], saved["history"]
        first = saved["step"] + 1
        torch.cuda.set_rng_state(saved["torch_rng"])
        kind, ints, pos, has_gauss, value = saved["numpy_rng"]
        np.random.set_state((kind, np.asarray(ints, np.uint32), pos, has_gauss, value))
        del saved
    if fit_from is not None:
        if args.hair_outline_from:
            from fit_vhap_hair_geometry import fit_hair_outline

            outline_fit = fit_hair_outline(mesh, train)
            print(f"Hair geometry refinement: {outline_fit}", flush=True)
        else:
            parent = json.loads((fit_from.parent / "metrics.json").read_text())
            outline_fit = parent["hair_outline_fit"]
            if not outline_fit["accepted"]:
                raise ValueError("Head color refinement requires a completed outline correction.")
            color_fit = refit_head_colors(actor, model, train, targets, args.head_color_steps, args.seed)
        kind, ints, pos, has_gauss, value = np.random.get_state()
        torch.save(dict(step=args.steps, head=actor.state_dict(), scene=model.capture(),
                        head_optimizer=head_optimizer.state_dict(), scene_events=controller.events,
                        head_events=head_events, culling=(model._culling, model.factor_culling),
                        blur=controller.blur, history=history, signature=signature,
                        torch_rng=torch.cuda.get_rng_state(),
                        numpy_rng=(kind, ints.tolist(), pos, has_gauss, value)), checkpoint)
    for step in range(first, args.steps + 1):
        epoch, index = divmod(step - 1, len(train))
        if not index or step == first:
            order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        native = train[order[index]]
        frame = controller.frame_for_step(native, step)
        if step > 3000 and step % 1000 == 0:
            model.oneupSHdegree()
        model.update_learning_rate(step)
        model.optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss, scene_2d, head_2d, radii, areas, head_radii = train_step(
            frame, native, model, actor, targets[native.name], controller, step, photo_only=mesh is not None)
        with torch.no_grad():
            if mesh is None and 300 <= step <= 7800:
                # The dedicated head screen gradient only sees the weighted
                # head render. Undo that blend before the original R15 threshold.
                actor.add_stats(head_2d.grad / HEAD_WEIGHT, head_radii, head_area[native.name])
            head_optimizer.step()
            actor.project_color(head_optimizer)
            if mesh is None:
                event = actor.densify(step, head_optimizer, guard=guard)
                if event:
                    head_events.append(event)
            controller.update(step, frame, scene_2d, radii, areas)
            if step == 1 or step % 100 == 0 or step == args.steps:
                value = float(loss)
                if not np.isfinite(value):
                    raise RuntimeError(f"Nonfinite combined training loss at step {step}")
                elapsed = monotonic() - start
                history.append(dict(step=step, loss=value, scene_points=len(model.get_xyz),
                                    head_points=len(actor.binding), elapsed=elapsed))
                print(f"step {step}/{args.steps} loss={value:.5f} scene={len(model.get_xyz)} "
                      f"head={len(actor.binding)} elapsed={elapsed:.1f}s", flush=True)
                if mesh is not None and step in (100, 1000, 3000, 5000, 8000, args.steps):
                    print(f"Shared mesh offset max: {float(mesh.max_offset * mesh.offset.tanh().abs().max()) * 1000:.2f} mm",
                          flush=True)
            if step in (100, 1000, 1900, 3000, 5000, 8000) or step == args.steps:
                ckpt = args.output / "training-state.partial"
                kind, ints, pos, has_gauss, value = np.random.get_state()
                torch.save(dict(step=step, head=actor.state_dict(), scene=model.capture(),
                                head_optimizer=head_optimizer.state_dict(),
                                scene_events=controller.events, head_events=head_events,
                                culling=(model._culling, model.factor_culling), blur=controller.blur,
                                history=history, signature=signature, torch_rng=torch.cuda.get_rng_state(),
                                numpy_rng=(kind, ints.tolist(), pos, has_gauss, value)), ckpt)
                ckpt.replace(checkpoint)
            if step == 1000:
                review_layers(args.output / "review-01000", review, scene_tensors(model),
                              lambda f: actor.head(f, geometry=True), render=render, visible_head=visible_head)
                if mesh is not None:
                    from review_vhap_mesh_offset import review_mesh_offset

                    review_mesh_offset(args.output / "review-01000", review, mesh)

    with torch.no_grad():
        fitted_scene = scene_tensors(model)
        train_names = [frame.name for frame in train]
        per_frame = evaluate(args.output, frames, fitted_scene, source, actor, train_names, True, args.export_frame)
        review_layers(args.output, review, fitted_scene, lambda f: actor.head(f, geometry=True),
                      render=render, visible_head=visible_head)
        trained = args.output / "scene-final.ply"
        model.save_ply(str(trained))
        args.readback_frames = [frames[0], export_frame, frames[-1]]
        report = export(args, K, trained, fitted_scene, export_frame, actor.head(export_frame, geometry=True))
    report.update(steps=args.steps, seed=args.seed, scale=scale, image_size=size,
                  K_integer_pixel_centers=K.tolist(), frames=per_frame, training_frames=train_names,
                  holdout_frames=args.holdout, loss_history=history, head_events=head_events,
                  scene_events=controller.events, scene_initialization=initial,
                  head_config="20261111 R15 6785a212 (VHAP head adaptation)", head_sh_degree=0,
                  protected_head_triangles=int(protected.sum()), training_renderer="diff-gaussian-rasterization-ms",
                  cuda_principal_point_px=principal_error, scene_sh_degree=model.active_sh_degree,
                  resumed_step=first - 1, source_signature=signature)
    if mesh is not None:
        from review_vhap_mesh_offset import review_mesh_offset

        report["mesh_offset"] = review_mesh_offset(args.output, review, mesh)
        report["alpha_supervision"] = False
        report["head_densification"] = False
        report["surface_samples_per_triangle"] = args.surface_samples
    if posed_seed is not None:
        report["posed_seed"] = posed_seed
    if outline_fit is not None:
        report["hair_outline_fit"] = outline_fit
        report["parent_checkpoint"] = {"path": str(fit_from), "sha256": sha256(fit_from)}
    if color_fit is not None:
        report["head_color_fit"] = {"steps": args.head_color_steps, "loss_samples": color_fit}
    temp = args.output / "metrics.partial"
    temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(args.output / "metrics.json")
    print(f"Finished {args.output}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("scene", "background", "bound", "tracker", "landmarks", "registered-vhap", "vendor", "output", "flame-mask"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--export-frame", default="00560.jpg")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mesh-offset", action="store_true", help="Optimize the shared VHAP mesh with RGB supervision.")
    parser.add_argument("--surface-samples", type=int, choices=(1, 4), default=1,
                        help="Number of independently colored SH0 sites on each bound mesh triangle.")
    parser.add_argument("--freeze-mesh-offset", action="store_true", help="Keep exact VHAP geometry in RGB-only training.")
    parser.add_argument("--posed-mesh", type=Path, help="Use an externally fitted, registered posed FLAME sequence.")
    refine = parser.add_mutually_exclusive_group()
    refine.add_argument("--hair-outline-from", type=Path,
                        help="Refine hair geometry from a completed RGB mesh run in a separate output directory.")
    refine.add_argument("--head-color-from", type=Path,
                        help="Refit head SH0 colors from a completed outline checkpoint.")
    parser.add_argument("--head-color-steps", type=int, default=1200)
    args = parser.parse_args()
    if args.surface_samples != 1 and not args.mesh_offset:
        parser.error("Surface sampling needs a registered mesh.")
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if (args.hair_outline_from or args.head_color_from) and (not args.mesh_offset or args.resume):
        parser.error("Refinement requires --mesh-offset and a separate output.")
    if args.freeze_mesh_offset and (not args.mesh_offset or args.hair_outline_from or args.head_color_from):
        parser.error("Freezing mesh offsets requires RGB-only --mesh-offset and no mesh or color-refinement input.")
    if args.posed_mesh and not args.freeze_mesh_offset:
        parser.error("An externally fitted posed sequence requires --mesh-offset --freeze-mesh-offset.")
    if args.head_color_steps < 1:
        parser.error("--head-color-steps must be positive.")
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    run(args)
