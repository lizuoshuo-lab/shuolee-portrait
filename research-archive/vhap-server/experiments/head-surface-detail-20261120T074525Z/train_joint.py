"""Optimize a FLAME-bound head and the observed appearance of its COLMAP scene."""

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.loss_utils import ssim
from utils.sh_utils import C0

from fuse_scene import (SCALE_LANDMARKS, align_rigid, check_principal_point, estimate_scale,
                        local_head, make_rasterizer, prepare_scene, read_gaussians, render,
                        to_bgr, world_gaussians, write_ply)
from joint_review import assess, save_review


@dataclass
class Frame:
    name: str
    rasterizer: object
    reference: torch.Tensor
    masks: dict
    head: tuple
    axes: torch.Tensor
    crop: tuple
    rotation: np.ndarray
    translation: np.ndarray


def make_masks(scene, name, xy, size):
    width, height = size
    person = cv2.imread(str(scene / "mask_undist/images" / name), cv2.IMREAD_GRAYSCALE)
    coarse = cv2.imread(str(scene / "headmask_undist/images" / Path(name).with_suffix(".png")),
                        cv2.IMREAD_GRAYSCALE)
    if person is None or coarse is None or person.shape != (height, width) or coarse.shape != person.shape:
        raise ValueError(f"Invalid native actor support for {name}")
    hull = np.zeros_like(person)
    if xy is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(np.rint(xy).astype(np.int32)), 1)
    face = cv2.erode(hull, np.ones((7, 7), np.uint8)).astype(bool)
    head_allowed = cv2.dilate(((coarse > 128) | (hull > 0)).astype(np.uint8),
                              np.ones((11, 11), np.uint8)).astype(bool)
    near = cv2.dilate(head_allowed.astype(np.uint8), np.ones((45, 45), np.uint8)).astype(bool)
    head_rgb = head_allowed & (person >= 224)
    # The existing coarse head mask provides a support region. The neck/upper-shirt
    # band is evaluated separately using the chin and the person's original mask.
    if xy is None:
        x0, y0, bw, bh = cv2.boundingRect((coarse > 128).astype(np.uint8))
        x1, y1 = x0 + bw, y0 + bh
    else:
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
    h, w = y1 - y0, x1 - x0
    yy, xx = np.ogrid[:height, :width]
    seam = (yy >= y1 - 0.05 * h) & (yy <= y1 + 0.48 * h)
    seam = seam & (xx >= x0 - 0.28 * w) & (xx <= x1 + 0.28 * w) & (person >= 224)
    scene_safe = ~head_allowed
    body = scene_safe & (person >= 224)
    room = scene_safe & (person <= 16)
    masks = {"face": face, "seam": seam, "scene": scene_safe, "head_allowed": head_allowed,
             "head_rgb": head_rgb, "body": body, "room": room, "head_exterior": near & ~head_allowed}
    for key, value in masks.items():
        if value.sum() < 100 and not (key == "face" and xy is None):
            raise ValueError(f"Insufficient {key} support for {name}")
    masks["scalp"] = head_rgb & (yy < y0) & (xx >= x0 - 0.15 * w) & (xx <= x1 + 0.15 * w)
    sy, sx = np.nonzero(head_allowed)
    crop = (slice(max(0, sy.min() - 8), min(height, sy.max() + 9)),
            slice(max(0, sx.min() - 8), min(width, sx.max() + 9)))
    return {key: torch.as_tensor(value, device="cuda") for key, value in masks.items()}, crop


def legacy_inputs(args):
    K, size, cameras, names, xy, track, actor, keep, bgfile, scene = prepare_scene(args)
    if not args.holdout or len(set(args.holdout)) != len(args.holdout) or not set(args.holdout) < set(names):
        raise ValueError("Provide unique observed holdout frame filenames and at least one training frame.")
    if args.export_frame in args.holdout:
        raise ValueError("Export a training frame to keep the holdout evaluation separate.")
    train = np.array([name not in args.holdout for name in names])
    if train.sum() < 3:
        raise ValueError("Scale initialization requires at least three training views.")
    alignments, centers, offsets = [], [], []
    for i, name in enumerate(names):
        R, C = cameras[Path(name).stem]
        landmarks = local_head(actor, track, i)
        R_actor, t_actor, camera_points, _ = align_rigid(landmarks, xy[i], K)
        alignments.append((R, C, R_actor, t_actor))
        centers.append(C)
        offsets.append(camera_points[SCALE_LANDMARKS] @ R.T)
    scale, _ = estimate_scale(np.asarray(centers)[train, None], np.asarray(offsets)[train])
    return K, size, cameras, names, xy, track, actor, keep, bgfile, scene, scale, alignments


def registered_inputs(args):
    """Use the stock VHAP track in its original camera gauge, with no second PnP."""
    from scene.flame_gaussian_model import FlameGaussianModel

    from fuse_scene import read_scene_cameras

    K, size, cameras = read_scene_cameras(args.scene, args.background)
    sequence = args.landmarks.parent
    initialization = json.loads(args.landmarks.read_text())
    records = initialization["frames"]
    manifest = json.loads((args.scene / "source-frames.json").read_text())["frames"]
    names = [item["name"] for item in records]
    if (names != [item["name"] for item in manifest] or len(set(names)) != len(names)
            or set(cameras) != {Path(name).stem for name in names}
            or args.export_frame not in names):
        raise ValueError("VHAP and scene must identify the same ordered source frames.")
    fit = np.load(args.registered_vhap, allow_pickle=False)
    if (fit["names"].astype(str).tolist() != [Path(name).stem for name in names]
            or str(fit["checkpoint_sha256"]) != sha256(args.tracker.read_bytes()).hexdigest()):
        raise ValueError("Registered head provenance does not match the original VHAP track.")
    scale = float(fit["scene_scale"])
    rotation = fit["tracker_to_world_rotation"]
    translation = fit["scene_origin"]
    first = records[0]
    original_rotation = np.asarray(first["world_from_virtual"]) @ np.asarray(first["baseline"]["rotation"])
    if (not np.isfinite(scale) or scale <= 0 or translation.shape != (3,)
            or not np.isfinite(translation).all() or rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)
            or not np.isclose(scale, initialization["scene_scale"], atol=1e-7, rtol=0)
            or not np.allclose(rotation, original_rotation, atol=1e-7, rtol=0)
            or not np.allclose(translation, first["camera_center_world"], atol=1e-7, rtol=0)):
        raise ValueError("Invalid tracker-to-scene similarity.")
    star = np.load(sequence / "landmark2d/STAR.npz", allow_pickle=False)["face_landmark_2d"]
    if star.shape != (len(names), 68, 3):
        raise ValueError("STAR data does not match the ordered VHAP track.")
    observed = []
    alignments = []
    for i, (record, source) in enumerate(zip(records, manifest, strict=True)):
        name = record["name"]
        R, C = cameras[Path(name).stem]
        warp = np.asarray(record["warp"])
        V = np.asarray(record["virtual_from_native_cv"])
        if (record["timestep"] != i
                or not np.allclose(warp @ K, np.asarray(record["K_integer_pixel_centers"]) @ V,
                                   atol=1e-5, rtol=0)
                or not np.allclose(R @ V.T, record["world_from_virtual"], atol=1e-7, rtol=0)
                or not np.allclose(C, record["camera_center_world"], atol=1e-7, rtol=0)
                or sha256((args.scene / "orig_undist/images" / name).read_bytes()).hexdigest()
                != source["native_sha256"]):
            raise ValueError(f"Scene and VHAP source camera/image disagree: {name}")
        xy = None
        if record["has_landmarks"]:
            image = cv2.imread(str(sequence / "images" / f"{i:08}.jpg"))
            if image is None or not np.isfinite(star[i]).all() or not (star[i, :, 2] > 0).all():
                raise ValueError(f"Invalid observed facial landmarks: {name}")
            pixels = star[i, :, :2].astype(float) * np.asarray(image.shape[1::-1]) - 0.5
            xy = cv2.perspectiveTransform(pixels[None], np.linalg.inv(warp))[0]
        observed.append(xy)
        alignments.append((R, C, rotation, translation))
    actor = FlameGaussianModel(3)
    actor.load_ply(str(args.bound), has_target=False)
    triangles = len(actor.flame_model.faces)
    if (actor.binding is None or len(actor.binding) != triangles
            or not torch.equal(actor.binding.long().sort().values,
                               torch.arange(triangles, device=actor.binding.device))
            or not bool(actor._xyz.eq(0).all())):
        raise ValueError("Registered head needs one Gaussian per FLAME triangle.")
    keep = torch.ones_like(actor.binding, dtype=torch.bool)
    reference = fit["vertices_tracker"]
    if reference.shape != (len(names), len(actor.flame_model.v_template), 3):
        raise ValueError("Registered VHAP geometry does not match the GA template.")
    peak = 0.
    for i, verts in enumerate(reference):
        actor.select_mesh_by_timestep(i)
        if not np.isfinite(verts).all():
            raise ValueError(f"Non-finite VHAP vertices at {names[i]}")
        peak = max(peak, float((actor.verts[0] - torch.as_tensor(verts, device="cuda")).abs().max()))
    if not np.isfinite(peak) or peak > 1e-5:
        raise ValueError(f"GA did not preserve the posed VHAP mesh: max delta {peak:.6g}")
    print(f"Verified {len(names)} exact source cameras and posed VHAP meshes; max tracker delta {peak:.6g}",
          flush=True)
    bgfile = args.background_ply or args.background / "point_cloud/iteration_18000/point_cloud.ply"
    return K, size, cameras, names, observed, None, actor, keep, bgfile, read_gaussians(bgfile), scale, alignments


def prepare_frames(args, rasterizer_factory=make_rasterizer):
    registered = args.registered_vhap is not None
    K, size, cameras, names, xy, track, actor, keep, bgfile, scene, scale, alignments = (
        registered_inputs(args) if registered else legacy_inputs(args)
    )
    if len(set(args.holdout)) != len(args.holdout) or not set(args.holdout) < set(names):
        raise ValueError("Provide unique observed holdout frames and at least one training frame.")
    if args.export_frame in args.holdout:
        raise ValueError("Export a training frame to keep the holdout evaluation separate.")
    frames = []
    bindings = actor.binding[keep].long()
    reference_face_size = None
    for i, name in enumerate(names):
        R, C, R_actor, t_actor = alignments[i]
        if registered:
            actor.select_mesh_by_timestep(i)
            rotation, translation = R_actor, t_actor
        else:
            local_head(actor, track, i)
            rotation = R @ R_actor
            translation = C + scale * (R @ t_actor)
        head = world_gaussians(actor, rotation, translation, scale, keep)
        sizes = actor.face_scaling[bindings].reshape(-1, 1, 1)
        if reference_face_size is None:
            reference_face_size = sizes.clone()
            if float(reference_face_size.min()) <= 1e-8:
                raise ValueError("A bound triangle has a degenerate reference size.")
        axes = torch.as_tensor(rotation, dtype=torch.float32, device="cuda") @ actor.face_orien_mat[bindings]
        axes *= scale * sizes / reference_face_size
        original = cv2.imread(str(args.scene / "orig_undist/images" / name))
        if original is None or original.shape[1::-1] != size:
            raise ValueError(f"Invalid native RGB frame {name}")
        reference = torch.as_tensor(original[..., ::-1].copy(), device="cuda").permute(2, 0, 1).float() / 255
        masks, crop = make_masks(args.scene, name, xy[i], size)
        frames.append(Frame(name, rasterizer_factory(K, size, R, C), reference, masks, head, axes,
                            crop, rotation, translation))
    return K, size, scale, bgfile, scene, frames


def combine(scene, head):
    return tuple(torch.cat((a, b)) for a, b in zip(scene, head))


def visible_head(rasterizer, gaussians, scene_count):
    xyz, _, opacity, scales, rotations = gaussians
    labels = torch.ones_like(xyz)
    labels[:scene_count] = 0
    return rasterizer(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
                      colors_precomp=labels, opacities=opacity, scales=scales,
                      rotations=rotations, cov3D_precomp=None)[0][0]


def rgb_error(prediction, reference, mask):
    return (prediction[:, mask] - reference[:, mask]).abs().mean()


def head_rgb_loss(rgb, frame):
    sy, sx = frame.crop
    gt = frame.reference[:, sy, sx]
    mask = frame.masks["head_rgb"][sy, sx]
    patch = rgb[:, sy, sx]
    rgb_l1 = rgb_error(patch, gt, mask)
    if frame.masks["face"].any():
        rgb_l1 = 0.65 * rgb_error(rgb, frame.reference, frame.masks["face"]) + 0.35 * rgb_l1
    return 0.8 * rgb_l1 + 0.2 * (1 - ssim(torch.where(mask, patch, gt), gt))


class HeadParameters(torch.nn.Module):
    """Shared residuals on the posed FLAME triangles."""

    def __init__(self, head):
        super().__init__()
        self.head_dc = torch.nn.Parameter(torch.zeros_like(head[1][:, :1]))
        self.head_offset = torch.nn.Parameter(torch.zeros_like(head[0]))
        self.head_log_scale = torch.nn.Parameter(torch.zeros_like(head[3]))
        self.head_opacity = torch.nn.Parameter(torch.zeros_like(head[2]))

    def head(self, frame, geometry):
        xyz, features, opacity, scales, rotations = frame.head
        # A pale FLAME scalp must be able to acquire dark hair from observed RGB.
        dc = (features[:, :1] + self.head_dc).clamp(-0.5 / C0, 0.5 / C0)
        features = torch.cat((dc, features[:, 1:]), dim=1)
        if geometry:
            offset = 0.004 * self.head_offset.tanh()
            xyz = xyz + (frame.axes @ offset[..., None]).squeeze(-1)
            scales = scales * (0.18 * self.head_log_scale.tanh()).exp()
            opacity = (opacity.clamp(1e-6, 1 - 1e-6).logit() + self.head_opacity.tanh()).sigmoid()
        return xyz, features, opacity, scales, rotations

    def prior(self):
        return (0.012 * self.head_offset.tanh().square().mean()
                + 0.006 * self.head_log_scale.tanh().square().mean()
                + 0.006 * self.head_opacity.tanh().square().mean()
                + 0.004 * self.head_dc.tanh().square().mean())


class JointParameters(HeadParameters):
    """Residuals used by the pre-trained scene baseline."""

    def __init__(self, scene, head):
        super().__init__(head)
        self.scene_dc = torch.nn.Parameter(torch.zeros_like(scene[1][:, :1]))

    def scene(self, source):
        dc = source[1][:, :1] + 0.18 * self.scene_dc.tanh()
        return (source[0], torch.cat((dc, source[1][:, 1:]), dim=1), *source[2:])

    def prior(self):
        return super().prior() + 0.005 * self.scene_dc.tanh().square().mean()


def train_step(parameters, frame, source_scene, geometry, optimizer):
    optimizer.zero_grad(set_to_none=True)
    scene = parameters.scene(source_scene)
    head = parameters.head(frame, geometry)
    # Both terms render both nodes in depth order. Route face supervision to the
    # head, and valid body/room pixels to the scene, so the wall cannot learn a face.
    joined = combine(tuple(x.detach() for x in scene), head)
    head_rgb = render(frame.rasterizer, joined)
    head_loss = head_rgb_loss(head_rgb, frame)
    if geometry:
        alpha = visible_head(frame.rasterizer, joined, len(scene[0]))
        if frame.masks["face"].any():
            head_loss += 0.06 * (1 - alpha[frame.masks["face"]]).mean()
        head_loss += 0.12 * alpha[frame.masks["head_exterior"]].mean()
    head_loss.backward()

    scene_rgb = render(frame.rasterizer, combine(scene, tuple(x.detach() for x in head)))
    scene_loss = 0.5 * rgb_error(scene_rgb, frame.reference, frame.masks["body"])
    scene_loss += 0.5 * rgb_error(scene_rgb, frame.reference, frame.masks["room"])
    (0.40 * scene_loss + parameters.prior()).backward()
    loss = torch.stack((head_loss.detach(), scene_loss.detach()))
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError(f"Non-finite training loss at {frame.name}")
    torch.nn.utils.clip_grad_norm_(parameters.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss


def evaluate(output, frames, trained_scene, source_scene, parameters, train, preview, export_frame=None):
    per_frame = {}
    panels = []
    for frame in frames:
        initial = combine(source_scene, parameters.initial_head(frame) if hasattr(parameters, "initial_head") else frame.head)
        trained = combine(trained_scene, parameters.head(frame, geometry=True))
        baseline_rgb = render(frame.rasterizer, initial)
        trained_rgb = render(frame.rasterizer, trained)
        baseline_alpha = visible_head(frame.rasterizer, initial, len(source_scene[0]))
        trained_alpha = visible_head(frame.rasterizer, trained, len(trained_scene[0]))
        result = {}
        for key, rgb, alpha in [("initial", baseline_rgb, baseline_alpha), ("trained", trained_rgb, trained_alpha)]:
            result[key] = assess(rgb, frame.reference, alpha, frame.masks)
            for region in ("body", "room"):
                result[key][f"{region}_mae_255"] = float(rgb_error(rgb, frame.reference, frame.masks[region])) * 255
            result[key]["head_rgb_mae_255"] = float(rgb_error(rgb, frame.reference, frame.masks["head_rgb"])) * 255
            result[key]["scalp_mae_255"] = (
                float(rgb_error(rgb, frame.reference, frame.masks["scalp"])) * 255
                if int(frame.masks["scalp"].sum()) >= 100 else None
            )
        per_frame[frame.name] = result
        if frame.name not in train:
            save_review(output / f"{Path(frame.name).stem}-heldout.jpg", baseline_rgb, trained_rgb,
                        frame.reference, baseline_alpha, trained_alpha, frame.masks)
        elif frame.name == export_frame:
            save_review(output / f"{Path(frame.name).stem}-reference.jpg", baseline_rgb, trained_rgb,
                        frame.reference, baseline_alpha, trained_alpha, frame.masks)
        if preview:
            height, width = frame.reference.shape[1:]
            scaled = (round(720 * width / height / 2) * 2, 720)
            left, right = [cv2.resize(to_bgr(image), scaled, interpolation=cv2.INTER_AREA)
                           for image in (frame.reference, trained_rgb)]
            panel = cv2.copyMakeBorder(np.hstack((left, right)), 30, 0, 0, 0, cv2.BORDER_CONSTANT)
            tag = "Train" if frame.name in train else "Holdout"
            cv2.putText(panel, f"Source {frame.name}   {tag}", (9, 21), 0, .52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "Joint render", (scaled[0] + 9, 21), 0, .52, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
    if panels:
        height, width = panels[0].shape[:2]
        preview_path = output / "selected-frame-preview.mp4"
        staged = preview_path.with_suffix(".partial.mp4")
        subprocess.run(["ffmpeg", "-y", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                        "-video_size", f"{width}x{height}", "-framerate", "3", "-i", "pipe:0",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(staged)],
                       input=b"".join(panel.tobytes() for panel in panels), check=True)
        staged.replace(preview_path)
    return per_frame


def export(args, K, bgfile, scene, frame, head):
    output = args.output
    path = output / "point_cloud.ply.partial"
    count = write_ply(path, bgfile, head, scene_dc=scene[1][:, :1])
    records = json.loads((args.background / "cameras.json").read_text())
    for record in records:
        record["cx"] = float(K[0, 2] + 0.5)
        record["cy"] = float(K[1, 2] + 0.5)
    loaded = read_gaussians(path)
    max_delta = 0.0
    for view in args.readback_frames:
        delta = (render(view.rasterizer, loaded)
                 - render(view.rasterizer, combine(scene, head))).abs().max()
        max_delta = max(max_delta, float(delta))
    if max_delta > 1e-3:
        raise RuntimeError(f"Static export readback mismatch: {max_delta:.6g}")
    (output / "cameras.json").write_text(json.dumps(records, indent=2))
    path.replace(output / "point_cloud.ply")
    return {"reference": frame.name, "scene_and_head_point_counts": count,
            "readback_rgb_max": max_delta, "readback_frames": [v.name for v in args.readback_frames]}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output}")
    with torch.no_grad():
        principal = check_principal_point()
        K, size, scale, bgfile, source_scene, frames = prepare_frames(args)
    train = [frame for frame in frames if frame.name not in args.holdout]
    print(f"{len(train)} training frames, {len(args.holdout)} RGB holdouts, scale {scale:.6f}", flush=True)
    params = JointParameters(source_scene, train[0].head)
    optimizer = torch.optim.Adam([
        {"params": [params.head_dc], "lr": 0.025},
        {"params": [params.scene_dc], "lr": 0.013},
        {"params": [params.head_offset, params.head_log_scale, params.head_opacity], "lr": 0.018},
    ])
    generator = np.random.default_rng(args.seed)
    history = []
    for step in range(args.steps):
        index = step % len(train)
        if index == 0:
            order = generator.permutation(len(train))
        loss = train_step(params, train[order[index]], source_scene, step >= args.warmup, optimizer)
        if index == 0 or step + 1 == args.steps:
            head, scene = loss.tolist()
            history.append({"step": step + 1, "head_loss": head, "scene_rgb_l1": scene})
        if step % 52 == 0 or step + 1 == args.steps:
            print(f"step {step + 1}/{args.steps}: head {history[-1]['head_loss']:.4f},"
                  f" scene {history[-1]['scene_rgb_l1']:.4f}", flush=True)

    with torch.no_grad():
        fitted_scene = params.scene(source_scene)
        train_names = [frame.name for frame in train]
        per_frame = evaluate(args.output, frames, fitted_scene, source_scene, params, train_names, args.preview,
                             args.export_frame if args.registered_vhap else None)
        frame = next(f for f in frames if f.name == args.export_frame)
        args.readback_frames = [frames[0], frame, frames[-1]]
        report = export(args, K, bgfile, fitted_scene, frame, params.head(frame, geometry=True))
        fitted_dc = fitted_scene[1][:, :1] - source_scene[1][:, :1]
        report.update({"steps": args.steps, "warmup": args.warmup, "seed": args.seed,
                       "scale": scale, "image_size": size, "K_integer_pixel_centers": K.tolist(),
                       "cuda_principal_point_px": principal, "training_frames": train_names,
                       "holdout_frames": args.holdout, "scene_updated_dc_points": int(fitted_dc.ne(0).any(dim=(1, 2)).sum()),
                       "max_head_offset_reference": float((0.004 * params.head_offset.tanh()).norm(dim=1).max()),
                       "frames": per_frame, "loss_history": history,
                       "placement": {f.name: {"rotation": f.rotation.tolist(),
                                              "translation": f.translation.tolist()} for f in frames},
                       "input_paths": {**{k: str(getattr(args, k)) for k in
                                          ("scene", "background", "bound", "tracker", "landmarks", "flame_boundary_mask")},
                                       "background_ply": str(bgfile)}})
        if args.registered_vhap:
            report["input_paths"]["registered_vhap"] = str(args.registered_vhap)
            report["placement_contract"] = "stock VHAP posed vertices; fixed tracker-to-COLMAP similarity"
            report["no_landmark_frames"] = [f.name for f in frames if not f.masks["face"].any()]
    torch.save({"parameters": params.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": args.steps, "input_paths": report["input_paths"],
                "train": train_names, "holdout": args.holdout}, args.output / "training_state.pt")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Finished {args.output}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "background", "bound", "tracker", "landmarks", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--flame-boundary-mask", type=Path)
    parser.add_argument("--registered-vhap", type=Path,
                        help="Registered head prepared from stock VHAP; --landmarks is then initialization.json.")
    parser.add_argument("--background-ply", type=Path, help="Explicit scene checkpoint, with cameras under --background.")
    parser.add_argument("--preview", default=True, action=BooleanOptionalAction,
                        help="Save a side-by-side sampled-frame MP4.")
    parser.add_argument("--export-frame", required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=520)
    parser.add_argument("--warmup", type=int, default=104)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.steps:
        parser.error("Require 0 <= warmup < steps.")
    if bool(args.flame_boundary_mask) == bool(args.registered_vhap):
        parser.error("Provide --flame-boundary-mask for the legacy path or --registered-vhap for the stock track.")
    cv2.setNumThreads(1)
    run(args)
