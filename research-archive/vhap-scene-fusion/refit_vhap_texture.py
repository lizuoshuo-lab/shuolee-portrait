"""Relearn native UV color on a frozen, registered 87-frame VHAP mesh."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from compare_vhap_appearance import checksum
from fit_guided_vhap import camera_gauge
from fit_vhap_head_neck import STATE_KEYS, load_checkpoint
from prepare_video_avatar import setup_vhap_cameras


def read_observations(args, names):
    mats, rgb = [], []
    with np.load(args.support, allow_pickle=False) as z:
        if (z["names"].tolist() != names or
                z["initialization_sha256"].item() != sha256((args.sequence / "initialization.json").read_bytes()).hexdigest()):
            raise ValueError("Appearance evidence does not match the registered sequence")
        labels = (z["hair"].astype(bool), z["skin"].astype(bool))
        crop_hash, mask_hash = z["crop_sha256"].tolist(), z["matte_sha256"].tolist()
    for i in range(len(names)):
        paths = [args.sequence / folder / f"{i:08d}.jpg" for folder in ("images", "alpha_maps")]
        images = [p.read_bytes() for p in paths]
        if [sha256(x).hexdigest() for x in images] != [crop_hash[i], mask_hash[i]]:
            raise ValueError(f"Changed RGB or matte for {names[i]}")
        bgr, matte = [cv2.imdecode(np.frombuffer(x, np.uint8), mode)
                      for x, mode in zip(images, (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE), strict=True)]
        rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        mats.append(cv2.erode((matte >= 245).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    raw, fg = np.stack(rgb), np.stack(mats)
    if any(m.shape != fg.shape for m in labels):
        raise ValueError("The regional masks must use the actual training crop raster")
    return [torch.as_tensor(a) for a in (raw, fg, *labels)]


def run(args):
    cv2.setNumThreads(1)
    context = json.loads((args.sequence / "initialization.json").read_text())
    records = context["frames"]
    names = [r["name"] for r in records]
    if not records or [r["timestep"] for r in records] != list(range(len(records))):
        raise ValueError("Frames are not in the recorded order")
    raw, matte, hair, skin = read_observations(args, names)
    _, _, extrinsics, _, _ = camera_gauge(records, context["scene_scale"])
    setup_vhap_cameras(extrinsics)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "track"),
        "--batch-size", "4", "--log.interval-media", "1000000", "--log.interval-scalar", "55",
    ])
    stage = cfg.pipeline.rgb_global_tracking
    stage.optimizable_params = ("texture",)
    torch.manual_seed(20261117)
    np.random.seed(20261117)
    tracker = GlobalTracker(cfg)
    load_checkpoint(tracker, args.current)
    if not (cfg.model.tex_painted and cfg.model.tex_extra and cfg.model.residual_tex and tracker.calibrated):
        raise ValueError("Expected original native painted albedo plus shared UV residual")
    faces = tracker.flame.faces
    frozen = {k: getattr(tracker, k).detach().clone() for k in STATE_KEYS if k != "tex_extra"}
    for k in frozen:
        getattr(tracker, k).requires_grad_(False)
    tex = tracker.tex_extra
    tracker.global_step, tracker.timestep = 0, len(names)

    def sample_data(sample):
        t = sample["timestep_index"].cpu()
        return (raw[t].to(tracker.device, dtype=torch.float32) / 255.,
                matte[t].to(tracker.device), hair[t].to(tracker.device), skin[t].to(tracker.device))

    def render(sample, verts=None, albedo=None, raster=None):
        tracker.fill_cam_params_into_sample(sample)
        if verts is None:
            tracker.clear_cache()
            verts, _, _, albedo = tracker.forward_flame(sample["timestep_index"])
            raster = tracker.rasterize_flame(sample, verts, faces)
        out = tracker.render_rgba(raster, verts, faces, albedo, tracker.lights[None],
                                  background_color=[.12]*3, enable_disturbance=False)
        out["hard_coverage"] = raster["rast_out"][..., -1].detach() > 0
        return out

    def region(sample, rgba, normal):
        target, fg, h, s = sample_data(sample)
        inside = (rgba[:, 3].detach() >= .995) & fg
        if not (inside.sum(dim=(1, 2)) > 256).all():
            raise ValueError("One or more frames lack valid RGB surface observations")
        view_axis = sample["extrinsic"][:, 2, :3].to(normal)
        cosine = (normal.detach() * view_axis[:, :, None, None]).sum(1).abs().clamp(0, 1)
        weight = inside.float() * (.20+.80*cosine) * (1. + 2.0*h + .5*s)
        return target, inside, h, s, weight

    all_data = DataLoader(tracker.dataset, batch_size=cfg.batch_size, num_workers=2)
    mixed = DataLoader(tracker.dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                       generator=torch.Generator().manual_seed(20261117))

    @torch.no_grad()
    def score(folder, expected=None):
        folder.mkdir(exist_ok=False)
        result, hashes = [], []
        max_alpha_difference = 0.
        alphas = []
        for sample in all_data:
            out = render(sample)
            rgba = out["rgba"]
            target, support, h, s, _ = region(sample, rgba, out["normal"])
            for j, i in enumerate(sample["timestep_index"].tolist()):
                rgba_j = rgba[j].cpu().numpy()
                hashes.append(checksum(out["hard_coverage"][j].cpu().numpy()))
                if expected is not None:
                    max_alpha_difference = max(max_alpha_difference, float(np.max(np.abs(rgba_j[3] - expected[1][i]))))
                else:
                    alphas.append(rgba_j[3].copy())
                rgb = rgba_j[:3].transpose(1, 2, 0)
                jpg = np.rint(np.clip(rgb, 0., 1.)*255).astype(np.uint8)[..., ::-1]
                if not cv2.imwrite(str(folder / f"{i:08d}.jpg"), jpg, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                    raise RuntimeError("Failed to save native RGB")
                difference = (rgba[j, :3].clamp(0, 1) - target[j]).abs().mean(0)
                stats = {}
                for k, pixels in (("all", support[j]), ("hair", h[j] & support[j]), ("skin", s[j] & support[j])):
                    area = int(pixels.sum())
                    stats[k] = {"pixels": area, "rgb_mae_255": float(255*difference[pixels].mean()) if area >= 32 else None}
                result.append({"name": names[i], "yaw_deg": records[i]["baseline_yaw_deg"], **stats})
        if (len(result) != len(names) or (expected is not None and expected[0] != hashes)
                or max_alpha_difference > .001):
            raise RuntimeError(f"Rendering coverage or frame order changed; alpha max delta {max_alpha_difference}")
        print(f"Rendered {len(result)} frames to {folder.name}; edge alpha max delta {max_alpha_difference:.8f}", flush=True)
        return result, (hashes, alphas), max_alpha_difference

    before, original_alpha, _ = score(args.output / "previous-rgb")
    with np.load(args.seed, allow_pickle=False) as z, torch.no_grad():
        v = z["tex_extra"]
        if tuple(v.shape) != tuple(tex.shape) or not np.isfinite(v).all():
            raise ValueError("Archived texture does not match current FLAME UV topology")
        tex.copy_(torch.as_tensor(v, device=tracker.device))
    restored = tex.detach().clone()
    initial, _, restored_alpha_delta = score(args.output / "restored-rgb", original_alpha)

    # Visibility footprint at fixed geometry, including the native texture filter/mipmap support.
    visible = torch.zeros_like(tex[0], dtype=torch.bool)
    for sample in all_data:
        out = render(sample)
        _, support, _, _, _ = region(sample, out["rgba"], out["normal"])
        g = torch.autograd.grad((out["albedo"] * support[:, None]).sum(), tex)[0]
        visible |= g.abs().amax(dim=0) > 1e-10
    if not visible.any():
        raise RuntimeError("Native appearance samples have no UV gradient")
    tex.register_hook(lambda g: g * visible[None])
    visits = np.zeros(len(names), dtype=int)

    def photometric(sample, verts, faces, albedos, rast_dict, step_i=None, stage=None):
        out = render(sample, verts, albedos, rast_dict)
        rgba = out["rgba"]
        target, _, _, _, weight = region(sample, rgba, out["normal"])
        error = F.smooth_l1_loss(rgba[:, :3], target, beta=.025, reduction="none").mean(dim=1)
        photo = ((error * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2))).mean()
        out.update(gt_rgb=target, pred_rgb=rgba[:, :3], error_rgb=target - rgba[:, :3], pred_alpha=rgba[:, 3:])
        if stage == "rgb_global_tracking":
            visits[sample["timestep_index"].tolist()] += 1
        return photo, out

    tracker.compute_photometric_energy = photometric
    original_eval = tracker.evaluate
    tracker.evaluate = lambda make_visualization=True, epoch=0: original_eval(make_visualization=False, epoch=epoch)
    print(f"Restored legacy UV; native Adam will train all {len(names)} frames for {stage.num_epochs} epochs; {int(visible.sum())} UV texels supported", flush=True)
    tracker.optimize_stage("rgb_global_tracking", dataloader=mixed)
    if (not np.all(visits == stage.num_epochs) or tracker.global_step != len(mixed)*stage.num_epochs
            or any(not torch.equal(getattr(tracker, k), v) for k, v in frozen.items())
            or not torch.equal(tex[:, ~visible], restored[:, ~visible]) or not torch.isfinite(tex).all()):
        raise RuntimeError("Texture-only, visible support or full-sequence contract failed")
    tracker.save_result(fname="refit-texture")
    final, _, final_alpha_delta = score(args.output / "final-rgb", original_alpha)
    with torch.no_grad():
        atlas = tracker.get_albedo()[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]
        cv2.imwrite(str(args.output / "vhap-uv-albedo.png"), np.rint(atlas*255).astype(np.uint8))
    stats = dict(frame_count=len(names), visits=visits.tolist(), epochs=stage.num_epochs, steps=tracker.global_step,
                 frozen_geometry_camera_lights=True, same_raster_coverage_all_frames=True,
                 maximum_edge_alpha_difference=max(restored_alpha_delta, final_alpha_delta), observed_texels=int(visible.sum()),
                 total_texels=visible.numel(), checkpoint=str(tracker.out_dir / "refit-texture.npz"),
                 previous=before, restored=initial, fitted=final)
    (args.output / "color-fit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    tracker.tb_writer.close()
    print(f"Completed {tracker.global_step} steps; each frame used {stage.num_epochs} times; hard coverage identical", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("sequence", "current", "seed", "support", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
