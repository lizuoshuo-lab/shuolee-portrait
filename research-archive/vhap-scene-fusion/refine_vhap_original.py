"""Warm-start the original VHAP global stages with registered anatomy and hair."""

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

from fit_guided_vhap import camera_gauge, render_mesh
from fit_vhap_head_neck import export_mesh, load_checkpoint, neck_targets
from prepare_video_avatar import setup_vhap_cameras
from vhap_scalp_outline import normals_for, scalp_support


def targets(sequence, archive, records, shape):
    with np.load(archive, allow_pickle=False) as z:
        if (z["names"].tolist() != [r["name"] for r in records]
                or z["hair"].shape != (len(records), *shape)
                or z["initialization_sha256"].item() != sha256((sequence / "initialization.json").read_bytes()).hexdigest()):
            raise ValueError("The outline archive is not from the registered crops")
        for field, folder in (("matte_sha256", "alpha_maps"), ("crop_sha256", "images")):
            hashes = [sha256((sequence / folder / f"{i:08d}.jpg").read_bytes()).hexdigest() for i in range(len(records))]
            if z[field].tolist() != hashes:
                raise ValueError(f"Outline provenance differs for {folder}")
        roi = z["hair"].astype(bool)
    matte = np.stack([cv2.imread(str(sequence / "alpha_maps" / f"{i:08d}.jpg"), 0) >= 128 for i in range(len(records))])
    sdf = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
                    - cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) for m in matte])
    # Both missed foreground and leakage matter. Penalize spill more strongly than before.
    sdf = np.clip(sdf / 12, -1., 1.) * np.where(matte, 1., 1.7)
    return matte, roi, sdf.astype(np.float32)


@torch.no_grad()
def transplant_scalp(tracker, source):
    ids, _ = scalp_support(tracker.flame)
    with np.load(source, allow_pickle=False) as z:
        if not np.array_equal(z["native_vertex_ids"], ids.cpu().numpy()):
            raise ValueError("The saved scalp does not match the native topology")
        offset = torch.as_tensor(z["outward_offset"], device=tracker.device, dtype=torch.float32)
    frames = torch.arange(tracker.n_timesteps, device=tracker.device)
    canonical = torch.cat([tracker.forward_flame(ts)[1] for ts in frames.split(8)]).mean(0, keepdim=True)
    normal = normals_for(canonical, tracker.flame.faces.long(), ids)[0]
    tracker.static_offset[0, ids] += normal * offset[:, None]
    return ids, normal


@torch.no_grad()
def score(tracker, loader, matte, roi, held, output):
    scores = {"fit": {"missed": 0, "outside": 0, "intersection": 0, "union": 0, "frames": 0},
              "check": {"missed": 0, "outside": 0, "intersection": 0, "union": 0, "frames": 0}}
    output.mkdir()
    for sample in loader:
        rgba, _ = render_mesh(tracker, sample)
        for j, i in enumerate(sample["timestep_index"].tolist()):
            pred = rgba[j, 3] >= .5
            support, target = roi[i], matte[i]
            if support.any():
                stats = scores["check" if held[i] else "fit"]
                stats["frames"] += 1
                stats["missed"] += int((target & ~pred & support).sum())
                stats["outside"] += int((~target & pred & support).sum())
                stats["intersection"] += int((target & pred & support).sum())
                stats["union"] += int(((target | pred) & support).sum())
            alpha = np.clip(rgba[j, 3, ..., None], 0, 1)
            color = rgba[j, :3].transpose(1, 2, 0)
            rgb = (np.clip(color*alpha + .12*(1-alpha), 0, 1)*255).astype(np.uint8)[..., ::-1]
            cv2.imwrite(str(output / f"{i:08d}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 91])
    for stats in scores.values():
        stats["iou"] = stats["intersection"] / max(1, stats["union"])
    return scores


def main(args):
    cv2.setNumThreads(1)
    initial = json.loads((args.sequence / "initialization.json").read_text())
    records = initial["frames"]
    names = [r["name"] for r in records]
    scale = float(initial["scene_scale"])
    if [r["timestep"] for r in records] != list(range(len(records))) or scale <= 0:
        raise ValueError("Unregistered frame order or scene scale")
    _, _, extrinsics, origin, gauge = camera_gauge(records, scale)
    args.output.mkdir(parents=True, exist_ok=False)
    setup_vhap_cameras(extrinsics)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "track"),
        "--batch-size", "4", "--log.interval-media", "1000000", "--log.interval-scalar", "50",
    ])
    stage = cfg.pipeline.rgb_global_tracking
    torch.manual_seed(20261117)
    np.random.seed(20261117)
    tracker = GlobalTracker(cfg)
    if (len(tracker.dataset) != len(records) or not tracker.calibrated or cfg.model.use_dynamic_offset
            or not cfg.model.use_static_offset or stage.num_epochs != 30):
        raise ValueError("Expected the native 30-epoch schedule with the registered cameras and one shared offset")
    load_checkpoint(tracker, args.seed)
    scalp_ids, normals = transplant_scalp(tracker, args.scalp_offset)
    matte, roi, sdf = targets(args.sequence, args.outline_targets, records, tuple(tracker.image_size))
    held = np.arange(len(records)) % 5 == 1
    roi_fit = roi.copy()
    roi_fit[held] = False
    include = torch.as_tensor(roi_fit, device=tracker.device)
    signed = torch.as_tensor(sdf, device=tracker.device)
    fg = torch.as_tensor(matte, device=tracker.device)

    neck_ids = tracker.flame.mask.get_vid_by_region(("neck",)).cpu().numpy()
    inverse = {int(sm): i for i, sm in enumerate(np.load(args.correspondence, allow_pickle=False).reshape(-1))}
    with np.load(args.prior, allow_pickle=False) as prior:
        if (prior["names"].tolist() != names or float(prior["scene_scale"]) != scale
                or not np.allclose(prior["native_camera_center"], [r["camera_center_world"] for r in records], rtol=0., atol=1e-8)
                or not np.allclose(prior["virtual_crop_homographies"], [r["warp"] for r in records], rtol=0., atol=1e-8)):
            raise ValueError("Neck prior and the registered sequence differ")
        ends = np.array([[inverse[int(a)], inverse[int(b)]] for a, b in prior["source_vertex_endpoints"]])
        exact = {a: i for i, (a, b) in enumerate(ends) if a == b}
        target, weights = neck_targets(prior, records, args.sequence, np.array([exact[i] for i in neck_ids]), origin, gauge, scale)
        t = prior["source_edge_interpolation"][None, :, None]
        metadata = {k: prior[k] for k in ("names", "faces", "timestamps_s", "neck_boundary_vertex_ids", "scene_scale", "native_K",
                    "native_camera_to_world_rotation", "native_camera_center", "virtual_crop_homographies")}
        metadata["neck_vertex_ids"] = np.flatnonzero(np.isin(ends, neck_ids).all(axis=1))
    if not np.isfinite(target).all() or not weights.any(axis=1).all():
        raise ValueError("Anatomical neck prior is missing on one or more frames")
    target, weights, neck_ids = [torch.as_tensor(a, device=tracker.device) for a in (target, weights, neck_ids)]

    original = tracker.compute_energy
    guided_batches = 0

    def energy(sample, step_i=None, stage=None):
        current, logs, verts, *rest = original(sample, step_i, stage)
        if stage != "rgb_global_tracking":
            return current, logs, verts, *rest
        ts = sample["timestep_index"]
        neck_error = F.huber_loss(verts[:, neck_ids], target[ts], delta=.01, reduction="none").sum(dim=-1)
        neck = 2000 * (neck_error * weights[ts]).sum() / weights[ts].sum()
        if include[ts].any():
            alpha = rest[-1]["pred_alpha"][:, 0]
            loss = 10. * (((alpha - fg[ts].float()) * signed[ts]) * include[ts]).sum() / include[ts].sum()
            gradient = torch.autograd.grad(loss, tracker.static_offset, retain_graph=True)[0][0, scalp_ids]
            if not torch.isfinite(gradient).all():
                raise RuntimeError("Nonfinite hair contour gradient")
            # Route the added objective only to shared normal offsets at allowed scalp vertices.
            g = (gradient * normals).sum(-1, keepdim=True).detach() * normals
            routed = (tracker.static_offset[0, scalp_ids] * g).sum()
            current = current + routed - routed.detach() + loss.detach()
            logs["hair_outline"] = loss.detach()
            nonlocal guided_batches
            guided_batches += 1
        current = current + neck
        logs.update(neck_surface_prior=neck, total=current)
        return current, logs, verts, *rest

    tracker.compute_energy = energy
    tracker.timestep, tracker.global_step = len(records), 0
    loader = DataLoader(tracker.dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                        generator=torch.Generator().manual_seed(20261117))
    checks = DataLoader(tracker.dataset, batch_size=cfg.batch_size, num_workers=2)
    tracker.save_result(fname="actual-initialization")
    export = lambda name: export_mesh(tracker, args.output / name, metadata, ends, t, origin, gauge, scale)
    v0 = export("initialized-head-neck.npz")
    first = score(tracker, checks, matte, roi, held, args.output / "initial-rgb")
    print(f"Starting native {stage.num_epochs} epochs across {len(records)} initialized frames; initial contour={first}", flush=True)
    tracker.optimize_stage("rgb_global_tracking", dataloader=loader, lr_scale=.1)
    if tracker.global_step != len(loader)*stage.num_epochs or not guided_batches:
        raise RuntimeError("The original global training did not complete")
    tracker.save_result(fname="final-head-neck")
    v1 = export("final-head-neck.npz")
    last = score(tracker, checks, matte, roi, held, args.output / "final-rgb")
    tracker.tb_writer.close()
    result = dict(frames=names, frame_count=len(names), initial=str(args.seed), scale=scale, first_dataset_frame=names[0],
                  first_landmark_frame=next(r["name"] for r in records if r["has_landmarks"]),
                  epochs=stage.num_epochs, updates=tracker.global_step, guided_batches=guided_batches, lr_scale=.1,
                  checkpoint=str(tracker.out_dir / "final-head-neck.npz"),
                  shared_static_offset_shape=list(tracker.static_offset.shape),
                  geometry_delta_scene_p50_p95=np.percentile(np.linalg.norm(v1-v0, axis=-1), [50, 95]).tolist(),
                  outline_initial=first, outline_final=last)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(f"Completed original VHAP schedule: {tracker.global_step} steps; final contour={last}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("sequence", "seed", "scalp-offset", "prior", "correspondence", "outline-targets", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    main(parser.parse_args())
