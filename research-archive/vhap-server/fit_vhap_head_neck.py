"""Refine one shared VHAP head and neck across a registered RGB sequence."""

from argparse import ArgumentParser
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge, label, render_mesh, render_pass
from prepare_video_avatar import setup_vhap_cameras, write_jpeg


STATE_KEYS = ("rotation", "translation", "neck_pose", "jaw_pose", "eyes_pose", "expr",
              "shape", "static_offset", "tex_extra", "lights")


@torch.no_grad()
def load_checkpoint(tracker, path):
    ids = list(tracker.dataset.timestep_ids)
    with np.load(path, allow_pickle=False) as state:
        if int(state["n_processed_frames"]) != len(ids) or state["timestep_id"].tolist() != ids:
            raise ValueError("Checkpoint frame order differs")
        for key in STATE_KEYS:
            target = getattr(tracker, key)
            data = torch.from_numpy(state[key]).to(target)
            if data.shape != target.shape or not torch.isfinite(data).all():
                raise ValueError(f"Invalid checkpoint array: {key}")
            target.copy_(data)


@torch.no_grad()
def replay_profiles(tracker, paths, records, output, scalp=None):
    profiles = [i for i, r in enumerate(records) if abs(r["baseline_yaw_deg"]) >= 80]
    if not profiles:
        raise ValueError("No recorded pure-side views")
    frontal = []
    if scalp is not None:
        for split in ("train", "holdout"):
            choices = [i for i, r in enumerate(records) if r[split] and r["has_landmarks"]]
            if choices:
                frontal.append(min(choices, key=lambda i: abs(records[i]["baseline_yaw_deg"])))
    selected = profiles + frontal
    loader = DataLoader(Subset(tracker.dataset, selected), batch_size=4)
    columns = []
    forward = tracker.forward_flame
    for index, path in enumerate(paths):
        load_checkpoint(tracker, path)
        if index == 1 and scalp is not None:
            tracker.forward_flame = with_scalp(tracker, scalp)
        rgb, original = [], []
        for sample in loader:
            rgba, _ = render_mesh(tracker, sample)
            color = rgba[:, :3].transpose(0, 2, 3, 1)
            alpha = np.clip(rgba[:, 3, :, :, None], 0, 1)
            rgb.extend((255 * np.clip(color * alpha + .12 * (1 - alpha), 0, 1)).astype(np.uint8)[..., ::-1])
            original.extend((255 * sample["rgb"].permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)[..., ::-1])
        columns.append(rgb)
    tracker.forward_flame = forward
    rows = [np.hstack([label(frame, f"Original {records[i]['name']}"), label(before, "VHAP RGB | before"),
                       label(after, "VHAP RGB | after")])
            for i, frame, before, after in zip(selected, original, *columns, strict=True)]
    write_jpeg(output / "profile-native-rgb.jpg", np.vstack(rows[:len(profiles)]))
    if frontal:
        write_jpeg(output / "frontal-native-rgb.jpg", np.vstack(rows[len(profiles):]))


def with_scalp(tracker, scalp):
    from vhap_scalp_outline import normals_for

    original = tracker.forward_flame
    faces = tracker.flame.faces.long()
    ids, displacement = scalp

    def forward(timesteps):
        posed, *rest = original(timesteps)
        shift = normals_for(posed, faces, ids) * displacement[None, :, None]
        return posed.index_add(1, ids, shift), *rest

    return forward


def neck_targets(prior, records, sequence, slots, origin, gauge, scale):
    xyz = prior["vertices_world"][:, slots].astype(np.float64)
    eligible = []
    for i, record in enumerate(records):
        alpha = cv2.imread(str(sequence / "alpha_maps" / f"{i:08d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise ValueError(f"Missing foreground alpha: {record['name']}")
        cam = (xyz[i] - prior["native_camera_center"][i]) @ prior["native_camera_to_world_rotation"][i]
        pixels = cam @ (prior["virtual_crop_homographies"][i] @ prior["native_K"]).T
        x, y = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int).T
        h, w = alpha.shape
        valid = (cam[:, 2] > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        valid[valid] &= alpha[y[valid], x[valid]] >= 224
        eligible.append(valid)
    return ((xyz - origin) @ gauge.T / scale).astype(np.float32), np.array(eligible)


@torch.no_grad()
def export_mesh(tracker, path, metadata, endpoints, t, origin, gauge, scale):
    batches = []
    for ts in torch.arange(len(metadata["names"]), device=tracker.device).split(8):
        mesh, _, _, _ = tracker.forward_flame(ts)
        a, b = mesh.cpu().numpy()[:, endpoints].transpose(2, 0, 1, 3)
        batches.append(((a + t * (b - a)) @ gauge * scale + origin).astype(np.float32))
    vertices = np.concatenate(batches)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite posed VHAP vertices")
    np.savez_compressed(path, vertices_world=vertices, **metadata)
    return vertices


def main(args):
    initial = json.loads((args.sequence / "initialization.json").read_text())
    report = json.loads(args.baseline.read_text())
    records, scale = initial["frames"], float(initial["scene_scale"])
    names = [r["name"] for r in records]
    indices = list(range(len(names)))
    saved = report["variants"]["baseline"]
    _, _, extrinsics, origin, gauge = camera_gauge(records, scale)
    if (not names or len(set(names)) != len(names) or len(names) != report["frame_count"]
            or indices != [r["timestep"] for r in records] or set(names) != set(saved["fitted"])
            or not np.isfinite(scale) or scale <= 0 or scale != report["share"]["single_scene_scale"]
            or not np.allclose(origin, report["share"]["scene_origin"], rtol=0, atol=1e-8)
            or not np.allclose(gauge, report["share"]["scene_to_tracker_rotation"], rtol=0, atol=1e-8)):
        raise ValueError("Baseline and sequence do not match")

    args.output.mkdir(parents=True, exist_ok=False)
    setup_vhap_cameras(extrinsics)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "track"),
        "--batch-size", "4", "--log.interval-media", "1000000", "--log.interval-scalar", "20",
    ])
    stage = cfg.pipeline.rgb_global_tracking
    stage.num_epochs = args.epochs
    params = ("pose", "joints", "expr", "shape", "static_offset", "texture", "lights")
    stage.optimizable_params = params
    torch.manual_seed(2026)
    np.random.seed(2026)
    tracker = GlobalTracker(cfg)
    if (len(tracker.dataset) != len(names) or cfg.model.use_dynamic_offset
            or not cfg.model.use_static_offset or cfg.data.target_extrinsic_type != "w2c"):
        raise ValueError("Expected calibrated per-frame cameras and one shared static head mesh")

    if args.replay_checkpoint:
        replay_profiles(tracker, (saved["checkpoint"], args.replay_checkpoint), records, args.output)
        tracker.tb_writer.close()
        return
    load_checkpoint(tracker, saved["checkpoint"])
    shared_before = {key: getattr(tracker, key).detach().clone() for key in STATE_KEYS[-4:]}

    neck_ids = tracker.flame.mask.get_vid_by_region(("neck",)).cpu().numpy()
    mapping = np.load(args.correspondence, allow_pickle=False).reshape(-1)
    inverse = {int(src): i for i, src in enumerate(mapping)}
    with np.load(args.prior, allow_pickle=False) as prior:
        if prior["names"].tolist() != names or float(prior["scene_scale"]) != scale:
            raise ValueError("Neck prior and source do not match")
        for key, values in (("native_camera_center", [r["camera_center_world"] for r in records]),
                            ("virtual_crop_homographies", [r["warp"] for r in records])):
            if not np.allclose(prior[key], values, rtol=0, atol=1e-8):
                raise ValueError(f"Neck prior camera mismatch: {key}")
        source = prior["source_vertex_endpoints"]
        ends = np.array([[inverse[int(a)], inverse[int(b)]] for a, b in source])
        exact = {a: i for i, (a, b) in enumerate(ends) if a == b}
        target_slots = np.array([exact[i] for i in neck_ids])
        target, eligible = neck_targets(prior, records, args.sequence, target_slots, origin, gauge, scale)
        t = prior["source_edge_interpolation"][None, :, None]
        metadata = {k: prior[k] for k in ("names", "faces", "timestamps_s", "neck_boundary_vertex_ids",
                    "scene_scale", "native_K", "native_camera_to_world_rotation", "native_camera_center",
                    "virtual_crop_homographies")}
        metadata["neck_vertex_ids"] = np.flatnonzero(np.isin(ends, neck_ids).all(axis=1))

    target_count = eligible.sum(axis=1)
    if not np.isfinite(target).all() or np.any(target_count == 0):
        raise ValueError("The anatomical neck prior has no supported samples in one or more frames")
    target = torch.as_tensor(target, device=tracker.device)
    weight = torch.as_tensor(eligible, device=tracker.device)
    vid = torch.as_tensor(neck_ids, device=tracker.device)
    compute = tracker.compute_energy

    def compute_with_neck(sample, step_i=None, stage=None):
        base, logs, verts, *rest = compute(sample, step_i, stage)
        if stage == "rgb_global_tracking":
            ts = sample["timestep_index"]
            error = F.huber_loss(verts[:, vid], target[ts], delta=.01, reduction="none").sum(dim=-1)
            neck = 2000 * (error * weight[ts]).sum() / weight[ts].sum()
            base = base + neck
            logs.update(neck_surface_prior=neck, total=base)
        return base, logs, verts, *rest

    tracker.compute_energy = compute_with_neck
    loader = DataLoader(tracker.dataset, batch_size=cfg.batch_size, num_workers=2, shuffle=True,
                        generator=torch.Generator().manual_seed(2026))
    eval_loader = DataLoader(tracker.dataset, batch_size=cfg.batch_size, num_workers=2)
    print(f"{len(names)} calibrated frames; {args.epochs} joint RGB epochs; neck support {target_count.min()}–{target_count.max()}", flush=True)
    before = export_mesh(tracker, args.output / "head-neck-before.npz", metadata, ends, t, origin, gauge, scale)
    initial_scores = render_pass(tracker, records, eval_loader, args.sequence)[1]
    tracker.timestep, tracker.global_step = len(names), 0
    tracker.optimize_stage("rgb_global_tracking", dataloader=loader, lr_scale=.3)
    if tracker.global_step != len(loader) * args.epochs or any(
            not torch.isfinite(getattr(tracker, key)).all() for key in STATE_KEYS):
        raise RuntimeError("Incomplete optimization or nonfinite fitted parameters")
    tracker.save_result(fname="head-neck-fitted")
    checkpoint = tracker.out_dir / "head-neck-fitted.npz"
    scalp_info = None
    if args.outline_targets:
        from vhap_scalp_outline import fit_scalp_outline

        rebuilt = export_mesh(tracker, args.output / "head-neck-rebuilt-control.npz", metadata, ends, t, origin, gauge, scale)
        scalp, scalp_info = fit_scalp_outline(tracker, eval_loader, records, args.sequence, args.outline_targets, args.output)
        allowed = np.isin(ends, scalp[0].cpu().numpy()).any(axis=1)
        forward = tracker.forward_flame
        tracker.forward_flame = with_scalp(tracker, scalp)
    after = export_mesh(tracker, args.output / "head-neck-fitted.npz", metadata, ends, t, origin, gauge, scale)
    if scalp_info is not None:
        if not np.array_equal(rebuilt[:, ~allowed], after[:, ~allowed]):
            raise RuntimeError("Scalp refinement changed protected mesh vertices")
        scalp_info["protected_vertices_match_rebuilt_control"] = int((~allowed).sum())
    fitted_scores = render_pass(tracker, records, eval_loader, args.sequence)[1]
    if scalp_info is not None:
        tracker.forward_flame = forward
        if args.compare_checkpoint:
            replay_profiles(tracker, (args.compare_checkpoint, checkpoint), records, args.output, scalp)
    tracker.tb_writer.close()

    result = dict(frame_count=len(names), frames=names, shared_scale=scale, epochs=args.epochs,
                  steps=tracker.global_step, optimized=list(params), checkpoint=str(checkpoint),
                  shared_shapes={k: list(getattr(tracker, k).shape) for k in shared_before},
                  shared_parameter_delta_l2={k: float((getattr(tracker, k).detach() - v).norm()) for k, v in shared_before.items()},
                  neck_prior_eligible_points=target_count.tolist(),
                  posed_vertex_delta_scene_p50_p95_max=np.percentile(np.linalg.norm(after - before, axis=-1), [50, 95, 100]).tolist(),
                  initial=initial_scores, fitted=fitted_scores, scalp_outline=scalp_info)
    (args.output / "fit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed {tracker.global_step} update steps and exported {len(names)} head-neck meshes", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("sequence", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("prior", "correspondence", "replay-checkpoint", "outline-targets", "compare-checkpoint"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--epochs", type=int, default=8, choices=range(1, 10))
    args = parser.parse_args()
    if not args.replay_checkpoint and (args.prior is None or args.correspondence is None):
        parser.error("Fitting requires --prior and --correspondence")
    main(args)
