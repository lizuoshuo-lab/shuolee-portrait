"""Compare pose seeds or neck observations using the native VHAP optimizer."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from prepare_video_avatar import setup_vhap_cameras, write_jpeg


def camera_gauge(records, scale):
    world_from_virt = np.asarray([r["world_from_virtual"] for r in records])
    center = np.asarray([r["camera_center_world"] for r in records])
    pose0 = world_from_virt[0] @ np.asarray(records[0]["baseline"]["rotation"])
    world_to_tracker = pose0.T
    origin = center[0]
    locations = (center - origin) @ world_to_tracker.T / scale
    world_from_virt = world_to_tracker @ world_from_virt
    virtual_to_gl = np.diag([1., -1., -1.])
    w2c = virtual_to_gl @ world_from_virt.transpose(0, 2, 1)
    extrinsics = np.concatenate((w2c, -(w2c @ locations[..., None])), axis=2).astype(np.float32)
    return locations, world_from_virt, extrinsics, origin, world_to_tracker


def render_mesh(tracker, sample):
    tracker.clear_cache()
    tracker.fill_cam_params_into_sample(sample)
    verts, _, _, albedo = tracker.forward_flame(sample["timestep_index"])
    faces = tracker.flame.faces
    rast = tracker.rasterize_flame(sample, verts, faces)
    light = tracker.lights[None] if tracker.lights is not None else None
    rgba = tracker.render_rgba(rast, verts, faces, albedo, light)["rgba"]
    return rgba.cpu().numpy(), rast


@torch.no_grad()
def position_offsets(tracker, records, loader, virtual):
    offsets = np.zeros((len(records), 3))
    for sample in loader:
        ids = sample["timestep_index"].tolist()
        if all(records[i]["has_landmarks"] for i in ids):
            continue
        rgba, _ = render_mesh(tracker, sample)
        alphas = rgba[:, 3]
        for j, index in enumerate(ids):
            record = records[index]
            if record["has_landmarks"]:
                continue
            target = np.squeeze(sample["alpha_map"][j].cpu().numpy()) >= .5
            x, y, w, h = cv2.boundingRect(target.astype(np.uint8))
            valid = np.s_[:y + round(h * .85), :]
            n = int(target[valid].sum())
            if n < 400:
                continue
            limit = min(50., .19 * w)
            shape = (target.shape[1], target.shape[0])

            def cost(pixel):
                affine = np.column_stack((np.eye(2), pixel)).astype(np.float32)
                moved = cv2.warpAffine(alphas[j], affine, shape, flags=cv2.INTER_LINEAR)
                return float(np.abs(moved - target)[valid].sum() / n + .02 * np.square(pixel / limit).sum())

            shift = minimize(cost, np.zeros(2), method="Powell", bounds=[(-limit, limit)] * 2,
                             options={"maxfev": 48, "xtol": .14, "ftol": 1e-4})
            if shift.fun < cost(np.zeros(2)) - .005:
                fx, fy = np.diag(record["K_integer_pixel_centers"])[:2]
                depth = record["baseline"]["translation"][2]
                offsets[index] = virtual[index, :, :2] @ (shift.x * depth / [fx, fy])
    return offsets


def render_pass(tracker, records, loader, sequence, neck=None):
    panels, stats = {}, {}
    with torch.no_grad():
        for sample in loader:
            timestep = sample["timestep_index"]
            rgba, raster = render_mesh(tracker, sample)
            checks = {}
            if neck is not None:
                for split in ("fit", "check"):
                    for point in neck.score(sample, raster, split=split):
                        checks.setdefault(point["timestep"], []).append(point)
            rgb = np.clip(rgba[:, :3], 0, 1).transpose(0, 2, 3, 1)
            alpha = rgba[:, 3]
            original = sample["rgb"].permute(0, 2, 3, 1).cpu().numpy()
            mattes = sample["alpha_map"].cpu().numpy()
            for j, index in enumerate(timestep.tolist()):
                name = records[index]["name"]
                truth = np.squeeze(mattes[j]) >= .5
                pred = alpha[j] >= .5
                valid = np.ones_like(truth)
                rows = np.flatnonzero(truth.any(axis=1))
                if len(rows) < 20:
                    raise RuntimeError(f"Missing head alpha at {name}")
                cutoff = int(rows[-1] - .15 * (rows[-1] - rows[0]))
                valid[cutoff:] = False
                union = (truth | pred) & valid
                overlap = (truth & pred) & valid
                if not union.any() or not (pred & valid).any():
                    raise RuntimeError(f"Empty mesh projection at {name}")
                rgb_error = np.abs(rgb[j] - original[j])[overlap].mean() if overlap.any() else None
                stats[name] = {"iou": float(overlap.sum() / union.sum()),
                               "photometric_mae": float(rgb_error) if rgb_error is not None else None}
                if neck is not None:
                    stats[name]["neck_points"] = checks.get(index, [])
                bg = cv2.imread(str(sequence / "images" / f"{index:08d}.jpg"))
                if bg is None or bg.shape[:2] != pred.shape:
                    raise RuntimeError(f"Original crop does not match the mesh render: {name}")
                overlay = bg.copy()
                overlay[pred] = np.uint8(overlay[pred] * .53 + np.array([230., 185., 32.]) * .47)
                edge = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                overlay[edge] = [235, 194, 70]
                panels[name] = (bg, overlay)
    if len(stats) != len(records):
        raise RuntimeError("The evaluation did not render the full timeline")
    return panels, stats


def label(image, text):
    panel = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (319, 29), (21, 21, 21), -1)
    cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .50, (250, 250, 250), 1, cv2.LINE_AA)
    return panel


def reviews(output, records, renderings, sample_fps, guidance):
    neck_comparison = "neck" in renderings
    profile = sorted(records, key=lambda r: abs(r["baseline_yaw_deg"]), reverse=True)
    selected = []
    for row in profile:
        if (not row["has_landmarks"] and abs(row["baseline_yaw_deg"]) >= 65
                and all(abs(row["timestamp_s"] - r["timestamp_s"]) >= .6 for r in selected)):
            selected.append(row)
        if len(selected) == 5:
            break
    selected.sort(key=lambda r: r["timestamp_s"])
    names = [row["name"] for row in selected]
    if neck_comparison:
        names = [r["name"] for r in records if abs(r["baseline_yaw_deg"]) >= 80 and not r["has_landmarks"]]

    def row_image(name, include_initial=False):
        img, base = renderings["baseline"]["fitted"][name]
        variant = "neck" if neck_comparison else "guided"
        guided = renderings[variant]["fitted"][name][1]
        columns = [label(img, f"Original {name}")]
        if include_initial:
            columns.extend(label(renderings[v]["initial"][name][1], title) for v, title in
                           (("baseline", "Pose prior seed"), ("guided", f"6DRepNet360 {guidance} seed")))
        columns.extend((label(base, "VHAP baseline fit"),
                        label(guided, "VHAP with neck points" if neck_comparison else "VHAP with 6D seed")))
        return np.hstack(columns)

    if names:
        write_jpeg(output / "profile-comparison.jpg", np.vstack([row_image(name, not neck_comparison) for name in names]))
    video = cv2.VideoWriter(str(output / "all-frames.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (960, 320))
    if not video.isOpened():
        raise OSError("Could not open the full-timeline review video")
    try:
        for record in records:
            video.write(row_image(record["name"]))
    finally:
        video.release()
    return names


def main(args):
    initialization = (args.sequence / "initialization.json").read_bytes()
    initial = json.loads(initialization)
    records, scale = initial["frames"], float(initial["scene_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Expected one finite positive world scale")
    if not (args.neck_observations or args.baseline_only) and not initial["conditioning"]["calibration_passed"]:
        raise RuntimeError("6DRepNet360 guidance did not pass independent frontal calibration")
    loc, virtual, extrinsics, origin, gauge = camera_gauge(records, scale)
    setup_vhap_cameras(extrinsics)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "track"),
        "--batch-size", "4", "--log.interval-media", "1000000", "--log.interval-scalar", "40",
    ])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tracker = GlobalTracker(cfg)
    if cfg.model.use_dynamic_offset or not cfg.model.use_static_offset or len(tracker.dataset) != len(records):
        raise RuntimeError("Expected every frame and one static, shared mesh")
    if cfg.data.target_extrinsic_type != "w2c":
        raise RuntimeError("Expected the VHAP world-to-camera convention")
    poses = ("neck_pose", "jaw_pose", "eyes_pose", "expr")
    shared = ("shape", "static_offset", "tex_extra", "lights")
    with np.load(args.shared_checkpoint) as state, torch.no_grad():
        for key in shared:
            target = getattr(tracker, key)
            values = torch.from_numpy(state[key]).to(target)
            if target.shape != values.shape or not torch.isfinite(values).all():
                raise ValueError(f"Invalid shared assets: {key}")
            target.copy_(values)
        joint_seed = {key: torch.from_numpy(np.median(state[key], axis=0)).to(getattr(tracker, key)) for key in poses}
        _, neutral, _, _ = tracker.forward_flame(torch.zeros(1, device=tracker.device, dtype=torch.long))
        root = (tracker.flame.J_regressor[0] @ neutral[0]).cpu().numpy()
        extent = (neutral[0].amax(dim=0) - neutral[0].amin(dim=0)).cpu().tolist()
    frozen = {key: getattr(tracker, key).detach().clone() for key in shared}
    for key in shared:
        getattr(tracker, key).requires_grad_(False)
    for index, record in enumerate(records):
        sample = tracker.dataset[index]
        if (int(sample["timestep_index"]) != index or int(record["timestep"]) != index
                or bool(np.any(np.asarray(sample["lmk2d"])[:, -1] > 0)) != record["has_landmarks"]):
            raise RuntimeError(f"VHAP dataset disagrees with the frame and landmark manifest at {index}")

    cfg.pipeline.rgb_global_tracking.num_epochs = args.epochs
    cfg.pipeline.rgb_global_tracking.optimizable_params = ("pose", "joints", "expr")
    neck = None
    if args.neck_observations:
        from vhap_neck_loss import configure

        neck_input = json.loads(args.neck_observations.read_text())
        detected = neck_input["frames"]
        if set(detected) != {record["name"] for record in records}:
            raise ValueError("Neck detections must refer to the same ordered source manifest")
        if (neck_input["schema"] != "vhap-neck-targets-v1"
                or neck_input["source"]["initialization_sha256"] != sha256(initialization).hexdigest()):
            raise ValueError("Neck detections must match the VHAP crop and pixel conventions")
        for index, record in enumerate(records):
            row = detected[record["name"]]
            crop = args.sequence / "images" / f"{index:08d}.jpg"
            if (row["timestep"] != index or row["timestamp_s"] != record["timestamp_s"]
                    or row["virtual_crop_sha256"] != sha256(crop.read_bytes()).hexdigest()):
                raise ValueError(f"Neck detections and VHAP input identity differ at {index}")
        observations = {i: detected[record["name"]]["points"] for i, record in enumerate(records)}
        fit_count = sum(point["split"] == "fit" for points in observations.values() for point in points)
        if not fit_count:
            raise RuntimeError("No visible automatic neck points passed the detection checks")
        neck = configure(tracker, observations, weight=args.neck_weight)
    render_loader = DataLoader(tracker.dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    report = {"frame_count": len(records), "share": {"shape": list(tracker.shape.shape),
              "static_offset": list(tracker.static_offset.shape), "dynamic_offsets": False,
              "canonical_bbox_xyz_flame_units": extent, "single_scene_scale": scale,
              "scene_origin": origin.tolist(), "scene_to_tracker_rotation": gauge.tolist()},
              "landmark_frames": sum(r["has_landmarks"] for r in records),
              "calibration": initial["conditioning"], "epochs": args.epochs, "guidance": args.guidance, "variants": {}}
    if neck is not None:
        report["neck"] = {"weight": args.neck_weight, "fit_points": fit_count,
                          "check_points": sum(p["split"] == "check" for rows in observations.values() for p in rows),
                          "frames_with_points": sum(bool(rows) for rows in observations.values())}
    renderings = {}
    offsets = None
    baseline_matrix = virtual @ np.asarray([record["baseline"]["rotation"] for record in records])
    baseline_translation = loc + (virtual @ np.asarray([r["baseline"]["translation"] for r in records])[..., None])[..., 0]
    variants = ("baseline",) if args.baseline_only else ("baseline", "neck" if neck is not None else "guided")
    for variant in variants:
        if neck is not None:
            neck.enabled = variant == "neck"
        seed = "baseline" if variant == "neck" else variant
        matrix = virtual @ np.asarray([r[seed]["rotation"] for r in records])
        translate = loc + (virtual @ np.asarray([r[seed]["translation"] for r in records])[..., None])[..., 0]
        if variant == "guided" and args.guidance == "yaw":
            relative = baseline_matrix.transpose(0, 2, 1) @ matrix
            yaw = np.arctan2(relative[:, 0, 2] - relative[:, 2, 0], relative[:, 0, 0] + relative[:, 2, 2])
            yaw_vectors = np.zeros((len(records), 3))
            yaw_vectors[:, 1] = yaw
            matrix = baseline_matrix @ Rotation.from_rotvec(yaw_vectors).as_matrix()
            translate = baseline_translation + (baseline_matrix - matrix) @ initial["root_flame_units"]
        with torch.no_grad():
            tracker.rotation.copy_(torch.from_numpy(Rotation.from_matrix(matrix).as_rotvec()).to(tracker.rotation))
            tracker.translation.copy_(torch.from_numpy(translate + (matrix - np.eye(3)) @ root).to(tracker.translation))
            for key in poses:
                getattr(tracker, key).copy_(joint_seed[key])
            if offsets is None:
                offsets = position_offsets(tracker, records, render_loader, virtual)
                report["share"]["initial_position_refined_frames"] = int(np.any(offsets != 0, axis=1).sum())
            tracker.translation.add_(torch.from_numpy(offsets).to(tracker.translation))
        tracker.global_step = 0
        tracker.timestep = len(records)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"{variant}: {len(records)} RGB frames, {args.epochs} epochs", flush=True)
        before, before_stats = render_pass(tracker, records, render_loader, args.sequence, neck)
        loader = DataLoader(tracker.dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                            generator=torch.Generator().manual_seed(args.seed))
        tracker.optimize_stage("rgb_global_tracking", dataloader=loader, lr_scale=.3)
        if any(not torch.isfinite(getattr(tracker, key)).all() for key in ("rotation", "translation", *poses)):
            raise RuntimeError(f"VHAP returned nonfinite tracking results for {variant}")
        if any(not torch.equal(getattr(tracker, key), value) for key, value in frozen.items()):
            raise RuntimeError("The shared shape or appearance changed during tracking")
        after, after_stats = render_pass(tracker, records, render_loader, args.sequence, neck)
        final_rotation = Rotation.from_rotvec(tracker.rotation.detach().cpu().numpy())
        jumps = np.degrees((final_rotation[1:] * final_rotation[:-1].inv()).magnitude())
        tracker.save_result(fname=variant, epoch=1000)
        report["variants"][variant] = {"checkpoint": str(tracker.out_dir / f"{variant}_1000.npz"),
                                       "steps": tracker.global_step, "frozen_arrays_match": True,
                                       "motion_p95_deg_per_sample": float(np.percentile(jumps, 95)),
                                       "motion_max_deg_per_sample": float(jumps.max()),
                                       "initial": before_stats, "fitted": after_stats}
        renderings[variant] = {"initial": before, "fitted": after}
    tracker.tb_writer.close()
    if neck is not None:
        report["neck"]["training_batches_with_points"] = neck.training_batches
        report["neck"]["training_point_uses_by_frame"] = {
            records[index]["name"]: count for index, count in sorted(neck.training_counts.items())
        }
        if not neck.training_counts:
            raise RuntimeError("No observed neck point matched a visible FLAME neck edge during training")
    if not args.baseline_only:
        report["review_frames"] = reviews(args.output, records, renderings, args.sample_fps, args.guidance)
    (args.output / "guided-vhap-report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "shared-checkpoint", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=3.)
    parser.add_argument("--guidance", choices=("full", "yaw"), default="full")
    parser.add_argument("--neck-observations", type=Path, help="Automatic visible neck skin contour points in the VHAP crop.")
    parser.add_argument("--neck-weight", type=float, default=.6)
    parser.add_argument("--baseline-only", action="store_true", help="Fit one sequence using the existing pose prior.")
    parser.add_argument("--epochs", type=int, default=6, choices=range(1, 10))
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    if args.baseline_only and args.neck_observations:
        parser.error("The neck observations option runs a paired comparison")
    if not np.isfinite(args.neck_weight) or args.neck_weight <= 0:
        parser.error("Neck loss weight must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    main(args)
