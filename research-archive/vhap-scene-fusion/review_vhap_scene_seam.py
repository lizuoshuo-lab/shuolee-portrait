"""Replay a completed scene/head fit at the same native cameras for seam diagnosis."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from export_head_neck_frames import boundary_loops
from fuse_scene import read_gaussians, render, save_jpg, to_bgr
from msv2_scene_engine import make_rasterizer
from prepare_video_scene import sha256
from review_layer_visibility import measure_visibility
from review_smplx_head_neck import project
from train_joint import combine, export, prepare_frames, visible_head
from train_msv2_vhap import MeshBoundHead, protected_triangles, raster_arguments
from vhap_mesh_offset import VhapMeshOffset


def depth_layer(frame, points):
    raw = frame.rasterizer.depth(**raster_arguments(points, torch.zeros_like(points[0])))
    return raw["rendered_depth"].squeeze(), raw["gidx"].squeeze(), 1 - raw["accum_alpha"].squeeze()


def count_ratio(mask, support):
    return float(mask[support].float().mean()) if support.any() else None


def stats(values):
    if not len(values):
        return None
    return [float(x) for x in torch.quantile(values.float(), values.new_tensor([.10, .50, .90]))]


@torch.no_grad()
def run(args):
    final = args.run / "joint"
    metadata = json.loads((final / "metrics.json").read_bytes())
    seed = json.loads((args.run / "registered-head/avatar.json").read_bytes())
    source = metadata["source_signature"]
    if (sha256(Path(seed["tracker"])) != source["tracker_sha256"]
            or sha256(Path(seed["bound"])) != source["bound_sha256"]
            or sha256(Path(seed["registration"])) != source["mesh_offset"]["registered"]
            or not source["mesh_offset"]["frozen"]):
        raise ValueError("Expected the exact completed frozen-mesh Gaussian fit")
    state_path = final / "training-state.pt"
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
        state = torch.load(state_path, weights_only=True, map_location="cpu")
    if state["signature"] != source or state["step"] != metadata["steps"]:
        raise ValueError("Checkpoint and final run receipt differ")
    frame_args = Namespace(scene=args.scene, background=args.background,
                           background_ply=args.background / "point_cloud.ply", tracker=Path(seed["tracker"]),
                           landmarks=Path(seed["initialization"]), bound=Path(seed["bound"]),
                           registered_vhap=Path(seed["registration"]), holdout=metadata["holdout_frames"],
                           export_frame=metadata["reference"])
    K, size, _, _, _, frames = prepare_frames(frame_args, make_rasterizer)
    views = {frame.name: frame for frame in frames}
    ref = views[metadata["reference"]]
    mesh = VhapMeshOffset(frame_args.registered_vhap, frames, args.flame_mask)
    actor = MeshBoundHead(ref, protected_triangles(frame_args.registered_vhap, args.flame_mask), mesh)
    actor.load_state_dict(state["head"])
    if torch.count_nonzero(mesh.offset) or torch.count_nonzero(actor.head_offset):
        raise ValueError("The diagnosed run must preserve the exact VHAP mesh and head centers")
    scene = read_gaussians(final / "scene-final.ply")
    static_ply = read_gaussians(final / "point_cloud.ply")
    counts = [len(scene[0]), len(actor.binding)]
    if (counts != metadata["scene_and_head_point_counts"] or len(static_ply[0]) != sum(counts)):
        raise ValueError("Scene and head PLY point counts do not match the final run")
    tape = np.load(frame_args.registered_vhap, allow_pickle=False)
    vertices = tape["vertices_tracker"].astype(np.float64)
    mean = vertices.mean(axis=0)
    # The neck ring is the only mesh boundary with 30 vertices in the official topology.
    neck, = [v for v in boundary_loops(tape["faces"]) if len(v) == 30]
    if not np.isfinite(mean[neck]).all():
        raise ValueError("Invalid lower neck boundary")
    world = (vertices @ tape["tracker_to_world_rotation"].T) * float(tape["scene_scale"]) + tape["scene_origin"]
    frame_index = {name: i for i, name in enumerate(tape["filenames"])}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {"counts": counts, "neck_boundary_vertices": len(neck), "camera_native_resolution": size,
              "integer_center_K": K.tolist(), "static_ply_pose": metadata["reference"],
              "output_readback_max_unit_rgb": metadata["readback_rgb_max"], "frames": {}}
    side_ply = None
    if args.export_pose is not None:
        frame = views[args.export_pose]
        frame_args.output = output / Path(frame.name).stem
        frame_args.output.mkdir()
        # Only the selected frame is a same-pose reference for this snapshot.
        frame_args.readback_frames = [frame]
        same_pose_head = actor.head(frame)
        exported = export(frame_args, K, final / "scene-final.ply", scene, frame, same_pose_head)
        camera_path = frame_args.output / "cameras.json"
        cameras = json.loads(camera_path.read_bytes())
        side_index = next(i for i, pose in enumerate(cameras) if Path(pose["img_name"]).stem == Path(frame.name).stem)
        cameras.insert(0, cameras.pop(side_index))
        camera_path.write_text(json.dumps(cameras, indent=2) + "\n")
        side_ply = read_gaussians(frame_args.output / "point_cloud.ply")
        combined = combine(scene, same_pose_head)
        comparison = {}
        for view in (ref, frames[-1]):
            error = (render(view.rasterizer, side_ply) - render(view.rasterizer, combined)).abs()
            comparison[view.name] = {"max_rgb": float(error.max()), "mean_rgb": float(error.mean())}
        exported.update(steps=metadata["steps"], training_renderer=metadata["training_renderer"],
                        source_training_state_sha256=sha256(state_path), same_world_camera=True,
                        secondary_view_serialization=comparison)
        (frame_args.output / "metrics.json").write_text(json.dumps(exported, indent=2, allow_nan=False) + "\n")
        report["exported_pose"] = exported
    for name in args.frames:
        frame = views[name]
        _, _, R, C = frame.rasterizer.calibration
        rim = world[frame_index[name], neck]
        xy, _ = project((rim - C) @ R, K)
        head = actor.head(frame)
        joined = combine(scene, head)
        source_rgb, scene_rgb, head_rgb, joint_rgb, static_rgb = [frame.reference, *(
            render(frame.rasterizer, g) for g in (scene, head, joined, static_ply))]
        head_alpha = render(frame.rasterizer, head, alpha_only=True)
        shown = visible_head(frame.rasterizer, joined, len(scene[0]))
        scene_z, _, scene_alpha = depth_layer(frame, scene)
        head_z, _, _ = depth_layer(frame, head)
        _, owner, alpha = depth_layer(frame, joined)
        ring_rec = np.zeros((size[1], size[0]), np.uint8)
        cv2.polylines(ring_rec, [np.rint(xy).astype(np.int32)], True, 255, 17)
        person = cv2.imread(str(args.scene / "mask_undist/images" / name), 0)
        seam = torch.as_tensor((ring_rec > 0) & (person >= 224), device="cuda")
        if not seam.any():
            raise RuntimeError(f"No source person support near the open neck edge: {name}")
        over = seam & (head_alpha >= .5)
        paired = over & (scene_alpha >= .5) & (scene_z > 0) & (head_z > 0)
        difference = (scene_z - head_z)[paired]
        layer_rgb = {
            "01-source": source_rgb, "02-scene": scene_rgb, "03-head": head_rgb,
            "04-native-joint": joint_rgb, "05-static-joint": static_rgb,
        }
        if side_ply is not None and name == args.export_pose:
            layer_rgb["07-exported-side"] = render(frame.rasterizer, side_ply)
        report["frames"][name] = {
            "whole_head_support": measure_visibility(head_alpha, shown, frame.masks["head_rgb"]),
            "rim_band_person_pixels": int(seam.sum()),
            "rim_head_alpha_under_point_one": count_ratio(head_alpha < .1, seam),
            "rim_joined_alpha_under_point_nine": count_ratio(alpha < .9, seam),
            "rim_layer_with_largest_weight_scene": count_ratio((owner >= 0) & (owner < counts[0]), seam),
            "rim_opaque_head_pixels": int(over.sum()),
            "rim_blocked_head_alpha_fraction": float(1 - shown[over].sum()/head_alpha[over].sum()) if over.any() else None,
            "rim_scene_minus_head_dominant_depth_p10_p50_p90": stats(difference),
            "rim_front_scene_ratio_of_paired": float((difference < 0).float().mean()) if len(difference) else None,
            "rim_abs_rgb_255": {k: float((v[:, seam] - source_rgb[:, seam]).abs().mean() * 255)
                                 for k, v in layer_rgb.items() if k != "01-source"},
        }
        left, top = (np.floor(xy.min(axis=0)).astype(int) - [42, 88]).clip(0, size)
        right, bottom = (np.ceil(xy.max(axis=0)).astype(int) + [42, 105]).clip(0, size)
        prefix = Path(name).stem
        for label, rgb in layer_rgb.items():
            image = to_bgr(rgb[:, top:bottom, left:right])
            save_jpg(output / f"{prefix}-{label}.jpg", image)
        overlay = to_bgr(source_rgb[:, top:bottom, left:right])
        cv2.polylines(overlay, [np.rint(xy - [left, top]).astype(np.int32)], True, (110, 250, 70), 1, cv2.LINE_AA)
        save_jpg(output / f"{prefix}-06-open-rim.jpg", overlay)
    (output / "seam-review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("run", "scene", "background", "flame-mask", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", default=["00850.jpg", "00560.jpg"])
    parser.add_argument("--export-pose", help="Export one verified static frame to a new PLY and camera package directory.")
    cv2.setNumThreads(1)
    run(parser.parse_args())
