"""Compare fitted FLAME, head Gaussians, and the scene under one recorded camera."""

from argparse import ArgumentParser
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from fit_side_frames import alpha_maps, mask_scores
from fuse_scene import (local_head, make_rasterizer, prepare_scene, project, render, to_bgr,
                        world_gaussians)
from train_joint import JointParameters, combine


def mesh_mask(vertices, faces, K, camera, pose, scale, size):
    R, C = camera
    A, t = pose
    xy = project((scale * vertices @ A.T + t - C) @ R, K)
    polygons = np.rint(xy[faces] * 16).astype(np.int32)
    mask = np.zeros(size[::-1], np.uint8)
    for poly in polygons:
        cv2.fillConvexPoly(mask, poly, 1, lineType=cv2.LINE_8, shift=4)
    return mask


def draw_boundary(image, mask, valid, color):
    band = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    image[(band != 0) & valid] = color


def make_panel(image, label):
    h, w = image.shape[:2]
    scale = min(236 / w, 210 / h)
    crop = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)
    result = np.full((238, 242, 3), 20, np.uint8)
    y, x = (210 - crop.shape[0]) // 2 + 23, (242 - crop.shape[1]) // 2
    result[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    cv2.putText(result, label, (6, 16), 0, .44, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def audit(args):
    m = json.loads((args.run / "model/metrics.json").read_text())
    side = json.loads((args.run / "side/side-poses.json").read_text())
    settings = {k: Path(v) for k, v in m["input_paths"].items()}
    settings["export_frame"] = m["reference"]
    K, size, cameras, front, _, track, actor, keep, _, scene = prepare_scene(SimpleNamespace(**settings))
    names = args.frames or side["review_frames"]
    evidence, _, warp_iou, _, _ = alpha_maps(settings["scene"], size, K, names)
    median = {k: torch.as_tensor([v], device="cuda", dtype=torch.float32)
              for k, v in side["propagated_articulation"].items()}
    binding = actor.binding[keep].long()
    local_head(actor, track, 0)
    baseline_sizes = actor.face_scaling[binding].reshape(-1, 1, 1).clone()
    checkpoint = torch.load(args.run / "model/training_state.pt", weights_only=True, map_location="cpu")
    all_rows, scores = [], {}
    for name in names:
        record = side["frames"][name]
        camera = cameras[Path(name).stem]
        R, C = camera
        A, t = np.asarray(record["rotation"]), np.asarray(record["translation"])
        scale = side["scale"]
        if record["mode"] == "front_landmarks":
            local_head(actor, track, front.index(name))
        else:
            local_head(actor, median, 0)
        head = world_gaussians(actor, A, t, scale, keep)
        axes = head[0].new_tensor(A) @ actor.face_orien_mat[binding]
        axes *= scale * actor.face_scaling[binding].reshape(-1, 1, 1) / baseline_sizes
        params = JointParameters(scene, head)
        params.load_state_dict(checkpoint["parameters"])
        head = params.head(SimpleNamespace(head=head, axes=axes), geometry=True)
        scene_fit = params.scene(scene)
        raster = make_rasterizer(K, size, R, C)
        head_rgb = render(raster, head)
        alpha = render(raster, head, alpha_only=True)
        rgba_gray = to_bgr(head_rgb + .10 * (1 - alpha)[None])
        scene_rgb = to_bgr(render(raster, scene_fit))
        joint_rgb = to_bgr(render(raster, combine(scene_fit, head)))
        original = cv2.imread(str(settings["scene"] / "orig_undist/images" / name))
        obs = evidence[name]
        measured = obs["target"]
        valid = obs["boundary_valid"]
        crop = obs["crop"]
        mesh = mesh_mask(actor.verts[0].cpu().numpy(), actor.flame_model.faces[binding].cpu().numpy(),
                         K, camera, (A, t), scale, size)
        alpha_np = alpha.cpu().numpy()
        base = original.copy()
        draw_boundary(base, measured, valid, (72, 242, 74))
        mesh_overlay = base.copy()
        draw_boundary(mesh_overlay, mesh, valid, (255, 169, 37))
        splat_overlay = base.copy()
        draw_boundary(splat_overlay, alpha_np >= .5, valid, (40, 141, 255))
        scores[name] = {"pose_method": record["mode"], "mesh": mask_scores(mesh, obs),
                        "head_gaussians": mask_scores(alpha_np, obs)}
        panels = [(base, f"{name} original + matte"), (mesh_overlay, "FLAME outline (blue)"),
                  (splat_overlay, "Gaussian outline (orange)"), (rgba_gray, "Head only"),
                  (scene_rgb, "Scene only"), (joint_rgb, "Joint render")]
        all_rows.append(np.hstack([make_panel(img[crop], label) for img, label in panels]))
    cv2.imwrite(str(args.output / "side-geometry-audit.jpg"), np.vstack(all_rows),
                [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    summary = {"source": str(args.run), "matte_mapping_min_iou": warp_iou, "frames": scores}
    (args.output / "side-geometry-audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    with torch.no_grad():
        audit(args)
