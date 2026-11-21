"""Seed the joint scene from original RGB with frozen COLMAP cameras and no body template."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import cv2
import numpy as np

from initialize_joint_scene import read_colmap, read_scene, sample_observations, write_gaussians
from prepare_video_scene import sha256


def run(args):
    records, camera, _, original, _, cameras, digests = read_scene(args.scene, args.gaussian_repo)
    holdout = set(args.holdout)
    if len(holdout) != len(args.holdout) or not holdout < set(original):
        raise ValueError("Invalid RGB holdout for frozen cameras.")
    records = [record for record in records if record["name"] not in holdout]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    images, masks, known, sparse = tuple(args.output / name for name in ("images", "masks", "known", "sparse"))
    for folder in (images, masks, known, sparse):
        folder.mkdir()
    db = args.output / "database.db"
    for record in records:
        name = record["name"]
        source = args.scene / "orig_undist/images" / name
        if sha256(source) != record["native_sha256"]:
            raise ValueError(f"Native frame changed since registration: {name}")
        (images / name).symlink_to(source)
        head = cv2.imread(str(args.scene / "headmask_undist/images" / Path(name).with_suffix(".png")), 0)
        if head is None or head.shape != (camera.height, camera.width):
            raise ValueError(f"No native head mask: {name}")
        excluded = cv2.dilate((head > 128).astype(np.uint8), np.ones((9, 9), np.uint8))
        if not cv2.imwrite(str(masks / (name + ".png")), (1 - excluded) * 255):
            raise OSError(f"Mask write failed: {name}")
    camera_values = ",".join(map(str, camera.params))
    colmap_env = {key: value for key, value in os.environ.items() if not key.startswith("QT_")}
    subprocess.run(["xvfb-run", "-a", "colmap", "feature_extractor", "--database_path", db, "--image_path", images,
                    "--ImageReader.mask_path", masks, "--ImageReader.single_camera", "1",
                    "--ImageReader.camera_model", camera.model, "--ImageReader.camera_params", camera_values,
                    "--SiftExtraction.num_threads", "6", "--SiftExtraction.use_gpu", "1"],
                   env=colmap_env, check=True)
    with sqlite3.connect(db) as database:
        rows = database.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()
    if {name for _, name, _ in rows} != {record["name"] for record in records} or len({c for _, _, c in rows}) != 1:
        raise ValueError("COLMAP extraction dropped a frame or changed camera groups.")
    cam_id = rows[0][2]
    (known / "cameras.txt").write_text(
        f"{cam_id} PINHOLE {camera.width} {camera.height} {' '.join(map(str, camera.params))}\n")
    lines = []
    for image_id, name, camera_id in rows:
        image = original[name]
        pose = " ".join(map(str, (*image.qvec, *image.tvec)))
        lines.append(f"{image_id} {pose} {camera_id} {name}\n\n")
    (known / "images.txt").write_text("".join(lines))
    (known / "points3D.txt").touch()
    subprocess.run(["xvfb-run", "-a", "colmap", "sequential_matcher", "--database_path", db,
                    "--SequentialMatching.overlap", "12", "--SiftMatching.num_threads", "6",
                    "--SiftMatching.use_gpu", "1"], env=colmap_env, check=True)
    subprocess.run(["colmap", "point_triangulator", "--database_path", db, "--image_path", images,
                    "--input_path", known, "--output_path", sparse, "--Mapper.fix_existing_images", "1",
                    "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0"], check=True)
    cal, frames, points = read_colmap(args.gaussian_repo, sparse)
    if set(cal) != {cam_id} or not np.allclose(cal[cam_id].params, camera.params, rtol=0, atol=1e-9):
        raise RuntimeError("Triangulation changed camera intrinsics.")
    if {image.name for image in frames.values()} != {record["name"] for record in records}:
        raise RuntimeError("Triangulation dropped a registered training camera.")
    records_by_name = {record["name"]: record for record in records}
    observations = {}
    for image_id, image in frames.items():
        fixed = original[image.name]
        if not (np.allclose(fixed.qvec2rotmat(), image.qvec2rotmat(), atol=1e-7, rtol=0)
                and np.allclose(fixed.tvec, image.tvec, atol=1e-7, rtol=0)):
            raise RuntimeError(f"Triangulation changed fixed world camera {image.name}")
        observed, colors, _ = sample_observations(args.scene, records_by_name[image.name], image,
                                                   (camera.height, camera.width))
        person = cv2.imread(str(args.scene / "mask_undist/images" / image.name), 0)
        body = np.zeros_like(observed, dtype=bool)
        valid = observed == 3
        x, y = np.floor(image.xys[valid]).astype(int).T
        body[valid] = person[y, x] >= 224
        observed[body] = 4
        observations[image_id] = observed, colors
    xyz, rgb, body_count = [], [], 0
    for point in points.values():
        if not np.isfinite(point.xyz).all() or not np.isfinite(point.error) or point.error > 1.5:
            continue
        labels, support = [], []
        for image_id, index in zip(point.image_ids, point.point2D_idxs, strict=True):
            image = frames[image_id]
            if image.point3D_ids[index] != point.id:
                raise RuntimeError("Invalid triangulation track association.")
            flags, colors = observations[image_id]
            labels.append(flags[index])
            support.append(colors[index])
        if len(labels) < 3 or labels[0] not in (1, 4) or len(set(labels)) != 1:
            continue
        xyz.append(point.xyz)
        rgb.append(np.median(support, axis=0) / 255)
        body_count += labels[0] == 4
    if body_count < 30 or len(xyz) - body_count < 1000:
        raise RuntimeError(f"Too few independently triangulated scene or body seeds: {len(xyz)}, {body_count}")
    centers = np.array([row["position"] for row in cameras if row["img_name"] + ".jpg" not in holdout])
    extent = float(1.1 * np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())
    write_gaussians(args.output / "point_cloud.ply", args.gaussian_repo, np.asarray(xyz, np.float32),
                    np.asarray(rgb, np.float32), extent)
    summary = dict(holdout=sorted(holdout), scene_extent=extent, body_sparse_points=int(body_count),
                   room_sparse_points=len(xyz) - int(body_count), trained_steps=0, camera_count=len(cameras),
                   colmap_binary_sha256=digests, body_prior="fixed-camera RGB triangulation", head_regions_excluded=True)
    (args.output / "cameras.json").write_text(json.dumps(cameras, indent=2))
    (args.output / "initialization.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout", required=True, nargs="+")
    parser.add_argument("--gaussian-repo", type=Path, default=Path.home() / "gaussian-splatting")
    args = parser.parse_args()
    args.scene = args.scene.resolve(strict=True)
    cv2.setNumThreads(2)
    run(args)
