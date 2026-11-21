"""Seed a static clothed-body layer from registered SMPL-X and training RGB."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from numba import njit


@njit
def depth_buffer(pixels, depth, faces, height, width):
    """Rasterize the mesh at integer pixel centers, interpolating reciprocal depth."""
    inverse_depth = np.zeros((height, width), np.float32)
    for i, j, k in faces:
        if min(depth[i], depth[j], depth[k]) <= 0:
            continue
        x0, y0 = pixels[i]
        x1, y1 = pixels[j]
        x2, y2 = pixels[k]
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-9:
            continue
        left = max(0, int(np.ceil(min(x0, x1, x2))))
        right = min(width - 1, int(np.floor(max(x0, x1, x2))))
        top = max(0, int(np.ceil(min(y0, y1, y2))))
        bottom = min(height - 1, int(np.floor(max(y0, y1, y2))))
        z0, z1, z2 = 1 / depth[i], 1 / depth[j], 1 / depth[k]
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                b = ((x - x0) * (y2 - y0) - (y - y0) * (x2 - x0)) / area
                c = ((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / area
                a = 1 - b - c
                if min(a, b, c) >= -1e-7:
                    d = a * z0 + b * z1 + c * z2
                    if d > inverse_depth[y, x]:
                        inverse_depth[y, x] = d
    return inverse_depth


def project(camera_xyz, K):
    homogeneous = camera_xyz @ K.T
    denominator = homogeneous[:, 2:3]
    return homogeneous[:, :2] / np.where(np.abs(denominator) > 1e-8, denominator, 1e-8)


def read_view(scene, frame):
    name = frame["name"]
    source = (scene / "orig_undist/images" / name).read_bytes()
    if sha256(source).hexdigest() != frame["native_sha256"]:
        raise ValueError(f"Native image differs from the recorded source: {name}")
    image = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    person = cv2.imread(str(scene / "mask_undist/images" / name), cv2.IMREAD_GRAYSCALE)
    head = cv2.imread(str(scene / "headmask_undist/images" / Path(name).with_suffix(".png")),
                      cv2.IMREAD_GRAYSCALE)
    if (image is None or person is None or head is None or person.shape != image.shape[:2]
            or head.shape != person.shape or person.max() < 224 or head.max() <= 128):
        raise ValueError(f"Invalid aligned RGB, person mask or head mask: {name}")
    kernel = np.ones((3, 3), np.uint8)
    allowed = cv2.erode((person >= 224).astype(np.uint8), kernel)
    allowed &= 1 - cv2.dilate((head > 128).astype(np.uint8), kernel)
    return image, allowed


def initialize(fit_path, scene, holdout, count):
    manifest = json.loads((scene / "source-frames.json").read_text(encoding="utf-8"))["frames"]
    names = [record["name"] for record in manifest]
    if len(set(names)) != len(names) or len(set(holdout)) != len(holdout) or not set(holdout) < set(names):
        raise ValueError("Expected unique native source names and recorded holdouts with training frames left")
    with np.load(fit_path, allow_pickle=False) as fit:
        if fit["names"].tolist() != names:
            raise ValueError("SMPL-X and the source raster must identify the same ordered frames")
        vertices = fit["vertices_world"]
        joints = fit["joints_world"]
        faces = fit["faces"].astype(np.int32)
        K = fit["native_K"]
        rotations = fit["native_camera_to_world_rotation"]
        centers = fit["native_camera_center"]
        neck, head = int(fit["neck_joint_index"]), int(fit["head_joint_index"])
    train = [i for i, name in enumerate(names) if name not in holdout]
    # A medoid is an actual observed pose. Use body joints 0–21, leaving fingers out.
    skeletons = joints[train, :22].astype(np.float64)
    differences = skeletons[:, None] - skeletons[None, :]
    pose_distances = np.sqrt(np.square(differences).sum(axis=-1).mean(axis=-1))
    representative = train[int(np.argmin(np.median(pose_distances, axis=1)))]
    triangles = vertices[representative, faces]
    neck_direction = joints[representative, head] - joints[representative, neck]
    # Remove head/upper neck by anatomy even where a coarse mask misses their outline.
    below_neck = ((triangles - joints[representative, neck]) @ neck_direction <= 0).all(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    weights = np.linalg.norm(normals, axis=1) * below_neck
    if count < 1 or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("No finite body surface below the registered neck")
    rng = np.random.default_rng(0)
    sampled_faces = faces[rng.choice(len(faces), size=count, p=weights / weights.sum())]
    bary = rng.random((count, 2)).astype(np.float32)
    bary = np.where(bary.sum(axis=1, keepdims=True) > 1, 1 - bary, bary)
    bary = np.column_stack((bary, 1 - bary.sum(axis=1)))[:, :, None]
    xyz = (vertices[representative, sampled_faces] * bary).sum(axis=1)
    best = np.zeros(count)
    views = np.zeros(count, np.int32)
    rgb = np.zeros((count, 3), np.uint8)
    for i in train:
        image, allowed = read_view(scene, manifest[i])
        height, width = allowed.shape
        camera_vertices = (vertices[i] - centers[i]) @ rotations[i]
        mesh_xy = project(camera_vertices, K)
        buffer = depth_buffer(mesh_xy, camera_vertices[:, 2], faces, height, width)
        triangle = camera_vertices[sampled_faces]
        sample = (triangle * bary).sum(axis=1)
        pixels = np.rint(project(sample, K)).astype(np.int32)
        x, y = pixels.T
        eligible = np.flatnonzero((sample[:, 2] > 0) & (x >= 0) & (x < width)
                                  & (y >= 0) & (y < height))
        xx, yy = x[eligible], y[eligible]
        closest = buffer[yy, xx]
        depth_ratio = sample[eligible, 2] * closest
        visible = (closest > 0) & (np.abs(depth_ratio - 1) <= 0.003) & (allowed[yy, xx] > 0)
        observed = eligible[visible]
        surface = triangle[observed]
        normal = np.cross(surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0])
        ray = -sample[observed]
        facing = np.sum(normal * ray, axis=1) / np.maximum(
            np.linalg.norm(normal, axis=1) * np.linalg.norm(ray, axis=1), 1e-12)
        oriented = facing > 0
        observed, facing = observed[oriented], facing[oriented]
        views[observed] += 1
        improved = facing > best[observed]
        updated = observed[improved]
        best[updated] = facing[improved]
        rgb[updated] = image[y[updated], x[updated], ::-1]
    keep = views > 0
    if not keep.any():
        raise ValueError("No body samples were visible within training RGB support")
    return {
        "xyz": xyz[keep].astype(np.float32), "rgb": rgb[keep], "support_count": views[keep],
        "representative_frame": np.array(names[representative]),
        "training_frames": np.array([names[i] for i in train]), "holdout_frames": np.array(holdout),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--smplx", type=Path, required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = initialize(args.smplx, args.scene, args.holdout, args.points)
    with args.output.open("xb") as output:
        np.savez_compressed(output, **result)
    print(json.dumps({"output": str(args.output), "points": len(result["xyz"]),
                      "representative_frame": result["representative_frame"].item(),
                      "minimum_rgb_views": int(result["support_count"].min()),
                      "median_rgb_views": float(np.median(result["support_count"]))}))


if __name__ == "__main__":
    main()
