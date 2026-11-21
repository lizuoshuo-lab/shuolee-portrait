"""Compare neck-inclusive meshes under the exact same registered source camera."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def project(camera, projection):
    homogeneous = camera @ projection.T
    xy = np.full_like(homogeneous[:, :2], np.nan)
    np.divide(homogeneous[:, :2], homogeneous[:, 2:], out=xy, where=homogeneous[:, 2:] > 1e-6)
    return xy, homogeneous[:, 2]


def mesh_overlay(image, camera, projection, faces, tint=(246, 210, 87)):
    """Rasterize visible projected triangles with perspective-correct depth."""
    height, width = image.shape[:2]
    xy, depth = project(camera, projection)
    tri = xy[faces]
    p0, p1, p2 = tri.transpose(1, 0, 2)
    u, v = p1 - p0, p2 - p0
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    bounds = np.array([width - 1, height - 1])
    visible = ((hi >= 0).all(axis=1) & (lo <= bounds).all(axis=1) & (np.abs(det) > 1e-8))
    ids = np.flatnonzero(visible)
    start = np.ceil(np.maximum(lo[ids], 0)).astype(int)
    stop = np.floor(np.minimum(hi[ids], bounds)).astype(int)
    dims = np.maximum(stop - start + 1, 0)
    counts = dims.prod(axis=1)
    prefix = np.r_[0, counts.cumsum()]
    z_buffer = np.full(height * width, -np.inf)
    owner = np.full(height * width, -1, dtype=np.int32)
    reciprocal = 1 / np.maximum(depth[faces], 1e-6)

    offset = 0
    while offset < len(ids):
        end = max(offset + 1, np.searchsorted(prefix, prefix[offset] + 180_000, side="right") - 1)
        indices = np.repeat(np.arange(offset, end), counts[offset:end])
        relative = np.arange(prefix[end] - prefix[offset]) - (prefix[indices] - prefix[offset])
        x = start[indices, 0] + relative % dims[indices, 0]
        y = start[indices, 1] + relative // dims[indices, 0]
        face = ids[indices]
        dx, dy = x - p0[face, 0], y - p0[face, 1]
        b = (dx * v[face, 1] - dy * v[face, 0]) / det[face]
        c = (u[face, 0] * dy - u[face, 1] * dx) / det[face]
        a = 1 - b - c
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        pixel, face = (x[inside] + y[inside] * width), face[inside]
        z = np.einsum("ij,ij->i", np.column_stack((a[inside], b[inside], c[inside])), reciprocal[face])
        np.maximum.at(z_buffer, pixel, z)
        nearest = z == z_buffer[pixel]
        owner[pixel[nearest]] = face[nearest]
        offset = end

    owner = owner.reshape(height, width)
    inside = owner >= 0
    xyz = camera[faces]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-8)
    light = .48 + .52 * np.abs(normals @ np.array([-.23, -.28, .93])).clip(0, 1)
    result = image.copy()
    tint = np.asarray(tint)
    foreground_tint = tint if tint.ndim == 1 else tint[owner[inside]]
    result[inside] = np.rint(image[inside] * .52 + foreground_tint * light[owner[inside], None] * .48).astype(np.uint8)
    edge = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if tint.ndim == 1:
        result[edge > 0] = tint
    else:
        part = np.zeros_like(owner, dtype=np.uint8)
        part[inside] = np.unique(tint, axis=0, return_inverse=True)[1][owner[inside]] + 1
        border = inside & (cv2.morphologyEx(part, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
        result[border] = tint[owner[border]]
    return result


def panel(image, label):
    frame = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (383, 27), (21, 21, 21), -1)
    cv2.putText(frame, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .49, (248, 248, 248), 1, cv2.LINE_AA)
    return frame


def run(args):
    source = json.loads((args.scene / "source-frames.json").read_text(encoding="utf-8"))["frames"]
    records = json.loads((args.sequence / "initialization.json").read_text(encoding="utf-8"))["frames"]
    names = [row["name"] for row in records]
    with np.load(args.flame_world, allow_pickle=False) as old, np.load(args.smplx_frames, allow_pickle=False) as new:
        if any(names != observed for observed in ([row["name"] for row in source], old["names"].tolist(), new["names"].tolist())):
            raise ValueError("The registered images and the two meshes must share the same frame order")
        flame, flame_faces = old["vertices_world"], old["faces"]
        smpl, smpl_faces = new["vertices_world"], new["faces"]
        if int(new["neck_joint_index"]) != 12:
            raise ValueError("Expected native SMPL-X neck joint 12")
        joints, K = new["joints_world"][:, 12], np.asarray(new["native_K"])
        rotation, center = new["native_camera_to_world_rotation"], new["native_camera_center"]

    example = cv2.imread(str(args.sequence / "images/00000000.jpg"))
    if example is None:
        raise FileNotFoundError("Missing first neck-inclusive source crop")
    size = example.shape[1::-1]
    fixed = [r["name"] for r in records if abs(r["baseline_yaw_deg"]) >= 80.][:4]
    if len(fixed) != 4:
        raise ValueError("Expected four views at or beyond 80 degrees in this review")
    args.output.mkdir(parents=True, exist_ok=False)
    video_path = args.output / "all-frames.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
               "-video_size", "1152x384", "-framerate", "3", "-i", "-", "-c:v", "libx264", "-preset", "fast",
               "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)]
    sheet, details = [], []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as video:
        try:
            for i, (row, src) in enumerate(zip(records, source, strict=True)):
                name = row["name"]
                path = args.scene / "orig_undist/images" / name
                if Path(name).name != name or sha256(path.read_bytes()).hexdigest() != src["native_sha256"]:
                    raise ValueError(f"Registered image changed: {name}")
                original = cv2.imread(str(path))
                if original is None:
                    raise OSError(f"Cannot decode {path}")
                warp = np.asarray(row["warp"])
                projection = warp @ K
                virtual = np.asarray(row["virtual_from_native_cv"])
                if (not np.allclose(projection, np.asarray(row["K_integer_pixel_centers"]) @ virtual, atol=1e-6, rtol=1e-6)
                        or not np.allclose(rotation[i] @ virtual.T, row["world_from_virtual"], atol=1e-6)
                        or not np.allclose(center[i], row["camera_center_world"], atol=1e-6)):
                    raise ValueError(f"Camera or crop is inconsistent at {name}")
                crop = cv2.warpPerspective(original, warp, size, flags=cv2.INTER_LINEAR)
                previous = mesh_overlay(crop, (flame[i] - center[i]) @ rotation[i], projection, flame_faces)
                current = mesh_overlay(crop, (smpl[i] - center[i]) @ rotation[i], projection, smpl_faces)
                xy, depth = project((joints[i:i + 1] - center[i]) @ rotation[i], projection)
                p = xy[0]
                in_crop = (depth[0] > 0 and np.isfinite(p).all() and (p >= 0).all() and (p < size).all())
                if in_crop:
                    center_px = tuple(np.rint(p).astype(int))
                    cv2.circle(current, center_px, 7, (17, 17, 17), 2, cv2.LINE_AA)
                    cv2.circle(current, center_px, 5, (40, 150, 250), -1, cv2.LINE_AA)
                line = np.hstack((panel(crop, f"{name} | original ({row['baseline_yaw_deg']:+.0f} deg)"),
                                  panel(previous, "Previous FLAME"), panel(current, "SMPL-X | dot: neck joint 12")))
                video.stdin.write(line.tobytes())
                if name in fixed:
                    sheet.append(line)
                details.append({"name": name, "baseline_yaw_deg": row["baseline_yaw_deg"],
                                "neck_joint_crop_xy": p.tolist() if np.isfinite(p).all() else None,
                                "neck_joint_in_crop": bool(in_crop)})
        finally:
            video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the mesh review")
    if not cv2.imwrite(str(args.output / "profile-comparison.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not save the side-view comparison")
    report = {"frames": details, "profile_views": fixed, "neck_joint_index": 12,
              "camera": "shared registered PINHOLE native rays warped into the approved virtual crop",
              "visibility": "mesh z-buffer; skeleton marker denotes projection even when internal"}
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("scene", "sequence", "flame-world", "smplx-frames", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
