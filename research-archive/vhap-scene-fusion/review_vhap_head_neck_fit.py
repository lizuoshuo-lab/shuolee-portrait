"""Review a native VHAP head and neck fit with paired cameras and source crops."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import cv2
import numpy as np

from review_smplx_head_neck import mesh_overlay, panel, project


def load_meshes(before, after, records):
    names = [r["name"] for r in records]
    keys = ("native_K", "native_camera_to_world_rotation", "native_camera_center", "virtual_crop_homographies")
    shapes = ((3, 3), (len(names), 3, 3), (len(names), 3), (len(names), 3, 3))
    arrays = [before[key] for key in keys]
    for key, value, shape in zip(keys, arrays, shapes, strict=True):
        other = after[key]
        if (value.shape != shape or other.shape != shape or not np.isfinite(value).all()
                or not np.allclose(value, other, rtol=0, atol=1e-8)):
            raise ValueError(f"The two meshes disagree on the registered camera: {key}")
    if any(a["names"].tolist() != names for a in (before, after)):
        raise ValueError("The two meshes and crop archive must have the same frame order")
    vertices = [a["vertices_world"] for a in (before, after)]
    faces = before["faces"]
    if (vertices[0].shape != vertices[1].shape or vertices[0].shape[0] != len(names)
            or any(v.ndim != 3 or v.shape[2] != 3 or not np.isfinite(v).all() for v in vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= vertices[0].shape[1]
            or not np.array_equal(faces, after["faces"])):
        raise ValueError("Expected a shared topology and finite vertices for the paired fit")
    return names, vertices, faces, arrays


def static_frames(records):
    profiles = [r["name"] for r in records if abs(r["baseline_yaw_deg"]) >= 80][:4]
    front = [r for r in records if abs(r["baseline_yaw_deg"]) <= 12]
    frontal = []
    for group in ("train", "holdout"):
        choices = [r for r in front if r.get(group)]
        if choices:
            frontal.append(min(choices, key=lambda r: abs(r["baseline_yaw_deg"]))["name"])
    for r in sorted(front, key=lambda r: abs(r["baseline_yaw_deg"])):
        if len(frontal) == 2:
            break
        if r["name"] not in frontal:
            frontal.append(r["name"])
    if len(profiles) != 4 or len(frontal) != 2:
        raise ValueError("Expected four pure side frames and two frontal comparisons")
    return profiles, frontal


def contour_measurements(points, camera, K, H, size, edges, anatomical):
    """Measure the full mesh's left/right extremum at an observed native scanline."""
    if not points:
        return []
    xy, depth = project(camera, K)
    a, b = xy[edges].transpose(1, 0, 2)
    good = (depth[edges] > 1e-6).all(axis=1)
    measurements = []
    for point in points:
        x, y = point["native_xy"]
        side = point["side"]
        if side not in ("left", "right") or not np.isfinite([x, y]).all():
            raise ValueError("Invalid native contour annotation")
        if anatomical is None:
            measurements.append({"status": "mesh_has_no_anatomical_neck_ids", "error_px": None})
            continue
        active = np.flatnonzero(good & ((a[:, 1] > y) != (b[:, 1] > y)))
        if not len(active):
            measurements.append({"status": "no_visible_mesh_edge", "error_px": None})
            continue
        t = (y - a[active, 1]) / (b[active, 1] - a[active, 1])
        intersection = a[active, 0] + t * (b[active, 0] - a[active, 0])
        selected = int(np.argmax(intersection) if side == "right" else np.argmin(intersection))
        value = intersection[selected]
        crop = cv2.perspectiveTransform(np.array([[[x, y], [value, y]]], dtype=np.float64), H)[0]
        if not ((crop > 0) & (crop < np.asarray(size) - 1)).all():
            status = "edge_outside_review_crop"
        else:
            status = "measured" if anatomical[active[selected]] else "visible_edge_is_not_anatomical_neck"
        measurements.append({"status": status, "error_px": abs(float(value - x)) if status == "measured" else None})
    return measurements


def summarize(observations):
    metrics = {"target_count": len(observations)}
    for condition in ("before", "after"):
        metrics[f"{condition}_valid_count"] = sum(r[condition]["error_px"] is not None for r in observations)
    paired = [r for r in observations if r["before"]["error_px"] is not None and r["after"]["error_px"] is not None]
    metrics["paired_count"] = len(paired)
    for condition in ("before", "after"):
        metrics[f"{condition}_median_paired_px"] = (float(np.median([r[condition]["error_px"] for r in paired]))
                                                     if paired else None)
    return metrics


def run(args):
    with ZipFile(args.frames) as archive, np.load(args.before, allow_pickle=False) as before, np.load(args.after, allow_pickle=False) as after:
        records = json.loads(archive.read("initialization.json"))["frames"]
        names, vertices, faces, (K, rotations, centers, homographies) = load_meshes(before, after, records)
        profiles, frontal = static_frames(records)
        edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
        anatomies = []
        for mesh in (before, after):
            if "neck_vertex_ids" not in mesh:
                anatomies.append(None)
                continue
            anatomy = np.setdiff1d(mesh["neck_vertex_ids"], mesh.get("neck_boundary_vertex_ids", []))
            anatomies.append(np.isin(edges, anatomy).all(axis=1))
        detected = {}
        if args.neck_observations:
            annotations = json.loads(args.neck_observations.read_text(encoding="utf-8"))
            detected = annotations["frames"]
            if (annotations["schema"] != "vhap-neck-targets-v1" or set(detected) != set(names)
                    or annotations["source"]["pixel_convention"] != "pixel_integer_centers"
                    or annotations["source"]["side_convention"] != "native_image_left_and_right"
                    or any(not np.isclose(detected[r["name"]]["timestamp_s"], r["timestamp_s"], rtol=0, atol=1e-8)
                           for r in records)):
                raise ValueError("Neck observations do not match the native frame names and times")

        args.output.mkdir(parents=True, exist_ok=False)
        command = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                   "-pixel_format", "bgr24", "-video_size", "1152x384", "-framerate", "3", "-i", "-",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags",
                   "+faststart", str(args.output / "all-frames.mp4")]
        sheet, measurements = {}, []
        min_depth = np.inf
        with subprocess.Popen(command, stdin=subprocess.PIPE) as video:
            try:
                for i, (name, record) in enumerate(zip(names, records, strict=True)):
                    H = homographies[i]
                    V = np.asarray(record["virtual_from_native_cv"])
                    projection = H @ K
                    if (not np.allclose(H, record["warp"], rtol=0, atol=1e-8)
                            or not np.allclose(projection, np.asarray(record["K_integer_pixel_centers"]) @ V, atol=1e-6)
                            or not np.allclose(rotations[i] @ V.T, record["world_from_virtual"], atol=1e-6)
                            or not np.allclose(centers[i], record["camera_center_world"], atol=1e-6)):
                        raise ValueError(f"Camera and crop disagree at {name}")
                    rgb = cv2.imdecode(np.frombuffer(archive.read(f"images/{i:08d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
                    if rgb is None:
                        raise ValueError(f"Cannot read source crop {name}")
                    cameras = [(mesh[i] - centers[i]) @ rotations[i] for mesh in vertices]
                    depth = min(float((camera @ projection[2]).min()) for camera in cameras)
                    if depth <= 1e-6:
                        raise ValueError(f"Head and neck vertices behind the shared crop camera: {name}")
                    min_depth = min(min_depth, depth)
                    overlays = [mesh_overlay(rgb, camera, projection, faces) for camera in cameras]
                    row = np.hstack((panel(rgb, f"{name} | original ({record['baseline_yaw_deg']:+.0f} deg)"),
                                     panel(overlays[0], "Native VHAP | before"), panel(overlays[1], "Native VHAP | after")))
                    video.stdin.write(row.tobytes())
                    if name in profiles or name in frontal:
                        sheet[name] = row
                    points = detected.get(name, {}).get("points", [])
                    if points:
                        measures = [contour_measurements(points, camera, K, H, rgb.shape[1::-1], edges, region)
                                    for camera, region in zip(cameras, anatomies, strict=True)]
                        measurements.extend({"name": name, "native_xy": p["native_xy"], "side": p["side"],
                                             "source_split": p["split"], "before": a, "after": b}
                                            for p, a, b in zip(points, *measures, strict=True))
            finally:
                video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("FFmpeg could not encode the paired VHAP review")

    for filename, selected in (("profile-comparison.jpg", profiles), ("frontal-comparison.jpg", frontal)):
        if not cv2.imwrite(str(args.output / filename), np.vstack([sheet[n] for n in selected]), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Cannot write {filename}")
    report = {"frame_count": len(names), "fps": 3, "profile_frames": profiles, "frontal_frames": frontal,
              "camera_proof": "both meshes and source archive agree; shared native K, pose, and virtual homography",
              "minimum_virtual_depth_world_units": min_depth, "shared_topology": True}
    if detected:
        report["neck_contour"] = {"domain": "same-video native pixel scanlines; evaluation measurements",
                                  "all": summarize(measurements),
                                  "profiles": summarize([r for r in measurements if r["name"] in profiles]),
                                  "points": measurements}
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("before", "after", "frames", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--neck-observations", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    cv2.setNumThreads(1)
    run(parser.parse_args())
