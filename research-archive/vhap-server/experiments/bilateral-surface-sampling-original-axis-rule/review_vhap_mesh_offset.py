"""Review learned VHAP mesh geometry at the exact native capture camera."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from review_smplx_head_neck import mesh_overlay, project


def as_array(values):
    return values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)


def quantiles(values):
    if len(values) == 0:
        return None
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(values, (0, .05, .5, .95, 1)).tolist()))


def vertex_normals(vertices, faces):
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(normals, corner, face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12), norm[:, 0] > 1e-12


def label(image, caption):
    result = cv2.copyMakeBorder(image, 27, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(result, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .44,
                (240, 240, 240), 1, cv2.LINE_AA)
    return result


def write_obj(path, vertices, faces):
    with path.open("w", encoding="ascii", newline="\n") as stream:
        np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")


def exported_pose_vertices(run, reference, registered, faces):
    joint = run / "joint"
    report = json.loads((joint / "metrics.json").read_bytes())
    if "mesh_offset" not in report:
        return registered
    if report["mesh_offset"]["obj_reference_frame"] != reference:
        raise ValueError("The exported mesh belongs to another pose")
    def read(version):
        path = joint / f"{Path(reference).stem}-mesh-{version}.obj"
        lines = path.read_text(encoding="ascii").splitlines()
        verts = np.loadtxt((line[2:] for line in lines if line.startswith("v ")))
        indices = np.loadtxt((line[2:] for line in lines if line.startswith("f ")), dtype=int)
        if verts.shape != registered.shape or not np.isfinite(verts).all() or not np.array_equal(indices - 1, faces):
            raise ValueError(f"Invalid mesh topology or geometry in {path}")
        return verts
    np.testing.assert_allclose(read("zero"), registered, rtol=0, atol=1e-5)
    return read("learned")


def outer_contour(mask, stop):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(outline, [max(contours, key=cv2.contourArea)], 0, 1, thickness=1)
    outline[stop:] = 0
    return outline.astype(bool)


def coarse_boundary(observed, silhouettes):
    """Compare the upper external boundary only because the lower mask cut is synthetic."""
    rows, cols = np.where(observed)
    if (len(rows) < 100 or rows.min() == 0 or cols.min() == 0 or cols.max() == observed.shape[1] - 1):
        return None
    stop = int(rows.min() + .65 * (rows.max() - rows.min() + 1))
    target = outer_contour(observed, stop)
    distance = lambda edge: cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    target_distance = distance(target)
    result = {"upper_rows": stop, "target_boundary_pixels": int(target.sum())}
    for name, silhouette in zip(("zero", "learned"), silhouettes):
        contour = outer_contour(silhouette, stop)
        if not contour.any() or not target.any():
            result[name] = None
            continue
        error = np.concatenate((target_distance[contour], distance(contour)[target]))
        result[name] = dict(symmetric_boundary_distance_px=quantiles(error),
                            mean_symmetric_boundary_distance_px=float(error.mean()),
                            upper_silhouette_iou=float((silhouette[:stop] & observed[:stop]).sum()
                                                       / (silhouette[:stop] | observed[:stop]).sum()))
    return result


@torch.no_grad()
def review_mesh_offset(output, frames, mesh):
    """Save native overlays and summarize geometric change by official FLAME region."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    if not frames:
        raise ValueError("Expected at least one fixed native review camera")
    obj_frame = next((frame.name for frame in frames if Path(frame.name).stem == "00560"), frames[0].name)
    faces = as_array(mesh.faces).astype(np.int64)
    regions = {name: as_array(mesh.region_mask(name)).astype(bool) for name in ("scalp", "neck")}
    editable = as_array(mesh.editable).astype(bool)
    scale = float(mesh.track_scale) if getattr(mesh, "track_scale", None) is not None else None
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("Expected a positive tracker-to-world scale")
    source = {
        "projection": "native camera with integer pixel centers, same unscaled crop in all panels",
        "signed_normal": "displacement dotted with the zero-offset, area-weighted FLAME winding normal",
        "binding_anchor": "arithmetic mean of each posed triangle's three vertices",
        "coarse_boundary_reference": "native RGB head support from person matte and coarse head mask; upper 65% excludes neck cut",
        "obj_reference_frame": obj_frame,
        "max_offset_tracker_units": float(mesh.max_offset),
        "tracker_to_world_scale": scale,
        "frames": {},
    }
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0:
        raise ValueError("Expected nonempty FLAME triangle indices")
    for frame in frames:
        if frame.name in source["frames"]:
            raise ValueError(f"Duplicate review frame: {frame.name}")
        before = as_array(mesh.vertices_world(frame, deformed=False)).astype(np.float64)
        after = as_array(mesh.vertices_world(frame, deformed=True)).astype(np.float64)
        if (before.ndim != 2 or before.shape[1] != 3 or before.shape != after.shape
                or not np.isfinite(before).all() or not np.isfinite(after).all()
                or faces.max() >= len(before) or editable.shape != (len(before),)
                or any(region.shape != editable.shape for region in regions.values())):
            raise ValueError(f"Invalid posed mesh or region masks: {frame.name}")

        K, (width, height), R, C = frame.rasterizer.calibration
        cameras = [(vertices - C) @ R for vertices in (before, after)]
        xy = [project(camera, K)[0] for camera in cameras]
        valid = np.isfinite(xy[0]).all(axis=1) & np.isfinite(xy[1]).all(axis=1)
        if not valid.all():
            raise ValueError(f"A review mesh vertex projects behind its native camera: {frame.name}")
        pixels = np.concatenate(xy, axis=0)
        lo = np.floor(pixels.min(axis=0)).astype(int) - 14
        hi = np.ceil(pixels.max(axis=0)).astype(int) + 15
        sy, sx = frame.crop
        lo = np.maximum(0, np.minimum(lo, [sx.start, sy.start]))
        hi = np.minimum([width, height], np.maximum(hi, [sx.stop, sy.stop]))
        if np.any(hi <= lo):
            raise ValueError(f"Projected review is outside its camera: {frame.name}")
        x0, y0 = lo.tolist()
        x1, y1 = hi.tolist()
        projection = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ K
        rgb = as_array(frame.reference[:, y0:y1, x0:x1])
        bgr = np.ascontiguousarray(np.rint(np.clip(rgb[::-1], 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0))
        black = np.full_like(bgr, 18)
        overlays = [mesh_overlay(bgr, camera, projection, faces) for camera in cameras]
        # A per-face tint limits the rasterizer's outline to inside the true mesh coverage.
        tint = np.full((len(faces), 3), 230, np.uint8)
        shaded = [mesh_overlay(np.zeros_like(black), camera, projection, faces, tint=tint) for camera in cameras]
        upper = np.hstack((label(bgr, f"Source {frame.name}"),
                           label(overlays[0], "Zero offset | overlay"), label(overlays[1], "Learned offset | overlay")))
        lower = np.hstack((label(black, "Same native camera"),
                           label(shaded[0], "Zero offset | mesh"), label(shaded[1], "Learned offset | mesh")))
        name = f"{Path(frame.name).stem}-mesh-offset.jpg"
        if not cv2.imwrite(str(output / name), np.vstack((upper, lower)), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write mesh review {output / name}")
        if frame.name == obj_frame:
            for version, vertices in zip(("zero", "learned"), (before, after)):
                write_obj(output / f"{Path(frame.name).stem}-mesh-{version}.obj", vertices, faces)

        delta = after - before
        normals, normal_valid = vertex_normals(before, faces)
        signed = np.einsum("vi,vi->v", delta, normals)
        magnitude = np.linalg.norm(delta, axis=1)
        projection_magnitude = np.linalg.norm(xy[1] - xy[0], axis=1)
        report = {
            "preview": name, "crop_xyxy": [x0, y0, x1, y1],
            "frozen_vertex_max_displacement_world": float(np.max(magnitude[~editable])) if (~editable).any() else None,
            "triangle_anchor_displacement_world": quantiles(np.linalg.norm(delta[faces].mean(axis=1), axis=1)),
            "regions": {},
        }
        support = getattr(frame, "masks", {}).get("head_rgb")
        if support is not None:
            observed = as_array(support[y0:y1, x0:x1]).astype(bool)
            report["coarse_upper_boundary"] = coarse_boundary(observed, [(item > 0).any(axis=2) for item in shaded])
        for region, mask in regions.items():
            selected = mask & editable
            signed_selected = signed[selected & normal_valid]
            report["regions"][region] = {
                "vertices": int(mask.sum()), "editable_vertices": int(selected.sum()),
                "valid_normal_vertices": len(signed_selected),
                "signed_normal_displacement_world": quantiles(signed_selected),
                "signed_normal_displacement_tracker": quantiles(signed_selected / scale) if scale else None,
                "vertex_displacement_world": quantiles(magnitude[selected]),
                "projected_shift_px": quantiles(projection_magnitude[selected]),
            }
        source["frames"][frame.name] = report
    (output / "mesh-offset-review.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return source
