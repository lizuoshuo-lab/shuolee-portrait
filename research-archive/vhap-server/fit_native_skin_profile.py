"""Test one fixed-camera FLAME translation against visible skin profile contours."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from scipy.optimize import least_squares


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def camera(record):
    virtual = np.asarray(record["virtual_from_native_cv"], dtype=np.float64)
    r = np.asarray(record["world_from_virtual"], dtype=np.float64) @ virtual
    k = np.linalg.solve(np.asarray(record["warp"]), np.asarray(record["K_integer_pixel_centers"]) @ virtual)
    k /= k[2, 2]
    c = np.asarray(record["camera_center_world"], dtype=np.float64)
    if (not np.isfinite([*k.ravel(), *r.ravel(), *c]).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1.)
            or not np.allclose(k[[0, 1, 2, 2], [1, 0, 0, 1]], 0, atol=1e-7)
            or min(k[0, 0], k[1, 1]) <= 0):
        raise ValueError(f"Invalid recorded native camera: {record['name']}")
    return k, r, c


def project(vertices, k, r, c):
    xyz = (vertices - c) @ r
    if not np.isfinite(xyz).all() or (xyz[:, 2] <= 0).any():
        raise ValueError("FLAME mesh projects behind its registered camera")
    pixels = xyz @ k.T
    return pixels[:, :2] / pixels[:, 2:3], xyz[:, 2]


def alpha_points(alpha, skin, excluded, origin, side):
    """Use both a native person-matte crossing and visible-skin semantic support."""
    points = []
    for y in range(3, alpha.shape[0] - 3):
        row = alpha[y]
        foreground = np.flatnonzero(row >= .5)
        if not len(foreground):
            continue
        x = foreground[-1] if side == 1 else foreground[0]
        if x < 6 or x >= len(row) - 6 or row[x] - row[x + side] <= 1e-5:
            continue
        if excluded[y - 2:y + 3, x - 2:x + 3].any():
            continue
        inside = x - side * np.arange(1, 5)
        outside = x + side * np.arange(2, 6)
        if not (skin[y - 1:y + 2, inside].mean() >= .5 and (row[outside] < .1).all()):
            continue
        points.append((x + side * (row[x] - .5) / (row[x] - row[x + side]), y))
    xy = np.asarray(points, dtype=float).reshape(-1, 2) + origin
    if len(xy) < 3:
        return xy, np.ones(len(xy))
    normal_x = 1 / np.sqrt(1 + np.gradient(xy[:, 0], xy[:, 1])**2)
    okay = normal_x >= .35
    return xy[okay], normal_x[okay]


def mesh_intersections(planes, depth, vertices, edges, eligible, rows, side):
    """Return the visible silhouette extremum along each source image row."""
    a, b = planes[edges[:, 0]], planes[edges[:, 1]]
    low, high = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    out_x = np.full(len(rows), np.nan)
    material = np.full((len(rows), 3), np.nan)
    for i, y in enumerate(rows):
        crossing = np.flatnonzero((low <= y) & (y < high))
        if not len(crossing):
            continue
        p, q = a[crossing], b[crossing]
        t = (y - p[:, 1]) / (q[:, 1] - p[:, 1])
        x = p[:, 0] + t * (q[:, 0] - p[:, 0])
        j = int(np.argmax(side*x))
        edge = crossing[j]
        if not eligible[edge]:
            continue
        endpoints = edges[edge]
        w = np.array([1 - t[j], t[j]]) / depth[endpoints]
        material[i] = w @ vertices[endpoints] / w.sum()
        out_x[i] = x[j]
    return out_x, material


def face_edges(faces, mask_path):
    with mask_path.open("rb") as handle:
        regions = pickle.load(handle, encoding="latin1")
    included = np.concatenate([regions[name] for name in ("face", "nose", "lips", "forehead")])
    excluded = np.concatenate([regions[name] for name in ("neck", "left_ear", "right_ear", "boundary")])
    skin = np.setdiff1d(included, excluded)
    edges = np.sort(np.concatenate([faces[:, pair] for pair in ([0, 1], [1, 2], [2, 0])]), axis=1)
    edges = np.unique(edges, axis=0)
    return edges, np.isin(edges, skin).all(axis=1)


def prepare_view(record, vertices, path, edges, eligible):
    with np.load(path, allow_pickle=False) as data:
        alpha, skin, excluded = (data[key] for key in ("alpha", "skin", "excluded"))
        origin = data["origin_xy"]
        colors = data["bgr"] if "bgr" in data.files else None
        for name in ("nose", "mouth"):
            if name in data.files:
                skin = skin.astype(bool) | data[name].astype(bool)
        ear_pixels = int(np.count_nonzero(data["ear"])) if "ear" in data.files else 0
    has_rgb = colors is not None
    if (alpha.ndim != 2 or not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1
            or skin.shape != alpha.shape or excluded.shape != alpha.shape or origin.shape != (2,)
            or (origin < 0).any() or not np.array_equal(origin, origin.astype(int))
            or (colors is not None and (colors.shape != (*alpha.shape, 3) or colors.dtype != np.uint8))):
        raise ValueError(f"Invalid native crop channels or origin in {path}")
    side = 1 if record["baseline_yaw_deg"] < 0 else -1
    k, r, c = camera(record)
    pixels, depth = project(vertices, k, r, c)
    xy, weight = alpha_points(alpha, skin.astype(bool), excluded.astype(bool), origin, side)
    if colors is None:
        colors = cv2.cvtColor(np.uint8(np.rint(alpha*255)), cv2.COLOR_GRAY2BGR)
    view = dict(xy=xy, held=np.zeros(len(xy), dtype=bool), geometry=vertices,
                k=k, r=r, c=c, colors=colors, origin=origin)
    x, material = mesh_intersections(pixels, depth, vertices, edges, eligible, xy[:, 1], side)
    gap = abs(x - xy[:, 0])
    valid = np.isfinite(gap) & (gap <= 24)
    detail = dict(stem=Path(record["name"]).stem, visible_side="right" if side == 1 else "left", side=side,
                  yaw=float(record["baseline_yaw_deg"]), observed=len(xy), compatible=int(valid.sum()),
                  ear_semantic_pixels=ear_pixels, source_sha256=file_hash(path), accepted_for_analysis=False)
    view["detail"] = detail
    if valid.sum() < 12:
        detail["reason"] = "fewer than 12 skin crossings with compatible visible FLAME anatomy"
        return detail, view
    xy, weight, material, x = xy[valid], weight[valid], material[valid], x[valid]
    lo, hi = np.quantile(xy[:, 1], [.35, .65])
    held = (xy[:, 1] >= lo) & (xy[:, 1] <= hi)
    if held.sum() < 3 or (~held).sum() < 8 or hi - lo < 3:
        detail["reason"] = "insufficient separate fitting and held contour arcs"
        return detail, view
    if has_rgb:
        gray = cv2.cvtColor(colors, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gradient = abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.125))
        ij = np.rint(xy - origin).astype(int)
        detail["median_native_rgb_edge"] = float(np.median([gradient[j, i-2:i+3].max() for i,j in ij]))
    detail["accepted_for_analysis"] = True
    detail["held_arc_rows_native"] = [float(lo), float(hi)]
    view.update(xy=xy, weight=weight, points=material, original_x=x, held=held)
    return detail, view


def split_frames(details, fit, holdout):
    listed = fit is not None or holdout is not None
    if listed:
        if not fit or not holdout:
            raise ValueError("Specify both fit and holdout frame lists")
        fit, holdout = (set(Path(name).stem for name in names) for names in (fit, holdout))
        if fit & holdout:
            raise ValueError("A profile frame cannot be used for both fitting and holdout")
    else:
        fit, holdout = set(), set()
        for side in (-1, 1):
            names = [x["stem"] for x in details if x["side"] == side]
            fit.update(name for i, name in enumerate(names) if i % 3 != 1)
            holdout.update(name for i, name in enumerate(names) if i % 3 == 1)
    return fit, holdout


def save_panel(path, view, candidate_vertices, faces):
    shape = view["colors"].shape[:2]
    rows = []
    estimate = "Candidate FLAME" if candidate_vertices is not None else "No candidate for this frame"
    for caption, vertices in (("Native skin support", None), ("Original FLAME", view["geometry"]),
                              (estimate, candidate_vertices)):
        rgb = view["colors"].copy()
        if vertices is not None:
            pixels, _ = project(vertices, view["k"], view["r"], view["c"])
            pixels = np.rint(pixels - view["origin"]).astype(np.int32)
            mask = np.zeros(shape, np.uint8)
            for tri in pixels[faces]:
                cv2.fillConvexPoly(mask, tri, 255)
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contour, -1, (255, 255, 0), 1, cv2.LINE_AA)
        for point, held in zip(view["xy"] - view["origin"], view["held"], strict=True):
            cv2.circle(rgb, tuple(np.rint(point).astype(int)), 1, (0, 190, 255) if held else (0, 220, 0), -1)
        rgb = cv2.copyMakeBorder(rgb, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(rgb, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1)
        rows.append(rgb)
    if not cv2.imwrite(str(path), np.hstack(rows)):
        raise OSError(f"Could not write {path}")


def run(args):
    context = json.loads(args.initialization.read_bytes())
    records = context["frames"]
    with np.load(args.registered, allow_pickle=False) as data:
        if (data["names"].tolist() != [Path(x["name"]).stem for x in records]
                or data["initialization_sha256"].item() != file_hash(args.initialization)):
            raise ValueError("Recorded mesh and native crop-camera registration differ")
        transform = data["tracker_to_world_rotation"]
        scale = float(data["scene_scale"])
        vertices = data["vertices_tracker"] @ transform.T * scale + data["scene_origin"]
        faces = data["faces"]
    edges, eligible = face_edges(faces, args.flame_masks)
    details, views, inspected = [], [], {}
    reference_k = camera(records[0])[0]
    for record, surface in zip(records, vertices, strict=True):
        if not np.allclose(camera(record)[0], reference_k, rtol=0, atol=1e-6):
            raise ValueError("Recovered native intrinsics disagree across crop cameras")
        stem = Path(record["name"]).stem
        path = args.semantic_dir / f"{stem}.npz"
        yaw = float(record["baseline_yaw_deg"])
        if abs(yaw) < 55:
            continue
        if not path.is_file():
            details.append(dict(stem=stem, side=1 if yaw < 0 else -1, yaw=yaw,
                                accepted_for_analysis=False, reason="Missing native semantic crop"))
            continue
        detail, view = prepare_view(record, surface, path, edges, eligible)
        details.append(detail)
        inspected[stem] = view
        if detail["accepted_for_analysis"]:
            views.append(view)
    fit, holdout = split_frames(details, args.fit_frames, args.holdout_frames)
    train = [x for x in views if x["detail"]["stem"] in fit]
    test = [x for x in views if x["detail"]["stem"] in holdout]
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(method="Native visible skin-matte contour, orthogonal known-camera translation only",
                  outside_hair_eyewear_and_neck=True, color_edge_is_diagnostic=True,
                  metrics="Median absolute horizontal gaps and cosine-weighted local normal distances; native px",
                  registration_sha256=file_hash(args.registered), initialization_sha256=file_hash(args.initialization),
                  native_K_integer_pixel_centers=reference_k.tolist(), fit_frames=sorted(fit), holdout_frames=sorted(holdout),
                  frames=details, accepted=False)
    world_shift = None
    if {x["detail"]["side"] for x in train} != {-1, 1} or {x["detail"]["side"] for x in test} != {-1, 1}:
        report["reason"] = "both signed profile sides are required in fitting and untouched held frames"
    else:
        # A two-axis common tracker translation keeps yaw, depth, scale and geometry fixed.
        world_axes = (transform[:, :2] * scale) @ np.diag([.012, .012])

        def residual(value):
            shift = world_axes @ value
            values = []
            for view in train:
                kept = ~view["held"]
                xy, _ = project(view["points"][kept] + shift, view["k"], view["r"], view["c"])
                matched = view["xy"][kept]
                # Normal distance uses y displacement too, unlike a horizontal x-only fit.
                pitch = np.gradient(view["xy"][:, 0], view["xy"][:, 1])[kept]
                values.extend((xy[:, 0] - matched[:, 0] - pitch*(xy[:, 1] - matched[:, 1])) * view["weight"][kept])
            return np.asarray(values)

        jac = np.column_stack([(residual(x*.001) - residual(-x*.001)) / .002 for x in np.eye(2)])
        _, singular, axes = np.linalg.svd(jac, full_matrices=False)
        supported = axes[singular > max(.10*singular[0], 1.)].T
        report["singular_sensitivity_native_pixels"] = singular.tolist()
        report["supported_translation_directions"] = supported.T.tolist()
        if not supported.shape[1]:
            report["reason"] = "no independently observable profile translation direction"
        else:
            limit = 1 / np.sqrt(supported.shape[1])
            model = least_squares(lambda x: residual(supported @ x), np.zeros(supported.shape[1]),
                                  bounds=(-limit, limit), loss="soft_l1", f_scale=2., max_nfev=25)
            tracker_shift = supported @ model.x * .012
            world_shift = world_axes @ supported @ model.x
            report.update(candidate_translation_tracker_xy=tracker_shift.tolist(), candidate_translation_world=world_shift.tolist())
            for view in train + test:
                moved = view["geometry"] + world_shift
                pixels, depth = project(moved, view["k"], view["r"], view["c"])
                x, _ = mesh_intersections(pixels, depth, moved, edges, eligible, view["xy"][:, 1], view["detail"]["side"])
                candidate = abs(x - view["xy"][:, 0])
                original = abs(view["original_x"] - view["xy"][:, 0])
                view["detail"]["lost_candidate_correspondences"] = int((~np.isfinite(candidate)).sum())
                for name, keep in (("fit_arc", ~view["held"]), ("held_arc", view["held"]), ("all_arcs", np.ones_like(view["held"]))):
                    for suffix, weight in (("", view["weight"]), ("_native_horizontal_px", np.ones_like(view["weight"]))):
                        view["detail"][name + suffix] = [float(np.median(original[keep] * weight[keep])),
                            float(np.median(candidate[keep] * weight[keep])) if np.isfinite(candidate[keep]).all() else None]
                if view["detail"]["stem"] in holdout or len(train) <= 5:
                    save_panel(args.output / f"{view['detail']['stem']}-skin-profile.jpg", view, moved, faces)
            tests = [x["detail"] for x in test]
            arcs = [x["detail"]["held_arc_native_horizontal_px"] for x in train]
            bidirectional = all(any(x["side"] == side for x in tests) for side in (-1, 1))
            valid = all(not x["detail"]["lost_candidate_correspondences"] for x in train + test)
            if valid and bidirectional:
                improvements = {side: [a-b for a,b in (x["all_arcs_native_horizontal_px"] for x in tests if x["side"] == side)] for side in (-1, 1)}
                report["held_median_improvement_px_per_side"] = {str(side): float(np.median(values)) for side, values in improvements.items()}
                report["accepted"] = (all(np.median(v) >= .25 and min(v) >= -.75 for v in improvements.values())
                                      and np.median([a-b for a,b in arcs]) >= 0 and model.success)
            if not report["accepted"]:
                report["reason"] = "translation failed the held arc/opposite-side check or lost anatomical support"
    for stem in ("00790", "00850"):
        path = args.output / f"{stem}-skin-profile.jpg"
        if stem in inspected and not path.is_file():
            view = inspected[stem]
            moved = view["geometry"] + world_shift if world_shift is not None else None
            save_panel(path, view, moved, faces)
    (args.output / "skin-profile-fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Analysed {len(views)} profile frames; accepted={report['accepted']}; {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("registered", "initialization", "semantic-dir", "flame-masks", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fit-frames", nargs="+")
    parser.add_argument("--holdout-frames", nargs="+")
    run(parser.parse_args())
