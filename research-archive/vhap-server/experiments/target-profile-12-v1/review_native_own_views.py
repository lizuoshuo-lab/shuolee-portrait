"""Compare a fresh native VHAP mesh to its own camera in decoded MP4 pixels."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.spatial import cKDTree
from vhap.util.landmark_detector_star import LandmarkDetectorSTAR

from fit_native_skin_profile import camera, project
from fit_torso_neck import VideoPixels, contour_x
from fit_vhap_neck_sequence import contour_edges, load_observations
from run_video import input_hash, write_json
from side_masks import profile_mask


def read(path):
    return json.loads(path.read_bytes())


def witnesses(video, observed, predicted, labels):
    src = video.map(observed)
    pred = video.map(predicted)
    rows = []
    for a, b, label in zip(src, pred, labels, strict=True):
        valid = np.isfinite(b).all()
        rows.append(dict(region=label, source=a.tolist(), mesh=b.tolist() if valid else None,
                         error_px=float(np.linalg.norm(a-b)) if valid else None))
    return rows


def stats(rows):
    values = [r["error_px"] for r in rows if r["error_px"] is not None]
    return dict(measured=len(values), over3=sum(v > 3 for v in values),
                missing=len(rows)-len(values), max_px=max(values, default=None))


def face_landmarks(star, record, crop_path, mesh, faces, embed, view, video):
    ids = np.flatnonzero((np.arange(len(star)) >= 17) & (star[:, 2] > 0))
    if not len(ids):
        return []
    image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    crop_xy = star[ids, :2].astype(float)*image.shape[::-1]-.5
    source = cv2.perspectiveTransform(crop_xy[None], np.linalg.inv(record["warp"]))[0]
    tris = faces[embed["full_lmk_faces_idx"][0, ids]]
    xyz = np.einsum("nv,nvc->nc", embed["full_lmk_bary_coords"][0, ids], mesh[tris])
    return witnesses(video, source, project(xyz, *view)[0], [f"STAR-{i}" for i in ids])


def raw_face(detector, image, lo, hi, mesh, view, video, faces, embed):
    """Read the detector's pixel coordinates directly on the unscaled video crop."""
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    box, lm = detector.detect_single_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if box[-1] <= 0 or not (lm[17:68, 2] > 0).all():
        return []
    points = lm[17:68, :2]*crop.shape[1::-1]+lo
    tri = faces[embed["full_lmk_faces_idx"][0, 17:68]]
    xyz = np.einsum("nv,nvc->nc", embed["full_lmk_bary_coords"][0, 17:68], mesh[tri])
    predicted = video.map(project(xyz, *view)[0])
    return [dict(region=f"STAR-{i}", source=a.tolist(), mesh=b.tolist(), error_px=float(np.linalg.norm(b-a)))
            for i, (a,b) in enumerate(zip(points, predicted, strict=True), 17)]


def upper_outline(scene, name, mesh, faces, view, video):
    raw_alpha = cv2.imread(str(scene/"alpha_maps"/name), 0)
    coarse = cv2.imread(str(scene/"headmask_undist/images"/Path(name).with_suffix(".png")), 0)
    if not np.allclose(raw_alpha.shape[::-1]*video.scale, video.size):
        raise ValueError("Unexpected source resolution for undilated alpha")
    mx, my = cv2.initUndistortRectifyMap(video.k, video.dist, None, view[0], coarse.shape[::-1], cv2.CV_32FC1)
    matte = cv2.remap(raw_alpha, mx, my, cv2.INTER_LINEAR)
    evidence = profile_mask(matte, coarse)
    ys,xs = evidence["crop"]
    pixels = project(mesh, *view)[0]
    lo = np.maximum(np.minimum([xs.start,ys.start], np.floor(pixels.min(0)).astype(int))-8, 0)
    hi = np.minimum(np.maximum([xs.stop,ys.stop], np.ceil(pixels.max(0)).astype(int))+9, coarse.shape[::-1])
    scale = 4
    patch = matte[lo[1]:hi[1],lo[0]:hi[0]]
    alpha = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    predicted = np.zeros_like(alpha)
    tris = np.rint(((pixels-lo+.5)*scale-.5)*16).astype(np.int32)[faces]
    for tri in tris:
        cv2.fillConvexPoly(predicted, tri, 255, shift=4)
    # Evaluate both directions so missing hair and excess mesh can both be seen.
    def boundary(mask):
        loops = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        nodes = max(loops, key=cv2.contourArea)[:,0]
        interior = ((nodes > 0) & (nodes < np.array(mask.shape[::-1])-1)).all(axis=1)
        return (nodes[interior]+.5)/scale-.5+lo
    observed = boundary(alpha >= 128)
    model = boundary(predicted)
    rows = []
    for src, dst, label in ((observed, model, "matte to mesh"), (model, observed, "mesh to matte")):
        ix, iy = np.rint(src[::4]).astype(int).T
        valid = evidence["boundary_valid"][iy,ix]
        a = video.map(src[::4][valid])
        dense = video.map(dst)
        if not len(a):
            continue
        dist, nearest = cKDTree(dense).query(a)
        for x, y, d in zip(a, dense[nearest], dist, strict=True):
            reference, fit = (x,y) if label == "matte to mesh" else (y,x)
            rows.append(dict(region=label, source=reference.tolist(), mesh=fit.tolist(), error_px=float(d)))
    box = video.map([[x,y] for x in (xs.start,xs.stop) for y in (ys.start,ys.stop)])
    lo = np.maximum(np.floor(box.min(0)).astype(int)-16, 0)
    hi = np.minimum(np.ceil(box.max(0)).astype(int)+17, video.size)
    return rows, lo, hi


def title(tile, lines):
    w = tile.shape[1]
    top = np.full((58, w, 3), (30, 29, 27), np.uint8)
    for line, y, font in zip(lines, (23, 45), (.56, .42)):
        scale = min(font, font*(w-12)/max(1, cv2.getTextSize(line, 0, font, 1)[0][0]))
        cv2.putText(top, line, (6, y), 0, scale, (240, 238, 235), 1, cv2.LINE_AA)
    return np.vstack((top, tile))


def draw_grid_frame(raw, name, mesh, faces, view, video, observations):
    pix, depth = project(mesh, *view)
    xy = video.map(pix)
    support = [pt["source"] for group in observations.values() for pt in group]
    extent = np.vstack((xy, support)) if support else xy
    lo = np.maximum(np.floor(extent.min(0)).astype(int)-32, 0)
    hi = np.minimum(np.ceil(extent.max(0)).astype(int)+33, raw.shape[1::-1])
    source = raw[lo[1]:hi[1], lo[0]:hi[0]].copy()
    triangles = np.rint(xy-lo).astype(np.int32)[faces]
    xyz = ((mesh-view[2]) @ view[1])[faces]
    normals = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.abs(normals @ np.asarray([-.35, -.25, -.90]))
    paint = source.copy()
    alpha = np.zeros(source.shape[:2], np.uint8)
    colors = ((.32+.68*light[:,None])*np.array([215, 214, 106])).astype(int).tolist()
    for i in np.argsort(depth[faces].mean(1))[::-1]:
        cv2.fillConvexPoly(paint, triangles[i], colors[i])
        cv2.fillConvexPoly(alpha, triangles[i], 255)
    overlay = cv2.addWeighted(source, .42, paint, .58, 0)
    edges = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(overlay, edges, -1, (226, 240, 100), 1, cv2.LINE_AA)
    # The center of each mark is at the original decoded pixel coordinate.
    groups = [("neck", "Neck"), ("face", "Inner face (STAR)"), ("outline", "Upper head incl. hair")]
    tiles = [title(source, [name, "Decoded video"]), title(overlay, ["Native VHAP mesh", "Own pose / own camera"])]
    for key, label in groups:
        rows = observations[key]
        tile = source.copy()
        for pt in sorted(rows, key=lambda q: q["error_px"] if q["error_px"] is not None else -1):
            a = tuple(np.rint(np.asarray(pt["source"])-lo).astype(int))
            err = pt["error_px"]
            if err is None:
                cv2.circle(tile, a, 2, (0, 182, 255), 1, cv2.LINE_AA)
                continue
            b = tuple(np.rint(np.asarray(pt["mesh"])-lo).astype(int))
            color = (38, 48, 255) if err > 3 else (78, 220, 70)
            cv2.line(tile, a, b, color, 1, cv2.LINE_AA)
            cv2.circle(tile, a, 2, color, -1, cv2.LINE_AA)
            cv2.drawMarker(tile, b, color, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
        m = stats(rows)
        subtitle = (f"{m['over3']}/{m['measured']} >3px   max {m['max_px']:.1f}px"
                    if m["measured"] else "No measurable pairs")
        if m["missing"]:
            subtitle += f"   unmatched {m['missing']}"
        tiles.append(title(tile, [label, subtitle]))
    merged = np.hstack(tiles)
    legend = np.full((30, merged.shape[1], 3), (30,29,27), np.uint8)
    cv2.putText(legend, "Dot = observed   + = mesh    Red > 3 px    Green <= 3 px    Amber = no mesh match",
                (7,20), 0, .47, (240,238,235), 1, cv2.LINE_AA)
    return np.vstack((merged,legend))


def run(root, out, selected, flame):
    settings = read(root/"run.json")
    trained = read(root/"vhap-rigid/result.json")
    seed = read(root/"vhap-rigid/input-neck.json")
    sequence = root/"head-neck"
    init = sequence/"initialization.json"
    records = read(init)["frames"]
    names = [r["name"] for r in records]
    if (trained["seeded_count"] != len(names) or trained["frame_names"] != names
        or seed["frames"] != names or trained["initial_neck_posture"] != "template-neutral"
        or settings["configuration"]["previous_run_inputs"] or trained["trained_steps"] <= 0):
        raise ValueError("Unexpected input provenance, frame order or initialization")
    chk = Path(trained["checkpoint"]).resolve(strict=True)
    chk.relative_to(root)
    src_hash = input_hash(chk)
    tape = np.load(root/"geometry-binding/registered-head.npz")
    if (tape["filenames"].tolist() != names or tape["checkpoint_sha256"].item() != src_hash
        or tape["initialization_sha256"].item() != input_hash(init)):
        raise ValueError("Registered surface does not match this camera sequence and checkpoint")
    world = tape["vertices_tracker"] @ tape["tracker_to_world_rotation"].T * float(tape["scene_scale"]) + tape["scene_origin"]
    faces = tape["faces"]
    views = [camera(row) for row in records]
    video = VideoPixels(root, views, names)
    neck_edges, neck_eligible, _, _ = contour_edges(faces, flame/"FLAME_masks.pkl")
    embed = np.load(flame/"landmark_embedding_with_eyes.npy", allow_pickle=True, encoding="latin1").item()
    star = np.load(sequence/"landmark2d/STAR.npz")["face_landmark_2d"]
    frozen = read(root/"split.json")
    frames_held = set()
    for block in frozen["neck_blocks"]:
        first, last = map(int, block.split(":"))
        frames_held.update(range(first, last+1))
    observed = load_observations(Namespace(scene=root/"scene", sequence=sequence,
             targets=root/"neck-expanded.json", native_fallback=root/"neck-narrow.json",
             supplement=root/"front-contours.json"), records)
    by_frame = [[] for _ in names]
    for i, sign, x, y, split, region, _ in observed:
        if i in frames_held or split == "check":
            by_frame[i].append(dict(native_xy=(x,y), side="left" if sign < 0 else "right", region=region))
    out.mkdir(exist_ok=True)
    native_grids = Path(trained["image_grid"]).resolve(strict=True)
    native_grids.relative_to(root)
    (out/"native").mkdir(exist_ok=True)
    selected = [Path(n).stem+".jpg" for n in selected]
    detector = LandmarkDetectorSTAR()
    panels, report = {}, {}
    for i, (name, rec, mesh, view, obs) in enumerate(zip(names, records, world, views, by_frame, strict=True)):
        loc = np.array([p["native_xy"] for p in obs]).reshape(-1, 2)
        x = contour_x(mesh, view, obs, neck_edges, neck_eligible)
        neck = witnesses(video, loc, np.column_stack((x, loc[:,1])), [p["region"] for p in obs])
        measured = dict(neck=neck, face=face_landmarks(star[i], rec, sequence/"images"/f"{i:08d}.jpg",
                            mesh, faces, embed, view, video), outline=[])
        face_source = "VHAP input STAR" if measured["face"] else "unavailable"
        if name in selected:
            raw = video.frame(name)
            outline, lo, hi = upper_outline(root/"scene", name, mesh, faces, view, video)
            measured["outline"] = outline
            if abs(rec["baseline_yaw_deg"]) < 55:
                measured["face"] = raw_face(detector, raw, lo, hi, mesh, view, video, faces, embed)
                face_source = "STAR redetected on original MP4 head crop" if measured["face"] else "unavailable"
            img = draw_grid_frame(raw, Path(name).stem, mesh, faces, view, video, measured)
            if not cv2.imwrite(str(out/(Path(name).stem+".png")), img):
                raise IOError(name)
            shutil.copyfile(native_grids/f"frame_{i:05d}.jpg", out/"native"/name)
            panels[name] = img
        report[name] = dict(index=i, yaw_deg=rec["baseline_yaw_deg"], face_source=face_source,
                            metrics={k:stats(v) for k,v in measured.items()}, witnesses=measured)
    width = max(p.shape[1] for p in panels.values())
    grid = np.vstack([cv2.copyMakeBorder(panels[n], 0, 0, 0, width-panels[n].shape[1],
                                        cv2.BORDER_CONSTANT, value=(30,29,27)) for n in selected])
    if not cv2.imwrite(str(out/"grid.png"), grid):
        raise IOError("grid.png")
    video.video.release()
    write_json(out/"result.json", dict(input_name=str(root), checkpoint_sha256=src_hash,
               review_script_sha256=input_hash(Path(__file__)), source_video_sha256=settings["configuration"]["video_sha256"],
               source_size=video.size.tolist(), max_mapping_roundtrip_px=video.roundtrip_px,
               threshold_video_px=3, native_steps=trained["trained_steps"], seed_deg=seed["neutral_template_deg"],
               seeded=len(names), selected=selected, frames=report,
               all_metrics={k:stats([q for v in report.values() for q in v["witnesses"][k]]) for k in measured},
               outline_source="scene/alpha_maps (undilated RVM at 0.5 opacity). The coarse head mask only selects the vertical evaluation band.",
               measures="Neck: horizontal silhouette intersections. Upper head: bidirectional nearest outline comparisons limited to observed upper-head matte support on named views; includes hair and accessories. All distances in decoded video pixels. STAR detector estimates are not physical ground truth."))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", nargs="+", required=True)
    p.add_argument("--flame", type=Path, required=True)
    args = p.parse_args()
    run(args.root.resolve(), args.output, args.frames, args.flame)
