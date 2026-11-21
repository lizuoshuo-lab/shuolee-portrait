"""Propose unobserved anterior neck boundaries from native RGB and independent source views."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from detect_vhap_neck import _split_side


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_crop(path, image):
    with np.load(path, allow_pickle=False) as z:
        bgr, alpha, origin = z["bgr"], z["alpha"], z["origin_xy"]
    x, y = origin.tolist()
    h, w = alpha.shape
    if not np.array_equal(bgr, image[y:y + h, x:x + w]) or not np.isfinite(alpha).all():
        raise ValueError(f"Native crop and source RGB mismatch: {path}")
    return bgr, alpha, origin


def largest_support(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        return np.zeros(mask.shape, dtype=bool)
    return labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()


def background_lines(gray, alpha, start, stop):
    """Recognize a structural line continuing into visible background outside the skin band."""
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return []
    output = []
    for x0, y0, x1, y1 in lines[:, 0]:
        if abs(y1 - y0) < 28 or abs(x1 - x0) > .30 * abs(y1 - y0):
            continue
        samples = np.linspace((x0, y0), (x1, y1), 24)
        x, y = np.rint(samples).astype(int).T
        valid = (x >= 0) & (x < gray.shape[1]) & (y >= 0) & (y < gray.shape[0])
        x, y = x[valid], y[valid]
        exposed = ((y < start - 3) | (y > stop + 3)) & (alpha[y, x] < .12)
        if exposed.sum() >= 6:
            output.append((x0, y0, x1, y1))
    return output


def discover(bgr, alpha, origin, posterior, direction):
    """Find candidate skin crossings using a local native-RGB trimap at a relative neck band."""
    if len(posterior) < 4:
        return {}, {"reason": "insufficient_posterior_skin_reference"}
    h, w = alpha.shape
    if direction == -1:
        bgr, alpha = bgr[:, ::-1].copy(), alpha[:, ::-1].copy()
    posterior = np.asarray([point["native_xy"] for point in posterior], dtype=float) - origin
    if direction == -1:
        posterior[:, 0] = w - 1 - posterior[:, 0]
    posterior = posterior[np.argsort(posterior[:, 1])]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)
    support = largest_support(alpha >= .5)
    start, end = map(int, posterior[[0, -1], 1])
    mid = (start + end) // 2
    span = np.flatnonzero(support[mid])
    if len(span) < 25:
        return {}, {"reason": "no_native_neck_span"}
    noseward = min(w - 3, int(span[-1] + 17))
    rear = int(np.median(posterior[:, 0]))
    stop = min(h - 9, end + min(18, max(6, (span[-1] - rear) // 5)))
    y0, y1 = max(0, start - 7), min(h, stop + 8)
    x0, x1 = max(0, rear - 5), min(w, noseward + 7)
    if y1 <= y0 or x1 <= x0:
        return {}, {"reason": "clipped_neck_window"}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rear_x = np.interp(yy, posterior[:, 1], posterior[:, 0])
    foreground = support[y0:y1, x0:x1]
    a = alpha[y0:y1, x0:x1]
    crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
    # Fixed seeds lie inside the certified posterior skin band, away from the throat edge.
    interior = (yy >= start) & (yy <= end) & (xx >= rear_x + 10) & (xx <= rear_x + 24)
    interior &= a >= .97
    background = (a <= .05) & (xx >= rear_x + 30)
    if interior.sum() < 50 or background.sum() < 50:
        return {}, {"reason": "insufficient_hard_trimap_support"}
    skin_color = np.median(lab[y0:y1, x0:x1][interior], axis=0)
    color_distance = np.linalg.norm(lab[y0:y1, x0:x1, 1:] - skin_color[1:], axis=2)
    probable = (foreground & (color_distance <= 15)) | ((a >= .93) & (xx < rear_x + 40))
    labels = np.where(probable, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[interior], labels[background] = cv2.GC_FGD, cv2.GC_BGD
    cv2.setRNGSeed(20261118)
    cv2.grabCut(crop, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    cut = largest_support((labels & 1) != 0)
    structural = background_lines(gray, alpha, start, stop)
    accepted, order = {}, {"trimap_rows": 0, "line_veto": 0, "color_or_matte_veto": 0}
    for y in range(start + 2, stop + 1):
        indexes = np.flatnonzero(cut[y - y0])
        if not len(indexes):
            continue
        x = x0 + indexes[-1]
        if x < x0 + 8 or x >= x1 - 6:
            continue
        order["trimap_rows"] += 1
        if any(min(a0, a1) - 4 <= y <= max(a0, a1) + 4
               and abs(x - (b0 + (y - a0) * (b1 - b0) / (a1 - a0))) <= 3
               for b0, a0, b1, a1 in structural):
            order["line_veto"] += 1
            continue
        inside = lab[y, x - np.array([2, 4, 6])]
        skin_like = np.linalg.norm(inside[:, 1:] - skin_color[1:], axis=1).max() <= 12
        skin_like &= abs(inside[:, 0] - skin_color[0]).max() <= 35
        exterior = alpha[y, x + np.arange(2, 6)]
        # The native matte has to independently witness background near this RGB boundary.
        if not skin_like or alpha[y, x - 4] < .90 or (exterior > .25).any():
            order["color_or_matte_veto"] += 1
            continue
        cross = float(x) + .5
        if alpha[y, x] >= .5 > alpha[y, x + 1]:
            cross = x + (alpha[y, x] - .5) / (alpha[y, x] - alpha[y, x + 1])
        accepted[y + int(origin[1])] = float((cross if direction == 1 else w - 1 - cross) + origin[0])
    return accepted, order


def track_control(images, points, frames, target, direction):
    """Transport an interior patch, with bidirectional agreement, near a checked contour site."""
    xy = np.asarray([point["native_xy"] for point in points], np.float32).reshape(-1, 1, 2)
    xy[..., 0] -= direction * 8
    if not len(xy):
        return xy[:, 0]
    roi = np.array([[0., 0.], [images[target].shape[1] - 1, images[target].shape[0] - 1]])
    for before, after in zip(frames, frames[1:]):
        lk = dict(winSize=(21, 21), maxLevel=3)
        new, found, _ = cv2.calcOpticalFlowPyrLK(images[before], images[after], xy, None, **lk)
        if new is None:
            return xy[:0, 0]
        old, back, _ = cv2.calcOpticalFlowPyrLK(images[after], images[before], new, None, **lk)
        if old is None:
            return xy[:0, 0]
        valid = (found[:, 0] > 0) & (back[:, 0] > 0) & (np.linalg.norm(xy[:, 0] - old[:, 0], axis=-1) <= .9)
        valid &= ((new[:, 0] >= roi[0]) & (new[:, 0] <= roi[1])).all(-1)
        xy = new[valid]
        if not len(xy):
            return xy[:, 0]
    xy[:, 0, 0] += direction * 8
    return xy[:, 0]


def run(args):
    current_hash = digest(args.initialization)
    records = json.loads(args.initialization.read_text())["frames"]
    scene_source = json.loads((args.scene / "source-frames.json").read_text())
    primary = json.loads(args.neck_points.read_text())
    crops = json.loads((args.native_crops / "source.json").read_text())
    names = [row["name"] for row in records]
    if (names != [row["name"] for row in scene_source["frames"]]
            or set(names) != set(primary["frames"]) or len(names) != len(set(names))
            or current_hash != crops["vhap_initialization_sha256"]
            or primary["source"]["video_sha256"] != scene_source["video_sha256"]):
        raise ValueError("Inputs disagree on camera/crop, source video or frame identity")
    manifest = {x["name"]: x for x in scene_source["frames"]}
    crop_manifest = {x["name"]: x for x in crops["frames"]}
    gray, observed, candidate, records_out = {}, {}, {}, {}
    for row in records:
        name = row["name"]
        if name not in crop_manifest:
            continue
        known = primary["frames"][name]
        proof = crop_manifest[name]
        native_path = args.scene / "orig_undist/images" / name
        matte_path = args.scene / "alpha_maps" / name
        crop_path = args.initialization.parent / "images" / f'{row["timestep"]:08}.jpg'
        if (digest(native_path) != proof["native_sha256"] or proof["native_sha256"] != known["native_sha256"]
                or manifest[name]["native_sha256"] != known["native_sha256"]
                or digest(matte_path) != proof["raw_matte_sha256"] or proof["raw_matte_sha256"] != known["raw_matte_sha256"]
                or digest(crop_path) != proof["corresponding_vhap_crop_sha256"]
                or row["timestamp_s"] != known["timestamp_s"] or row["timestep"] != known["timestep"]):
            raise ValueError(f"Source image, matte, or exact virtual crop identity differs: {name}")
        image = cv2.imread(str(native_path))
        if image is None or image.shape[1::-1] != tuple(crops["native_raster_size"]):
            raise ValueError(f"Unexpected registered native image: {name}")
        bgr, alpha, origin = read_crop(args.native_crops / f"{Path(name).stem}.npz", image)
        direction = 1 if row["baseline_yaw_deg"] < 0 else -1
        front = "right" if direction == 1 else "left"
        back = [point for point in known["points"] if point["side"] != front]
        observed[name] = [point for point in known["points"] if point["side"] == front]
        item, stats = discover(bgr, alpha, origin, back, direction)
        candidate[name] = (direction, front, item)
        gray[name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        records_out[name] = {"native_sha256": known["native_sha256"], "raw_matte_sha256": known["raw_matte_sha256"],
                             "virtual_crop_sha256": proof["corresponding_vhap_crop_sha256"],
                             "timestamp_s": row["timestamp_s"], "timestep": row["timestep"], "front_side": front,
                             "primary_front_points": len(observed[name]), "native_candidate_rows": len(item),
                             "points": [], "discovery": stats}
    controls, errors = {}, []
    for name, points in observed.items():
        rows = candidate[name][2]
        checks = [point["native_xy"] for point in points if point["split"] == "check"]
        delta = [abs(rows[int(round(y))] - x) for x, y in checks if int(round(y)) in rows]
        if checks:
            controls[name] = {"held_sites": len(checks), "recovered_sites": len(delta), "abs_px": delta}
        errors.extend(delta)
    validated = (sum(bool(row["recovered_sites"]) for row in controls.values()) >= 3
                 and np.median(errors) <= 2.0 and np.quantile(errors, .9) <= 4.0)
    timeline = list(gray)
    times = {name: row["timestamp_s"] for name, row in records_out.items()}
    for name, (direction, side, rows) in candidate.items():
        if observed[name] or not rows or not validated:
            continue
        here = timeline.index(name)
        votes = dict.fromkeys(rows, 0)
        for control, points in observed.items():
            if candidate[control][0] != direction or not points or abs(times[control] - times[name]) > 2.1:
                continue
            there = timeline.index(control)
            journey = timeline[min(here, there):max(here, there) + 1]
            if max(abs(times[b] - times[a]) for a, b in zip(journey, journey[1:])) > .45:
                continue
            if here < there:
                journey.reverse()
            tracked = track_control(gray, [p for p in points if p["split"] == "check"], journey, name, direction)
            for y, x in rows.items():
                votes[y] += int(((abs(tracked[:, 1] - y) <= 2) & (abs(tracked[:, 0] - x) <= 3)).any())
        supported = [y for y in sorted(votes) if votes[y] >= 2]
        records_out[name]["temporally_supported_rows"] = len(supported)
        if len(supported) < 8:
            continue
        record = records_out[name]
        H = np.asarray(records[record["timestep"]]["warp"], dtype=float)
        uv = np.array([(rows[y], y) for y in supported], dtype=float)
        warped = cv2.perspectiveTransform(uv[None], H)[0]
        points = [{"native_xy": [rows[y], y], "xy": position.tolist(), "side": side,
                   "independent_control_views": votes[y]} for y, position in zip(supported, warped, strict=True)]
        record["points"] = _split_side(points)
    result = {"schema": "vhap-neck-front-supplement-v1", "source": {
        "initialization_sha256": current_hash, "video_sha256": scene_source["video_sha256"],
        "primary_targets_sha256": digest(args.neck_points), "native_crop_manifest_sha256": digest(args.native_crops / "source.json"),
        "method": "native RGB trimap, matte and structural-line rejection, checked temporal skin patches",
    }, "controls_passed": bool(validated), "controls": controls, "frames": records_out}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"{len(records_out)} profile frames, {len(errors)} recovered check sites, "
          f"{sum(len(row['points']) for row in records_out.values())} validated added sites", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "initialization", "neck-points", "native-crops", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
