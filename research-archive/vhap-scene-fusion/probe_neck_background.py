"""Check native throat observations against room pixels transferred from other cameras."""

from argparse import ArgumentParser
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from detect_vhap_neck import _split_side
from initialize_joint_scene import read_colmap


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def native_camera(view, camera, world):
    local = world @ view.qvec2rotmat().T + view.tvec
    xy = local[:, :2] / np.maximum(local[:, 2:], 1e-9) * camera.params[:2] + camera.params[2:] - .5
    return xy, local[:, 2] > 0


def donor_patch(source, ids, xyz, points, pixels, load, bounds, target):
    x0, y0, width, height = bounds
    rgb, matte = load(source.name)
    valid_room = cv2.erode((matte <= 16).astype(np.uint8), np.ones((7, 7), np.uint8))
    matches = {}
    for j in ids:
        point = points[xyz[j]]
        use = np.flatnonzero(point.image_ids == source.id)
        if not len(use):
            continue
        pixel = source.xys[point.point2D_idxs[use[0]]] - .5
        x, y = np.rint(pixel).astype(int)
        if 0 <= y < matte.shape[0] and 0 <= x < matte.shape[1] and valid_room[y, x]:
            matches.setdefault(tuple(np.rint(pixels[j] / 2).astype(int)), (pixel, pixels[j]))
    if len(matches) < 22:
        return None
    src, dst = np.asarray([matches[key] for key in sorted(matches)]).transpose(1, 0, 2)
    check = np.arange(len(src)) % 4 == 0
    cv2.setRNGSeed(20261118)
    H, inliers = cv2.findHomography(src[~check], dst[~check], cv2.RANSAC, 1.5, maxIters=2000, confidence=.999)
    if H is None or int(inliers.sum()) < 12:
        return None
    track_error = np.linalg.norm(cv2.perspectiveTransform(src[check, None], H)[:, 0] - dst[check], axis=1)
    p50, p90 = np.quantile(track_error, (.5, .9))
    if p50 > 1.5 or p90 > 2.5:
        return None
    H = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]]) @ H
    warp = cv2.warpPerspective(rgb, H, (width, height)).astype(float)
    available = cv2.warpPerspective(valid_room, H, (width, height), flags=cv2.INTER_NEAREST) > 0
    target_rgb, target_matte = target
    y, x = np.indices((height, width))
    held = ((x // 8 + y // 8) % 2) == 1
    room = available & (target_matte <= 16)
    if min((room & held).sum(), (room & ~held).sum()) < 100:
        return None
    correction = np.median((target_rgb - warp)[room & ~held], axis=0)
    error = np.abs(target_rgb - warp - correction).mean(axis=2)
    p50_rgb, p90_rgb = np.quantile(error[room & held], (.5, .9))
    if p50_rgb > 12 or p90_rgb > 35:
        return None
    return available, error, {"name": source.name, "tracks_fit": int(inliers.sum()), "tracks_check": int(check.sum()),
                              "tracks_check_p50_p90_px": [float(p50), float(p90)],
                              "exposed_room_check_pixels": int((room & held).sum()),
                              "exposed_room_check_p50_p90_rgb8": [float(p50_rgb), float(p90_rgb)]}


def probe(record, view, camera, donors, points, load, known):
    name = record["name"]
    direction = 1 if record["baseline_yaw_deg"] < 0 else -1
    front = "right" if direction == 1 else "left"
    posterior = sorted([row["native_xy"] for row in known if row["side"] != front], key=lambda xy: xy[1])
    report = {"side": front, "new_rows": [], "room_donors": [], "known_front_count": sum(x["side"] == front for x in known)}
    if len(posterior) < 4:
        return dict(report, reason="no_certified_posterior_band")
    photo, matte = load(name)
    start, stop = [round(posterior[i][1]) for i in (0, -1)]
    middle = (start + stop) // 2
    row = np.flatnonzero(matte[middle] >= 128)
    if not len(row):
        return dict(report, reason="no_native_target_foreground")
    x_front = int(row[-1] if direction == 1 else row[0])
    if x_front <= 66 or x_front >= matte.shape[1] - 67 or start <= 60 or stop >= matte.shape[0] - 66:
        return dict(report, reason="clipped_native_band")
    x0, y0, width, height = x_front - 66, start - 60, 133, stop - start + 126
    keys = list(points)
    pixels, positive = native_camera(view, camera, np.asarray([points[key].xyz for key in keys]))
    ids = np.flatnonzero(positive & ((pixels >= [x0, y0]) & (pixels < [x0 + width, y0 + height])).all(axis=1))
    rgb = photo[y0:y0 + height, x0:x0 + width]
    alpha = matte[y0:y0 + height, x0:x0 + width]
    witness = [patch for source in donors if (patch := donor_patch(source, ids, keys, points, pixels, load,
                                                                   (x0, y0, width, height), (rgb, alpha))) is not None]
    report["room_donors"] = [a[2] for a in witness]
    if len(witness) < 3:
        return dict(report, reason="too_few_independently_checked_room_views")
    valid = np.stack([a[0] for a in witness])
    delta = np.stack([a[1] for a in witness])
    base = np.asarray([a[2]["exposed_room_check_p50_p90_rgb8"][1] for a in witness])[:, None, None]
    votes = valid.sum(0)
    foreground = (votes >= 3) & ((valid & (delta > base + 6)).sum(0) >= np.ceil(.75 * votes)) & (alpha >= 140)
    background = (votes >= 3) & ((valid & (delta <= base)).sum(0) >= np.ceil(.75 * votes))
    major = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(float)
    profile = []
    for y in range(start, min(stop + 12, y0 + height - 2) + 1):
        r = y - y0
        rear = round(np.interp(y, *np.asarray(posterior).T[::-1]))
        inside_color = major[y, rear + direction * np.arange(10, 24)]
        color = np.median(inside_color, axis=0)
        if np.ptp(inside_color[:, 0]) > 35:
            continue
        offsets = np.arange(-32, 17)
        columns = x_front + direction * offsets
        local_columns = columns - x0
        foreground_slots = np.flatnonzero(foreground[r, local_columns])
        if not len(foreground_slots) or foreground_slots[-1] >= len(columns) - 5:
            continue
        i = int(foreground_slots[-1])
        b = np.flatnonzero(background[r, local_columns[i + 1:]])
        if not len(b) or b[0] >= 4:
            continue
        j = i + 1 + int(b[0])
        if j + 3 > len(columns) or not background[r, local_columns[j:j + 3]].all():
            continue
        c = major[y, columns[i] - direction * np.array([2, 4, 6])]
        if (np.linalg.norm(c[:, 1:] - color[1:], axis=1).max() > 12
                or np.min(c[:, 0]) < color[0] - 24
                or not (alpha[r, columns[i] - x0 - direction * np.array([2, 4, 6])] >= 200).all()):
            continue
        x = float((columns[i] + columns[j]) / 2)
        profile.append({"native_xy": [x, y], "side": front, "bracket_px": int(j - i),
                        "room_donor_agreement": int(votes[r, local_columns[j]])})
    checks = sorted([row["native_xy"] for row in known if row["side"] == front and row["split"] == "check"])
    proposed = {row["native_xy"][1]: row["native_xy"][0] for row in profile}
    report.update(new_rows=profile, checked_sites=len(checks), check_abs_px=[abs(proposed[round(y)] - x) for x, y in checks if round(y) in proposed],
                  corridor_pixels_with_three_donors=int((votes >= 3).sum()), reason="room_transfer_evaluated")
    return report


def run(args):
    frames = json.loads(args.initialization.read_text())["frames"]
    records = {row["name"]: row for row in frames}
    scene = json.loads((args.scene / "source-frames.json").read_text())
    source = {row["name"]: row for row in scene["frames"]}
    prior = json.loads(args.neck_points.read_text())
    sparse = args.scene / "orig_undist/sparse"
    cameras, images, points = read_colmap(args.gaussian_repo, sparse)
    views = {view.name: view for view in images.values()}
    if set(views) != set(records) or set(source) != set(records) or prior["source"]["video_sha256"] != scene["video_sha256"]:
        raise ValueError("Room camera map, registered frames and certified native observations disagree")

    @lru_cache(maxsize=24)
    def load(name):
        native = args.scene / "orig_undist/images" / name
        if digest(native) != source[name]["native_sha256"]:
            raise ValueError(f"Native frame changed: {name}")
        photo = cv2.imread(str(native))
        matte = cv2.imread(str(args.scene / "mask_undist/images" / name), 0)
        if photo is None or matte is None or photo.shape[:2] != matte.shape or tuple(matte.shape[::-1]) != tuple(scene["native_size"]):
            raise ValueError(f"Native photo and matte disagree: {name}")
        return photo, matte

    report, errors = {}, []
    for name, row in records.items():
        old = prior["frames"][name]
        if name != args.target and (abs(row["baseline_yaw_deg"]) < 55 or len({x["side"] for x in old["points"]}) != 2):
            continue
        if (source[name]["native_sha256"] != old["native_sha256"] or row["timestamp_s"] != old["timestamp_s"]
                or digest(args.scene / "alpha_maps" / name) != old["raw_matte_sha256"]):
            raise ValueError(f"Certified neck provenance mismatch: {name}")
        donors = [views[other] for other in records if other != name and abs(records[other]["timestamp_s"] - row["timestamp_s"]) <= 2.7]
        item = probe(row, views[name], cameras[views[name].camera_id], donors, points, load, old["points"])
        report[name] = item
        if name != args.target:
            errors.extend(item.get("check_abs_px", []))
    passed = (sum(bool(item.get("check_abs_px")) for name, item in report.items() if name != args.target) >= 3
              and np.median(errors) <= 2 and np.quantile(errors, .9) <= 4)
    target = report[args.target]
    targets = []
    if passed and len(target["new_rows"]) >= 12 and not target["known_front_count"]:
        H = np.asarray(records[args.target]["warp"])
        native = np.asarray([row["native_xy"] for row in target["new_rows"]], dtype=float)
        uv = cv2.perspectiveTransform(native[None], H)[0]
        targets = _split_side([dict(row, xy=p.tolist()) for row, p in zip(target["new_rows"], uv, strict=True)])
    target["points"] = targets
    result = {"target": args.target, "controls_passed": bool(passed), "admitted_target_points": len(targets), "frames": report,
              "source": {"initialization_sha256": digest(args.initialization), "video_sha256": scene["video_sha256"],
                         "known_native_neck_sha256": digest(args.neck_points),
                         "sparse_sha256": {path.name: digest(path) for path in (sparse / f for f in ("cameras.bin", "images.bin", "points3D.bin"))}}}
    with args.output.open("x", encoding="utf8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(f"{len(report) - 1} control frames, {len(errors)} measured control sites, {len(target['new_rows'])} target rows, {len(targets)} admitted target points", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "initialization", "neck-points", "gaussian-repo", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--target", required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
