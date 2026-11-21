"""Choose actual opposing cameras and measure a missed neck outline in the receiving image."""

from pathlib import Path

import cv2
import numpy as np

from detect_vhap_neck import _split_side
from supplement_native_neck_front import digest, read_crop, track_control


def fit_rows(rows):
    return [q for q in rows if q["split"] == "fit"]


def has_profile(rows):
    return len({q["native_xy"][1] for q in fit_rows(rows)}) >= 3


def bracketed_posterior(root, records, rows, target):
    """Neighbors identify visible skin. Only the receiving frame supplies silhouette coordinates."""
    yaw = np.array([r["baseline_yaw_deg"] for r in records])
    direction = int(np.sign(yaw[target]))  # outward from the posterior edge
    side = "right" if direction > 0 else "left"
    name = records[target]["name"]
    def donor(i):
        return [q for q in fit_rows(rows[i]) if q["region"] == "posterior" and q["side"] == side
                and q["provider"] in ("current_crop", "native_fallback")]
    times = np.array([r["timestamp_s"] for r in records])
    eligible = [i for i in range(len(records)) if abs(times[i] - times[target]) <= 2.1
                and direction * yaw[i] >= 55 and len(donor(i)) >= 3]
    earlier = [i for i in eligible if i < target]
    later = [i for i in eligible if i > target]
    info = dict(target=name, donors=[], used_images=[], points=[])
    if not earlier or not later:
        return [], dict(info, reason="no_bracketing_semantic_fit_observations")
    left, right = max(earlier), min(later)
    journey = list(range(left, right + 1))
    if np.max(np.diff(times[journey])) > .45:
        return [], dict(info, reason="nonconsecutive_source_frames")
    def rgb(i):
        return root/"scene/orig_undist/images"/records[i]["name"]
    gray = {i:cv2.imread(str(rgb(i)), cv2.IMREAD_GRAYSCALE) for i in journey}
    info.update(donors=[records[i]["name"] for i in (left, right)],
                used_images=[records[i]["name"] for i in journey], source_sha256=digest(rgb(target)))
    paths = (list(range(left,target+1)), list(range(right,target-1,-1)))
    flows = [track_control(gray, donor(path[0]), path, target, direction) for path in paths]
    info["bidirectional_skin_sites"] = [f.tolist() for f in flows]
    if min(map(len, flows)) < 2:
        return [], dict(info,reason="insufficient_bidirectional_skin_tracks")
    archive = root/"native-regions"/Path(name).with_suffix(".npz")
    bgr, alpha, origin = read_crop(archive, cv2.imread(str(rgb(target))))
    info["crop_sha256"] = digest(archive)
    lab = cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(float)
    h, w = alpha.shape
    lo = max(0, int(np.ceil(max(f[:,1].min() for f in flows) - origin[1])))
    hi = min(h-1, int(np.floor(min(f[:,1].max() for f in flows) - origin[1])))
    accepted = []
    for y in range(lo, hi+1):
        yy = int(y+origin[1])
        span = np.flatnonzero(alpha[y] >= .5)
        if not len(span):
            continue
        x = int(span[-1] if direction > 0 else span[0])
        if x < 7 or x >= w-7:
            continue
        also = x + direction
        cross = x + direction*(alpha[y,x]-.5)/(alpha[y,x]-alpha[y,also])
        xx = float(cross + origin[0])
        # Both tracks must independently localize this neck band in native pixels.
        if any(abs(xx - np.interp(yy, f[np.argsort(f[:,1]),1], f[np.argsort(f[:,1]),0])) > 3 for f in flows):
            continue
        inside, outside = x-direction*np.arange(3,7), x+direction*np.arange(3,7)
        contrast = np.linalg.norm(np.median(lab[y,inside],0) - np.median(lab[y,outside],0))
        if min(alpha[y,inside]) < .85 or max(alpha[y,outside]) > .20 or contrast < 15:
            continue
        accepted.append(dict(native_xy=[xx,yy], side=side, region="posterior",
                             provider="receiving_matte_rgb_bracketed_skin"))
    groups = np.split(np.arange(len(accepted)), np.flatnonzero(np.diff([q["native_xy"][1] for q in accepted]) > 2)+1)
    segment = max(groups,key=len)
    points = _split_side([accepted[int(i)] for i in segment]) if len(segment) >= 12 else []
    info.update(verified_band_rows=len(segment), points=points)
    return points, info


def select_both(root, records, rows):
    yaw = np.array([r["baseline_yaw_deg"] for r in records])
    train, held, used, receipt = [], [], set(), {}
    for sign in (1,-1):
        cameras = [i for i in range(len(yaw)) if sign*yaw[i] >= 55]
        if not cameras:
            receipt[str(sign)] = {"reason":"no_observed_side_camera"}
            continue
        farthest = max(cameras,key=lambda i:sign*yaw[i])
        profiles = [i for i in cameras if 80 <= sign*yaw[i] <= 100 and has_profile(rows[i])]
        target = min(profiles,key=lambda i:(abs(abs(yaw[i])-90),i)) if profiles else farthest
        item = dict(farthest=records[farthest]["name"],farthest_yaw=float(yaw[farthest]),
                    selected=records[target]["name"],selected_yaw=float(yaw[target]))
        if not has_profile(rows[target]):
            extra, evidence = bracketed_posterior(root, records, rows, target)
            rows[target].extend(extra)
            item["receiving_frame_measurement"] = evidence
            used.update(evidence["used_images"])
        if has_profile(rows[target]):
            train.append(target)
        receipt[str(sign)] = item
        # Hold whole cameras outside the semantic donors and their flow paths.
        candidates = [i for i in cameras if i != target and records[i]["name"] not in used
                      and abs(yaw[i]-yaw[target]) <= 10 and has_profile(rows[i])]
        held.extend(sorted(candidates,key=lambda i:(abs(yaw[i]-yaw[target]),i))[:2])
    return train, list(dict.fromkeys(train+held)), receipt
