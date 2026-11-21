"""Detect visible neck edges as agreement between face parsing and a native matte."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np


NECK = 14
EXCLUDED = (15, 16, 17, 18)  # necklace, clothes, hair, hat
MIN_NECK_PROBABILITY = 0.55
MAX_EXCLUDED_PROBABILITY = 0.25
MIN_SPLIT_ROW_DISTANCE = 4
MAX_BRIDGE_DISTANCE = 16
MAX_BRIDGE_CHROMA_DISTANCE = 12.0
MAX_BRIDGE_LIGHTNESS_DISTANCE = 35.0


def _skin_bridge(x, direction, component, neck, excluded, lab):
    columns = x + direction * np.arange(1, MAX_BRIDGE_DISTANCE + 1)
    columns = columns[(columns >= 0) & (columns < len(component))]
    blocked = ~component[columns] | (excluded[columns] > MAX_EXCLUDED_PROBABILITY)
    if blocked.any():
        columns = columns[:np.flatnonzero(blocked)[0]]
    high = neck[columns] >= MIN_NECK_PROBABILITY
    anchors = np.flatnonzero(high[:-2] & high[1:-1] & high[2:]) if len(high) >= 3 else ()
    for start in anchors:
        end = start + 3
        pixels = lab[columns[:end]]
        reference = np.median(pixels[start:end], axis=0)
        difference = np.abs(pixels - reference)
        if (np.linalg.norm(difference[:, 1:], axis=1).max() <= MAX_BRIDGE_CHROMA_DISTANCE
                and difference[:, 0].max() <= MAX_BRIDGE_LIGHTNESS_DISTANCE):
            return float(np.median(neck[columns[start:end]])), float(excluded[columns[:end]].max()), end
    return None


def _side_rows(component, alpha, neck, excluded, side, lab=None):
    rows = np.flatnonzero(component.any(axis=1))
    direction = 1 if side == "left" else -1
    if not len(rows):
        return []
    edge = np.full(len(component), -1, dtype=np.int32)
    mask = component[rows]
    edge[rows] = mask.argmax(axis=1) if direction == 1 else mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    points = []
    for y in rows[(rows >= 2) & (rows < len(component) - 2)]:
        x = edge[y]
        # A near-horizontal edge is unreliable as a left or right neck silhouette.
        if np.any(edge[[y - 2, y + 2]] < 0) or np.ptp(edge[y - 2:y + 3]) > 6:
            continue
        inside = x + direction * np.array([2, 4, 6])
        outside = x - direction
        if min(inside.min(), outside) < 0 or max(inside.max(), outside) >= component.shape[1]:
            continue
        if not component[y, inside].all() or alpha[y, outside] >= 128:
            continue
        confidence = float(np.median(neck[y, inside]))
        contamination = float(np.max(excluded[y, inside]))
        bridge_distance = None
        if confidence < MIN_NECK_PROBABILITY or contamination > MAX_EXCLUDED_PROBABILITY:
            bridge = None if lab is None else _skin_bridge(x, direction, component[y], neck[y], excluded[y], lab[y])
            if bridge is None:
                continue
            confidence, contamination, bridge_distance = bridge
        a_in, a_out = int(alpha[y, x]), int(alpha[y, outside])
        crossing = x - direction * (a_in - 127.5) / (a_in - a_out)
        point = {"native_xy": [float(crossing), float(y)], "neck_probability": confidence,
                 "excluded_probability": contamination}
        if bridge_distance is not None:
            point["skin_bridge_inward_px"] = int(bridge_distance)
        points.append(point)
    return points


def _split_side(points):
    middle = len(points) // 2
    check = np.arange(middle - 1, middle + 1)
    rows = np.array([point["native_xy"][1] for point in points])
    before = np.flatnonzero(rows <= rows[check[0]] - MIN_SPLIT_ROW_DISTANCE)
    after = np.flatnonzero(rows >= rows[check[-1]] + MIN_SPLIT_ROW_DISTANCE)
    if not len(before) or not len(after):
        return []
    fit = [i for band in (before, after) for i in np.unique(band[[0, -1]])]
    return [dict(points[i], split="fit") for i in fit] + [dict(points[i], split="check") for i in check]


def neck_edge_points(neck_probability, excluded_probability, person_alpha, warp, virtual_size, *, native_bgr=None):
    """Return separate fit and check points in the virtual integer-center raster."""
    if (person_alpha.ndim != 2 or person_alpha.dtype != np.uint8
            or neck_probability.shape != person_alpha.shape or excluded_probability.shape != person_alpha.shape):
        raise ValueError("Native matte and probability maps must have the same raster")
    if warp.shape != (3, 3) or not np.isfinite(warp).all():
        raise ValueError("The native-to-virtual camera warp must be a finite 3x3 matrix")
    lab = (cv2.cvtColor(native_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
           if native_bgr is not None else None)
    if lab is not None and lab.shape[:2] != person_alpha.shape:
        raise ValueError("The original RGB and person alpha must be aligned")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((person_alpha >= 128).astype(np.uint8))
    if count < 2:
        return []
    component = labels == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    result = []
    for side in ("left", "right"):
        candidates = _side_rows(component, person_alpha, neck_probability, excluded_probability, side, lab)
        if not candidates:
            continue
        native = np.asarray([point["native_xy"] for point in candidates], dtype=np.float64)
        pixels = cv2.perspectiveTransform(native[None], warp)[0]
        width, height = virtual_size
        valid = ((pixels >= 2) & (pixels <= [width - 3, height - 3])).all(axis=1)
        usable = [dict(point, xy=pixel.tolist(), side=side) for point, pixel, keep in
                  zip(candidates, pixels, valid, strict=True) if keep]
        if len(usable) < 12:
            continue
        # Keep only a contiguous portion of the visible skin edge.
        segments = np.split(np.arange(len(usable)), np.flatnonzero(np.diff(
            [point["native_xy"][1] for point in usable]) > 2) + 1)
        segment = max(segments, key=len)
        if len(segment) < 12:
            continue
        result.extend(_split_side([usable[i] for i in segment]))
    return result


def _load_parser(repo, weights, device):
    import torch

    sys.path.insert(0, str(repo.resolve(strict=True)))
    import models.bisenet as net

    if not Path(net.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError("The loaded face parser is outside the supplied repository")
    original = net.resnet18
    try:
        net.resnet18 = lambda: original(weights=None)
        model = net.BiSeNet(19, "resnet18")
    finally:
        net.resnet18 = original
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def _native_probabilities(model, rgb, warp, native_size, device):
    import torch

    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = torch.as_tensor(resized, device=device).permute(2, 0, 1).float() / 255.
    mean = x.new_tensor([.485, .456, .406]).reshape(3, 1, 1)
    std = x.new_tensor([.229, .224, .225]).reshape(3, 1, 1)
    with torch.inference_mode():
        probabilities = model(((x - mean) / std)[None])[0].softmax(1)[0]
        channels = torch.stack((probabilities[NECK], probabilities[list(EXCLUDED)].amax(dim=0))).cpu().numpy()
    return [cv2.warpPerspective(cv2.resize(channel, (w, h), interpolation=cv2.INTER_LINEAR),
                               warp, native_size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            for channel in channels]


def run(args):
    model = _load_parser(args.parser_repo, args.weights, args.device)
    from fit_side_frames import alpha_maps
    from guided_vhap_input import _native_cameras

    source = json.loads((args.scene / "source-frames.json").read_text(encoding="utf-8"))
    initialization = json.loads(args.initialization.read_text(encoding="utf-8"))
    records = initialization["frames"]
    frames = {frame["name"]: frame for frame in source["frames"]}
    if (not records or len(records) != len(frames) or {r["name"] for r in records} != set(frames)
            or [r["timestep"] for r in records] != list(range(len(records)))):
        raise ValueError("VHAP frames and native source manifest disagree")
    K, size, _ = _native_cameras(args.scene)
    _, _, parity, remap, raw_size = alpha_maps(args.scene, size, K, [records[0]["name"]])
    detections = {}
    for record in records:
        name, index = record["name"], record["timestep"]
        native_bytes = (args.scene / "orig_undist/images" / name).read_bytes()
        source_hash = sha256(native_bytes).hexdigest()
        if source_hash != frames[name]["native_sha256"] or record["timestamp_s"] != frames[name]["timestamp_s"]:
            raise ValueError(f"The native source frame or its timestamp changed: {name}")
        raw = (args.scene / "alpha_maps" / name).read_bytes()
        matte = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        native_bgr = cv2.imdecode(np.frombuffer(native_bytes, np.uint8), cv2.IMREAD_COLOR)
        crop = (args.sequence / "images" / f"{index:08d}.jpg").read_bytes()
        bgr = cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR)
        if matte is None or matte.shape[::-1] != raw_size or bgr is None or native_bgr is None:
            raise ValueError(f"Missing valid RVM or VHAP crop input: {name}")
        warp = np.asarray(record["warp"], dtype=np.float64)
        warped = cv2.remap(matte, *remap, interpolation=cv2.INTER_LINEAR)
        neck, excluded = _native_probabilities(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                                warp, size, args.device)
        points = neck_edge_points(neck, excluded, warped, warp, bgr.shape[1::-1], native_bgr=native_bgr)
        detections[name] = {"timestep": index, "timestamp_s": record["timestamp_s"],
                            "native_sha256": source_hash, "raw_matte_sha256": sha256(raw).hexdigest(),
                            "virtual_crop_sha256": sha256(crop).hexdigest(), "points": points}
        print(f"{name}: {len(points)} confident neck-edge points", flush=True)
    result = {"schema": "vhap-neck-targets-v1", "source": {
        "video_sha256": source["video_sha256"], "weights_sha256": sha256(args.weights.read_bytes()).hexdigest(),
        "initialization_sha256": sha256(args.initialization.read_bytes()).hexdigest(),
        "matte_remap_iou_first_frame": parity, "pixel_convention": "pixel_integer_centers",
        "side_convention": "native_image_left_and_right", "neck_class": NECK, "excluded_classes": EXCLUDED,
        "fit_check_min_row_distance_px": MIN_SPLIT_ROW_DISTANCE,
        "minimum_neck_probability": MIN_NECK_PROBABILITY,
        "maximum_excluded_probability": MAX_EXCLUDED_PROBABILITY,
        "skin_bridge_max_distance_px": MAX_BRIDGE_DISTANCE,
        "skin_bridge_max_chroma_distance_lab8": MAX_BRIDGE_CHROMA_DISTANCE,
        "skin_bridge_max_lightness_distance_lab8": MAX_BRIDGE_LIGHTNESS_DISTANCE}, "frames": detections}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "sequence", "initialization", "parser-repo", "weights", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    cv2.setNumThreads(1)
    run(parser.parse_args())
