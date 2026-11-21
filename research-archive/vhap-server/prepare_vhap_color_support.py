"""Freeze reliable hair and face-skin RGB support on the original VHAP crops."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from prepare_vhap_outline_targets import MIN_CONFIDENCE, PARSER_WEIGHTS, consistent_probabilities, load_parser


def color_support(alpha, probability, landmarks):
    h, w = alpha.shape
    scale = min(h, w) / 512
    person = alpha >= 245
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, round(4 * scale)) + 1,) * 2)
    core = person & (probability[17] >= MIN_CONFIDENCE)
    core = cv2.erode(core.astype(np.uint8), erode, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    hair = np.zeros(alpha.shape, dtype=bool)
    if count > 1:
        component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[component, cv2.CC_STAT_AREA] >= 64 * scale ** 2:
            hair = labels == component

    protected = probability[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].max(axis=0) >= .40
    protected = cv2.dilate(protected.astype(np.uint8), erode) != 0
    rois = np.zeros(alpha.shape, dtype=np.uint8)
    inner = landmarks[17:68]
    observed = (inner[:, 2] > 0) & np.isfinite(inner).all(axis=1)
    if observed.sum() >= 20:
        # The original STAR data uses normalized pixel-boundary coordinates.
        xy = np.rint(inner[observed, :2] * [w, h] - .5).astype(np.int32)
        cv2.fillConvexPoly(rois, cv2.convexHull(xy), 1)
        rois = cv2.dilate(rois, erode)
    else:
        # Give skin support near recognized features if STAR has no observation.
        # A narrow lower extent excludes the neck when parser class 1 spills below the jaw.
        facial = probability[[2, 3, 4, 5, 10, 11, 12, 13]].max(axis=0) >= MIN_CONFIDENCE
        x, y, width, height = cv2.boundingRect((facial & person).astype(np.uint8))
        if width >= 5 * scale and height >= 7 * scale:
            lateral = round(46 * scale)
            top, bottom = max(0, y - round(34 * scale)), min(h, y + height + round(15 * scale))
            rois[top:bottom, max(0, x - lateral):min(w, x + width + lateral)] = 1

    stable = (probability[1] >= MIN_CONFIDENCE) & person & (rois != 0) & ~protected
    skin = cv2.erode(stable.astype(np.uint8), erode, borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    if skin.sum() < 64 * scale ** 2:
        skin[:] = False
    return hair, skin


def run(args):
    if args.output.exists() or args.output.suffix != ".npz":
        raise ValueError("Choose a new .npz output for the color support")
    weights_hash = sha256(args.weights.read_bytes()).hexdigest()
    if weights_hash != PARSER_WEIGHTS:
        raise ValueError("Expected the pinned BiSeNet appearance parser checkpoint")
    parser = load_parser(args.parser_repo, args.weights, "cpu")
    names, hair, skin, crop_hashes, matte_hashes = [], [], [], [], []
    with ZipFile(args.frames) as source:
        initialization = source.read("initialization.json")
        records = json.loads(initialization)["frames"]
        with np.load(BytesIO(source.read("landmark2d/STAR.npz")), allow_pickle=False) as points:
            landmarks = points["face_landmark_2d"]
        if landmarks.shape != (len(records), 68, 3):
            raise ValueError("The STAR archive must contain the registered source frames")
        for i, row in enumerate(records):
            if row["timestep"] != i:
                raise ValueError("The VHAP crop archive is out of frame order")
            encoded_crop = source.read(f"images/{i:08d}.jpg")
            encoded_matte = source.read(f"alpha_maps/{i:08d}.jpg")
            image = cv2.imdecode(np.frombuffer(encoded_crop, np.uint8), cv2.IMREAD_COLOR)
            alpha = cv2.imdecode(np.frombuffer(encoded_matte, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or alpha is None or image.shape[:2] != alpha.shape:
                raise ValueError(f"Incompatible RGB and alpha: {row['name']}")
            probability = consistent_probabilities(parser, image, "cpu")
            h, s = color_support(alpha, probability, landmarks[i])
            names.append(row["name"])
            hair.append(h)
            skin.append(s)
            crop_hashes.append(sha256(encoded_crop).hexdigest())
            matte_hashes.append(sha256(encoded_matte).hexdigest())
            if (i + 1) % 20 == 0:
                print(f"Parsed {i + 1}/{len(records)} registered crops", flush=True)
    if not names or len(set(names)) != len(names):
        raise ValueError("Expected unique registered frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, names=names, hair=hair, skin=skin, crop_sha256=crop_hashes,
                            matte_sha256=matte_hashes, initialization_sha256=sha256(initialization).hexdigest(),
                            weight_sha256=weights_hash)
    print(json.dumps({"frames": len(names), "hair_frames": sum(bool(m.any()) for m in hair),
                      "skin_frames": sum(bool(m.any()) for m in skin), "hair_pixels": sum(int(m.sum()) for m in hair),
                      "skin_pixels": sum(int(m.sum()) for m in skin), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    for key in ("frames", "parser-repo", "weights", "output"):
        cli.add_argument(f"--{key}", type=Path, required=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    run(cli.parse_args())
