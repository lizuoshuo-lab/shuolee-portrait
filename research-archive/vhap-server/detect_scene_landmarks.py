"""Recover frame IDs by source bytes and detect STAR landmarks on the native scene raster."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from vhap.util.landmark_detector_star import LandmarkDetectorSTAR


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("scene", "selected", "tracker", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    raw = {}
    for path in (args.scene / "images").glob("*.jpg"):
        raw.setdefault(sha256(path.read_bytes()).digest(), []).append(path.name)
    selected = sorted(args.selected.glob("*.jpg"))
    with np.load(args.tracker) as track:
        if [int(s.stem) for s in selected] != list(track["timestep_id"].astype(int)):
            raise ValueError("Selected images do not match the tracker timesteps.")
    detector = LandmarkDetectorSTAR()
    names, landmarks, native_hashes = [], [], []
    for path in selected:
        candidates = raw.get(sha256(path.read_bytes()).digest(), [])
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one unmodified source image for {path.name}.")
        name = candidates[0]
        native = args.scene / "orig_undist/images" / name
        img = cv2.imread(str(native))
        if img is None:
            raise ValueError(f"Missing undistorted scene frame: {native}")
        bbox, points = detector.detect_single_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if bbox[-1] <= 0 or (points[:, 2] <= 0).any():
            raise ValueError(f"Missing STAR landmarks: {name}")
        names.append(name)
        landmarks.append(points[:, :2].astype(np.float64) * img.shape[1::-1])
        native_hashes.append(sha256(native.read_bytes()).hexdigest())
        print(f"{path.stem} -> {name}, native {img.shape[1]}x{img.shape[0]}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, names=names, xy=landmarks, native_sha256=native_hashes,
             tracker_sha256=sha256(args.tracker.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
