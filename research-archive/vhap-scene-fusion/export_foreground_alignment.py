"""Export identical foreground observations in scene and VHAP image coordinates."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def export(run, output):
    def read(path):
        return json.loads((run / path).read_bytes())

    sequence = run / "head-neck"
    frames = read("head-neck/initialization.json")["frames"]
    names = [row["name"] for row in frames]
    split = read("split.json")
    if names != split["source_names"] or len(set(names)) != len(names):
        raise ValueError("Source frames and registered VHAP cameras differ")
    held = set(split["held_frames"])
    if not held.issubset(names):
        raise ValueError("Unknown held-out frame")
    data = {k: [] for k in ("scene_xy", "vhap_xy", "label", "scene_alpha_u8",
                             "scene_usable", "in_vhap_crop", "in_vhap_hair_roi")}
    starts = [0]
    warps = []
    with np.load(run / "outline-targets/outline-targets.npz", allow_pickle=False) as targets:
        if targets["names"].tolist() != names:
            raise ValueError("VHAP hair observations and frame names differ")
        hair_roi = targets["hair"]
    for index, row in enumerate(frames):
        stem = Path(row["name"]).stem
        path = run / "native-outer-contours" / (stem + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["edge"]
            y, x = np.nonzero(labels)
            native = np.column_stack((x, y)) + archive["origin"]
            data["label"].append(labels[y, x])
            data["scene_alpha_u8"].append(archive["alpha"][y, x])
            data["scene_usable"].append(archive["usable"][y, x].astype(bool))
        warp = np.asarray(row["warp"], dtype=np.float64)
        warps.append(warp)
        if row["timestep"] != index or warp.shape != (3, 3) or not np.isfinite(warp).all():
            raise ValueError(f"Invalid frame transform for {stem}")
        crop = cv2.imread(str(sequence / "alpha_maps" / f"{index:08d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if crop is None or crop.shape != hair_roi[index].shape:
            raise ValueError(f"Incompatible VHAP alpha or hair region for {stem}")
        projection = np.column_stack((native, np.ones(len(native)))) @ warp.T
        if (np.abs(projection[:, 2]) < 1e-10).any():
            raise ValueError(f"Undefined crop coordinates for {stem}")
        uv = projection[:, :2] / projection[:, 2:]
        height, width = crop.shape
        valid = ((uv >= 0).all(1) & (uv <= [width - 1, height - 1]).all(1)
                 & (native[:, 1] <= row["native_head_neck_bottom_px"]))
        in_roi = np.zeros(len(uv), bool)
        u, v = np.rint(uv[valid]).astype(np.int32).T
        in_roi[valid] = hair_roi[index, v, u]
        data["scene_xy"].append(native.astype(np.int32))
        data["vhap_xy"].append(uv.astype(np.float32))
        data["in_vhap_crop"].append(valid)
        data["in_vhap_hair_roi"].append(in_roi)
        starts.append(starts[-1] + len(native))
    # starts[i]:starts[i+1] addresses the same physical image samples in every field.
    np.savez_compressed(output, **{k: np.concatenate(v) for k, v in data.items()},
                        names=np.asarray(names), frame_starts=starts,
                        scene_to_vhap=np.stack(warps), scene_holdout=[n in held for n in names],
                        label_names=["unlabeled", *read("native-outer-contours/manifest.json")["labels"]],
                        pixel_centers="zero-based integer centers; xy order")
    return starts[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Exported {export(args.run, args.output)} paired foreground samples")
