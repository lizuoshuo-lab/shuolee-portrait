"""Extract fixed native-pixel face-skin witnesses for calibrated VHAP review."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fit_side_frames import alpha_maps
from guided_vhap_input import _native_cameras
from learned_profile_pose import profile_pose_crop
from prepare_vhap_outline_targets import PARSER_WEIGHTS, consistent_probabilities, load_parser


def decode(path, mode):
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), mode)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image, sha256(payload).hexdigest()


def run(args):
    if sha256(args.weights.read_bytes()).hexdigest() != PARSER_WEIGHTS:
        raise ValueError("Unexpected parser weights")
    init_bytes = (args.sequence / "initialization.json").read_bytes()
    records = json.loads(init_bytes)["frames"]
    source = json.loads((args.scene / "source-frames.json").read_text())["frames"]
    if [row["name"] for row in source] != [row["name"] for row in records]:
        raise ValueError("Native images and VHAP camera order disagree")
    K, size, views = _native_cameras(args.scene)
    _, _, parity, remap, raw_size = alpha_maps(args.scene, size, K, [records[0]["name"]])
    model = load_parser(args.parser_repo, args.weights, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for i, (row, provenance) in enumerate(zip(records, source, strict=True)):
        if abs(row["baseline_yaw_deg"]) < 55:
            continue
        name = row["name"]
        if int(row["timestep"]) != i or name != Path(name).name:
            raise ValueError("Bad frame identity")
        R, C = views[name]
        H = np.asarray(row["warp"])
        V = np.asarray(row["virtual_from_native_cv"])
        if (not np.allclose(H @ K, np.asarray(row["K_integer_pixel_centers"]) @ V, atol=1e-6)
                or not np.allclose(R @ V.T, row["world_from_virtual"], atol=1e-6)
                or not np.allclose(C, row["camera_center_world"], atol=1e-6)):
            raise ValueError(f"Crop is not registered: {name}")
        bgr, rgb_hash = decode(args.scene / "orig_undist/images" / name, cv2.IMREAD_COLOR)
        raw_alpha, matte_hash = decode(args.scene / "alpha_maps" / name, cv2.IMREAD_GRAYSCALE)
        crop_bytes = (args.sequence / "images" / f"{i:08d}.jpg").read_bytes()
        coarse, coarse_hash = decode(args.scene / "headmask_undist/images" / Path(name).with_suffix(".png"),
                                     cv2.IMREAD_GRAYSCALE)
        if (rgb_hash != provenance["native_sha256"] or bgr.shape[:2][::-1] != size
                or raw_alpha.shape[::-1] != raw_size):
            raise ValueError(f"Native RGB or alpha changed: {name}")

        _, (x0, y0, x1, y1) = profile_pose_crop(bgr, coarse >= 128)
        if x0 < 0 or y0 < 0 or x1 > size[0] or y1 > size[1]:
            raise ValueError(f"Native head crop touches the image limit: {name}")
        crop = bgr[y0:y1, x0:x1]
        p = consistent_probabilities(model, crop, args.device)
        skin = p[[1, 10, 11, 12, 13]].max(axis=0)
        glasses, ear, nose, mouth = p[6], p[[7, 8]].max(axis=0), p[10], p[[11, 12, 13]].max(axis=0)
        other = p[[14, 15, 16, 17, 18]].max(axis=0)
        eyewear = cv2.dilate((glasses >= .20).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        excluded = eyewear | (other >= .45)
        alpha = cv2.remap(raw_alpha, remap[0][y0:y1, x0:x1], remap[1][y0:y1, x0:x1],
                          interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.
        np.savez_compressed(args.output / f"{Path(name).stem}.npz",
                            origin_xy=np.array((x0, y0)), alpha=alpha, skin=(skin >= .55) & ~excluded,
                            excluded=excluded, eyewear=eyewear, ear=ear >= .55, nose=nose >= .45,
                            mouth=mouth >= .40, background=p[0] >= .50, bgr=crop)
        results.append({"name": name, "yaw_deg": row["baseline_yaw_deg"], "native_sha256": rgb_hash,
                        "raw_matte_sha256": matte_hash, "native_head_mask_sha256": coarse_hash,
                        "corresponding_vhap_crop_sha256": sha256(crop_bytes).hexdigest(),
                        "skin_pixels": int(((skin >= .55) & ~excluded).sum()),
                        "eyewear_pixels": int(eyewear.sum())})
    (args.output / "source.json").write_text(json.dumps({
        "native_raster_size": size, "first_matte_remap_iou": parity,
        "vhap_initialization_sha256": sha256(init_bytes).hexdigest(),
        "parser_sha256": PARSER_WEIGHTS, "frames": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("scene", "sequence", "parser-repo", "weights", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    run(parser.parse_args())
