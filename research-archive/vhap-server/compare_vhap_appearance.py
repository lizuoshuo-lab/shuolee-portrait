"""Replay a fixed VHAP mesh with current and archived native UV textures."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge, label, render_mesh
from fit_vhap_head_neck import load_checkpoint
from prepare_video_avatar import setup_vhap_cameras, write_jpeg


def checksum(array):
    return sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


@torch.no_grad()
def run(args):
    context = json.loads((args.sequence / "initialization.json").read_text())
    records = context["frames"]
    scale = context["scene_scale"]
    _, _, extrinsics, _, _ = camera_gauge(records, scale)
    setup_vhap_cameras(extrinsics)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "track"),
        "--batch-size", "4",
    ])
    tracker = GlobalTracker(cfg)
    load_checkpoint(tracker, args.current)
    names = ["00450.jpg", "00560.jpg", "00840.jpg"]
    indices = [[r["name"] for r in records].index(name) for name in names]
    sample = default_collate([tracker.dataset[i] for i in indices])
    truth = np.rint(sample["rgb"].cpu().numpy().transpose(0, 2, 3, 1)*255).astype(np.uint8)[..., ::-1]
    images = []
    log = {"source_frames": names, "tex_extra_sha256": {}, "settings": "Only the native shared UV residual is swapped; geometry, poses, lights, camera and output are fixed."}
    for name, path in [("Current", args.current), ("Old front 18", args.front18), ("Old front 17", args.front17)]:
        with np.load(path, allow_pickle=False) as z:
            tex = z["tex_extra"]
            if list(tex.shape) != list(tracker.tex_extra.shape) or not np.isfinite(tex).all():
                raise ValueError(f"Invalid native UV residual at {path}")
            log["tex_extra_sha256"][name] = checksum(tex)
            tracker.tex_extra.copy_(torch.as_tensor(tex, device=tracker.device))
        rgba, _ = render_mesh(tracker, sample)
        a = np.clip(rgba[:, 3], 0, 1)
        if images and not np.array_equal(a, visibility):
            raise RuntimeError("Texture swap changed rendered coverage")
        visibility = a
        rgb = rgba[:, :3].transpose(0, 2, 3, 1)
        images.append(np.rint(np.clip(rgb*a[..., None] + .12*(1-a[..., None]), 0, 1)*255).astype(np.uint8)[..., ::-1])
    for path in args.fingerprint:
        with np.load(path, allow_pickle=False) as z:
            log["tex_extra_sha256"][str(path)] = checksum(z["tex_extra"])
    with (args.output / "appearance-check.json").open("x", encoding="utf8") as file:
        json.dump(log, file, indent=2)
    for j, name in enumerate(names):
        row = [label(truth[j], f"{name} | source")]
        row += [label(frame[j], title) for frame, title in zip(images, ("Current texture", "Old 18-view texture", "Old 17-view texture"), strict=True)]
        write_jpeg(args.output / name, np.hstack(row))
    tracker.tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "current", "front18", "front17", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, nargs="*", default=[])
    run(parser.parse_args())
