"""Render the original VHAP image_grid for a verified posed mesh archive."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch

from transfer_vhap_neck_anchor import load_neck_track, replay_neck_track
from vhap.model.lbs import vertices2landmarks


@torch.no_grad()
def run(args):
    cold = json.loads(args.cold_report.read_bytes())
    trained = json.loads(args.gaussian_report.read_bytes())
    digest = sha256(args.posed_mesh.read_bytes()).hexdigest()
    if digest != trained["posed_seed"]["sha256"]:
        raise ValueError("The requested mesh differs from the one used for Gaussian training")
    tracker, records, faces, basis, origin, scale = load_neck_track(args)
    names = [record["name"] for record in records]
    if names != cold["frame_names"] or trained["posed_seed"]["frames"] != len(names):
        raise ValueError("Tracking, Gaussian training and source frame lists disagree")
    before, _ = replay_neck_track(tracker)
    with np.load(args.posed_mesh, allow_pickle=False) as tape:
        if (tape["frames"].tolist() != names or not np.array_equal(faces, tape["faces"])
                or not np.allclose(before, tape["old"], atol=1e-6, rtol=0.)
                or not np.allclose(basis, tape["basis"], atol=1e-8, rtol=0.)
                or not np.allclose(origin, tape["origin"], atol=1e-8, rtol=0.)
                or not np.isclose(scale, float(tape["scale"]), atol=1e-8, rtol=0.)):
            raise ValueError("Posed mesh archive does not match the registered tracker")
        vertices = torch.as_tensor(tape["vertices"], device=tracker.device, dtype=tracker.translation.dtype)
    if tuple(vertices.shape) != before.shape or not torch.isfinite(vertices).all():
        raise ValueError("Invalid posed vertices")

    source_grid = Path(cold["image_grid"])
    original = tracker.forward_flame
    flame = tracker.flame
    rendered = set()

    def forward(timesteps):
        _, canonical, _, albedo = original(timesteps)
        posed = vertices[timesteps]
        count = len(posed)
        landmarks = vertices2landmarks(posed, flame.faces, flame.full_lmk_faces_idx.repeat(count, 1),
                                        flame.full_lmk_bary_coords.repeat(count, 1, 1))
        rendered.update(torch.as_tensor(timesteps).tolist())
        return posed, canonical, landmarks, albedo

    tracker.forward_flame = forward
    # The external posed mesh cannot be represented by the native pose checkpoint.
    # Evaluate its vertices directly and keep the input checkpoint untouched.
    tracker.save_result = lambda **_: None
    tracker.evaluate(make_visualization=True, epoch=0)
    tracker.tb_writer.close()
    fitted_grid = tracker.out_dir / "eval_0/image_grid"
    expected = {f"frame_{i:05d}.jpg" for i in range(len(names))}
    if (rendered != set(range(len(names))) or {p.name for p in source_grid.glob("*.jpg")} != expected
            or {p.name for p in fitted_grid.glob("*.jpg")} != expected):
        raise RuntimeError("Missing original or posed native image grid frames")
    info = {"frames": names, "original_grid": str(source_grid), "posed_grid": str(fitted_grid),
            "posed_geometry_sha256": digest, "source": "VHAP stock evaluation and image_grid compositor",
            "gaussian_steps": trained["steps"], "texture": "existing VHAP albedo and lighting"}
    (args.output / "grids.json").write_text(json.dumps(info, indent=2)+"\n")
    with ZipFile(args.output / "vhap-native-grids.zip", "x", compression=ZIP_STORED) as out:
        for i, name in enumerate(names):
            for folder, images in (("01_vhap_original", source_grid), ("02_vhap_shared_neck", fitted_grid)):
                out.write(images / f"frame_{i:05d}.jpg", f"{folder}/{name}")
        out.write(args.output / "grids.json", "grids.json")
    print(json.dumps({key: value for key, value in info.items() if key != "frames"}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "registered", "posed-mesh", "cold-report", "gaussian-report", "output"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--pose-report", type=Path)
    parser.add_argument("--rim-report", type=Path)
    args = parser.parse_args()
    if bool(args.pose_report) != bool(args.rim_report):
        parser.error("Pose and rim reports must be paired")
    run(args)

