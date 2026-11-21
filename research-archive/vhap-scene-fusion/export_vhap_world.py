"""Export a guided VHAP checkpoint's original posed FLAME meshes in scene coordinates."""

import json
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from prepare_video_avatar import setup_vhap_cameras


@torch.no_grad()
def export_world(sequence, report_path, variant, output):
    sequence, output = sequence.resolve(), output.resolve()
    if output.exists() or output.suffix != ".npz":
        raise ValueError("Choose a new .npz mesh export path")
    initial = json.loads((sequence / "initialization.json").read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = initial["frames"]
    scale = float(report["share"]["single_scene_scale"])
    saved = report["variants"][variant]
    names = [record["name"] for record in records]
    indices = list(range(len(names)))
    if (not names or len(names) != report["frame_count"] or len(set(names)) != len(names)
            or set(names) != set(saved["fitted"]) or [r["timestep"] for r in records] != indices):
        raise ValueError("VHAP sequence and tracking report disagree on source frames")
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale, initial["scene_scale"], rtol=0, atol=1e-9):
        raise ValueError("Invalid shared scene scale")
    _, _, extrinsics, origin, gauge = camera_gauge(records, scale)
    if (not np.allclose(origin, report["share"]["scene_origin"], rtol=0, atol=1e-8)
            or not np.allclose(gauge, report["share"]["scene_to_tracker_rotation"], rtol=0, atol=1e-8)):
        raise ValueError("VHAP sequence and tracking report disagree on the scene gauge")

    setup_vhap_cameras(extrinsics)
    with TemporaryDirectory(prefix="vhap-world-export-") as temp:
        cfg = tyro.cli(BaseTrackingConfig, args=[
            "--data.root-folder", str(sequence.parent), "--data.sequence", sequence.name,
            "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", temp,
            "--batch-size", "4",
        ])
        tracker = GlobalTracker(cfg)
        try:
            if len(tracker.dataset) != len(names) or cfg.model.use_dynamic_offset:
                raise ValueError("Expected a complete shared FLAME sequence")
            poses = ("rotation", "translation", "neck_pose", "jaw_pose", "eyes_pose", "expr")
            with np.load(saved["checkpoint"], allow_pickle=False) as state:
                if (int(state["n_processed_frames"]) != len(names)
                        or state["timestep_id"].tolist() != [f"{i:08d}" for i in indices]):
                    raise ValueError("Checkpoint frame order does not match the VHAP input")
                for key in (*poses, "shape", "static_offset"):
                    target = getattr(tracker, key)
                    values = torch.from_numpy(state[key]).to(target)
                    if target.shape != values.shape or not torch.isfinite(values).all():
                        raise ValueError(f"Invalid checkpoint parameter: {key}")
                    target.copy_(values)

            vertices, roots = [], []
            for start in range(0, len(names), cfg.batch_size):
                ts = torch.as_tensor(indices[start:start + cfg.batch_size], device=tracker.device)
                mesh, canonical, _, _ = tracker.forward_flame(ts)
                # Native LBS rotates about J0, so the posed root only takes translation.
                root = torch.einsum("v,bvc->bc", tracker.flame.J_regressor[0], canonical) + tracker.translation[ts]
                vertices.append(mesh.cpu().numpy() @ gauge * scale + origin)
                roots.append(root.cpu().numpy() @ gauge * scale + origin)
            faces = tracker.flame.faces.cpu().numpy()
            parts = tracker.flame.mask
            face_ids = parts.get_vid_by_region(("face",)).cpu().numpy()
            eyes = parts.get_vid_by_region(("left_eyeball", "right_eyeball")).cpu().numpy()
            face_ids = np.setdiff1d(face_ids, eyes)
            neck_ids = parts.get_vid_by_region(("neck",)).cpu().numpy()
        finally:
            tracker.tb_writer.close()

    vertices_world = np.concatenate(vertices).astype(np.float32)
    root_world = np.concatenate(roots).astype(np.float32)
    if not np.isfinite(vertices_world).all() or not np.isfinite(root_world).all():
        raise RuntimeError("Native VHAP produced non-finite world coordinates")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez_compressed(file, vertices_world=vertices_world, faces=faces, names=np.asarray(names),
                            face_vertex_ids=face_ids, neck_vertex_ids=neck_ids, root_world=root_world,
                            scale=scale)
    print(f"Exported {len(names)} original posed meshes with {len(faces)} faces each to {output}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="baseline")
    args = parser.parse_args()
    export_world(args.sequence, args.report, args.variant, args.output)
