"""Export the fitted VHAP profile correction as a static scene PLY."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fuse_scene import (local_head, make_rasterizer, prepare_scene, read_gaussians,
                        render, world_gaussians)
from profile_shape import PROFILE_LIMITS, profile_basis, profile_weights
from train_joint import JointParameters, combine, export


@torch.no_grad()
def run(run_dir, output):
    model = run_dir / "model"
    metrics = json.loads((model / "metrics.json").read_text())
    profile = run_dir / "profile-refinement/profile-refinement.json"
    profile_bytes = profile.read_bytes()
    fitted = json.loads(profile_bytes)
    coefficients = np.asarray(fitted["scalp_and_posterior_neck_flame_units"])
    if (not fitted["shape_candidate_passed_contour_comparison"] or coefficients.shape != (2,)
            or not all(lo <= value <= hi for value, (lo, hi) in zip(coefficients, PROFILE_LIMITS))):
        raise ValueError("Expected a validated, bounded profile correction.")

    reference = metrics["reference"]
    settings = SimpleNamespace(export_frame=reference,
                               **{key: Path(value) for key, value in metrics["input_paths"].items()})
    K, size, cameras, names, _, track, actor, keep, bgfile, source_scene = prepare_scene(settings)
    placement = metrics["placement"][reference]
    rotation, translation = (np.asarray(placement[key]) for key in ("rotation", "translation"))
    scale = metrics["scale"]
    binding = actor.binding[keep].long()
    local_head(actor, track, 0)
    reference_sizes = actor.face_scaling[binding].reshape(-1, 1, 1).clone()
    local_head(actor, track, names.index(reference))

    def reference_head():
        head = world_gaussians(actor, rotation, translation, scale, keep)
        axes = head[0].new_tensor(rotation) @ actor.face_orien_mat[binding]
        axes *= scale * actor.face_scaling[binding].reshape(-1, 1, 1) / reference_sizes
        return SimpleNamespace(head=head, axes=axes)

    frame = reference_head()
    params = JointParameters(source_scene, frame.head)
    checkpoint = torch.load(model / "training_state.pt", weights_only=True, map_location="cpu")
    params.load_state_dict(checkpoint["parameters"])
    scene = params.scene(source_scene)
    views = [SimpleNamespace(name=name, rasterizer=make_rasterizer(K, size, *cameras[Path(name).stem]))
             for name in [reference, fitted["heldout_frames"][0], fitted["heldout_frames"][-2]]]
    old_ply = read_gaussians(model / "point_cloud.ply")
    baseline_delta = float((render(views[0].rasterizer, old_ply)
                            - render(views[0].rasterizer, combine(scene, params.head(frame, geometry=True))))
                           .abs().max())
    if not np.isfinite(baseline_delta) or baseline_delta > 1e-3:
        raise RuntimeError(f"Saved reference PLY differs from the reconstructed base: {baseline_delta:.6g}")

    weights, protected = profile_weights(actor, settings.flame_boundary_mask)
    basis = profile_basis(actor, weights)
    vertices = actor.verts + (basis * basis.new_tensor(coefficients)[None, :, None]).sum(dim=1)[None]
    if not torch.isfinite(vertices).all() or not torch.equal(vertices[:, protected], actor.verts[:, protected]):
        raise RuntimeError("Invalid deformation or movement of protected landmarks.")
    actor.update_mesh_properties(vertices, actor.verts_cano.clone())

    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(output=output, background=settings.background, readback_frames=views)
    report = export(args, K, bgfile, scene, views[0], params.head(reference_head(), geometry=True))
    if not np.isfinite(report["readback_rgb_max"]):
        raise RuntimeError("Non-finite read-back error.")
    report.update(static_pose=True, coordinate_system="source_COLMAP_world",
                  zero_correction_baseline_rgb_max=baseline_delta, profile_sha256=sha256(profile_bytes).hexdigest(),
                  scalp_and_posterior_neck_flame_units=coefficients.tolist())
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)
