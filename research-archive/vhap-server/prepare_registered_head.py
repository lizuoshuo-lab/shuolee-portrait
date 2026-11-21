"""Convert a calibrated VHAP checkpoint to a GaussianAvatars seed without fitting."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


def check_checkpoint_frames(state, ordered_ids, filenames, checkpoint, training_result, pose_result, initialization):
    if state["timestep_id"].tolist() != ordered_ids:
        raise ValueError("Checkpoint timestep IDs differ from the registered images")
    expected = {f"frame_{i:05d}.jpg" for i in range(len(ordered_ids))}
    if pose_result is not None:
        result = json.loads(pose_result.read_bytes())
        grid = Path(result["image_grid"])
        if (result["status"] != "accepted_pose_candidate"
                or result["frames"] != len(ordered_ids) or list(result["frame_results"]) != filenames
                or result["source"]["initialization_sha256"] != sha256(initialization).hexdigest()
                or Path(result["checkpoint"]).resolve() != checkpoint
                or {p.name for p in grid.glob("*.jpg")} != expected):
            raise ValueError("The accepted pose result does not match the calibrated sequence")
        allowed = {"rotation", "translation", "neck_pose"}
        if result["previous_neck_mesh_frozen"]:
            previous_path = Path(result["previous_checkpoint"]).resolve(strict=True)
        else:
            if set(result["updated_fields"]) != allowed | {"static_offset"}:
                raise ValueError("Expected a verified local neck mesh refinement")
            previous_path = Path(result["source"]["checkpoint"]).resolve(strict=True)
        with np.load(previous_path, allow_pickle=False) as previous:
            if not result["previous_neck_mesh_frozen"]:
                from scipy.spatial.transform import Rotation

                check_checkpoint_frames(previous, ordered_ids, filenames, previous_path, training_result, None, initialization)
                pose = Rotation.from_rotvec(state["rotation"]) * Rotation.from_rotvec(state["neck_pose"])
                before = Rotation.from_rotvec(previous["rotation"]) * Rotation.from_rotvec(previous["neck_pose"])
                changed = np.any(state["static_offset"] != previous["static_offset"], axis=(0, 2))
                if (np.count_nonzero(changed) > result["shape_affected_vertices"]
                        or np.max((before.inv() * pose).magnitude()) > 1e-6):
                    raise ValueError("Local neck refinement changed the protected head transform or mesh support")
                allowed.add("static_offset")
            if set(previous.files) != set(state.files) or any(
                    not np.array_equal(previous[key], state[key]) for key in previous.files if key not in allowed):
                raise ValueError("The accepted pose refinement changed a frozen input")
        neck_x = [result["frame_results"][name]["flame_neck_x_deg"][1] for name in filenames]
        if not np.allclose(np.degrees(state["neck_pose"][:, 0]), neck_x, atol=1e-5, rtol=0):
            raise ValueError("Checkpoint pitch differs from the verified pose result")
        return
    processed = int(state["n_processed_frames"])
    if processed == len(ordered_ids):
        return
    # Stock global VHAP exports self.timestep as zero even after fitting the full clip.
    if processed != 0 or training_result is None:
        raise ValueError("Provide --training-result for a complete stock global VHAP checkpoint")
    cold = json.loads(training_result.read_bytes())
    image_grid = Path(cold["image_grid"])
    if (Path(cold["checkpoint"]).resolve() != checkpoint or cold["frame_names"] != filenames
            or cold["seeded_count"] != len(ordered_ids) or cold["trained_steps"] <= 0
            or {p.name for p in image_grid.glob("*.jpg")} != expected):
        raise ValueError("The completed VHAP run report does not match this checkpoint and source sequence")


def extract(sequence, checkpoint, output, reference, training_result, pose_result):
    import torch
    import torch.nn.functional as F
    import tyro
    from vhap.config.base import BaseTrackingConfig
    from vhap.model import tracker as upstream_tracker

    from fit_guided_vhap import camera_gauge
    from fit_vhap_head_neck import STATE_KEYS
    from prepare_video_avatar import setup_vhap_cameras

    if Path(upstream_tracker.__file__).resolve() != Path("vhap/model/tracker.py").resolve():
        raise RuntimeError("VHAP loaded from a different checkout")
    initialization = (sequence / "initialization.json").read_bytes()
    context = json.loads(initialization)
    records = context["frames"]
    filenames = [record["name"] for record in records]
    names = [Path(name).stem for name in filenames]
    indices = list(range(len(names)))
    if (len(set(names)) != len(names) or reference not in names
            or [record["timestep"] for record in records] != indices):
        raise ValueError("Expected one registered camera per frame and a valid reference")
    scale = float(context["scene_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid scene scale")
    cameras, orientations, extrinsics, origin, gauge = camera_gauge(records, scale)
    if (not np.isfinite(extrinsics).all()
            or not np.allclose(gauge.T @ gauge, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(gauge), 1., rtol=0, atol=1e-6)):
        raise ValueError("Invalid rotation or camera in the VHAP scene gauge")
    setup_vhap_cameras(extrinsics)

    with TemporaryDirectory(prefix="vhap-registered-head-") as temp, torch.no_grad():
        cfg = tyro.cli(BaseTrackingConfig, args=[
            "--data.root-folder", str(sequence.parent), "--data.sequence", sequence.name,
            "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", temp,
            "--no-async-func",
        ])
        tracker = upstream_tracker.GlobalTracker(cfg)
        try:
            if (not tracker.calibrated or tracker.n_timesteps != len(names) or cfg.model.use_dynamic_offset
                    or not (cfg.model.tex_painted and cfg.model.tex_extra and cfg.model.residual_tex)):
                raise ValueError("Expected the calibrated stock FLAME model and shared residual texture")
            with np.load(checkpoint, allow_pickle=False) as state:
                check_checkpoint_frames(state, list(tracker.dataset.timestep_ids), filenames, checkpoint,
                                        training_result, pose_result, initialization)
                for key in STATE_KEYS:
                    target = getattr(tracker, key)
                    value = torch.from_numpy(state[key]).to(target)
                    if value.shape != target.shape or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid VHAP parameter: {key}")
                    target.copy_(value)
            ref = names.index(reference)
            flame = tracker.flame
            posed, roots = [], []
            for batch in torch.arange(len(names), device=tracker.device).split(cfg.batch_size):
                vertices, canonical, _, _ = tracker.forward_flame(batch)
                roots.append((flame.J_regressor[0] @ canonical + tracker.translation[batch]).cpu().numpy())
                posed.append(vertices.cpu().numpy())
            template = flame.v_template
            radius = float((template - template.mean(dim=0)).norm(dim=1).max())
            faces = flame.faces.cpu().numpy()
            uv = flame.verts_uvs[flame.textures_idx.long()].mean(dim=1)
            # Match native VHAP's image-row UV convention and bilinear texel centers.
            grid = torch.stack((2 * uv[:, 0] - 1, 1 - 2 * uv[:, 1]), dim=-1)
            colors = F.grid_sample(tracker.get_albedo(), grid[None, :, None], mode="bilinear",
                                   padding_mode="border", align_corners=False)[0, :, :, 0].T
            colors = colors.clamp(0, 1).cpu().numpy()
            image_size = np.asarray(tracker.image_size)[::-1]
        finally:
            tracker.tb_writer.close()

    vertices_tracker = np.concatenate(posed)
    root_tracker = np.concatenate(roots)
    if radius <= 0 or any(not np.isfinite(x).all() for x in (radius, vertices_tracker, root_tracker, colors)):
        raise ValueError("Nonfinite seed geometry, texture, or FLAME template radius")
    registration = output / "registered-head.npz"
    np.savez_compressed(registration, names=np.asarray(names), filenames=np.asarray(filenames),
                        faces=faces, vertices_tracker=vertices_tracker, root_tracker=root_tracker,
                        scene_scale=scale, scene_origin=origin, tracker_to_world_rotation=gauge.T,
                        checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                        initialization_sha256=sha256(initialization).hexdigest(), reference_index=ref,
                        reference_K=np.asarray(records[ref]["K_integer_pixel_centers"]),
                        reference_tracker_from_camera=orientations[ref], reference_camera_tracker=cameras[ref],
                        reference_image_size=image_size)
    texture = output / "vhap-face-seed.npz"
    np.savez(texture, faces=faces, colors=colors)
    return registration, texture, radius


def seed_and_verify(checkpoint, texture, bound, registration, radius):
    import cv2
    import torch
    from scene import FlameGaussianModel
    from utils.sh_utils import RGB2SH

    from fuse_scene import make_rasterizer, render, to_bgr
    from prepare_video_avatar import HEAD_SEED_AXES, HEAD_SEED_OPACITY, write_bound_head

    write_bound_head(checkpoint, texture, bound, radius)
    with np.load(registration, allow_pickle=False) as registered, np.load(texture, allow_pickle=False) as appearance, torch.no_grad():
        actor = FlameGaussianModel(3)
        actor.load_ply(str(bound), has_target=False)
        device = actor.binding.device
        faces = torch.as_tensor(registered["faces"], device=device, dtype=torch.long)
        vertices = torch.as_tensor(registered["vertices_tracker"], device=device)
        colors = torch.as_tensor(appearance["colors"], device=device)
        expected_binding = torch.arange(len(faces), device=device)
        features = actor.get_features
        if (not torch.equal(actor.binding, expected_binding)
                or not torch.equal(actor.flame_model.faces, faces)
                or not torch.allclose(features[:, 0], RGB2SH(colors), atol=1e-7, rtol=0)
                or torch.count_nonzero(features[:, 1:]).item() != 0
                or not torch.allclose(actor.get_opacity, torch.full_like(actor.get_opacity, HEAD_SEED_OPACITY))):
            raise RuntimeError("The bound PLY changed seed topology, RGB, or opacity on reload")

        max_vertex_delta = 0.
        max_center_delta = 0.
        for i, mesh in enumerate(vertices):
            actor.select_mesh_by_timestep(i)
            max_vertex_delta = max(max_vertex_delta, float((actor.verts[0] - mesh).abs().max()))
            max_center_delta = max(max_center_delta, float((actor.get_xyz - mesh[faces].mean(dim=1)).abs().max()))
        if max(max_vertex_delta, max_center_delta) > 1e-5:
            raise RuntimeError(f"GA does not reproduce native VHAP poses: {max_vertex_delta=}, {max_center_delta=}")

        ref = int(registered["reference_index"])
        actor.select_mesh_by_timestep(ref)
        projector = make_rasterizer(registered["reference_K"], tuple(map(int, registered["reference_image_size"])),
                                    registered["reference_tracker_from_camera"], registered["reference_camera_tracker"])
        model = (actor.get_xyz, features, actor.get_opacity, actor.get_scaling, actor.get_rotation)
        if any(not torch.isfinite(x).all() for x in model) or (model[3] <= 0).any():
            raise RuntimeError("Bound seed contains invalid Gaussian transforms")
        rgb = render(projector, model)
        coverage = render(projector, model, alpha_only=True)
        supported_pixels = int((coverage > .5).sum())
        if not torch.isfinite(rgb).all() or not torch.isfinite(coverage).all() or supported_pixels < 256:
            raise RuntimeError("Bound seed did not render in the registered reference camera")
        for name, value in (("reference-rgb.jpg", to_bgr(rgb)),
                            ("reference-alpha.png", np.rint(coverage.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))):
            if not cv2.imwrite(str(bound.parent.parent / name), value):
                raise OSError(f"Could not write {name}")
        return dict(gaussians=len(faces), frames=len(vertices), reference=str(registered["names"][ref]),
                    maximum_tracker_vertex_delta=max_vertex_delta, maximum_bound_center_delta=max_center_delta,
                    reference_pixels_alpha_gt_half=supported_pixels, opacity=HEAD_SEED_OPACITY,
                    scale_in_triangle_units=HEAD_SEED_AXES)


def run(args):
    sequence, checkpoint, output = (path.resolve() for path in (args.sequence, args.checkpoint, args.output))
    if not sequence.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("The registered VHAP input or checkpoint does not exist")
    home = Path.home()
    vhap_repo = args.vhap_repo.resolve(strict=True)
    if not (vhap_repo / "vhap/model/tracker.py").is_file():
        raise ValueError(f"Expected the upstream VHAP checkout at {vhap_repo}")
    sys.path.insert(0, str(vhap_repo))
    os.chdir(vhap_repo)
    output.mkdir(parents=True, exist_ok=False)
    training_result = args.training_result.resolve(strict=True) if args.training_result is not None else None
    pose_result = args.pose_result.resolve(strict=True) if args.pose_result is not None else None
    registration, texture, radius = extract(sequence, checkpoint, output, Path(args.reference).stem,
                                            training_result, pose_result)
    repo = home / "GaussianAvatars"
    env = dict(os.environ)
    env.pop("PYTHONNOUSERSITE", None)
    env["PYTHONPATH"] = os.pathsep.join((str(home / "ga_pkgs"), str(repo)))
    bound = output / "bound-head" / "point_cloud.ply"
    subprocess.run([str(home / "miniconda3/envs/fastavatar/bin/python"), "-B", str(Path(__file__).resolve()),
                    "--write-bound", str(checkpoint), str(texture), str(bound), str(registration), str(radius)],
                   cwd=repo, env=env, check=True)
    proof = json.loads((output / "bound-verification.json").read_bytes())
    manifest = dict(tracker=str(checkpoint), bound=str(bound), registration=str(registration),
                    initialization=str(sequence / "initialization.json"), gaussian_steps=0,
                    head_source="vhap_flame_initialization", color="native VHAP albedo sampled at triangle UV centers",
                    vhap_repo=str(vhap_repo), training_result=str(training_result) if training_result else None,
                    template_radius=radius, verification=proof)
    if pose_result is not None:
        manifest["pose_result"] = {"path": str(pose_result), "sha256": sha256(pose_result.read_bytes()).hexdigest()}
    (output / "avatar.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Registered {proof['frames']} exact VHAP poses and {proof['gaussians']} seed Gaussians in {output}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--write-bound"]:
        checkpoint, texture, bound, registration = map(Path, sys.argv[2:6])
        proof = seed_and_verify(checkpoint, texture, bound, registration, float(sys.argv[6]))
        (bound.parent.parent / "bound-verification.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    else:
        parser = ArgumentParser(description=__doc__)
        parser.add_argument("--sequence", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--reference", default="00560")
        parser.add_argument("--training-result", type=Path, help="Completed native VHAP training receipt.")
        parser.add_argument("--pose-result", type=Path, help="Verified neck refinement with a complete native grid.")
        parser.add_argument("--vhap-repo", type=Path, default=Path(__file__).resolve().parent / "upstream/VHAP-6e1e770")
        run(parser.parse_args())
