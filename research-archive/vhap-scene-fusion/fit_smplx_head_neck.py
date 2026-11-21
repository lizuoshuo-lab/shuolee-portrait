"""Fit a continuous SMPL-X head, neck and shoulders in the existing VHAP world."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import smplx
from smplx.lbs import batch_rodrigues
import torch
import torch.nn.functional as F

from initial_smplx_body import initialize


NECK = 12
HEAD = 15
COCO_BODY = list(range(5, 17))
SMPLX_BODY = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


def native_camera(records):
    virtual_rotation = np.asarray([r["virtual_from_native_cv"] for r in records])
    virtual_in_world = np.asarray([r["world_from_virtual"] for r in records])
    native_in_world = virtual_in_world @ virtual_rotation
    camera_centers = np.asarray([r["camera_center_world"] for r in records])
    intrinsics = np.linalg.inv(np.asarray([r["warp"] for r in records])) @ np.asarray(
        [r["K_integer_pixel_centers"] for r in records]) @ virtual_rotation
    intrinsics /= intrinsics[:, 2:3, 2:3]
    if not np.allclose(intrinsics, intrinsics[0], atol=1e-6, rtol=0):
        raise RuntimeError("The frozen native camera intrinsics do not agree")
    return intrinsics[0], native_in_world, camera_centers


def weighted_mean(loss, weights):
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def run(args):
    device = "cuda"
    torch.manual_seed(20261117)
    initialization = json.loads(args.initialization.read_text())
    records = initialization["frames"]
    names = [r["name"] for r in records]
    scale = float(initialization["scene_scale"])
    K, camera_rotation, camera_center = native_camera(records)
    with np.load(args.body_observations) as obs, np.load(args.vhap_world) as flame:
        if names != obs["names"].tolist() or names != flame["names"].tolist():
            raise ValueError("SMPL-X, VHAP and observations must use identical ordered frames")
        if not np.isclose(scale, float(flame["scale"]), rtol=1e-8):
            raise ValueError("The fitted head and the registered cameras must share one scale")
        model = smplx.SMPLX(str(args.model), gender="neutral", num_betas=10, use_pca=False).to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        initial_rotation, initial_translation, initial_body = initialize(model, obs, records, scale, device, K)
        mapping = np.load(args.correspondence)
        face_ids = np.unique(flame["face_vertex_ids"].astype(int))
        face_ids = face_ids[(face_ids >= 0) & (face_ids < len(mapping))]
        if (len(mapping) != 5023 or len(face_ids) < 100 or not np.issubdtype(mapping.dtype, np.integer)
                or np.min(mapping) < 0 or np.max(mapping) >= model.v_template.shape[0]):
            raise ValueError("Expected the official FLAME to SMPL-X head vertex map")
        face_ids = face_ids[np.linspace(0, len(face_ids) - 1, min(len(face_ids), 350)).round().astype(int)]
        head_targets = flame["vertices_world"][:, face_ids].copy()
        native_targets = obs["keypoints"][:, COCO_BODY].copy()
        scores = obs["scores"][:, COCO_BODY]
        reliability = np.clip(scores / np.maximum(np.quantile(scores, .85, axis=0), 1e-6), 0, 1)**2
        width = obs["boxes"][:, 2] - obs["boxes"][:, 0]
        reliability *= np.exp(-.5 * (obs["flip_disagreement"][:, COCO_BODY] / np.maximum(5., .065 * width[:, None]))**2)
    reliability[:, 6:8] *= .35
    face_indices = torch.as_tensor(mapping[face_ids], dtype=torch.long, device=device)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    cameras, centers, intrinsics = (tensor(x) for x in (camera_rotation, camera_center, K))
    target_joints, joint_weights, target_face = (tensor(x) for x in (native_targets, reliability, head_targets))
    rv = torch.nn.Parameter(tensor(initial_rotation))
    translation = torch.nn.Parameter(tensor(initial_translation))
    beta = torch.nn.Parameter(torch.zeros((1, 10), device=device))
    initial_pose = tensor(initial_body).reshape(21, 3)
    shared_pose = torch.nn.Parameter(initial_pose.clone())
    per_frame = torch.nn.Parameter(torch.zeros((len(records), 2, 3), device=device))
    head_joints = (NECK - 1, HEAD - 1)

    def forward(body_pose):
        batch = len(body_pose)
        zeros = torch.zeros((batch, 3), device=device)
        hands = torch.zeros((batch, 45), device=device)
        body = model(betas=beta.expand(batch, -1), body_pose=body_pose.reshape(batch, 63),
                     global_orient=zeros, transl=zeros, jaw_pose=zeros, leye_pose=zeros, reye_pose=zeros,
                     left_hand_pose=hands, right_hand_pose=hands, expression=zeros.new_zeros(batch, 10))
        R = batch_rodrigues(rv[None])[0]
        world = lambda point: scale * (point @ R.T) + translation
        return world(body.vertices), world(body.joints[:, :55])

    def project(points):
        camera = (points - centers[:, None]) @ cameras
        homogeneous = camera @ intrinsics.T
        return homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(.05)

    def compute_static():
        vertices, joints = forward(shared_pose[None])
        predicted = project(joints[:, SMPLX_BODY])
        pixel = torch.linalg.vector_norm(predicted - target_joints, dim=-1)
        body = weighted_mean(F.huber_loss(pixel, torch.zeros_like(pixel), delta=8., reduction="none"), joint_weights) / 100
        face_error = (vertices[:, face_indices] - target_face).norm(dim=-1) / scale
        face = F.huber_loss(face_error, torch.zeros_like(face_error), delta=.018) * 4300
        regularization = .015 * beta.square().mean() + .07 * (shared_pose - initial_pose).square().mean()
        return body + face + regularization

    optimizer = torch.optim.Adam([{"params": [rv, shared_pose], "lr": .007},
                                  {"params": [translation], "lr": .004}, {"params": [beta], "lr": .024}])
    for step in range(args.shared_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_static()
        if not torch.isfinite(loss):
            raise RuntimeError("Shared SMPL-X fit diverged")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f"shared SMPL-X step {step + 1}/{args.shared_steps}: {loss.item():.5f}", flush=True)

    for param in (rv, translation, beta, shared_pose):
        param.requires_grad_(False)
    batches = torch.arange(len(records), device=device).split(12)
    optimizer = torch.optim.Adam([per_frame], lr=.008)
    for step in range(args.neck_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch in batches:
            pose = shared_pose[None].expand(len(batch), -1, -1).clone()
            pose[:, head_joints] = pose[:, head_joints] + per_frame[batch]
            vertices, _ = forward(pose)
            displacement = (vertices[:, face_indices] - target_face[batch]).norm(dim=-1) / scale
            term = F.huber_loss(displacement, torch.zeros_like(displacement), delta=.012, reduction="sum")
            term = term * 5000 / (len(records) * len(face_indices))
            term.backward()
            losses.append(term.detach())
        temporal = (per_frame[1:] - per_frame[:-1]).square().mean()
        (15. * temporal + .25 * per_frame.square().mean()).backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"per-frame neck/head step {step + 1}/{args.neck_steps}: {torch.stack(losses).sum().item():.5f}", flush=True)

    with torch.no_grad():
        posed_meshes, posed_joints, poses = [], [], []
        for batch in batches:
            pose = shared_pose[None].expand(len(batch), -1, -1).clone()
            pose[:, head_joints] = pose[:, head_joints] + per_frame[batch]
            vertices, joints = forward(pose)
            posed_meshes.append(vertices.cpu().numpy())
            posed_joints.append(joints.cpu().numpy())
            poses.append(pose.cpu().numpy().reshape(len(batch), 63))
        meshes = np.concatenate(posed_meshes)
        joints = np.concatenate(posed_joints)
        body_pose = np.concatenate(poses)
        reprojection = project(tensor(joints[:, SMPLX_BODY])).cpu().numpy()
        pixel = np.linalg.norm(reprojection - native_targets, axis=-1)
        face_offset = np.linalg.norm(meshes[:, mapping[face_ids]] - head_targets, axis=-1)
        if not all(np.isfinite(x).all() for x in (meshes, joints, body_pose, pixel)):
            raise RuntimeError("SMPL-X export contains non-finite geometry")
        if int(model.parents[HEAD]) != NECK or len(meshes) != len(names):
            raise RuntimeError("Unexpected SMPL-X head/neck kinematic chain")
        lower = torch.as_tensor([value for value in range(21) if value not in head_joints], device=device)
        if not torch.equal(tensor(body_pose.reshape(-1, 21, 3))[:, lower], shared_pose[lower][None].expand(len(records), -1, -1)):
            raise RuntimeError("Non-head joint pose changed across the fixed body sequence")
        kwargs = {"names": names, "vertices_world": meshes, "joints_world": joints, "faces": model.faces,
                  "betas": beta.cpu().numpy()[0], "body_pose": body_pose, "global_orient": rv.cpu().numpy(),
                  "world_translation": translation.cpu().numpy(), "scene_scale": scale,
                  "neck_joint_index": NECK, "head_joint_index": HEAD, "native_K": K,
                  "native_camera_to_world_rotation": camera_rotation, "native_camera_center": camera_center}
        np.savez_compressed(args.output / "smplx-frames.npz", **kwargs)
        report = {"frame_count": len(names), "vertex_count": meshes.shape[1], "triangle_count": len(model.faces),
                  "shared_betas": len(kwargs["betas"]), "scene_scale": scale, "neck_joint_index": NECK,
                  "head_joint_index": HEAD, "shared_steps": args.shared_steps, "neck_steps": args.neck_steps,
                  "body_reprojection_median_px_on_observed": float(np.median(pixel[reliability >= .45])),
                  "head_face_offset_median_model_units": float(np.median(face_offset) / scale),
                  "head_face_offset_p90_model_units": float(np.percentile(face_offset, 90) / scale)}
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for arg in ("initialization", "body-observations", "vhap-world", "model", "correspondence", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--shared-steps", type=int, default=200)
    parser.add_argument("--neck-steps", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)
