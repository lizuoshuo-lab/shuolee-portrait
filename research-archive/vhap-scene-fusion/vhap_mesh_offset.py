"""Move one shared FLAME surface along its registered posed vertex normals."""

from dataclasses import replace
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.nn import functional as F


class VhapMeshOffset(torch.nn.Module):
    def __init__(self, registered_npz, frames, mask_path, *, max_offset=.03,
                 movable_regions=("scalp", "neck"), protected_vertex_ids=(), landmark_path=None,
                 posed_mesh=None):
        super().__init__()
        from utils.graphics_utils import compute_face_orientation

        frames = tuple(frames)
        self.max_offset = float(max_offset)
        if not frames or not np.isfinite(self.max_offset) or not 0 < self.max_offset <= .03:
            raise ValueError("Registered frames and a positive FLAME offset bound up to 0.03 are required")
        self._face_orientation = compute_face_orientation
        device, dtype = frames[0].head[0].device, frames[0].head[0].dtype
        with np.load(registered_npz, allow_pickle=False) as data:
            names = data["names"].tolist()
            if (names != [Path(frame.name).stem for frame in frames] or len(set(names)) != len(names)):
                raise ValueError("Mesh sequence and cached frames do not agree")
            vertices = torch.as_tensor(data["vertices_tracker"], device=device, dtype=dtype)
            faces = torch.as_tensor(data["faces"], device=device, dtype=torch.long)
            rotation = torch.as_tensor(data["tracker_to_world_rotation"], device=device, dtype=dtype)
            origin = torch.as_tensor(data["scene_origin"], device=device, dtype=dtype)
            self.track_scale = float(data["scene_scale"])
        if (vertices.ndim != 3 or vertices.shape[0] != len(names) or vertices.shape[-1] != 3
                or not torch.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                or (faces < 0).any() or (faces >= vertices.shape[1]).any()
                or rotation.shape != (3, 3) or origin.shape != (3,) or not torch.isfinite(origin).all()
                or not torch.allclose(rotation.T @ rotation, torch.eye(3, device=device, dtype=dtype), atol=1e-6, rtol=0)
                or not torch.isclose(rotation.det(), rotation.new_tensor(1.), atol=1e-6, rtol=0)
                or not np.isfinite(self.track_scale) or self.track_scale <= 0):
            raise ValueError("Invalid registered FLAME topology or tracker-to-world similarity")
        self._indices = {frame.name: i for i, frame in enumerate(frames)}
        self.register_buffer("faces", faces, persistent=False)
        self.register_buffer("_vertices", vertices, persistent=False)
        self.register_buffer("_rotation", rotation, persistent=False)
        self.register_buffer("_origin", origin, persistent=False)
        self.register_buffer("_posed_seed", None, persistent=False)
        if posed_mesh is not None:
            with np.load(posed_mesh, allow_pickle=False) as tape:
                if (tape["frames"].tolist() != [frame.name for frame in frames]
                        or not np.array_equal(tape["faces"], faces.cpu().numpy())
                        or not np.allclose(tape["basis"], rotation.cpu(), atol=1e-7, rtol=0)
                        or not np.allclose(tape["origin"], origin.cpu(), atol=1e-6, rtol=0)
                        or not np.isclose(float(tape["scale"]), self.track_scale, atol=1e-8, rtol=0)):
                    raise ValueError("Posed mesh does not match the ordered registration and world gauge")
                seed = torch.as_tensor(tape["vertices"], device=device, dtype=dtype)
            if seed.shape != vertices.shape or not torch.isfinite(seed).all() or torch.equal(seed, vertices):
                raise ValueError("Expected a distinct, finite registered posed mesh")
            self._posed_seed = seed

        mask_path = Path(mask_path)
        with mask_path.open("rb") as file:
            regions = pickle.load(file, encoding="latin1")
        count = vertices.shape[1]
        self._region_names = frozenset(regions)
        for name, ids in regions.items():
            ids = torch.as_tensor(np.asarray(ids), device=device, dtype=torch.long)
            if ids.ndim != 1 or (ids < 0).any() or (ids >= count).any():
                raise ValueError(f"FLAME mask {name} is incompatible with the registered topology")
            mask = torch.zeros(count, device=device, dtype=torch.bool)
            mask[ids] = True
            self.register_buffer(f"_region_{name}", mask, persistent=False)
        protected = torch.stack([self.region_mask(name) for name in (
            "face", "forehead", "neck_base", "left_ear", "right_ear", "left_eyeball", "right_eyeball", "boundary",
        ) if name in self._region_names]).any(0)
        embedding = Path(landmark_path) if landmark_path is not None else mask_path.with_name("landmark_embedding_with_eyes.npy")
        indices = np.load(embedding, allow_pickle=True, encoding="latin1").item()["full_lmk_faces_idx"]
        indices = torch.as_tensor(indices.reshape(-1), device=device, dtype=torch.long)
        if (indices < 0).any() or (indices >= len(faces)).any():
            raise ValueError("Facial landmarks refer to a different FLAME topology")
        protected[faces[indices].flatten()] = True
        additional = torch.as_tensor(protected_vertex_ids, device=device, dtype=torch.long).reshape(-1)
        if (additional < 0).any() or (additional >= count).any():
            raise ValueError("Additional protected vertices refer to a different topology")
        protected[additional] = True
        # Keep one ring around protected facial identities and the open neck rim.
        protected[faces[protected[faces].any(dim=1)].flatten()] = True
        editable = torch.stack([self.region_mask(name) for name in movable_regions]).any(0) & ~protected
        if not editable.any():
            raise ValueError("No editable FLAME scalp/neck surface after protecting anatomy")
        self.register_buffer("editable", editable, persistent=False)
        ids = editable.nonzero().flatten()
        self.register_buffer("_ids", ids, persistent=False)
        edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]))
        edges = torch.unique(edges.sort(dim=1).values, dim=0)
        self.register_buffer("_edges", edges[editable[edges].any(dim=1)], persistent=False)

        with torch.no_grad():
            triangles = vertices[:, faces]
            cross = torch.cross(triangles[:, :, 1] - triangles[:, :, 0],
                                triangles[:, :, 2] - triangles[:, :, 0], dim=-1)
            normals = torch.zeros_like(vertices)
            for corner in range(3):
                normals.index_add_(1, faces[:, corner], cross)
            normals = F.normalize(normals[:, ids], dim=-1)
            if not torch.isfinite(normals).all() or (normals.norm(dim=-1) < .99).any():
                raise ValueError("Editable FLAME vertices require finite, nonzero posed normals")
            orientation, distance = compute_face_orientation(vertices, faces, return_scale=True)
            if not torch.isfinite(distance).all() or (distance < 1e-8).any():
                raise ValueError("Registered FLAME triangles must have finite, positive size")
        self.register_buffer("_normals", normals, persistent=False)
        self.register_buffer("_old_orientation", orientation, persistent=False)
        self.register_buffer("_old_distance", distance, persistent=False)
        self.register_buffer("_old_centers", triangles.mean(dim=2), persistent=False)
        self.offset = torch.nn.Parameter(vertices.new_zeros(len(ids)))

    def region_mask(self, name):
        if name not in self._region_names:
            raise ValueError(f"Unknown FLAME region: {name}")
        return getattr(self, f"_region_{name}")

    def _posed(self, i):
        if self._posed_seed is not None:
            return self._posed_seed[i]
        displacement = self.max_offset * self.offset.tanh()
        return self._vertices[i].index_add(0, self._ids, self._normals[i] * displacement[:, None])

    def vertices_world(self, frame, *, deformed=True):
        i = self._indices[frame.name]
        vertices = self._posed(i) if deformed else self._vertices[i]
        return self.track_scale * (vertices @ self._rotation.T) + self._origin

    def deform(self, frame):
        from roma import quat_product, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rotmat_to_unitquat

        i = self._indices[frame.name]
        xyz, features, opacity, scales, rotations = frame.head
        posed = self._posed(i)
        orientation, distance = self._face_orientation(posed, self.faces, return_scale=True)
        centers = posed[self.faces].mean(dim=1)
        shift = self.track_scale * ((centers - self._old_centers[i]) @ self._rotation.T)
        ratio = distance / self._old_distance[i]
        world_orientation = self._rotation @ orientation
        rotation_change = world_orientation @ (self._rotation @ self._old_orientation[i]).transpose(-1, -2)
        q = quat_product(rotmat_to_unitquat(rotation_change), quat_wxyz_to_xyzw(rotations))
        rotation = quat_xyzw_to_wxyz(F.normalize(q, dim=-1))
        # Transport the original cached frame, including its reference divisor.
        # Rebuilding the base frame from rounded vertices can rotate thin faces.
        axes = (rotation_change @ frame.axes) * ratio[:, None]
        return replace(frame, head=(xyz + shift, features, opacity, scales * ratio, rotation), axes=axes)

    def regularizer(self):
        # Penalize an edge connecting an editable vertex to a protected one as well.
        relative = self.offset.tanh()
        field = relative.new_zeros(len(self.editable)).index_copy(0, self._ids, relative)
        left, right = self._edges.T
        return .04 * (field[left] - field[right]).square().mean() + .006 * relative.square().mean()

    @torch.no_grad()
    def validate_zero(self, frames):
        if self.offset.count_nonzero():
            raise ValueError("Zero-replay verification requires an untrained mesh offset")
        errors = {"centers": 0., "scales": 0., "quaternions": 0., "axes": 0.}
        for source in frames:
            target = self.deform(source)
            for key, before, after in zip(("centers", "scales", "quaternions", "axes"),
                                          (source.head[0], source.head[3], source.head[4], source.axes),
                                          (target.head[0], target.head[3], target.head[4], target.axes), strict=True):
                if key == "quaternions":
                    after = torch.where((before * after).sum(-1, keepdim=True) < 0, -after, after)
                delta = float((before - after).abs().max())
                if not np.isfinite(delta) or delta > 1e-5:
                    raise RuntimeError(f"Zero mesh offset changed {key} of {source.name}: {delta}")
                errors[key] = max(errors[key], delta)
        return errors

    @torch.no_grad()
    def validate_posed_seed(self, frames):
        maximum = 0.
        if self._posed_seed is None or self.offset.requires_grad:
            raise ValueError("Posed seed validation requires an explicitly frozen mesh")
        for frame in frames:
            xyz, _, _, radii, rotations = self.deform(frame).head
            expected = self.vertices_world(frame)[self.faces].mean(dim=1)
            error = float((xyz - expected).abs().max())
            if (not np.isfinite(error) or error > 1e-5 or
                    not (torch.isfinite(radii) & (radii > 0)).all() or not torch.isfinite(rotations).all()):
                raise RuntimeError(f"Posed mesh and transported Gaussians disagree at {frame.name}")
            maximum = max(maximum, error)
        return {"frames": len(frames), "max_bound_center_world_delta": maximum,
                "max_change_flame_mm": 1000. * float((self._posed_seed - self._vertices).norm(dim=-1).max())}
