"""Shared scalp and back-neck surface displacements after FLAME skinning."""

import pickle

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn.functional as F


# Bounds are displacements in FLAME model units, before the world similarity.
PROFILE_LIMITS = ((0.0, 0.025), (-0.010, 0.010))
_RAMP = 0.018


def _smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def _geodesic_ramp(graph, allowed):
    depth = dijkstra(graph, directed=False,
                     indices=np.flatnonzero(~allowed), min_only=True)
    depth[~np.isfinite(depth)] = 0.
    return np.where(allowed, _smoothstep(depth / _RAMP), 0.)


def profile_weights(actor, mask_path):
    """Return canonical support (V,2) and a protected-vertex boolean mask.

    Use the trusted FLAME part-mask asset matching the actor topology.
    The scalp field represents visible hair thickness when fitted to an RGB matte.
    """
    flame = actor.flame_model
    vertices = flame.v_template.detach().cpu().numpy()
    faces = flame.faces.detach().cpu().numpy()
    count = len(vertices)
    with open(mask_path, "rb") as file:
        regions = pickle.load(file, encoding="latin1")

    scalp = np.zeros(count, dtype=bool)
    scalp[np.asarray(regions["scalp"], dtype=np.int64)] = True
    neck = np.zeros(count, dtype=bool)
    neck[np.asarray(regions["neck"], dtype=np.int64)] = True
    protected = np.zeros(count, dtype=bool)
    for region in ("face", "forehead", "nose", "lips", "eye_region", "left_eye_region",
                   "right_eye_region", "left_ear", "right_ear", "left_eyeball",
                   "right_eyeball", "boundary"):
        protected[np.asarray(regions[region], dtype=np.int64)] = True
    embedded = flame.full_lmk_faces_idx.detach().cpu().numpy().reshape(-1)
    protected[faces[embedded].reshape(-1)] = True

    z = vertices[:, 2]
    posterior = np.median(z[neck]) - z
    scalp_allowed = scalp & ~neck & ~protected
    # The posterior neck is labeled in both masks. Assign the overlap to the neck.
    neck_allowed = neck & ~protected & (posterior > 0.)
    if not scalp_allowed.any() or not neck_allowed.any():
        raise ValueError("FLAME profile regions lack protected, adjustable surface")
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)
    start, end = edges.T
    lengths = np.linalg.norm(vertices[start] - vertices[end], axis=1)
    graph = coo_matrix((np.maximum(lengths, 1e-9), (start, end)), shape=(count, count)).tocsr()
    graph = graph + graph.T
    weights = np.column_stack((_geodesic_ramp(graph, scalp_allowed),
                               _geodesic_ramp(graph, neck_allowed) * _smoothstep(posterior / .030)))
    return (torch.as_tensor(weights, dtype=flame.v_template.dtype, device=flame.v_template.device),
            torch.as_tensor(protected, device=flame.v_template.device))


def profile_basis(actor, weights):
    """Return the posed (V,2,3) per-unit basis without changing joints or mesh state."""
    vertices = actor.verts[0]
    faces = actor.flame_model.faces.long()
    triangles = vertices[faces]
    face_normals = torch.cross(triangles[:, 1] - triangles[:, 0],
                               triangles[:, 2] - triangles[:, 0], dim=-1)
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    normals = F.normalize(normals, dim=-1, eps=1e-12)
    return weights[..., None] * normals[:, None, :]
