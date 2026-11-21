"""Build the complementary upper torso patch for a fitted VHAP head and neck."""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from export_head_neck_frames import boundary_loops, clip


def _edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)


def _adjacency(faces, canonical, weighted=False):
    left, right = _edges(faces).T
    if weighted:
        length = np.linalg.norm(canonical[left] - canonical[right], axis=1)
        value = 1. / np.maximum(length, 1e-5)
    else:
        value = np.ones(len(left))
    adjacency = coo_matrix((np.tile(value, 2), (np.r_[left, right], np.r_[right, left])),
                           shape=(len(canonical), len(canonical))).tocsr()
    return adjacency


def build_patch(smplx_path, original_head_path, model_path, aligned_head_world):
    """Return a mesh aligned at the exact existing neck ring and rooted at the torso."""
    with np.load(smplx_path, allow_pickle=False) as full, np.load(
            original_head_path, allow_pickle=False) as head, np.load(model_path, allow_pickle=False) as model:
        names = full["names"]
        source_faces = full["faces"]
        neutral = model["v_template"]
        posed = full["vertices_world"]
        if not np.array_equal(names, head["names"]) or not np.array_equal(source_faces, model["f"]):
            raise ValueError("The torso and head must come from the same SMPL-X frames and model")
        for key in ("native_K", "native_camera_center", "native_camera_to_world_rotation", "scene_scale"):
            if not np.array_equal(full[key], head[key]):
                raise ValueError(f"The torso and head must share the same registered coordinates: {key}")
        if aligned_head_world.shape != head["vertices_world"].shape or not np.isfinite(aligned_head_world).all():
            raise ValueError("The aligned head must preserve all original frame and vertex indices")
        ring = head["neck_boundary_vertex_ids"]
        ring_edges = head["source_vertex_endpoints"][ring]
        ring_fraction = head["source_edge_interpolation"][ring]
        neck_joints = {12}
        for child, parent in enumerate(model["kintree_table"][0]):
            if int(parent) in neck_joints:
                neck_joints.add(child)
        weight = model["weights"][:, sorted(neck_joints)].sum(axis=1)
        joints = model["J_regressor"] @ neutral

    seam_weights = weight[ring_edges[:, 0]] * (1. - ring_fraction) + weight[ring_edges[:, 1]] * ring_fraction
    cutoff = float(np.median(seam_weights))
    if not np.allclose(seam_weights, cutoff, atol=1e-7, rtol=0):
        raise ValueError("Head seam is not the expected SMPL-X skinning weight isocontour")
    endpoints, fraction, faces, _ = clip(enumerate(source_faces), -weight, -cutoff)
    lookup = {tuple(edge): i for i, edge in enumerate(endpoints)}
    try:
        seam = np.asarray([lookup[tuple(edge)] for edge in ring_edges], dtype=np.int32)
    except KeyError as exc:
        raise ValueError("A head seam edge has no complementary patch edge") from exc
    if not np.allclose(fraction[seam], ring_fraction, atol=1e-5, rtol=0):
        raise ValueError("Complementary cut and head disagree on their shared intersection")
    fraction[seam] = ring_fraction
    canonical = neutral[endpoints[:, 0]] * (1 - fraction[:, None]) + neutral[endpoints[:, 1]] * fraction[:, None]

    shoulder = joints[[16, 17]]
    shoulder_width = float(np.linalg.norm(shoulder[0] - shoulder[1]))
    # Keep the shoulder heads and upper chest while excluding the arms and lower torso.
    x_min = shoulder[:, 0].min() - .31 * shoulder_width
    x_max = shoulder[:, 0].max() + .31 * shoulder_width
    y_min = shoulder[:, 1].min() - .45 * shoulder_width
    allowed = (canonical[:, 0] >= x_min) & (canonical[:, 0] <= x_max) & (canonical[:, 1] >= y_min)
    admitted = faces[allowed[faces].all(axis=1)]
    _, component = connected_components(_adjacency(admitted, canonical), directed=False)
    if np.unique(component[seam]).size != 1:
        raise ValueError("The permitted shoulder patch does not connect the head seam")
    faces = admitted[component[admitted[:, 0]] == component[seam[0]]]
    used, remapped = np.unique(faces, return_inverse=True)
    seam = np.searchsorted(used, seam).astype(np.int32)
    faces = remapped.reshape(-1, 3).astype(np.int32)
    endpoints, fraction, canonical = endpoints[used], fraction[used], canonical[used]
    loops = boundary_loops(faces)
    outer_loops = [loop for loop in loops if not np.isin(loop, seam).any()]
    if (len(outer_loops) != len(loops) - 1 or not outer_loops
            or not any(np.array_equal(np.sort(loop), np.sort(seam)) for loop in loops)):
        raise ValueError("The shoulder patch must have one complete attachment ring and open torso edges")
    outer = np.concatenate(outer_loops)

    # Move the upper patch with the fitted neck, decaying to the original torso at
    # its outer boundaries. Solve one mesh-connected interpolation per sequence.
    vertices = posed[:, endpoints[:, 0]] * (1 - fraction[None, :, None]) + posed[:, endpoints[:, 1]] * fraction[None, :, None]
    vertices = vertices.astype(np.result_type(vertices.dtype, aligned_head_world.dtype), copy=False)
    alignment = aligned_head_world[:, ring] - vertices[:, seam]
    adjacency = _adjacency(faces, canonical, weighted=True)
    interior = np.setdiff1d(np.arange(len(canonical)), np.r_[seam, outer])
    laplacian = (diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()
    influence = spsolve(laplacian[interior][:, interior].tocsc(), adjacency[interior][:, seam].toarray())
    vertices[:, interior] += np.einsum("ik,fkd->fid", influence, alignment).astype(vertices.dtype)
    vertices[:, seam] = aligned_head_world[:, ring]
    a, b, c = np.moveaxis(vertices[:, faces], 2, 0)
    if not np.isfinite(vertices).all() or (np.linalg.norm(np.cross(b - a, c - a), axis=-1) <= 1e-10).any():
        raise ValueError("The aligned shoulder patch contains invalid triangles")
    return {"vertices_world": vertices, "faces": faces, "head_boundary_vertex_ids": ring,
            "patch_boundary_vertex_ids": seam, "outer_boundary_vertex_ids": outer,
            "canonical_vertices": canonical, "source_vertex_endpoints": endpoints,
            "source_edge_interpolation": fraction}
