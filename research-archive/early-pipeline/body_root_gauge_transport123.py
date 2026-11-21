"""Paired rigid frame transport after the upstream triangle frame is computed.

This optional input adapter is not loaded by the running baseline. It avoids
reconstructing near-degenerate triangle axes from newly rounded world vertices.
The caller must transform its camera with E_new = E_old @ inverse(G).
"""
import torch
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, quat_product


def transform_selected_triangle_frames(model, transform):
    """Call once after ordinary select_mesh_by_timestep; no trainable parameters."""
    rotation, translation = transform[:3, :3], transform[:3, 3]
    model.face_center = model.face_center @ rotation.T + translation
    model.face_orien_mat = rotation @ model.face_orien_mat
    q = rotmat_to_unitquat(rotation).expand(len(model.face_orien_quat), -1)
    model.face_orien_quat = quat_xyzw_to_wxyz(quat_product(q, quat_wxyz_to_xyzw(model.face_orien_quat)))
    # Existing face_scaling, local Gaussian xyz/scales/rotations, SH, opacity,
    # triangle binding and canonical vertex offsets are deliberately untouched.


def transform_vertices(vertices, transform):
    return vertices @ transform[:3, :3].T + transform[:3, 3]
