"""Face protection for the optional clothing geometry branch, not a fitter.

The original running Gaussian control does not import this adapter. A later
facial landmark branch has its own parameters; these masks do not freeze that
branch. Model-generated landmarks are not independent 3-D observations.
"""
import torch


def protected_face_vertices(weights, faces, static_faces, dynamic_faces):
    """Conservatively protect head, neck, jaw, eyes and all landmark support."""
    if weights.ndim != 2 or weights.shape[1] != 55:
        raise ValueError('Expected the original SMPL-X 55-joint skinning table')
    protected = torch.isin(weights.argmax(-1), weights.new_tensor(
        [12, 15, 22, 23, 24], dtype=torch.long))
    indices = torch.cat((static_faces.reshape(-1), dynamic_faces.reshape(-1))).long()
    if indices.numel() == 0 or indices.min() < 0 or indices.max() >= len(faces):
        raise ValueError('Landmark triangle identities do not match this mesh')
    protected[faces[indices].reshape(-1)] = True
    return protected


def clothing_only_offset(delta, clothing_vertices, protected_face):
    """Prevent direct and regularizer gradients from updating facial offsets."""
    if delta.ndim != 2 or delta.shape[-1] != 3:
        raise ValueError('Expected one shared canonical vertex offset, not per-frame offsets')
    if clothing_vertices.shape != delta.shape[:1] or protected_face.shape != delta.shape[:1]:
        raise ValueError('Vertex masks must use the same topology as the offset')
    active = clothing_vertices.bool() & ~protected_face.bool()
    return delta * active[:, None]
