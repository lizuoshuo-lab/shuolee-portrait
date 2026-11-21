"""Distribution-to-plane primitives for a fixed, validated SMPL-X surface.

SplatFace Sec. 3.3 motivates expected absolute distance. Here the bound
plane is fixed for each observation and the folded-normal integral is
evaluated analytically. This does not implement closest-face reassignment
or FLAME joint optimization.
"""
from math import sqrt, pi

import torch
from torch.nn.functional import normalize


def folded_normal_distance(center, sigma):
    """Expected absolute signed distance for N(center, sigma**2)."""
    scale = sigma.clamp_min(max(torch.finfo(sigma.dtype).tiny, 1e-12))
    # Saturate the negligible tail before division to keep its backward finite.
    ratio = torch.minimum(center.abs(), 16*scale)/scale
    return center.abs()*torch.erf(ratio/sqrt(2.)) + sigma*sqrt(2./pi)*torch.exp(-.5*ratio.square())


def plane_distribution(local_center, local_log_sigma, local_quat_wxyz, face_scale):
    """Metric signed mean, normal sigma and E|distance| in the bound face frame."""
    from roma import quat_wxyz_to_xyzw, unitquat_to_rotmat

    rotation = unitquat_to_rotmat(quat_wxyz_to_xyzw(normalize(local_quat_wxyz, dim=-1)))
    scales = local_log_sigma.exp()*face_scale[..., None]
    mean = local_center[..., 1]*face_scale
    # +Y (axis 1) is the face-frame normal, including the Gaussian's own rotation.
    sigma = torch.linalg.vector_norm(rotation[..., 1, :]*scales, dim=-1)
    return mean, sigma, folded_normal_distance(mean, sigma)


def choose_world_split_rows(distance, repeated_source_bins, stable_ids, *, threshold, budget):
    """Rank eligible distributions by metric off-plane distance, not image radius."""
    if threshold <= 0 or budget < 0 or not (distance.shape == repeated_source_bins.shape == stable_ids.shape):
        raise ValueError('World split needs a positive metric threshold and matching per-row evidence.')
    eligible = torch.where((repeated_source_bins >= 2) & torch.isfinite(distance) & (distance > threshold))[0]
    order = eligible[torch.argsort(stable_ids[eligible], stable=True)]
    return order[torch.argsort(distance[order], descending=True, stable=True)][:budget]

@torch.no_grad()
def tangent_split_seeds(local_center, local_log_sigma, local_quat_wxyz, face_scale, *,
                        center_clearance_m, normal_sigma_cap_m, tangent_shift_ratio=.5):
    """Propose two local children preserving the parent's tangent second moment.

    The children inherit the parent's face. The new frame has +Y as its thin
    normal direction; its tangent axes diagonalize the parent's projected
    tangent covariance. This only creates a proposal. Native RGB/alpha and
    multi-view support must decide whether to use it.
    """
    from roma import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rotmat_to_unitquat, unitquat_to_rotmat

    if not (0 < tangent_shift_ratio < 1 and center_clearance_m >= 0 and normal_sigma_cap_m > 0):
        raise ValueError('Invalid metric tangent-split tolerances.')
    if not bool((torch.isfinite(face_scale) & (face_scale > 0)).all()):
        raise ValueError('Tangent split requires a positive calibrated face scale.')
    rotations = unitquat_to_rotmat(quat_wxyz_to_xyzw(normalize(local_quat_wxyz, dim=-1)))
    scales = local_log_sigma.exp()
    local = rotations*scales[..., None, :]
    tangents = local[..., (0, 2), :]
    eig, vectors = torch.linalg.eigh(tangents @ tangents.transpose(-1,-2))
    minor, major = eig.clamp_min(1e-16).sqrt().unbind(-1)
    direction = vectors[..., :, 1]
    canonical = torch.where(direction[..., :1].abs() > 1e-6, direction[..., :1], direction[..., 1:2])
    direction = torch.where(canonical < 0, -direction, direction)
    x,z = direction.unbind(-1)
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    t1 = torch.stack((x,zero,z),-1)
    t2 = torch.stack((-z,zero,x),-1)
    n = torch.stack((zero,one,zero),-1)
    aligned = torch.stack((t1,n,t2),-1)
    quaternion = quat_xyzw_to_wxyz(rotmat_to_unitquat(aligned))
    shift = t1*(major*tangent_shift_ratio)[..., None]
    base = local_center.clone()
    clearance = center_clearance_m/face_scale
    base[..., 1] = base[..., 1].clamp(-clearance,clearance)
    positions = torch.stack((base-shift, base+shift),-2)
    old_normal = torch.linalg.vector_norm(local[..., 1, :],dim=-1)
    normal = torch.minimum(old_normal, normal_sigma_cap_m/face_scale)
    child = torch.stack((major*sqrt(1-tangent_shift_ratio**2),normal,minor),-1)
    return positions, child.clamp_min(1e-8).log()[..., None, :].expand_as(positions),         quaternion[..., None, :].expand(*positions.shape[:-1],4)
