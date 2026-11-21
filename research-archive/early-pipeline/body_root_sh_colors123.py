"""Optional body-root SH coordinates via the upstream override_color interface.

Geometry/cameras/rasterization stay in their original coordinates. This changes
radiance parameterization; existing world-SH checkpoints are diagnostic inputs,
not valid retrained models. The running baseline never imports this module.
"""
import torch
from utils.sh_utils import eval_sh


def root_frame_colors(model, camera_center, root_rotation):
    xyz = model.get_xyz
    # The existing full-K Camera keeps metadata on CPU; the CUDA rasterizer
    # also transfers campos explicitly. Match that device/dtype contract here.
    center = camera_center.to(device=xyz.device, dtype=xyz.dtype)
    rotation = root_rotation.to(device=xyz.device, dtype=xyz.dtype)
    direction = xyz - center
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    direction = direction @ rotation.T
    features = model.get_features.transpose(1, 2)
    return torch.clamp_min(eval_sh(model.active_sh_degree, features, direction) + .5, 0.)


def render_root_frame(upstream_render, camera, model, pipe, background, root_rotation, **kwargs):
    # Preserve white/diagnostic overrides and all geometry-related options.
    if kwargs.get('override_color') is None:
        kwargs['override_color'] = root_frame_colors(model, camera.camera_center, root_rotation)
    return upstream_render(camera, model, pipe, background, **kwargs)
