"""Real upstream RGB, depth/silhouette and camera-gradient CUDA acceptance.

The renderer's third depth output is checked only for finite forward values.
Depth supervision gradients use SplaTAM's actual (z, 1, z^2) color encoding.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/opt/gs-slam-lab/splatam')
sys.path.insert(0, str(ROOT / 'repos/SplaTAM'))
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import _C
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (transform_to_frame, transformed_params2rendervar,
                                transformed_params2depthplussilhouette)


def params(camera_grad):
    values = {
        'means3D': [[0.1, 0.05, 2.0], [-0.2, -0.1, 2.5]],
        'rgb_colors': [[0.8, 0.2, 0.1], [0.1, 0.4, 0.8]],
        'unnorm_rotations': [[1., 0., 0., 0.], [1., 0., 0., 0.]],
        'logit_opacities': [[0.4], [0.7]],
        'log_scales': np.log([[0.1, 0.15, 0.2], [0.13, 0.11, 0.16]]).tolist(),
        'cam_unnorm_rots': [[[1.], [0.], [0.], [0.]]],
        'cam_trans': [[[0.], [0.], [0.]]],
    }
    return {name: torch.tensor(value, device='cuda', dtype=torch.float32,
                               requires_grad=(name.startswith('cam_') == camera_grad))
            for name, value in values.items()}


def check_case(camera_grad):
    p = params(camera_grad)
    k = np.array([[24., 0., 16.], [0., 24., 16.], [0., 0., 1.]])
    w2c = np.eye(4)
    camera = setup_camera(32, 32, k, w2c)
    transformed = transform_to_frame(p, 0, gaussians_grad=not camera_grad, camera_grad=camera_grad)
    rgb, radii, raw_depth = GaussianRasterizer(raster_settings=camera)(
        **transformed_params2rendervar(p, transformed))
    depth_sil, depth_radii, _ = GaussianRasterizer(raster_settings=camera)(
        **transformed_params2depthplussilhouette(p, torch.eye(4, device='cuda'), transformed))
    assert rgb.shape == (3, 32, 32) and depth_sil.shape == (3, 32, 32)
    assert torch.isfinite(rgb).all() and torch.isfinite(raw_depth).all()
    assert torch.isfinite(depth_sil).all() and torch.count_nonzero(radii) == 2
    assert torch.count_nonzero(depth_radii) == 2 and depth_sil[1].max() > 0
    # Both terms invoke the genuine CUDA backward, with independent color weights.
    loss = (rgb * torch.tensor([0.2, 0.7, 0.4], device='cuda')[:, None, None]).square().mean()
    loss = loss + 0.1 * depth_sil[0].square().mean() + 0.3 * depth_sil[1].mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = {}
    for name, value in p.items():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all(), name
            norm = float(value.grad.norm())
            assert norm > 0, (name, norm)
            grads[name] = norm
    return {'loss': float(loss), 'rgb_shape': list(rgb.shape), 'raw_depth_shape': list(raw_depth.shape),
            'visible_gaussians': int(torch.count_nonzero(radii)),
            'raw_depth_finite': bool(torch.isfinite(raw_depth).all()),
            'depth_silhouette_shape': list(depth_sil.shape), 'gradient_norms': grads}


def main():
    started = time.time()
    assert torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(0)
    assert capability == (12, 0), capability
    mapping = check_case(camera_grad=False)
    tracking = check_case(camera_grad=True)
    # Import and run the real evaluation implementation and pretrained LPIPS.
    from utils.eval_helpers import loss_fn_alex
    lpips = loss_fn_alex(torch.zeros(1, 3, 64, 64, device='cuda'),
                         torch.ones(1, 3, 64, 64, device='cuda'))
    assert torch.isfinite(lpips).all() and float(lpips) > 0
    extension = Path(_C.__file__)
    receipt = {'status': 'passed', 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0), 'capability': list(capability),
               'torch_arch_list': torch.cuda.get_arch_list(), 'extension_path': str(extension),
               'extension_sha256': hashlib.sha256(extension.read_bytes()).hexdigest(),
               'mapping': mapping, 'tracking': tracking, 'lpips_alex': float(lpips),
               'third_depth_backward_claimed': False, 'evaluation_stub': False,
               'elapsed_s': time.time() - started}
    dest = ROOT / 'renderer-smoke.json'
    dest.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
