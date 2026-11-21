"""Optional, not yet executed: exact SplaTAM CUDA rasterizer forward/backward."""
import os
from pathlib import Path
import sys

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer

lab = Path(os.environ.get("GS_SLAM_LAB", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(lab / "repos/SplaTAM"))
from utils.recon_helpers import setup_camera

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
camera = setup_camera(32, 32, np.array([[24, 0, 16], [0, 24, 16], [0, 0, 1]]), np.eye(4))
means = torch.tensor([[0.1, 0.05, 2.0], [-0.2, -0.1, 2.5]], device="cuda", requires_grad=True)
means2d = torch.zeros_like(means, requires_grad=True)
colors = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.4, 0.8]], device="cuda", requires_grad=True)
opacity = torch.full((2, 1), 0.6, device="cuda", requires_grad=True)
scales = torch.full((2, 3), 0.15, device="cuda", requires_grad=True)
rotations = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda")
image, radii, depth = GaussianRasterizer(raster_settings=camera)(
    means3D=means, means2D=means2d, shs=None, colors_precomp=colors,
    opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None,
)
assert image.shape == (3, 32, 32) and torch.isfinite(image).all()
assert torch.count_nonzero(radii) > 0
image.square().mean().backward()
torch.cuda.synchronize()
for name, value in (("means", means), ("colors", colors), ("opacity", opacity), ("scales", scales)):
    assert value.grad is not None and torch.isfinite(value.grad).all(), name
print("PASS: tiny Gaussian RGB render and gradients. Does not validate SLAM accuracy.")
