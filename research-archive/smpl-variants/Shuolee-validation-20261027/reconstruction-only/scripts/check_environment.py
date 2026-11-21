"""Check independent imports and real CUDA kernels without model weights or training."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import site
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]


def check():
    env = Path(sys.prefix).resolve()
    assert sys.prefix != sys.base_prefix, 'Use an isolated virtual environment.'
    assert not site.ENABLE_USER_SITE, 'Set PYTHONNOUSERSITE=1 or invoke python -s.'
    assert not os.environ.get('SHUOLEE_EXTRA_PYTHONPATH'), 'External experiment dependencies are not allowed in this check.'
    foreign = [p for p in sys.path if 'site-packages' in p and env not in Path(p).resolve().parents]
    assert not foreign, f'Foreign package directories: {foreign}'
    sys.path[:0] = [str(PROJECT / p) for p in (
        'src', 'src/avatar_core', 'third_party/gvhmr', 'third_party/sam2', 'third_party/human_body_prior')]
    modules = {}
    names = [
        'torch', 'torchvision', 'cv2', 'pycolmap', 'smplx', 'roma', 'rtmlib',
        'ultralytics', 'onnxruntime', 'hydra', 'hydra_zen', 'pytorch_lightning',
        'timm', 'av', 'ffmpeg', 'wis3d', 'nvdiffrast.torch', '_nvdiffrast_c', 'pytorch3d._C',
        'simple_knn._C', 'diff_gaussian_rasterization._C',
        'sam2.build_sam', 'human_body_prior.models.vposer_model',
        'hmr4d.network.hmr2', 'hmr4d.utils.preproc.vitpose_pytorch',
        'hmr4d.model.gvhmr.gvhmr_pl_demo',
        'hmr4d.network.gvhmr.relative_transformer',
    ]
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert env in path.parents or PROJECT in path.parents, f'Borrowed module {name}: {path}'
        modules[name] = str(path)
    import cv2
    assert cv2.__version__ == '4.10.0', f'Unexpected OpenCV runtime: {cv2.__version__}'
    import torch
    torch.set_num_threads(4)
    assert torch.__version__ == '2.7.1+cu128'
    assert torch.cuda.is_available()
    device = 'cuda'
    # Exercise compiled kNN operators, including an actual backward call.
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], device=device)
    from simple_knn._C import distCUDA2
    distances = distCUDA2(xyz)
    assert torch.isfinite(distances).all() and (distances > 0).all()
    from pytorch3d.ops import knn_points
    query = xyz[None].clone().requires_grad_(True)
    out = knn_points(query, (xyz + .1)[None], K=1)
    out.dists.sum().backward()
    assert torch.isfinite(query.grad).all() and query.grad.abs().max() > 0
    # Exercise the mesh rasterizer on a synthetic triangle; no human data used.
    import nvdiffrast.torch as dr
    pos = torch.tensor([[[-.7, -.6, 0., 1.], [.7, -.6, 0., 1.], [0., .6, 0., 1.]]], device=device, requires_grad=True)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    raster, _ = dr.rasterize(dr.RasterizeCudaContext(), pos, tri, resolution=[32, 32])
    assert (raster[..., 3] > 0).any()
    raster[..., :2].square().sum().backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().max() > 0
    # Exercise Gaussian forward/backward with the same public extension API used by training.
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix
    proj = getProjectionMatrix(.01, 100., 1., 1.).transpose(0, 1).contiguous().cuda()
    settings = GaussianRasterizationSettings(32, 32, .54630249, .54630249,
        torch.zeros(3, device=device), 1., torch.eye(4, device=device), proj, 0,
        torch.zeros(3, device=device), False, False)
    centers = torch.tensor([[0., 0., 2.], [.15, .1, 2.2]], device=device, requires_grad=True)
    pixels = torch.zeros_like(centers, requires_grad=True)
    colors = torch.tensor([[1., .2, .1], [.1, .5, 1.]], device=device, requires_grad=True)
    rgb, radii = GaussianRasterizer(settings)(means3D=centers, means2D=pixels,
        opacities=torch.full((2, 1), .7, device=device), colors_precomp=colors,
        scales=torch.full((2, 3), .08, device=device),
        rotations=torch.tensor([[1., 0., 0., 0.]] * 2, device=device))
    assert torch.isfinite(rgb).all() and (radii > 0).all() and rgb.max() > 0
    rgb.square().sum().backward()
    assert torch.isfinite(centers.grad).all() and torch.isfinite(colors.grad).all()
    assert colors.grad.abs().max() > 0
    torch.cuda.synchronize()
    return dict(environment=str(env), python=sys.version, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, cuda=torch.version.cuda, opencv=cv2.__version__, module_paths=modules,
                isolated_package_imports=True, knn_cuda=True, mesh_forward_backward=True,
                gaussian_forward_backward=True, model_weights_loaded=False,
                training_started=False, reconstruction_quality_accepted=False,
                asset_dependent_modules_checked=False,
                new_operating_system_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'created': datetime.now(timezone.utc).isoformat(), 'passed': False}
    try:
        result.update(check(), passed=True)
    except Exception:
        result['error'] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
