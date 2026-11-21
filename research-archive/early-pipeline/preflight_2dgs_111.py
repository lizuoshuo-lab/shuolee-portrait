"""Bounded fresh analytic scene; does not load or modify a trained person."""
from pathlib import Path
import os, sys, time, json, hashlib, traceback

R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
O = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/surface-point-alignment-111')
sys.path.insert(0, str(R))
os.environ.setdefault('MAX_JOBS', '4')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.0')
t0 = time.perf_counter()
protocol = dict(scope='Isolated existing gsplat checkout, fresh analytic 64x64 plane and duplicate layers only',
                controls=['off-center intrinsics and slanted plane expected depth', 'distinct depth layers retain nonzero distortion and geometry gradient'],
                pass_limits=dict(plane_depth_max_abs=2e-4, plane_distortion_max_abs=2e-4, duplicate_distortion_min=1e-6, duplicate_geometry_grad_min=1e-6),
                seed=111, trained_person_loaded=False, training_started=False, installed_packages_modified=False,
                note='Native backend check does not accept subject geometry, general mapping correctness, or compatibility with the standard 3DGS viewer.')
(O/'2dgs-preflight-protocol.json').write_text(json.dumps(protocol, indent=2))
result = {'scope': protocol['scope'], 'training_started': False, 'passed': False}
try:
    import torch, gsplat
    from gsplat import rasterization_2dgs
    torch.manual_seed(111)
    result.update(gsplat_path=gsplat.__file__, gsplat_version=getattr(gsplat, '__version__', None), torch=torch.__version__)
    device = 'cuda'
    T = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    H = W = 64
    K = T([[[70., 0., 29.3], [0., 70., 35.1], [0., 0., 1.]]])
    V = torch.eye(4, device=device).unsqueeze(0)
    theta = .2
    c, s = torch.cos(T(theta / 2)), torch.sin(T(theta / 2))
    means = T([[-.16, 0, 2], [0, .08, 2], [.16, -.05, 2]])
    means[:, 2] -= torch.tan(T(theta)) * means[:, 0]
    means.requires_grad_(True)
    quats = T([[float(c), 0, float(s), 0]] * 3).requires_grad_(True)
    scales = T([[.16, .16, .001]] * 3)
    opacity = T([.8, .8, .8])
    colors = T([[.7, .2, .1], [.7, .2, .1], [.7, .2, .1]])
    rgbd, alpha, normals, surf, distort, median, meta = rasterization_2dgs(
        means, quats, scales, opacity, colors.unsqueeze(0), V, K, W, H,
        render_mode='RGB+ED', distloss=True)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    expected = (2 * torch.cos(T(theta))) / (torch.sin(T(theta)) * (xx + .5 - K[0,0,2]) / K[0,0,0] + torch.cos(T(theta)))
    valid = (alpha[0,:,:,0] > .1) & (xx > 4) & (xx < 59) & (yy > 4) & (yy < 59)
    err = (rgbd[0,:,:,3] - expected).abs()[valid]
    plane = dict(pixels=int(valid.sum()), depth_max_abs=float(err.max()), depth_mean_abs=float(err.mean()), distortion_max_abs=float(distort[0,:,:,0][valid].abs().max()), finite=bool(torch.isfinite(rgbd).all()))
    loss = (rgbd[0,:,:,3][valid] - expected[valid]).square().mean()
    grads = torch.autograd.grad(loss, (means, quats), allow_unused=True)
    plane['geometry_gradient_finite'] = all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    layer_means = T([[0,0,1.97],[0,0,2.03]]).requires_grad_(True)
    q = T([[1,0,0,0]] * 2)
    out = rasterization_2dgs(layer_means, q, T([[.16,.16,.001]] * 2), T([.7,.7]), colors[:2].unsqueeze(0), V, K, W, H, render_mode='RGB+ED', distloss=True)
    valid2 = out[1][0,:,:,0] > .1
    d = out[4][0,:,:,0][valid2].mean()
    g = torch.autograd.grad(d, layer_means)[0]
    layers = dict(distortion_mean=float(d), geometry_gradient_max_abs=float(g.abs().max()), gradient_finite=bool(torch.isfinite(g).all()), center_gradients=g.detach().cpu().tolist())
    limits = protocol['pass_limits']
    result.update(plane=plane, duplicate_layers=layers, passed=(plane['finite'] and plane['geometry_gradient_finite'] and plane['depth_max_abs'] < limits['plane_depth_max_abs'] and plane['distortion_max_abs'] < limits['plane_distortion_max_abs'] and layers['gradient_finite'] and layers['distortion_mean'] > limits['duplicate_distortion_min'] and layers['geometry_gradient_max_abs'] > limits['duplicate_geometry_grad_min']))
except Exception as e:
    result.update(error=type(e).__name__, message=str(e), traceback=traceback.format_exc())
result.update(seconds=time.perf_counter()-t0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'2dgs-preflight-result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'traceback'}, indent=2))
