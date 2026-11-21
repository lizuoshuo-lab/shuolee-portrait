"""Small forward/backward test of the actual gsplat CUDA extension."""
import json
from pathlib import Path
import time
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(0)
start = time.perf_counter()
means = torch.tensor([[0., 0., 2.], [.2, .1, 2.5]], device='cuda', requires_grad=True)
quats = torch.tensor([[1., 0., 0., 0.]] * 2, device='cuda', requires_grad=True)
scales = torch.full((2, 3), .1, device='cuda', requires_grad=True)
opacity = torch.full((2,), .8, device='cuda', requires_grad=True)
colors = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device='cuda', requires_grad=True)
K = torch.tensor([[[40., 0., 16.], [0., 40., 16.], [0., 0., 1.]]], device='cuda')
render, alpha, _ = gsplat.rasterization(means, quats, scales, opacity, colors,
                                        torch.eye(4, device='cuda')[None], K, 32, 32,
                                        packed=False)
loss = render.square().mean()
loss.backward()
torch.cuda.synchronize()
gradients = {key: {'finite': bool(torch.isfinite(value.grad).all()),
                   'l1': float(value.grad.abs().sum())}
             for key, value in {'means': means, 'quats': quats, 'scales': scales,
                                'opacity': opacity, 'colors': colors}.items()}
assert torch.isfinite(render).all() and all(v['finite'] for v in gradients.values())
assert gradients['means']['l1'] > 0 and gradients['colors']['l1'] > 0
record = {'status': 'passed', 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gsplat': gsplat.__version__, 'gpu': torch.cuda.get_device_name(0),
          'shape': list(render.shape), 'elapsed_seconds_including_first_compile': time.perf_counter()-start,
          'loss': float(loss), 'gradients': gradients}
(ROOT / 'manifests' / 'gsplat-cuda-check.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))
