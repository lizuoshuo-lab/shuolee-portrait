"""Small independent CPU audit; no original test suite or model execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path('/mnt/d/Projects/Shuolee-torch-clothing-consumer-123')
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait.free_clothing_transport import FreeClothingTransport
from shuolee_portrait.free_clothing_torch import FreeClothingTorch

torch.set_num_threads(1)
axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
S = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
Q = np.eye(3) + np.sin(.37) * S + (1 - np.cos(.37)) * S @ S
A = 1.7
SHIFT = np.array([.2, .3, .4])
V = np.array([[0., 0., 3.], [1., 0., 3.], [.3, .8, 3.]])
X = np.array([[.2, .3, 3.35], [-.2, 1.3, 2.7]])
QUAT = np.array([[.8, .2, -.1, .3], [.6, -.2, .1, .7]])
LOGS = np.log([[.1, .07, .03], [.05, .1, .08]])
CAM = np.array([.5, -1.2, -.3])
IDS = ('free_a', 'free_b')
FRAMES = ('reference', 'posed')
E = np.eye(4); E[:3, :3] = Q; E[:3, 3] = -Q @ CAM
CONTRACT = dict(translation_unit='meters', meters_per_camera_unit=1., basis='synthetic_known', evidence='Independent declared synthetic geometry only.')


def source():
    return FreeClothingTransport(frame_ids=FRAMES, reference_frame_id='reference',
        vertices_world=np.stack([V, A * V @ Q.T + SHIFT]), faces=np.array([[0, 1, 2]]),
        motion_support_triangle_ids=np.array([0, 0]), gaussian_ids=IDS,
        frame_source_sha256={'reference': 'a' * 64, 'posed': 'b' * 64}, translation_unit='meters')


def consumer(dtype=torch.float64):
    return FreeClothingTorch.from_cpu(source(), camera_length_contract=CONTRACT,
        frame_source_w2c={f: E for f in FRAMES}, device='cpu', dtype=dtype)


def parameters(dtype=torch.float64, grad=False, quats=QUAT):
    return tuple(torch.tensor(v, dtype=dtype, device='cpu', requires_grad=grad) for v in (X, quats, LOGS))


def matrix_independent(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1); zero = torch.zeros_like(w)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
    return torch.eye(3, dtype=q.dtype, device='cpu') + 2 * w[:, None, None] * skew + 2 * skew @ skew


def loss(out):
    w = torch.arange(1, 7, dtype=out['centers'].dtype, device='cpu').reshape(2, 3) / 10
    cw = torch.tensor([[.3, .12, -.08], [.12, .5, .04], [-.08, .04, .9]], dtype=w.dtype, device='cpu')
    return (out['centers'] * w).sum() + 7 * (out['covariance'] * cw).sum() + out['reference_view_directions'][:, 0].sum()


positives = []
for dtype in (torch.float32, torch.float64):
    c = consumer(dtype); values = parameters(dtype, True)
    out = c('posed', *values, gaussian_ids=IDS, return_covariance=True)
    actual_grad = torch.autograd.grad(loss(out), values)
    x, q, logs = parameters(dtype, True)
    rot = torch.tensor(Q, dtype=dtype, device='cpu')
    direct_x = A * x @ rot.T + torch.tensor(SHIFT, dtype=dtype, device='cpu')
    local_factor = matrix_independent(q) * logs.exp()[:, None, :]
    direct_factor = (A * rot) @ local_factor
    camera_ref = torch.tensor((CAM - SHIFT) @ Q / A, dtype=dtype, device='cpu')
    ray = x - camera_ref; directions = ray / torch.linalg.vector_norm(ray, dim=-1, keepdim=True)
    direct = dict(centers=direct_x, covariance=direct_factor @ direct_factor.transpose(-1, -2), reference_view_directions=directions)
    expected_grad = torch.autograd.grad(loss(direct), (x, q, logs))
    positives.append(dict(dtype=str(dtype), output_max_error={k:float((out[k] - direct[k]).detach().abs().max()) for k in direct}, gradient_max_error={k:float((a-b).abs().max()) for k,a,b in zip(('XYZ', 'quaternion', 'log_scales'), actual_grad, expected_grad)}))

flag_probes = []
for field in ('_rotations', '_quaternions', '_scales', '_centroids', '_camera_centers'):
    c = consumer(); table = getattr(c, field); old = table._version
    table.requires_grad_(True)
    try:
        out = c('posed', *parameters(), gaussian_ids=IDS, return_covariance=True)
        gradient = torch.autograd.grad(loss(out), table, allow_unused=True)[0]
        flag_probes.append(dict(field=field, version_before=old, version_after=table._version, accepted=True, gradient_max_abs=None if gradient is None else float(gradient.abs().max())))
    except Exception as error:
        flag_probes.append(dict(field=field, version_before=old, version_after=table._version, accepted=False, error_type=type(error).__name__, message=str(error)))

small_q = []
for magnitude in (1e-18, 1e-21, 1e-22):
    raw = np.tile([1., 2., 3., 4.], (2, 1)) * magnitude
    c = consumer(torch.float32)
    try:
        out = c('reference', *parameters(torch.float32, quats=raw), gaussian_ids=IDS, return_covariance=True)
        actual_q = out['quats'].double().numpy()
        expected_q = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
        small_q.append(dict(input_scale=magnitude, accepted=True, quaternion_norm= np.linalg.norm(actual_q, axis=-1).tolist(), quaternion_max_error=float(np.max(np.abs(actual_q-expected_q)))))
    except Exception as error:
        small_q.append(dict(input_scale=magnitude, accepted=False, error_type=type(error).__name__, message=str(error)))

try:
    with torch.inference_mode():
        consumer()
    inference = dict(accepted=True)
except Exception as error:
    inference = dict(accepted=False, error_type=type(error).__name__, message=str(error))

result = dict(schema='free-clothing-torch-independent-audit/1', environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']), source_sha256=hashlib.sha256((ROOT/'src/shuolee_portrait/free_clothing_torch.py').read_bytes()).hexdigest(), positive_controls=positives, frozen_table_requires_grad_probes=flag_probes, tiny_quaternion_reference_probes=small_q, inference_mode_construction=inference, original_14_tests_run=False, model_forwards=0, GPU_jobs=0, training_steps=0, physical_quality_accepted=False)
outpath=Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/independent-torch-audit.json')
with outpath.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result,indent=2))
