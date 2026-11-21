"""Quantify the original consumer's quaternion normalization at small norms."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/mnt/d/Projects/Shuolee-torch-clothing-consumer-123')
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait.free_clothing_transport import FreeClothingTransport

SOURCE = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/torch-clothing-consumer-123-20261029/source/src/shuolee_portrait/free_clothing_torch.py')
spec = importlib.util.spec_from_file_location('shuolee_portrait.free_clothing_torch_original_audit', SOURCE)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
torch.set_num_threads(1)
vertices = np.array([[[0., 0., 3.], [1., 0., 3.], [.3, .8, 3.]]])
source = FreeClothingTransport(frame_ids=('reference',), reference_frame_id='reference',
    gaussian_ids=('gaussian',), vertices_world=vertices, faces=np.array([[0, 1, 2]]),
    motion_support_triangle_ids=np.array([0]), frame_source_sha256={'reference': 'a' * 64}, translation_unit='meters')
contract = dict(translation_unit='meters', meters_per_camera_unit=1., basis='synthetic_known', evidence='Known synthetic fixture only.')
rows = []
for dtype, magnitudes in ((torch.float32, [1., 1e-18, 1e-21, 1e-22, 1e-23, 1e20]),
                         (torch.float64, [1., 1e-161, 1e-162, 1e200])):
    consumer = module.FreeClothingTorch.from_cpu(source, camera_length_contract=contract,
        frame_source_w2c={'reference': np.eye(4)}, device='cpu', dtype=dtype)
    normal = None
    for magnitude in magnitudes:
        raw = torch.tensor([[1., 2., 3., 4.]], dtype=dtype, device='cpu') * magnitude
        try:
            out = consumer('reference', torch.tensor([[.2, .3, 3.35]], dtype=dtype, device='cpu'), raw,
                torch.tensor(np.log([[.13, .08, .04]]), dtype=dtype, device='cpu'),
                gaussian_ids=('gaussian',), return_covariance=True)
            rotation = module.normalized_quaternion_matrix(out['quats']).double().numpy()[0]
            covariance = out['covariance'].double().numpy()[0]
            quaternion = out['quats'].double().numpy()[0]
            if magnitude == 1.:
                normal = dict(rotation=rotation.copy(), covariance=covariance.copy(), quaternion=quaternion.copy())
            rows.append(dict(dtype=str(dtype), input_multiplier=magnitude, accepted=True,
                normalized_quaternion_norm=float(np.linalg.norm(quaternion)),
                quaternion_max_abs_difference_vs_normal=float(np.max(np.abs(quaternion-normal['quaternion']))),
                rotation_max_abs_difference_vs_normal=float(np.max(np.abs(rotation-normal['rotation']))),
                rotation_orthogonality_max_abs_error=float(np.max(np.abs(rotation.T@rotation-np.eye(3)))),
                rotation_determinant=float(np.linalg.det(rotation)),
                covariance_max_abs_difference_vs_normal=float(np.max(np.abs(covariance-normal['covariance']))),
                covariance_relative_frobenius_difference_vs_normal=float(np.linalg.norm(covariance-normal['covariance'])/np.linalg.norm(normal['covariance']))))
        except Exception as error:
            rows.append(dict(dtype=str(dtype), input_multiplier=magnitude, accepted=False,
                error_type=type(error).__name__, message=str(error)))
report = dict(schema='free-clothing-quaternion-normalization-audit/1',
    original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    fixture=dict(raw_quaternion_direction=[1., 2., 3., 4.], physical_scale_radii=[.13, .08, .04],
                 selected_frame='reference', camera_w2c='identity', center=[.2, .3, 3.35]),
    comparisons=rows, CPU_only=True, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    original_14_tests_run=False, model_forwards=0, GPU_jobs=0, training_steps=0,
    interpretation='Quaternion amplitude must not change represented orientation or covariance. Positive-definite covariance alone does not validate orthonormal rotation.',
    physical_quality_accepted=False)
path=Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/tiny-quaternion-audit.json')
with path.open('x',encoding='utf-8') as stream:
    json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report,indent=2))
