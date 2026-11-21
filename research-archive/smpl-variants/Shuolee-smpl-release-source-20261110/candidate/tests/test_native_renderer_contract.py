"""CPU interface checks only; these do not execute or simulate native CUDA."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait.native_renderer import ResidualRenderContract, render_native


class NativeRendererInterface(unittest.TestCase):
    def test_import_and_manifest_without_torch_or_gsplat(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'gsplat'):
        raise AssertionError('Heavy dependency imported while reading the camera contract')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from shuolee_portrait.native_renderer import ResidualRenderContract
manifest = ResidualRenderContract('processed_output_pixels').manifest()
assert manifest['pixels'] == 'pixel_corners'
assert manifest['gaussian_evaluation'] == 'ray-3d'
assert manifest['camera_optimization'] is False
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        subprocess.run([sys.executable, '-c', code], env=env, check=True,
                       capture_output=True, text=True)

    def test_contract_has_explicit_brown_order_and_no_image_warp(self):
        contract = ResidualRenderContract('processed_output_pixels')
        manifest = contract.manifest()
        self.assertEqual(manifest['coefficient_order'],
                         ['k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                          'p1', 'p2', 's1', 's2', 's3', 's4'])
        self.assertEqual(manifest['mapping'], 'ideal_normalized_to_observed_normalized')
        self.assertFalse(manifest['image_resampled'])
        manifest['coefficient_order'].clear()
        self.assertEqual(len(contract.manifest()['coefficient_order']), 12)
        for invalid in (ResidualRenderContract('raw_fisheye'),
                        ResidualRenderContract('processed_output_pixels', 'integer_centers')):
            with self.assertRaises(ValueError):
                invalid.manifest()

    def test_cpu_inputs_fail_before_loading_native_backend(self):
        # A CPU tensor must produce the actionable device error even when gsplat
        # is absent, rather than trying a renderer fallback or importing CUDA.
        xyz = torch.tensor([[0., 0., 3.]])
        quats = torch.tensor([[1., 0., 0., 0.]])
        scales = torch.full((1, 3), .1)
        E = torch.eye(4)[None]
        K = torch.tensor([[[30., 0., 4.2], [0., 31., 3.3], [0., 0., 1.]]])
        with mock.patch.dict(sys.modules, {'gsplat': None, 'gsplat.cuda._wrapper': None}):
            with self.assertRaisesRegex(ValueError, 'CUDA device and float32'):
                render_native(xyz, quats, scales, torch.tensor([.3]),
                              torch.full((1, 3), .4), E, K, 8, 6,
                              contract=ResidualRenderContract('synthetic_known_camera'),
                              radial_coeffs=torch.zeros((1, 6)),
                              tangential_coeffs=torch.zeros((1, 2)),
                              thin_prism_coeffs=torch.zeros((1, 4)))


if __name__ == '__main__':
    unittest.main()
