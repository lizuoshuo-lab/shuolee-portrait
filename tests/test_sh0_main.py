import contextlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from shuolee_portrait import cli, runtime as rt
from shuolee_portrait.sh0_main import frozen_surface, initialization, main_schedule
from shuolee_portrait.training import smpl_sh0_effective_config


class UnifiedSH0Tests(unittest.TestCase):
    def test_schedule_finishes_topology_and_resets_before_terminal(self):
        effective = smpl_sh0_effective_config()
        schedule = effective['unified_main']
        self.assertEqual(schedule, main_schedule())
        self.assertEqual(effective['iterations'], 10000)
        self.assertEqual(effective['head_appearance_steps'], 0)
        self.assertLess(schedule['warmup_steps'], min(schedule['coverage_steps']))
        self.assertLess(max(effective['opacity_reset_steps']), max(schedule['coverage_steps']))
        self.assertLess(max(schedule['coverage_steps']), effective['iterations'])
        self.assertEqual(effective['optimization']['densify_until_iter'], schedule['stock_densify_through'])

    def test_mesh_only_reuse_validates_content_and_protected_vertices(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'fit'/'frozen-surface.npz'
            path.parent.mkdir()
            faces = np.array([[0, 1, 2]])
            offsets = np.zeros((3, 3), dtype=np.float32)
            fields = dict(vertices_world=offsets, faces=faces, reference_frame_id='frame_000001',
                          reference_pose_index=0, static_offset=offsets)
            fit, layout = dict(faces=faces), dict(reference_frame_id='frame_000001', reference_pose_index=0)
            clothes = np.array([False, True, True])
            np.savez(path, **fields)
            spec = dict(file='fit/frozen-surface.npz', sha256=rt.sha(path))
            source, delta = frozen_surface(root, spec, fit, layout, clothes)
            self.assertEqual(source, path)
            np.testing.assert_array_equal(delta, offsets)
            np.savez(path, **fields, prior_gaussians=np.ones((3, 3)))
            with self.assertRaises(ValueError):
                frozen_surface(root, dict(spec, sha256=rt.sha(path)), fit, layout, clothes)

    def test_fresh_initializer_rejects_displaced_centers(self):
        model = SimpleNamespace(_xyz=torch.zeros((2, 3)), _scaling=torch.zeros((2, 3)),
            _rotation=torch.zeros((2, 4)), _features_dc=torch.zeros((2, 1, 3)),
            _features_rest=torch.zeros((2, 0, 3)), _opacity=torch.zeros((2, 1)),
            binding=torch.arange(2), binding_counter=torch.ones(2), faces=torch.zeros((2, 3)),
            optimizer=SimpleNamespace(state={}))
        self.assertEqual(initialization(model)['initial_points'], 2)
        model._xyz[0, 0] = 1
        with self.assertRaises(RuntimeError):
            initialization(model)

    def test_public_train_fitted_dispatches_to_normal_delivery(self):
        with patch.object(sys, 'argv', ['shuolee_portrait.cli', 'train-fitted', '--run', 'staged']), \
             patch('shuolee_portrait.sequence_pipeline.train_fitted', return_value={'state': 'completed'}) as train, \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        train.assert_called_once_with(Path('staged'))
