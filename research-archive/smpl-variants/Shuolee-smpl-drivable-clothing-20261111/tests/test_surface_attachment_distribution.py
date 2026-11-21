"""Exercise visible skin selection, weighted backward and main optimizer auditing."""
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch

from shuolee_portrait.surface_attachment import SurfaceAttachment, compare_skin_pairs


class BoundSkinModel:
    def __init__(self):
        self.faces = torch.tensor([[0, 1, 2]])
        vertices = torch.tensor([[.24, .24, 1.], [.24, .30, 1.], [.30, .24, 1.]])
        self.vertices_sequence = [vertices, vertices]
        self.num_timesteps = 2
        self.static_offset = torch.zeros_like(vertices)
        self.binding = torch.zeros(4, dtype=torch.long)
        self.densification_ids = torch.tensor([10, 11, 12, 13])
        self.get_opacity = torch.full((4, 1), .5)
        self._xyz = torch.nn.Parameter(torch.tensor([[0., .004, 0.], [.006, .005, .004],
                                                    [0., .040, 0.], [.10, .004, 0.]]))
        self._scaling = torch.nn.Parameter(torch.tensor([[.004, .004, .004], [.008, .001, .008],
                                                        [.004, .004, .004], [.004, .004, .004]]).log())
        tilted = [math.cos(math.pi/8), math.sin(math.pi/8), 0., 0.]
        self._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.], tilted, tilted, tilted]))
        self.select_mesh_by_timestep(0)

    def select_mesh_by_timestep(self, index):
        self.timestep = index
        self.face_center = torch.tensor([[.26, .26, 1.]])
        self.face_scaling = torch.ones(1, 1)
        self.face_orien_mat = torch.tensor([[[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]])


class VisibleSkinDistributionTests(unittest.TestCase):
    def test_all_visible_material_rows_enter_distribution_and_real_adam_audit(self):
        model = BoundSkinModel()
        intrinsics = torch.diag(torch.tensor([100., 100., 1.]))
        constraints = SimpleNamespace(rows=[dict(frame='f0'), dict(frame='f1')], slots=[0, 1], bins=[0, 1],
            matrices=[(intrinsics, torch.eye(4))]*2, views=[SimpleNamespace(mesh_depth=torch.ones(48, 48))]*2)
        material_mask = torch.ones(32, 32, dtype=torch.bool)

        def skin_pixels(view, row, observed, projected):
            return ((0, 0, 32, 32), material_mask), dict(frame=row['frame'], native_material_mask_sha256='mask')

        # Supply a fixed accepted native skin mask so the test isolates geometry, depth and optimizer behavior.
        with patch('shuolee_portrait.surface_attachment._skin_pixels', side_effect=skin_pixels):
            attachment = SurfaceAttachment(model, constraints, torch.tensor([True]), np.zeros((2, 118, 3)),
                                           projected=np.zeros((2, 118, 2)), distribution=True)

        optimizer = torch.optim.Adam([model._xyz, model._scaling, model._rotation], lr=.001)
        with TemporaryDirectory() as directory:
            first, last = Path(directory)/'first.npz', Path(directory)/'last.npz'
            first_measure = attachment.measure(pairs_path=first)
            self.assertEqual(first_measure['stable_pairs']['rows'], 4)
            self.assertEqual(model.timestep, 0)
            loss = attachment.loss(0, torch.ones(4, dtype=torch.bool), audit_step=101)
            # The main loss also moves all four x coordinates. That term is excluded from the prior gradient audit.
            (.15*loss + model._xyz[:, 0].sum()).backward()
            self.assertGreater(float(model._scaling.grad[0, 1]), 0.)
            self.assertGreater(float(model._rotation.grad[1].norm()), 0.)
            torch.testing.assert_close(model._scaling.grad[2:], torch.zeros(2, 3))
            expected_position_prior_l2 = float(model._xyz.grad[:2, 1].norm())
            optimizer.step()
            attachment.record_optimizer_step()
            model.select_mesh_by_timestep(1)
            self.assertEqual(float(attachment.loss(1, torch.zeros(4, dtype=torch.bool), audit_step=102)), 0.)
            attachment.record_optimizer_step()
            report = attachment.report()
            self.assertEqual(report['coverage']['eligible_row_draws'], 2)
            self.assertEqual(report['coverage']['distribution_row_draws'], 2)
            self.assertEqual(report['coverage']['distribution_nonsheet_row_draws'], 1)
            self.assertEqual(report['coverage']['fraction'], 1.)
            audit, empty = report['gradient_update_audit']['steps']
            self.assertEqual(audit['parameters']['log_scale']['updated_rows_with_prior_gradient'], 2)
            self.assertEqual(audit['parameters']['quaternion']['updated_rows_with_prior_gradient'], 1)
            self.assertEqual(audit['parameters']['position']['updated_rows_with_prior_gradient'], 2)
            self.assertAlmostEqual(audit['parameters']['position']['prior_gradient_l2'], expected_position_prior_l2)
            self.assertEqual(empty['state'], 'no_visible_skin')
            attachment.measure(pairs_path=last)
            pairs = compare_skin_pairs(first, last)
            self.assertEqual(pairs['paired_rows'], 4)
            self.assertEqual(pairs['thin_axis_angle_deg']['paired_rows'], 2)
            self.assertLess(pairs['expected_abs_distance_mm']['paired_delta']['p95'], 0.)


if __name__ == '__main__':
    unittest.main()
