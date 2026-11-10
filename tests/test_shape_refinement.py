import unittest

import torch

from shuolee_portrait.head_appearance import snapshot_name
from shuolee_portrait.shape_refinement import _balanced_alpha, _tangent_distance, shape_config


class ShapeRefinementTests(unittest.TestCase):
    def test_zero_far_background_does_not_dilute_spill(self):
        options = shape_config()
        alpha = torch.tensor([.50, .50, .80, .50], requires_grad=True)
        inside = torch.tensor([True, False, False, False])
        outside = torch.tensor([False, True, False, False])
        loss = _balanced_alpha(alpha, inside, outside, options)
        loss.backward()
        self.assertLess(alpha.grad[0], 0)
        self.assertGreater(alpha.grad[1], 0)
        self.assertTrue((alpha.grad[2:] == 0).all())
        zeros = torch.zeros(1024)
        padded = _balanced_alpha(torch.cat((alpha.detach(), zeros)),
            torch.cat((inside, zeros.bool())), torch.cat((outside, ~zeros.bool())), options)
        torch.testing.assert_close(loss.detach(), padded)

    def test_planar_offset_is_measured_from_actual_triangle(self):
        vertices = torch.tensor([[-1., -1.], [1., -1.], [0., 1.]]).expand(3, -1, -1)
        locations = torch.tensor([[0., 0.], [0., 2.], [2., -1.]], requires_grad=True)
        result = _tangent_distance(locations, vertices)
        torch.testing.assert_close(result, torch.tensor([0., 1., 1.]))
        result.sum().backward()
        self.assertTrue(torch.isfinite(locations.grad).all())
        torch.testing.assert_close(locations.grad[1:], torch.tensor([[0., 1.], [1., 0.]]))

    def test_final_shape_receipt_preserves_coverage_count(self):
        head = 'step-010000-head-000600'
        surface = head+'-surface-001800'
        coverage = surface+'-coverage-001500'
        final = coverage+'-shape-001800'
        receipt = dict(steps=10000, points=16, final_snapshot=final,
            head_appearance=dict(state='completed', steps=600, snapshot=head),
            surface_refinement=dict(state='completed', steps=1800, snapshot=surface,
                baseline_snapshot=head, points=12, frozen_mesh_verified=True),
            coverage_refinement=dict(state='completed', steps=1500, snapshot=coverage,
                baseline_snapshot=surface, initial_points=12, points=15, frozen_mesh_verified=True),
            shape_refinement=dict(state='completed', steps=1800, snapshot=final,
                baseline_snapshot=coverage, initial_points=15, points=16, frozen_mesh_verified=True))
        self.assertEqual(snapshot_name(receipt), final)
        with self.assertRaises(ValueError):
            snapshot_name(dict(receipt, shape_refinement=dict(receipt['shape_refinement'], initial_points=14)))
