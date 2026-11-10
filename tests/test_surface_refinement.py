import unittest

import torch

from shuolee_portrait.head_appearance import snapshot_name
from shuolee_portrait.surface_refinement import _geometry_loss, _split_candidates, surface_config


class SurfaceRefinementTests(unittest.TestCase):
    def test_axial_penalty_shortens_only_the_long_axis(self):
        sigma = torch.tensor([[.005, .005, .00001], [.020, .002, .00001]], requires_grad=True)
        xyz = torch.tensor([[0., 0., 0.], [1.5, 0., 0.]], requires_grad=True)
        _geometry_loss(xyz, sigma, 2., surface_config()).backward()
        torch.testing.assert_close(sigma.grad[0], torch.zeros(3))
        self.assertGreater(sigma.grad[1, 0], 0)
        torch.testing.assert_close(sigma.grad[1, 1:], torch.zeros(2))
        self.assertGreater(xyz.grad[1, 0], 0)

    def test_split_budget_is_separate_for_head_and_body(self):
        head = torch.tensor([True, True, True, False, False, False])
        score = torch.tensor([.9, .6, .0, .05, .02, .001])
        options = dict(surface_config(), split_fraction=.5, minimum_screen_gradient=.01)
        torch.testing.assert_close(_split_candidates(score, head, options),
                                   torch.tensor([True, False, False, True, False, False]))

    def test_export_uses_only_a_bound_completed_refinement(self):
        base = 'step-010000-head-000600'
        terminal = base + '-surface-001800'
        receipt = dict(steps=10000, points=12, final_snapshot=terminal,
                       head_appearance=dict(state='completed', steps=600, snapshot=base),
                       surface_refinement=dict(state='completed', steps=1800, snapshot=terminal,
                                               baseline_snapshot=base, points=12, frozen_mesh_verified=True))
        self.assertEqual(snapshot_name(receipt), terminal)
        with self.assertRaises(ValueError):
            snapshot_name(dict(receipt, points=13))
        with self.assertRaises(ValueError):
            snapshot_name(dict(receipt, surface_refinement=dict(receipt['surface_refinement'],
                                                                 frozen_mesh_verified=False)))
