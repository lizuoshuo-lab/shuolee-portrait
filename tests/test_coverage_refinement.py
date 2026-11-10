import unittest

import torch

from shuolee_portrait.coverage_refinement import (_clip, _region_alpha, _reliable_regions,
                                                  _select_sites, coverage_config)
from shuolee_portrait.head_appearance import snapshot_name


class CoverageRefinementTests(unittest.TestCase):
    def test_alpha_pushes_only_certified_interior_and_exterior(self):
        options = coverage_config()
        mask = torch.zeros(1, 23, 23)
        mask[0, 6:17, 6:17] = 1
        inside, outside = _reliable_regions(mask, options)
        self.assertEqual(int(inside.sum()), 49)
        self.assertFalse((inside & outside).any())
        alpha = torch.full((23, 23), .5, requires_grad=True)
        _region_alpha(alpha, inside, outside, options).backward()
        self.assertTrue((alpha.grad[inside] < 0).all())
        self.assertTrue((alpha.grad[outside] > 0).all())
        self.assertTrue((alpha.grad[~(inside | outside)] == 0).all())

    def test_two_bins_veto_and_one_site_per_triangle(self):
        votes = torch.tensor([[2, 3, 1], [2, 2, 1], [1, 1, 1]])
        veto = torch.tensor([[False, True, False], [False, False, False], [False, False, False]])
        deficit = torch.tensor([[1.1, 1.9, .9], [1., 1.5, .9], [.5, .7, .8]])
        owners, sites = _select_sites(votes, veto, deficit, votes, coverage_config(), 3)
        torch.testing.assert_close(owners, torch.tensor([1, 0]))
        torch.testing.assert_close(sites, torch.tensor([1, 0]))

    def test_clip_preserves_native_pixel_corners_and_near_order(self):
        width, height = 160, 80
        projected = torch.tensor([[63.5, 31.5, 1.], [127., 63., 2.]])
        clip = _clip(projected, width, height)
        ndc = clip[:, :3]/clip[:, 3:]
        pixels = torch.stack(((ndc[:, 0]+1)*width/2, (1-ndc[:, 1])*height/2), dim=1)
        torch.testing.assert_close(pixels, torch.tensor([[63.5, 31.5], [63.5, 31.5]]))
        self.assertLess(ndc[0, 2], ndc[1, 2])

    def test_export_accepts_added_points_without_mutating_the_prior_count(self):
        head = 'step-010000-head-000600'
        surface = head + '-surface-001800'
        terminal = surface + '-coverage-001500'
        receipt = dict(steps=10000, points=15, final_snapshot=terminal,
                       head_appearance=dict(state='completed', steps=600, snapshot=head),
                       surface_refinement=dict(state='completed', steps=1800, snapshot=surface,
                                               baseline_snapshot=head, points=12, frozen_mesh_verified=True),
                       coverage_refinement=dict(state='completed', steps=1500, snapshot=terminal,
                                                baseline_snapshot=surface, initial_points=12,
                                                points=15, frozen_mesh_verified=True))
        self.assertEqual(snapshot_name(receipt), terminal)
        with self.assertRaises(ValueError):
            snapshot_name(dict(receipt, points=13))
