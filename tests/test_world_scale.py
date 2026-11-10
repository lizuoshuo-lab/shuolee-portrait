import math
from types import SimpleNamespace
import unittest

import torch

from shuolee_portrait.world_scale import (WorldScale, _children, _limits, _off_surface, _projected_widths,
                                          _source_conflict, scale_config)


class WorldScaleTests(unittest.TestCase):
    def test_cap_is_metric_and_per_owner_over_all_poses(self):
        face_max = torch.tensor([.004, .012, .031])
        cap, log_cap = _limits(face_max, 1.8, scale_config())
        torch.testing.assert_close(cap, torch.tensor([.0288, .036, .045]))
        for scale in (face_max, face_max*.5):
            self.assertTrue((log_cap.exp()*scale <= cap+1e-7).all())

    def test_projection_changes_only_overlarge_axes_and_outward_adam_momentum(self):
        parameter = torch.nn.Parameter(torch.tensor([[.3, .1, -.5], [1.1, .5, -.2]]))
        exp_avg = torch.tensor([[-2., -3., 4.], [5., -6., -7.]])
        model = SimpleNamespace(_scaling=parameter, binding=torch.tensor([0, 1]),
                                optimizer=SimpleNamespace(state={parameter: {'exp_avg': exp_avg}}))
        guard = WorldScale.__new__(WorldScale)
        guard.model, guard.log_cap = model, torch.tensor([.2, .7])
        guard.projections = torch.zeros(3, dtype=torch.long)
        with torch.no_grad():
            guard.project()
            guard.project()
        torch.testing.assert_close(parameter, torch.tensor([[.2, .1, -.5], [.7, .5, -.2]]))
        torch.testing.assert_close(exp_avg, torch.tensor([[0., -3., 4.], [5., -6., -7.]]))
        self.assertEqual(guard.projections.tolist(), [2, 2, 1])

    def test_replacement_preserves_midpoint_opacity_and_reduces_major_sigma(self):
        sigma = torch.tensor([.04, .04, .04])
        alpha = torch.tensor([.10, .50, .99995])
        options = scale_config()
        child, shift, opacity = _children(sigma, alpha, options)
        midpoint = opacity*torch.exp(-.5*(shift/child).square())
        torch.testing.assert_close(1-(1-midpoint).square(), alpha)
        torch.testing.assert_close(child[:2].square()+shift[:2].square(), sigma[:2].square())
        self.assertTrue((child < sigma).all())
        self.assertTrue((opacity <= options['child_opacity_max']).all())

    def test_thin_large_world_axis_can_have_small_native_projection(self):
        K = torch.diag(torch.tensor([1000., 1000., 1.]))
        axes = torch.diag(torch.tensor([.002, .003, .100]))[None]
        axis, ellipse = _projected_widths(torch.zeros((1, 2)), torch.ones(1), K, torch.eye(4), axes, torch.tensor([2]))
        self.assertEqual(axis.item(), 0)
        self.assertTrue(math.isclose(ellipse.item(), 3))


    def test_off_surface_distance_uses_triangle_instead_of_centroid(self):
        local = torch.tensor([[.8, 0., 0.], [.8, 1., 0.], [3., 0., 0.]])
        triangle = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]]).expand(3, -1, -1)
        gap, displaced = _off_surface(local, triangle, torch.full((3,), .025), 1.7, scale_config()['needle'])
        torch.testing.assert_close(gap, torch.tensor([0., .025, .05]))
        self.assertEqual(displaced.tolist(), [False, True, True])

    def test_native_conflict_is_distinct_from_occlusion_and_supported_geometry(self):
        fg, bg = torch.tensor([2, 3, 0, 1]), torch.tensor([5, 2, 0, 1])
        conflicted, unobserved = _source_conflict(fg, bg, 3)
        self.assertEqual(conflicted.tolist(), [True, False, False, False])
        self.assertEqual(unobserved.tolist(), [False, False, True, False])
