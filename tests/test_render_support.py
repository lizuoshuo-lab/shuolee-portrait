import math
from types import SimpleNamespace
import unittest

import torch

from shuolee_portrait.render_support import FootprintLimits, classify, color_weights


class RenderSupportTests(unittest.TestCase):
    def test_gradient_is_opaque_pixel_mass_after_ordered_occlusion(self):
        composite = torch.tensor([[[.60, .32], [.0, .80]], [[.0, .0], [.0, .0]]])
        inside = torch.tensor([[True, False], [False, False]])
        outside = torch.tensor([[False, True], [False, False]])
        with torch.no_grad():
            weights = color_weights(lambda color: torch.einsum('hwp,pc->chw', composite, color),
                                    inside, outside, 2)
        torch.testing.assert_close(weights, torch.tensor([[.60, .0, .60], [.32, .80, 1.12]]))

    def test_classification_does_not_consult_projected_centers(self):
        weights = torch.tensor([[[5., 0., 5.], [.0, .1, .2], [.0, 2., 2.], [.2, 0., 3.]],
                                [[0., 0., 0.], [.0, .1, .2], [.0, 2., 2.], [.2, 0., 3.]]])
        config = dict(min_equivalent_pixels=1., min_background_bins=2, background_to_foreground=.20)
        fg, unobserved, conflict = classify(weights, [0, 1], config)
        self.assertEqual(fg.tolist(), [True, False, False, False])
        self.assertEqual(unobserved.tolist(), [False, True, False, False])
        self.assertEqual(conflict.tolist(), [False, False, True, False])

    def test_cap_survives_clone_prune_and_following_optimizer_growth(self):
        param = torch.nn.Parameter(torch.zeros((2, 3)))
        limits = FootprintLimits(param)
        with torch.no_grad():
            limits.set(torch.tensor([0]), torch.tensor([[-.2, -.3, .1]]))
            limits.apply(param, SimpleNamespace(state={}), measure=False)
            self.assertEqual(limits.summary()['projected_point_updates'], 0)
            limits.append(2, torch.tensor([0, 1]))
            limits.append(1, None)
            limits.prune(torch.tensor([False, True, True, True, True]))
            param = torch.nn.Parameter(torch.tensor([[0., 0., 0.], [.2, -.2, -.8], [.6, 0., 0.], [0., 0., 0.]]))
            moment = -torch.ones_like(param)
            limits.apply(param, SimpleNamespace(state={param: {'exp_avg': moment}}))
        self.assertEqual(limits.lineage.tolist(), [-1, 0, -1, -1])
        torch.testing.assert_close(param[1], torch.tensor([-.2, -.3, -.8]))
        torch.testing.assert_close(moment[1], torch.tensor([0., 0., -1.]))
        self.assertEqual(limits.summary()['inherited_children'], 1)
        self.assertEqual(limits.summary()['pruned_points'], 1)
        self.assertTrue(math.isclose(limits.summary()['max_preprojection_growth_ratio'], math.exp(.4), rel_tol=1e-6))

