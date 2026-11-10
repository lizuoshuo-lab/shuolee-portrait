from types import SimpleNamespace
import unittest

import torch

from shuolee_portrait.render_support import _coverage_metrics, _rgb_error


class ReplacementCoverageTests(unittest.TestCase):
    def setUp(self):
        self.inside = torch.ones((20, 20), dtype=torch.bool)
        self.inside[:, 0] = False
        self.view = SimpleNamespace(box=(1, 0, 20, 20), inside=self.inside, outside=~self.inside,
                                    rgb=torch.full((3, 20, 19), 128, dtype=torch.uint8))
        self.baseline = torch.ones((20, 20))
        self.baseline[:, 0] = 0
        self.error = torch.zeros((20, 19))
        self.options = dict(coverage_tile_px=16, coverage_local_deficit_px=1.,
                            coverage_region_deficit_px=1., coverage_rgb_mae_increase=1/255,
                            coverage_local_rgb_mae_increase=1/255)

    def metrics(self, proposal, error=None, regions=None):
        return _coverage_metrics(self.baseline, proposal, self.error, self.error if error is None else error,
                                 self.view, regions or {}, .98, .65, 2., 2., self.options)

    def test_alpha_reduction_above_target_is_not_missing(self):
        proposal = self.baseline.clone()
        proposal[self.inside] = .99
        result = self.metrics(proposal)
        self.assertGreater(result['redundant_alpha_loss_px'], 2.)
        self.assertEqual(result['foreground_target_deficit_px'], 0.)
        self.assertEqual(result['failures'], [])

    def test_a_new_hole_cannot_be_hidden_by_existing_hole_repair(self):
        self.baseline[1, 1] = .30
        proposal = self.baseline.clone()
        proposal[1, 1], proposal[1, 2] = 1., .64
        result = self.metrics(proposal)
        self.assertLess(result['proposed_target_deficit_px'], result['baseline_target_deficit_px'])
        self.assertEqual(result['new_low_alpha_holes'], 1)
        self.assertEqual([row['item'] for row in result['failures']], ['new_low_alpha_holes'])

    def test_spatial_peak_and_ear_color_are_protected_independently(self):
        proposal = self.baseline.clone()
        proposal[:3, 1:4] = .85
        proposal[:3, 0] = .80
        error = self.error.clone()
        error[18, 17] = .02
        result = self.metrics(proposal, error, {'left_ear': torch.tensor([[18, 18], [18, 0]])})
        failures = {row['item']: row['region'] for row in result['failures']}
        self.assertEqual(failures, {'background_gain_px': None, 'worst_tile_target_deficit_px': [1, 0, 17, 16],
                                    'region_rgb_mae_increase': 'left_ear'})
        self.assertEqual(result['protected_regions']['left_ear']['pixels'], 1)

    def test_rgb_error_uses_straight_color_and_native_crop(self):
        alpha = self.baseline*.8
        rgb = alpha.expand(3, -1, -1)*(128/255)
        rgb[:, :, 0] = 1.
        torch.testing.assert_close(_rgb_error(rgb, alpha, self.view), self.error)


if __name__ == '__main__':
    unittest.main()
