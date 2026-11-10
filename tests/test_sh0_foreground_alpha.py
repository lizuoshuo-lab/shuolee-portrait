"""Regression controls for independent SH0 foreground color and coverage."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.training import _sh0_foreground_alpha_loss, foreground_loss


class SH0ForegroundAlphaTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(1, 64, 64)
        self.mask[:, 8:56, 8:56] = 1
        self.box = (8, 8, 56, 56)
        self.black = torch.zeros(3, 64, 64)

    def loss(self, rendered, alpha, target):
        return _sh0_foreground_alpha_loss(rendered, alpha, target, self.mask, self.box)

    def test_black_foreground_differs_from_a_hole(self):
        target = self.black[:, 8:56, 8:56]
        opaque = self.mask.clone()
        hole = self.mask.clone()
        hole[:, 27:37, 27:37] = 0
        opaque_loss = self.loss(self.black, opaque, target)
        hole_loss = self.loss(self.black, hole, target)
        torch.testing.assert_close(hole_loss[0], opaque_loss[0], rtol=0, atol=0)
        self.assertEqual(opaque_loss[3].item(), 0.)
        self.assertAlmostEqual(hole_loss[3].item(), .5 * 100 / (48 * 48))

    def test_background_rgb_does_not_affect_color(self):
        self.mask[:, 21:25, 8:16] = 0
        target = torch.full((3, 48, 48), .6)
        alpha = .7 * self.mask
        rendered = torch.full_like(self.black, .4) * alpha
        base = self.loss(rendered, alpha, target)
        target[:, 13:17, :8] = 1.
        rendered[:, 21:25, 8:16] = .05
        changed = self.loss(rendered, alpha, target)
        for before, after in zip(base, changed, strict=True):
            torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_crop_keeps_original_ssim_support(self):
        self.mask[:, 21:25, 8:16] = 0
        im = torch.rand((3, 64, 64), generator=torch.Generator().manual_seed(8))
        gt = torch.rand((3, 64, 64), generator=torch.Generator().manual_seed(9))
        old = foreground_loss(im, gt, self.mask)
        new = self.loss(im*self.mask, self.mask, gt[:, 8:56, 8:56])
        for before, after in zip(old, new[:3], strict=True):
            torch.testing.assert_close(after, before)

    def test_coverage_gradients_respect_soft_targets(self):
        self.mask[:, 7, 8:56] = .4
        alpha = .7 * self.mask
        alpha[:, 3, 3] = .8
        alpha.requires_grad_(True)
        target = self.black[:, 8:56, 8:56]
        self.loss(self.black, alpha, target)[3].backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertLess(alpha.grad[0, 32, 32].item(), 0.)
        self.assertLess(alpha.grad[0, 7, 30].item(), 0.)
        self.assertGreater(alpha.grad[0, 3, 3].item(), 0.)


if __name__ == '__main__':
    unittest.main()
