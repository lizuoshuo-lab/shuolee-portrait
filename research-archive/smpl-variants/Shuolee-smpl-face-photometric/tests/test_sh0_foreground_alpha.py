"""Regression controls for independent SH0 foreground color and coverage."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.training import (
    _face_interior_pixels, _sh0_foreground_alpha_loss, foreground_loss, smpl_sh0_effective_config, train_official,
)


class SH0ForegroundAlphaTests(unittest.TestCase):
    def test_opacity_resets_leave_the_terminal_interval_for_recovery(self):
        config = smpl_sh0_effective_config()
        self.assertEqual(config['iterations'], 10_000)
        self.assertEqual(config['optimization']['opacity_reset_interval'], 1_800)
        self.assertEqual(config['opacity_reset_steps'], [1_800, 3_600, 5_400, 7_200])

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

    def test_spill_penalty_does_not_shrink_with_background_size(self):
        alpha = self.mask.clone()
        alpha[:, 3, 3] = .8
        target = self.black[:, 8:56, 8:56]
        small = self.loss(self.black, alpha, target)[3]
        pad = lambda x: torch.nn.functional.pad(x, (0, 64, 0, 64))
        large = _sh0_foreground_alpha_loss(pad(self.black), pad(alpha), target, pad(self.mask), self.box)[3]
        torch.testing.assert_close(large, small, rtol=0, atol=0)
        self.assertAlmostEqual(small.item(), .5*.8/(48*48))

    def test_face_photometric_option_preserves_other_profile_values(self):
        enabled = smpl_sh0_effective_config()
        disabled = smpl_sh0_effective_config(face_photometric=False)
        self.assertTrue(enabled['face_photometric'])
        self.assertEqual(enabled['face_photometric_weight'], .05)
        self.assertEqual(dict(disabled, face_photometric=True), enabled)
        with patch('shuolee_portrait.training._train_smpl_sh0') as train:
            train_official('.', dict(enabled, face_photometric=False))
            train.assert_called_once_with(Path('.'), disabled)
        with self.assertRaisesRegex(ValueError, 'face_photometric must be a boolean'):
            train_official('.', dict(enabled, face_photometric='false'))

    def test_face_roi_uses_confident_in_raster_landmarks_and_eroded_mask(self):
        points = np.array([[20, 20], [29, 20], [38, 20], [38, 29], [38, 38], [29, 38], [20, 38], [20, 29]], float)
        landmarks = np.c_[points, np.ones(len(points))]
        unsupported = np.array([[1, 1, .2], [-7, 25, .9], [25, 100, .9], [30, np.nan, .9]])
        self.mask[:, 28:31, 28:31] = .5
        rows, columns = _face_interior_pixels(np.vstack((landmarks, unsupported)), self.mask, self.box)
        self.assertGreaterEqual(len(rows), 32)
        self.assertGreaterEqual(min(rows.min(), columns.min()), 22-self.box[0])
        self.assertLessEqual(max(rows.max(), columns.max()), 36-self.box[0])
        self.assertTrue((self.mask[0, rows+self.box[1], columns+self.box[0]] == 1).all())
        self.assertFalse(((rows == 21) & (columns == 18)).any())  # Two pixels outside the soft interior gap.
        self.assertIsNone(_face_interior_pixels(landmarks[:7], self.mask, self.box))
        self.assertIsNone(_face_interior_pixels(landmarks, self.mask*0, self.box))

    def test_face_composites_distinguish_dark_holes_and_opaque_grey(self):
        target = self.black[:, 8:56, 8:56]
        pixels = torch.tensor([[24], [24]])
        opaque = self.mask.clone()
        transparent = opaque.clone()
        transparent[0, 32, 32] = .6
        grey = self.black.clone()
        grey[:, 32, 32] = .25

        def extra(rgb, alpha):
            result = _sh0_foreground_alpha_loss(rgb, alpha, target, self.mask, self.box,
                                              face_pixels=pixels, face_weight=.05)
            base = self.loss(rgb, alpha, target)
            for after, before in zip(result[1:], base[1:], strict=True):
                self.assertTrue(torch.equal(after, before))
            return float(result[0]-base[0])

        self.assertEqual(extra(self.black, opaque), 0.)
        self.assertAlmostEqual(extra(self.black, transparent), .05*.2)
        self.assertAlmostEqual(extra(grey, opaque), .05*.25)

    def test_face_loss_has_gradients_only_at_selected_pixels(self):
        pixels = torch.tensor([[24], [24]])
        alpha = (.5*self.mask).requires_grad_()
        rgb = (.2*self.black + .1).requires_grad_()
        target = self.black[:, 8:56, 8:56]
        base = self.loss(rgb, alpha, target)[0]
        changed = _sh0_foreground_alpha_loss(rgb, alpha, target, self.mask, self.box,
                                            face_pixels=pixels, face_weight=.05)[0]
        base_grads = torch.autograd.grad(base, (rgb, alpha))
        changed_grads = torch.autograd.grad(changed, (rgb, alpha))
        for before, after, sign in zip(base_grads, changed_grads, (1, -1), strict=True):
            delta = after-before
            self.assertTrue(torch.isfinite(delta).all())
            self.assertTrue((sign*delta[:, 32, 32] > 0).all())
            delta[:, 32, 32] = 0
            self.assertEqual(delta.count_nonzero().item(), 0)

    def test_disabled_and_empty_face_support_match_values_and_gradients_exactly(self):
        rgb = (.3*self.mask.expand_as(self.black)).requires_grad_()
        alpha = (.6*self.mask).requires_grad_()
        target = torch.full((3, 48, 48), .7)
        options = ({}, dict(face_pixels=torch.tensor([[24], [24]]), face_weight=0.),
                   dict(face_pixels=torch.empty((2, 0), dtype=torch.long), face_weight=.05),
                   dict(face_pixels=None, face_weight=.05))
        expected = self.loss(rgb, alpha, target)
        baseline = torch.autograd.grad(expected[0]+.1*expected[3], (rgb, alpha))
        for opts in options:
            result = _sh0_foreground_alpha_loss(rgb, alpha, target, self.mask, self.box, **opts)
            gradient = torch.autograd.grad(result[0]+.1*result[3], (rgb, alpha))
            for before, after in zip((*expected, *baseline), (*result, *gradient), strict=True):
                self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
