"""CPU controls for the scoped head-appearance loss and snapshot handoff."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.head_appearance import editable_points, face_composite_loss, face_interior_pixels, polish, snapshot_name


class HeadAppearanceTests(unittest.TestCase):
    def test_native_landmarks_define_crop_relative_confident_face_interior(self):
        box = (11, 9, 43, 39)
        mask = torch.zeros(1, 48, 56)
        mask[:, 9:39, 11:43] = 1.
        mask[0, 25, 25] = .99
        landmarks = np.array([(14, 12, .9), (25, 12, .9), (40, 12, .9), (40, 25, .9),
                              (40, 36, .9), (25, 36, .9), (14, 36, .9), (14, 25, .9),
                              (0, 0, .49), (55, 47, .49), (56, 15, .95), (np.nan, 30, .95)])
        pixels = face_interior_pixels(landmarks, mask, box)
        self.assertIsNotNone(pixels)
        rows, columns = pixels
        self.assertEqual((rows.min(), rows.max(), columns.min(), columns.max()), (5, 25, 5, 27))
        self.assertEqual(len(rows), 21*23-13)
        self.assertFalse(((rows == 16) & (columns == 14)).any())
        self.assertTrue(((rows == 5) & (columns == 5)).any())
        self.assertIsNone(face_interior_pixels(landmarks[:7], mask, box))
        self.assertIsNone(face_interior_pixels(landmarks, mask*.99, box))

    def test_two_background_composite_separates_opaque_black_from_a_hole(self):
        box = (2, 1, 6, 5)
        rgb = torch.zeros(3, 7, 8)
        alpha = torch.ones(1, 7, 8)
        target = torch.zeros(3, 4, 4)
        pixels = torch.tensor([[0, 2], [1, 3]])
        self.assertEqual(face_composite_loss(rgb, alpha, target, box, pixels).item(), 0.)
        self.assertEqual(face_composite_loss(rgb, alpha*0, target, box, pixels).item(), .5)
        self.assertEqual(face_composite_loss(rgb+.25, alpha, target, box, pixels).item(), .25)

    def test_direct_face_composite_gradients_only_touch_selected_native_pixels(self):
        box = (2, 1, 6, 5)
        rgb = torch.full((3, 7, 8), .2, requires_grad=True)
        alpha = torch.full((1, 7, 8), .7, requires_grad=True)
        target = torch.full((3, 4, 4), .1)
        pixels = torch.tensor([[0, 2], [1, 3]])
        face_composite_loss(rgb, alpha, target, box, pixels).backward()
        rows, columns = pixels[0]+box[1], pixels[1]+box[0]
        expected_rgb = torch.zeros_like(rgb)
        expected_rgb[:, rows, columns] = 1/6
        expected_alpha = torch.zeros_like(alpha)
        expected_alpha[:, rows, columns] = -.25
        torch.testing.assert_close(rgb.grad, expected_rgb)
        torch.testing.assert_close(alpha.grad, expected_alpha)

    def test_editable_points_require_all_three_triangle_vertices_and_keep_binding_order(self):
        protected = np.array([True, True, True, False, False, False, True])
        faces = np.array([[0, 1, 2], [0, 1, 3], [3, 4, 5], [2, 1, 6]])
        binding = np.array([2, 3, 0, 1, 0, 3], dtype=np.int32)
        self.assertEqual(editable_points(protected, faces, binding).dtype, np.dtype(bool))
        np.testing.assert_array_equal(editable_points(protected, faces, binding), [False, True, True, False, True, True])

    def test_unsupported_head_appearance_returns_the_stock_replay_without_rendering(self):
        stock = Path('step-010000')
        replay = (SimpleNamespace(binding=torch.tensor([0])), {})
        fit = dict(faces=np.array([[0, 1, 2]]))
        for protected, reason in (([True, True, False], 'no_editable_head_points'),
                                  ([True, True, True], 'no_usable_face_observations')):
            with self.subTest(reason=reason), patch.dict(sys.modules, {'gaussian_renderer': SimpleNamespace(render=None)}):
                selected, receipt = polish(Path('.'), stock, replay, fit, [], dict(pose_indices=[0]), {},
                    np.array(protected), [], [(None, None, None)], {})
                self.assertIs(selected, replay)
                self.assertEqual(receipt, dict(state='skipped', reason=reason, steps=0, snapshot=stock.name))


class HeadAppearanceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.stock = dict(steps=10_000, final_snapshot='step-010000')
        self.tail = dict(state='completed', steps=600, snapshot='step-010000-head-000600')

    def test_snapshot_selects_legacy_and_completed_tail(self):
        self.assertEqual(snapshot_name(self.stock), self.stock['final_snapshot'])
        completed = dict(self.stock, head_appearance=self.tail, final_snapshot=self.tail['snapshot'])
        self.assertEqual(snapshot_name(completed), self.tail['snapshot'])

    def test_snapshot_keeps_stock_on_skip_or_failed_tail(self):
        for status in ('skipped', 'failed'):
            with self.subTest(status=status):
                outcome = dict(state=status, reason='insufficient frozen ROI coverage')
                self.assertEqual(snapshot_name(dict(self.stock, head_appearance=outcome)), self.stock['final_snapshot'])
                outcome['snapshot'] = self.stock['final_snapshot']
                self.assertEqual(snapshot_name(dict(self.stock, head_appearance=outcome)), self.stock['final_snapshot'])

    def test_snapshot_rejects_traversal_and_incomplete_tail_claims(self):
        completed = dict(self.stock, head_appearance=self.tail, final_snapshot=self.tail['snapshot'])
        invalid = (dict(self.stock, final_snapshot='../step-010000'),
                   dict(completed, head_appearance=dict(self.tail, snapshot='../step-010000-head-000600')),
                   dict(completed, head_appearance=dict(self.tail, steps=599)),
                   dict(completed, head_appearance=dict(self.tail, snapshot='step-009000-head-000600')),
                   dict(completed, final_snapshot=self.stock['final_snapshot']),
                   dict(self.stock, head_appearance=dict(state='failed', reason='fixture', snapshot=self.tail['snapshot'])))
        for receipt in invalid:
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                snapshot_name(receipt)


if __name__ == '__main__':
    unittest.main()
