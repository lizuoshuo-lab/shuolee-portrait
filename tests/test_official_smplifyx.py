"""Portable observation and single-frame interface checks, without fitting."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait import official_smplifyx as entry


class ObservationInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.points = np.arange(3 * 133 * 2).reshape(3, 133, 2).astype(np.float32)
        self.scores = np.tile(np.array([2., 5., 10.])[:, None], (1, 133))
        self.boxes = np.tile([0., 0., 100., 100.], (3, 1))
        self.flip = np.zeros((3, 133))

    def test_whole_roster_scaling_and_118_point_order(self):
        target, meta = entry.adapt_rtmw_observations(self.points, self.scores, self.boxes, self.flip)
        self.assertEqual(target.shape, (3, 118, 3))
        np.testing.assert_array_equal(target[:, 2, :2], self.points[:, 6])
        np.testing.assert_array_equal(target[:, 19:25, :2], self.points[:, 17:23])
        np.testing.assert_array_equal(target[:, 25:67, :2], self.points[:, 91:133])
        np.testing.assert_array_equal(target[:, 67:118, :2], self.points[:, 40:91])
        np.testing.assert_allclose(target[:, 0, 2], [2/9, 5/9, 1])
        self.assertFalse(np.array_equal(target[:, 0, 2], np.clip(self.scores[:, 0], 0, 1)))
        self.assertEqual(meta['per_joint_q90'], [9.] * 133)
        self.assertIn('not calibrated probability', meta['kind'])
        np.testing.assert_array_equal(target[:, [1, 8]], 0)

    def test_flip_confidence_squared_preserves_existing_weight(self):
        self.flip[0, 0] = 2.4
        target, _ = entry.adapt_rtmw_observations(self.points, self.scores, self.boxes, self.flip)
        self.assertAlmostEqual(float(target[0, 0, 2])**2, (2/9)**2*np.exp(-2), places=8)

    def test_unlabeled_raw_scores_rejected(self):
        with self.assertRaisesRegex(ValueError, 'explicit fit_confidence'):
            entry.require_relative_confidence(dict(scores=self.scores))

    def test_invalid_inputs_rejected(self):
        for kind in ('nonfinite', 'negative_score', 'negative_flip', 'box_height', 'point_count'):
            with self.subTest(kind=kind):
                points, scores, boxes, flip = (x.copy() for x in
                                              (self.points, self.scores, self.boxes, self.flip))
                if kind == 'nonfinite': points[0, 0, 0] = np.nan
                if kind == 'negative_score': scores[0, 0] = -1
                if kind == 'negative_flip': flip[0, 0] = -1
                if kind == 'box_height': boxes[0, 3] = 0
                if kind == 'point_count': points = points[:, :132]
                with self.assertRaises(ValueError):
                    entry.adapt_rtmw_observations(points, scores, boxes, flip)


class OfficialFrameInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kw = dict(official_root=self.root, image=np.zeros((8, 10, 3), np.float32),
            keypoints=np.zeros((1, 118, 3), np.float32),
            K_integer_centers=np.array([[10., .3, 4.2], [0., 11., 3.7], [0., 0., 1.]]),
            body_model=object(), vposer2=object(), result_path=self.root/'fit.pkl',
            diff_path=self.root/'input-output.diff', shared_betas=np.ones(10),
            freeze_betas=False, device='cuda')

    def test_core_called_once_with_original_arguments_and_explicit_shape_policy(self):
        for frozen in (False, True):
            with self.subTest(freeze_betas=frozen):
                self.kw['freeze_betas'] = frozen
                calls = []
                arrays = dict(vertices_camera=np.ones((1, 3, 3)))
                def core(**kwargs):
                    calls.append('fit')
                    for name, value in self.kw.items():
                        self.assertIs(kwargs[name], value)
                    return arrays, {'translation_applied_once': True}
                def cache(body):
                    self.assertIs(body, self.kw['body_model'])
                    calls.append('topology')
                    return {'optimization': 'static topology'}
                with patch.object(entry, 'fit_one', side_effect=core) as fit, patch(
                        'shuolee_portrait.smplifyx_runtime.cache_static_topology', side_effect=cache):
                    actual, report = entry.fit_frame(**self.kw)
                self.assertEqual(calls, ['topology', 'fit'])
                fit.assert_called_once()
                self.assertIs(actual, arrays)
                self.assertTrue(report['translation_applied_once'])

    def test_explicit_old_runtime_bypass(self):
        with patch.object(entry, 'fit_one', return_value=({}, {})) as fit, patch(
                'shuolee_portrait.smplifyx_runtime.cache_static_topology') as cache:
            _, report = entry.fit_frame(**self.kw, static_topology=False)
        cache.assert_not_called()
        fit.assert_called_once()
        self.assertIsNone(report['runtime_optimization'])

    def test_invalid_inputs_do_not_call_solver(self):
        for changes in (dict(image=np.zeros((8, 10, 3), np.uint8)),
                        dict(keypoints=np.ones((1, 118, 3))*2),
                        dict(shared_betas=None, freeze_betas=True), dict(device='cpu')):
            with self.subTest(changes=list(changes)):
                with patch.object(entry, 'fit_one') as fit:
                    with self.assertRaises(ValueError):
                        entry.fit_frame(**dict(self.kw, **changes))
                    fit.assert_not_called()

    def test_existing_output_is_preserved(self):
        self.kw['result_path'].write_bytes(b'original result')
        with patch.object(entry, 'fit_one') as fit:
            with self.assertRaisesRegex(ValueError, 'preserve previous fits'):
                entry.fit_frame(**self.kw)
            fit.assert_not_called()
        self.assertEqual(self.kw['result_path'].read_bytes(), b'original result')

    def test_upstream_yaml_numeric_types_restored(self):
        folder = self.root/'cfg_files'
        folder.mkdir()
        (folder/'fit_smplx.yaml').write_text('focal_length: 5e3\nrho: 1e2\n'
            'shape_weights: [1e2, 5e1]\nmaxiters: 30\njaw_pose_prior_weights: ["1,2,3"]\n')
        config = entry.load_official_config(self.root)
        self.assertEqual(config['focal_length'], 5000.)
        self.assertEqual(config['shape_weights'], [100., 50.])
        self.assertEqual(config['maxiters'], 30)
        self.assertEqual(config['jaw_pose_prior_weights'], ['1,2,3'])

    def test_unpinned_official_source_rejected(self):
        folder = self.root/'smplifyx'
        folder.mkdir()
        (folder/'fit_single_frame.py').write_text('def fit_single_frame(): pass\n')
        with self.assertRaisesRegex(ValueError, 'Pinned source differs'):
            entry.adapted_entry_source(self.root)

    def test_no_research_main_or_private_paths_in_module(self):
        source = Path(entry.__file__).read_text(encoding='utf-8')
        functions = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn('main', functions)
        for value in ('/mnt/c/', '/mnt/d/', 'frame_000032', 'full94', 'run_official_sequence'):
            self.assertNotIn(value, source)


if __name__ == '__main__':
    unittest.main()
