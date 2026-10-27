"""CPU checks for the 119 input boundary and portable Gaussian transforms.

These tests do not fit a person, run CUDA, verify third-party UI, or establish
physical calibration. They exercise real files/NPZ/PLY rather than mock images.
"""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait import route119
from shuolee_portrait.export import rigid_rotation64, rotate_ply
from shuolee_portrait.observation_support import image_support
from shuolee_portrait.prior import camera_intrinsics_for_prior


class PreparedFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.video = self.base / 'source.mp4'
        self.video.write_bytes(b'unit fixture video identity only; no decoder is invoked')
        self.root = self.base / 'prepared'
        self.cfg = route119.configuration(image_height=32)
        route119.initialize_run(self.root, self.video, self.cfg, self.base / 'assets')
        raw = self.root / 'raw'
        masks = self.root / 'data/masks'
        raw.mkdir()
        masks.mkdir()
        self.rows = []
        for i in range(12):
            frame = f'frame_{i:06d}'
            image, mask = raw / f'{frame}.png', masks / f'{frame}.png'
            rgb = np.zeros((32, 24, 3), np.uint8)
            rgb[5:28, 4:20] = [120, 150, i]
            Image.fromarray(rgb).save(image)
            alpha = np.zeros((32, 24), np.uint8)
            alpha[5:28, 4:20] = 255
            Image.fromarray(alpha).save(mask)
            E = np.eye(4)
            E[0, 3] = i / 10
            self.rows.append(dict(frame=frame, name=frame + '.png',
                source_frame_index=i * 5, time_seconds=i / 6,
                image=str(image), source_sha256=rt.sha(image), wh=[24, 32],
                mask=str(mask), mask_sha256=rt.sha(mask),
                initial_box_xyxy=[4., 5., 20., 28.],
                K=[[30., 0., 12.6], [0., 31., 17.1], [0., 0., 1.]], w2c=E.tolist(),
                official_prediction_slot=i, camera_kind='fresh_test_geometry',
                camera_registration=dict(interpolation_used=False), observation_weight=1.))
        self.save_rows()
        self.save_observations()
        rt.save(self.root / 'preparation.json', dict(
            video=str(self.video), video_sha256=rt.sha(self.video),
            from_this_video=True, old_subject_state_loaded=False))

    def tearDown(self):
        self.temp.cleanup()

    def save_rows(self):
        for name in ('inputs.json', 'training-inputs.json'):
            rt.save(self.root / 'data' / name, dict(records=self.rows))

    def save_observations(self, *, reverse=False):
        ids = [row['frame'] for row in self.rows]
        if reverse:
            ids.reverse()
        np.savez_compressed(self.root / 'data/observations.npz',
            frame_ids=np.asarray(ids), keypoints=np.full((12, 133, 2), 10.),
            scores=np.ones((12, 133)), flip_disagreement=np.zeros((12, 133)),
            boxes=np.asarray([row['initial_box_xyxy'] for row in self.rows]),
            joint_names=np.asarray([f'joint_{i}' for i in range(133)]))


class PreparedInputTests(PreparedFixture):
    def test_real_files_and_coco_observations_pass_without_quality_approval(self):
        report = route119.inspect_prepared(self.root)
        self.assertEqual(report['frames'], 12)
        self.assertTrue(report['pixels_and_masks_verified'])
        self.assertFalse(report['physical_calibration_accepted'])
        self.assertFalse(report['formal_training_admitted'])

    def test_changed_image_is_rejected(self):
        Image.new('RGB', (24, 32), 'red').save(self.rows[0]['image'])
        with self.assertRaisesRegex(ValueError, 'changed image'):
            route119.inspect_prepared(self.root)

    def test_observation_frame_permutation_is_rejected(self):
        self.save_observations(reverse=True)
        with self.assertRaisesRegex(ValueError, 'identities/order'):
            route119.inspect_prepared(self.root)

    def test_camera_frame_order_cannot_be_silently_repaired(self):
        self.rows[1]['source_frame_index'] = self.rows[0]['source_frame_index']
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'frame indices'):
            route119.inspect_prepared(self.root)

    def test_different_training_camera_is_rejected(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['K'][0][2] += 1
        rt.save(self.root / 'data/training-inputs.json', dict(records=changed))
        with self.assertRaisesRegex(ValueError, 'records differ'):
            route119.inspect_prepared(self.root)

    def test_measurement_manifest_cannot_point_to_another_video(self):
        for name in ('inputs.json', 'training-inputs.json'):
            rt.save(self.root / 'data' / name, dict(records=self.rows, video_sha256='a' * 64))
        with self.assertRaisesRegex(ValueError, 'different uploaded video'):
            route119.inspect_prepared(self.root)

    def test_bad_extrinsic_and_interpolated_pose_are_rejected(self):
        original = copy.deepcopy(self.rows[0])
        self.rows[0]['w2c'][0][0] = -1
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'proper rigid'):
            route119.inspect_prepared(self.root)
        self.rows[0] = original
        self.rows[0]['camera_registration']['interpolation_used'] = True
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'pose interpolation'):
            route119.inspect_prepared(self.root)



    def test_fresh_run_refuses_existing_directory(self):
        digest = rt.sha(self.root / 'run.json')
        with self.assertRaises(FileExistsError):
            route119.initialize_run(self.root, self.video, self.cfg, self.base / 'assets')
        self.assertEqual(digest, rt.sha(self.root / 'run.json'))


    def test_nonexistent_run_is_not_created_by_stage_status(self):
        absent = self.base / 'absent'
        with self.assertRaises(FileNotFoundError):
            route119.run_stage('prior', absent, self.base / 'assets')
        self.assertFalse(absent.exists())

    def test_status_reads_failed_preparation_without_model_imports(self):
        rt.save(self.root / 'preparation.json', dict(state='failed_preserved_for_review',
            error='No sufficient fresh seed pair', training_started=False))
        report = route119.recorded_status(self.root)
        self.assertEqual(report['preparation']['state'], 'failed_preserved_for_review')
        self.assertFalse(report['process_liveness_checked'])

    def test_busy_gpu_lock_does_not_create_a_new_run(self):
        target = self.base / 'busy-target'
        with patch('shuolee_portrait.cli.gpu_lock', side_effect=RuntimeError('existing task lock')):
            with self.assertRaisesRegex(RuntimeError, 'existing task lock'):
                route119.reconstruct(self.video, target, self.base / 'assets', self.cfg)
        self.assertFalse(target.exists())

    def test_failed_preparation_after_init_records_the_stage(self):
        target = self.base / 'failed-target'
        with patch('shuolee_portrait.cli.gpu_lock', return_value=nullcontext()):
            with patch('shuolee_portrait.video119.prepare_video', side_effect=RuntimeError('original decoder failure')):
                with self.assertRaisesRegex(RuntimeError, 'original decoder failure'):
                    route119.reconstruct(self.video, target, self.base / 'assets', self.cfg)
        report = route119.recorded_status(target)
        self.assertEqual(report['status']['state'], 'failed')
        self.assertEqual(report['status']['stage'], 'prepare')
        self.assertIn('original decoder failure', report['status']['error'])




class RecipeTests(unittest.TestCase):
    def test_119_hull_and_free_world_recipe_are_frozen(self):
        self.assertEqual(route119.configuration()['iterations'], 18000)
        for change in (dict(gaussian_representation='mesh_bound'), dict(iterations=1000),
                       dict(surface_vote_policy='visible_only'), dict(initial_points=100000),
                       dict(max_fit_p90_body_fraction=.9), dict(formal_training_admitted=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                route119.configuration(change)

    def test_prior_conversion_is_explicit_and_does_not_change_cameras(self):
        rows = [dict(K=[[100., 0., 45.2], [0., 101., 27.8], [0., 0., 1.]])]
        before = copy.deepcopy(rows)
        legacy = camera_intrinsics_for_prior(rows, {})
        converted = camera_intrinsics_for_prior(rows, dict(pixel_center_convention='renderer_corner_to_detector_integer'))
        np.testing.assert_array_equal(converted[:, :2, 2], legacy[:, :2, 2] - .5)
        self.assertEqual(rows, before)

    def test_mask_support_keeps_coordinates_and_rejects_off_image(self):
        mask = np.zeros((32, 24), bool)
        mask[10:20, 10:20] = True
        xy = np.asarray([[11., 11.], [8., 12.], [4., 12.], [-1., 12.]])
        before = xy.copy()
        np.testing.assert_array_equal(image_support(xy, mask), [1, 1, 0, 0])
        np.testing.assert_array_equal(xy, before)


class PortableGaussianTests(unittest.TestCase):
    def test_roundoff_normalization_rejects_scale_and_reflection(self):
        raw = Rotation.from_euler('xyz', [23, -44, 11], degrees=True).as_matrix().astype(np.float32)
        proper, report = rigid_rotation64(raw)
        np.testing.assert_allclose(proper @ proper.T, np.eye(3), atol=1e-14)
        self.assertLess(report['rotation_adjustment_max'], 1e-6)
        for invalid in (np.diag([-1., 1., 1.]), np.diag([1., 1.01, 1.])):
            with self.assertRaises(RuntimeError):
                rigid_rotation64(invalid)

    def test_real_ply_roundtrip_preserves_positions_covariance_and_directional_sh(self):
        from utils.sh_utils import eval_sh
        rng = np.random.default_rng(119)
        count = 9
        fields = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
        fields += [f'f_rest_{i}' for i in range(45)] + ['opacity']
        fields += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        vertices = np.zeros(count, dtype=[(name, '<f4') for name in fields])
        xyz = rng.normal(size=(count, 3))
        rotations = Rotation.random(count, random_state=rng)
        q = rotations.as_quat()[:, [3, 0, 1, 2]]
        scales = rng.uniform(.01, .08, (count, 3))
        features = rng.normal(scale=.1, size=(count, 16, 3))
        rest = features[:, 1:, :].transpose(0, 2, 1).reshape(count, 45)
        for i, key in enumerate(('x', 'y', 'z')):
            vertices[key] = xyz[:, i]
            vertices[f'f_dc_{i}'] = features[:, 0, i]
            vertices[f'scale_{i}'] = np.log(scales[:, i])
        for i in range(4):
            vertices[f'rot_{i}'] = q[:, i]
        for i in range(45):
            vertices[f'f_rest_{i}'] = rest[:, i]
        vertices['opacity'] = -.7
        R = Rotation.from_euler('xyz', [13, 62, -21], degrees=True).as_matrix()
        origin = np.asarray([.1, -.5, 1.])
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'world.ply', Path(directory) / 'portable.ply'
            PlyData([PlyElement.describe(vertices, 'vertex')], text=False, byte_order='<').write(source)
            report = rotate_ply(source, target, R, origin)
            out = PlyData.read(target, mmap=False)['vertex'].data
            self.assertEqual(report['points'], count)
            np.testing.assert_allclose(np.column_stack([out[x] for x in ['x', 'y', 'z']]), (xyz - origin) @ R.T, atol=3e-7)
            new_rotation = Rotation.from_quat(np.column_stack([out[f'rot_{i}'] for i in [1, 2, 3, 0]])).as_matrix()
            old_cov = np.einsum('nij,nj,nkj->nik', rotations.as_matrix(), scales ** 2, rotations.as_matrix())
            new_cov = np.einsum('nij,nj,nkj->nik', new_rotation, scales ** 2, new_rotation)
            np.testing.assert_allclose(new_cov, R @ old_cov @ R.T, atol=3e-9)
            new_dc = np.column_stack([out[f'f_dc_{i}'] for i in range(3)])[:, None, :]
            new_rest = np.column_stack([out[f'f_rest_{i}'] for i in range(45)]).reshape(count, 3, 15).transpose(0, 2, 1)
            new_features = np.concatenate([new_dc, new_rest], axis=1)
            directions = rng.normal(size=(count, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            expected = eval_sh(3, features.transpose(0, 2, 1), directions)
            actual = eval_sh(3, new_features.transpose(0, 2, 1), directions @ R.T)
            np.testing.assert_allclose(actual, expected, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
