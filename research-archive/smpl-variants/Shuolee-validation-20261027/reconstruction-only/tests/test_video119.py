"""CPU evidence for fresh video preparation; synthetic controls are not quality acceptance."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.video119 import (
    VideoPreparationNotReady, estimate_focal_intrinsics, fixed_intrinsics,
    prepare_video, settings_for, solve_camera_geometry, split_track_records,
    validate_video_stream,
)


def synthetic_tracks(n=6, count=330):
    rng = np.random.default_rng(6060)
    X = rng.uniform([-1., -.8, 4.], [1., .8, 8.], (count, 3))
    K = np.array([[600., 0, 319.5], [0, 600., 239.5], [0, 0, 1.]])
    xy, poses = [], []
    for i in range(n):
        R = Rotation.from_rotvec([.007 * i, .025 * i, -.005 * i]).as_matrix()
        center = np.array([.18 * i, .008 * i * i, -.02 * i])
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = -R @ center
        q = X @ R.T + E[:3, 3]
        pixels = q @ K.T
        xy.append(pixels[:, :2] / pixels[:, 2, None])
        poses.append(E)
    records = split_track_records([[(i, j) for i in range(n)] for j in range(count)])
    return xy, records, K, np.array(poses)


class Video119Tests(unittest.TestCase):
    def test_resize_contract_preserves_pixel_center_and_off_axis_principal_point(self):
        profile = dict(mode='fixed_pinhole', image_wh=[1920, 1080],
            provenance='Known exported-video K, reviewed crop mapping', pixel_convention='integer_centers',
            K=[[1200, 0, 923.25], [0, 1250, 508.75], [0, 0, 1]])
        K, A = fixed_intrinsics(profile, [1920, 1080], [960, 540])
        expected = np.array([[600, 0, 461.375], [0, 625, 254.125], [0, 0, 1]])
        np.testing.assert_allclose(K, expected)
        corner_profile = {**profile, 'pixel_convention': 'pixel_corners',
            'K': [[1200, 0, 923.75], [0, 1250, 509.25], [0, 0, 1]]}
        np.testing.assert_array_equal(fixed_intrinsics(corner_profile, [1920, 1080], [960, 540])[0], K)
        np.testing.assert_allclose(A @ [100, 200, 1], [49.75, 99.75, 1])

    def test_fixed_K_cannot_invent_crop_or_accept_distortion(self):
        profile = dict(mode='fixed_pinhole', image_wh=[640, 480], K=np.eye(3).tolist(),
            provenance='test fixture', pixel_convention='integer_centers')
        with self.assertRaisesRegex(VideoPreparationNotReady, 'match this video'):
            fixed_intrinsics(profile, [1280, 720], [640, 360])
        with self.assertRaisesRegex(VideoPreparationNotReady, 'nonzero distortion'):
            fixed_intrinsics({**profile, 'distortion': [.01, 0]}, [640, 480], [640, 480])
        with self.assertRaisesRegex(VideoPreparationNotReady, 'provenance'):
            fixed_intrinsics({**profile, 'provenance': ''}, [640, 480], [640, 480])

    def test_rotation_and_HDR_require_explicit_mapping(self):
        base = dict(width=640, height=480)
        for stream in [{**base, 'tags': {'rotate': '90'}},
            {**base, 'tags': {'rotate': '90'}, 'side_data_list': [{'rotation': -90}]},
            {**base, 'color_transfer': 'smpte2084'}]:
            with self.assertRaises(VideoPreparationNotReady):
                validate_video_stream(stream)
        validate_video_stream(base)

    def test_frozen_split_stable_and_ambiguous_tracks_rejected(self):
        groups = [[(i, j) for i in range(6)] for j in range(330)]
        records = split_track_records(groups)
        self.assertEqual(records, split_track_records(list(reversed(groups))))
        self.assertGreaterEqual(sum(row['held'] for row in records), 30)
        self.assertEqual(split_track_records([[(0, 1), (0, 2), (1, 1)]]), [])
        self.assertEqual(len({row['id'] for row in records}), len(records))

    def test_synthetic_fresh_focal_is_bounded_hypothesis_and_excludes_holdout(self):
        xy, records, K, _ = synthetic_tracks()
        estimated, evidence = estimate_focal_intrinsics(xy, records, [640, 480])
        self.assertLess(abs(estimated[0, 0] - K[0, 0]), 1.)
        self.assertFalse(evidence['physical_calibration_accepted'])
        self.assertFalse(evidence['held_tracks_used'])
        poisoned = [row.copy() for row in records]
        for row in poisoned:
            if row['held']:
                row['nodes'] = [(i, 0) for i in range(len(xy))]
        again, _ = estimate_focal_intrinsics(xy, poisoned, [640, 480])
        np.testing.assert_allclose(estimated, again, atol=1e-8)

    def test_real_essential_pnp_BA_register_all_synthetic_views_without_held_leakage(self):
        xy, records, K, expected = synthetic_tracks()
        settings = settings_for({'video119': {'max_ba_nfev': 10}})
        with tempfile.TemporaryDirectory() as directory:
            Es, report = solve_camera_geometry(xy, records, K, directory, settings)
            self.assertEqual(Es.shape, (6, 4, 4))
            self.assertLess(report['fitting_errors']['max'], .01)
            self.assertFalse(report['physical_calibration_accepted'])
            self.assertFalse(report['formal_training_admitted'])
            self.assertFalse(report['training_started'])
            np.testing.assert_allclose(np.linalg.det(Es[:, :3, :3]), 1, atol=1e-8)
            with np.load(Path(directory) / 'solver-output.npz') as solved:
                held = {row['id'] for row in records if row['held']}
                self.assertFalse(set(solved['track_ids'].tolist()) & held)
                self.assertEqual(set(solved['obs_cam'].tolist()), set(range(6)))
                np.testing.assert_array_equal(solved['K'], K)
            evaluation = json.loads((Path(directory) / 'held-evaluation.json').read_text())
            self.assertTrue(evaluation['whole_tracks_reserved'])
            self.assertFalse(evaluation['physical_identity_reviewed'])
            self.assertLess(evaluation['all_errors']['max'], .01)

    def test_camera_geometry_refuses_inadequate_holdouts_instead_of_reusing_old_camera(self):
        xy, records, K, _ = synthetic_tracks()
        records = [row for row in records if not row['held']]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'holdouts'):
                solve_camera_geometry(xy, records, K, directory)
            self.assertFalse((Path(directory) / 'solver-output.npz').exists())

    def test_registered_budgets_and_quality_cannot_be_preapproved(self):
        for cfg in [{'quality_accepted': True}, {'video119': {'max_ba_nfev': 61}},
            {'video119': {'max_frames': 301}}, {'frame_stride': 0}]:
            with self.assertRaises(ValueError):
                settings_for(cfg)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg not installed')
    def test_actual_new_video_decodes_but_textureless_capture_fails_preserving_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'fresh-video.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 12, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(np.full((240, 320, 3), 127, dtype=np.uint8))
            writer.release()
            run = root / 'run'
            (run / 'data').mkdir(parents=True)
            (run / 'logs').mkdir()
            cfg = {'frame_stride': 1, 'max_image_size': 320, 'min_frames': 12, 'max_frames': 300}
            with self.assertRaisesRegex(VideoPreparationNotReady, 'No fresh texture'):
                prepare_video(video, run, cfg, None, camera_only=True)
            decoded = json.loads((run / 'decoded.json').read_text())
            self.assertEqual(len(decoded['records']), 12)
            self.assertEqual([row['source_frame_index'] for row in decoded['records']], list(range(12)))
            failure = json.loads((run / 'preparation.json').read_text())
            self.assertEqual(failure['state'], 'failed')
            self.assertTrue(failure['from_this_video'])
            self.assertFalse(failure['old_subject_state_loaded'])
            self.assertFalse(failure['training_started'])
            self.assertFalse((run / 'data/observations.npz').exists())
            with self.assertRaises(FileExistsError):
                prepare_video(video, run, cfg, None, camera_only=True)


if __name__ == '__main__':
    unittest.main()
