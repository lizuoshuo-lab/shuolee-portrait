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
    prepare_video, rectified_center_crop_intrinsics, settings_for, solve_camera_geometry, split_track_records,
    validate_video_stream, retained_camera_indices, select_balanced_tracks, _fresh_tracks,
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
    def test_balanced_selection_prevents_long_tracks_starving_other_views(self):
        verified = {k: [(i, k) for i in range(5)] for k in range(5000)}
        verified.update({k: [(i, k) for i in (4, 5, 6)] for k in range(5000, 5060)})
        selected, counts, available = select_balanced_tracks(verified, list(range(7)), 5000)
        self.assertEqual(len(selected), 5000)
        self.assertEqual(len(set(selected)), 5000)
        self.assertGreaterEqual(min(counts.values()), 50)
        self.assertEqual(available[6], 60)
        old = sorted(verified, key=lambda k: (-len(verified[k]), k))[:5000]
        self.assertFalse(any(i == 6 for k in old for i, _ in verified[k]))

    def test_frame_selection_limits_gaps_and_keeps_original_indices(self):
        wanted = [i for i in range(59) if i not in (22, 23)]
        self.assertEqual(retained_camera_indices(59, wanted), wanted)
        for total, removed in ((20, [4, 5, 6]), (100, [40, 41, 42]), (12, [5])):
            with self.assertRaisesRegex(VideoPreparationNotReady, 'camera coverage'):
                retained_camera_indices(total, [i for i in range(total) if i not in removed])

    def test_real_geometry_recovers_an_isolated_missing_frame_without_interpolation(self):
        xy, records, K, _ = synthetic_tracks(n=20, count=400)
        bad = 9
        xy[bad] = np.empty((0, 2))
        records = [dict(r, nodes=[(i, j) for i, j in r['nodes'] if i != bad]) for r in records]
        with tempfile.TemporaryDirectory() as directory:
            Es, report = solve_camera_geometry(xy, records, K, directory,
                settings_for({'video119': {'max_ba_nfev': 10}}))
            self.assertEqual(Es.shape, (19, 4, 4))
            self.assertEqual(report['retained_input_indices'], [i for i in range(20) if i != bad])
            self.assertEqual(report['dropped_frames'], [dict(index=bad, reason='camera_unregistered')])
            self.assertLess(report['fitting_errors']['max'], .02)
            evidence = json.loads((Path(directory) / 'frame-selection.json').read_text())
            self.assertFalse(evidence['interpolation_used'])
            with np.load(Path(directory) / 'solver-output.npz') as solved:
                self.assertEqual(set(solved['obs_cam']), set(range(19)))

    def test_foreground_texture_cannot_supply_camera_features_through_exclusion_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw').mkdir(); (root / 'camera').mkdir(); (root / 'masks').mkdir()
            image = np.full((300, 400), 127, np.uint8)
            image[70:230, 100:300] = np.random.default_rng(123).integers(0, 255, (160, 200), dtype=np.uint8)
            mask = np.full_like(image, 255); mask[40:260, 70:330] = 0
            rows = [dict(frame=f'f{i}', name=f'f{i}.png') for i in range(3)]
            for row in rows:
                cv2.imwrite(str(root / 'raw' / row['name']), image)
                cv2.imwrite(str(root / 'masks' / row['name']), mask)
            with self.assertRaisesRegex(VideoPreparationNotReady, 'No fresh texture'):
                _fresh_tracks(root, rows, root / 'camera', lambda *a, **k: None, root / 'masks')

    def test_center_crop_maps_landscape_and_portrait_without_swapping_camera_axes(self):
        profile = dict(mode='rectified_center_crop', input_domain='rectified_pinhole',
            reference_wh=[4000, 3000], K=[[1600., 0, 2100.], [0, 1500., 1400.], [0, 0, 1]],
            pixel_convention='integer_centers', provenance='Synthetic rectified reference with off-center principal point')
        cases = [([2000, 1000], [1000, 500], 'landscape',
                  [[400, 0, 524.625], [0, 375, 224.625], [0, 0, 1]]),
                 ([1000, 2000], [500, 1000], 'portrait',
                  [[1600/3, 0, 283.], [0, 500, 1399/3], [0, 0, 1]])]
        for source, decoded, orientation, expected in cases:
            with self.subTest(orientation=orientation):
                K, contract = rectified_center_crop_intrinsics(profile, source, decoded)
                np.testing.assert_allclose(K, expected, atol=1e-10)
                mapped_center = np.asarray(contract['pixel_map']) @ [1999.5, 1499.5, 1.]
                np.testing.assert_allclose(mapped_center, [(decoded[0]-1)/2, (decoded[1]-1)/2, 1.], atol=1e-10)
                self.assertEqual(contract['orientation'], orientation)
                self.assertFalse(contract['exact_device_integer_ROI_verified'])
                self.assertFalse(contract['physical_calibration_accepted'])
                corners = {**profile, 'pixel_convention': 'pixel_corners',
                           'K': [[1600, 0, 2100.5], [0, 1500, 1400.5], [0, 0, 1]]}
                np.testing.assert_allclose(rectified_center_crop_intrinsics(corners, source, decoded)[0], K)

    def test_center_crop_rejects_undeclared_rectification_and_invalid_geometry(self):
        profile = dict(mode='rectified_center_crop', input_domain='rectified_pinhole',
            reference_wh=[4000, 3000], K=[[1600, 0, 2000], [0, 1600, 1500], [0, 0, 1]],
            pixel_convention='integer_centers', provenance='Synthetic rectified camera')
        for changed in ({'input_domain': 'raw_fisheye'}, {'reference_wh': [4000, 0]},
                        {'reference_wh': [4000., 3000]}, {'distortion': [.1, 0]}):
            with self.subTest(changed=changed), self.assertRaises(VideoPreparationNotReady):
                rectified_center_crop_intrinsics({**profile, **changed}, [1520, 2704], [899, 1600])
        for source, decoded in [([0, 2704], [899, 1600]), ([1520, 2704], [899, float('nan')])]:
            with self.assertRaises(VideoPreparationNotReady):
                rectified_center_crop_intrinsics(profile, source, decoded)

    def test_deployed_profile_preserves_original_factory_crop_mapping_for_both_orientations(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/video119.json').read_text())
        profile = settings_for(cfg)['intrinsics']
        self.assertEqual(profile['mode'], 'rectified_center_crop')
        for source, decoded in [([2704, 1520], [1600, 899]), ([1520, 2704], [899, 1600])]:
            # Independent original workflow expression: crop reference first,
            # then resize. It must equal resize-reference then center-crop video.
            fw, fh = profile['reference_wh']
            aspect = source[0] / source[1]
            cw, ch = (fw, fw/aspect) if aspect >= fw/fh else (fh*aspect, fh)
            x0, y0 = (fw-cw)/2, (fh-ch)/2
            w, h = decoded
            A = np.array([[w/cw, 0, w/cw*(.5-x0)-.5], [0, h/ch, h/ch*(.5-y0)-.5], [0, 0, 1.]])
            expected = A @ np.asarray(profile['K'])
            K, _ = rectified_center_crop_intrinsics(profile, source, decoded)
            np.testing.assert_allclose(K, expected, atol=1e-10)

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
