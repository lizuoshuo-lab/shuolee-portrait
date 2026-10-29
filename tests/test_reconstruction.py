"""Entry failure and real-array producer integration controls; no GPU model."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait import reconstruction as entry, runtime as rt, sequence_pipeline as pipeline


class ReconstructionEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.video = self.base / 'person.mp4'
        self.video.write_bytes(b'explicit synthetic input fixture')
        self.profile = self.base / 'deployment.json'
        rt.save(self.profile, dict(schema='smplx-deployment-v1', environment=sys.prefix,
                                  assets=str(self.base / 'assets'), runs=str(self.base / 'runs')))
        self.camera = self.base / 'camera.json'
        self.cfg = dict(schema='smplx-camera-v1', intrinsics=dict(mode='fixed_pinhole',
            input_domain='rectified_pinhole', image_wh=[32, 24],
            K=[[20., 0., 15.5], [0., 20., 11.5], [0., 0., 1.]],
            pixel_convention='integer_centers', distortion=[0., 0., 0., 0.],
            fixed_processing_across_frames=True, provenance='Synthetic pinhole camera fixture.'))
        rt.save(self.camera, self.cfg)

    def launch(self, root):
        with patch.object(entry, 'inspect_video', return_value=dict(source_wh=[32, 24],
                first_frame_decoded=True, full_preprocessing_complete=False, orientation='landscape')):
            return entry.train_video(self.video, self.profile, self.camera, output=root)

    def test_unknown_fisheye_domain_preserves_first_error_and_no_static_fallback(self):
        self.cfg['intrinsics']['input_domain'] = 'unresolved_fisheye_export'
        rt.save(self.camera, self.cfg)
        target = self.base / 'failed'
        with patch('shuolee_portrait.route119.reconstruct') as legacy:
            with self.assertRaisesRegex(entry.ReconstructionError, 'pixel domain is unresolved'):
                self.launch(target)
            legacy.assert_not_called()
        status = rt.read(target / 'status.json')
        self.assertEqual(status['stage'], 'camera_input')
        self.assertEqual(status['error_code'], 'CAMERA_PIXEL_DOMAIN_UNRESOLVED')
        self.assertFalse(status['training_started'])
        self.assertTrue((target / 'logs/error.log').is_file())

    def test_official_connection_does_not_fall_back_to_experimental_solver(self):
        target = self.base / 'scale'
        with patch('shuolee_portrait.video119.prepare_video') as prepare, \
             patch('shuolee_portrait.prior.infer') as experimental:
            with self.assertRaisesRegex(entry.ReconstructionError, 'official SMPLify-X'):
                self.launch(target)
            prepare.assert_not_called()
            experimental.assert_not_called()
        self.assertEqual(rt.read(target / 'status.json')['stage'], 'official_inputs')

    def test_existing_run_is_preserved(self):
        target = self.base / 'existing'; target.mkdir()
        (target / 'status.json').write_bytes(b'preserved')
        with self.assertRaises(FileExistsError):
            self.launch(target)
        self.assertEqual((target / 'status.json').read_bytes(), b'preserved')

    def test_no_historical_intrinsics_fallback(self):
        rt.save(self.camera, dict(schema='smplx-camera-v1'))
        with self.assertRaisesRegex(entry.ReconstructionError, 'pixel domain'):
            entry.camera_configuration(self.camera, [32, 24])

    def test_fixed_k_cannot_be_reused_for_rotated_dimensions(self):
        with self.assertRaisesRegex(RuntimeError, 'image_wh'):
            entry.camera_configuration(self.camera, [24, 32])

    def test_unknown_processing_or_nonzero_distortion_is_not_pinhole(self):
        for key, value in [('distortion', [0., .1]), ('fixed_processing_across_frames', False)]:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg); cfg['intrinsics'][key] = value
                rt.save(self.camera, cfg)
                with self.assertRaises(entry.ReconstructionError):
                    entry.camera_configuration(self.camera, [32, 24])

    def test_sampling_and_camera_units_reach_pipeline_unchanged(self):
        self.cfg['sampling'] = dict(frame_stride=3, max_image_size=800)
        length = dict(translation_unit='sfm_units', meters_per_camera_unit=.4,
                      basis='model_conditioned', evidence='Explicit synthetic test declaration.')
        self.cfg['camera_length_contract'] = length
        options = dict(source_root='/pinned/source', shape_policy=dict(mode='fit_reference'))
        self.cfg['official_smplifyx'] = options
        rt.save(self.camera, self.cfg)
        _, actual = entry.camera_configuration(self.camera, [32, 24])
        self.assertEqual(actual['camera_length_contract'], length)
        self.assertEqual(actual['official_smplifyx'], options)
        self.assertEqual(actual['frame_stride'], 3)
        self.assertEqual(actual['min_frames'], 12)



if __name__ == '__main__':
    unittest.main()
