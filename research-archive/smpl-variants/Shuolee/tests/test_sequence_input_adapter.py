"""CPU file/array controls only. No Torch or body model execution required."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait.alignment_inputs import COCO133_NAMES
from shuolee_portrait.sequence_input_adapter import (
    load_sequence_inputs, relative_detector_confidence, SOURCE_CHANNELS, TARGET_CHANNELS,
)


class SequenceInputsControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = ['actual-20', 'actual-4', 'actual-90']
        self.rows, crop = [], []
        K = np.array([[30., 1.25, 15.1], [0., 29., 14.4], [0., 0., 1.]])
        for i, name in enumerate(self.ids):
            image, mask = self.root/(name+'.png'), self.root/(name+'-mask.png')
            Image.new('RGB', (32, 32), (50+i, 60, 70)).save(image)
            Image.new('L', (32, 32), 255).save(mask)
            E = np.eye(4); E[0, 3] = i*.2
            self.rows.append(dict(frame=name, source_frame_index=[20, 4, 90][i], wh=[32, 32],
                image=str(image), source_sha256=self.sha(image), mask=str(mask), mask_sha256=self.sha(mask),
                K_integer_centers=K.tolist(), w2c=E.tolist(), official_prediction_slot=[2, 0, 1][i],
                camera_registration=dict(interpolation_used=False)))
        for i in [1, 2, 0]:
            r = self.rows[i]
            crop.append(dict(frame=r['frame'], source_sha256=r['source_sha256'], source_wh=r['wh'],
                source_to_crop=np.eye(3).tolist(), crop_to_source=np.eye(3).tolist(),
                pixel_convention='Integer pixel centres in both source image and crop'))
        self.xy = np.arange(3*133*2, dtype=float).reshape(3, 133, 2)/100 + 1
        self.obs = dict(frame_ids=np.array(self.ids), joint_names=np.array(COCO133_NAMES),
            keypoints=self.xy, scores=np.ones((3, 133))*2.1,
            boxes=np.tile([0., 0., 30., 32.], (3, 1)), flip_disagreement=np.ones((3, 133))*.1)
        self.pred = dict(incam_body_pose=np.arange(189).reshape(3, 63)/1000,
                         incam_global_orient=np.arange(9).reshape(3, 3)/100,
                         incam_transl=np.array([[1., 0., 5.], [2., 0., 5.], [3., 0., 5.]]),
                         incam_betas=np.ones((3, 10))*90)
        self.save('frames.json', dict(records=self.rows)); self.save('crop.json', dict(records=crop))
        np.savez(self.root/'obs.npz', **self.obs); np.savez(self.root/'prior.npz', **self.pred)
        self.m = dict(schema='shuolee-sequence-inputs/1', purpose='synthetic_contract_test',
            pixel_domain='rectified_pinhole', pixel_convention='integer_centers',
            root_policy='provided_initialization', initialization_kind='cached_same_video_gvhmr_incam',
            initialization_evidence='Known file fixture, not real inference.', reference_frame_id='actual-4',
            camera_length_contract=dict(translation_unit='sfm_units', meters_per_camera_unit=.4,
                                        basis='synthetic_known', evidence='Known test factor.'),
            frames=self.desc('frames.json'), observations=self.desc('obs.npz'), prior=self.desc('prior.npz'),
            crop_mapping=self.desc('crop.json'), reviews=None)
        self.manifest = self.root/'manifest.json'

    def tearDown(self):
        self.temp.cleanup()

    def sha(self, p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def desc(self, name):
        return dict(path=name, sha256=self.sha(self.root/name))

    def save(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding='utf-8')

    def load(self, **kw):
        self.save('manifest.json', self.m)
        return load_sequence_inputs(self.manifest, **kw)

    def add_reviews(self, records):
        self.save('reviews.json', dict(records=records))
        self.m['reviews'] = self.desc('reviews.json')

    def record(self, i, j, **kw):
        r = dict(frame=self.ids[i], ibug68_index=j, uv_native=self.xy[i, j+23].tolist(),
                 visibility='observed', split='FIT', family_id='named-family-'+str(j),
                 semantic_identity_verified=True, physical_identity_verified=False,
                 visibility_source='synthetic_known', review_evidence='Known synthetic fixture')
        r.update(kw)
        return r

    def fresh_receipt(self, **changes):
        self.m['initialization_kind'] = 'fresh_same_video_gvhmr_incam'
        self.m['video_sha256'] = 'b' * 64
        prediction_rows = sorted(self.rows, key=lambda row: row['official_prediction_slot'])
        receipt = dict(prediction_variant='original_incam', postprocessing_applied=False,
            fresh_inference=True, from_this_video=True, previous_person_parameters_loaded=False,
            video_sha256=self.m['video_sha256'], prediction_sha256=self.m['prior']['sha256'],
            inputs_sha256=self.m['frames']['sha256'],
            frame_ids=[row['frame'] for row in prediction_rows],
            source_frames=[row['source_frame_index'] for row in prediction_rows],
            image_sha256=[row['source_sha256'] for row in prediction_rows])
        receipt.update(changes)
        self.save('original-receipt.json', receipt)
        self.m['prior_provenance'] = self.desc('original-receipt.json')
        return receipt

    def test_fresh_original_receipt_keeps_explicit_prediction_slots(self):
        self.fresh_receipt()
        bundle = self.load()
        self.assertTrue(bundle.report()['prior_is_fresh_inference'])
        self.assertFalse(bundle.report()['cached_prior_is_fresh_inference'])
        np.testing.assert_array_equal(bundle.transl, self.pred['incam_transl'][[2, 0, 1]])
        self.assertFalse(bundle.initial_shared_betas.any())
        self.save('original-receipt.json', dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            bundle.assert_unchanged()

    def test_fresh_prior_cannot_be_postprocessed_or_old_person_state(self):
        for changes in (dict(postprocessing_applied=True), dict(prediction_variant='postprocessed_incam'),
                        dict(previous_person_parameters_loaded=True), dict(fresh_inference=False)):
            with self.subTest(changes=changes):
                self.fresh_receipt(**changes)
                with self.assertRaisesRegex(ValueError, 'original incam'):
                    self.load()

    def test_postprocessed_filename_is_not_relabelled_as_original(self):
        (self.root / 'prediction-postprocessed.npz').write_bytes((self.root / 'prior.npz').read_bytes())
        self.m['prior'] = self.desc('prediction-postprocessed.npz')
        self.fresh_receipt()
        with self.assertRaisesRegex(ValueError, 'postprocessed prediction file'):
            self.load()

    def test_fresh_prior_receipt_requires_this_video_and_producer_inputs(self):
        self.fresh_receipt(video_sha256='c' * 64)
        with self.assertRaisesRegex(ValueError, 'video SHA256'):
            self.load()
        self.fresh_receipt(inputs_sha256='d' * 64)
        with self.assertRaisesRegex(ValueError, 'input roster SHA256'):
            self.load()
        self.fresh_receipt(prediction_sha256='e' * 64)
        with self.assertRaisesRegex(ValueError, 'prediction SHA256'):
            self.load()

    def test_fresh_receipt_wrong_slot_identity_and_absence_are_rejected(self):
        self.fresh_receipt(frame_ids=self.ids)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.load()
        self.fresh_receipt()
        self.m.pop('prior_provenance')
        with self.assertRaisesRegex(ValueError, 'fresh prior provenance'):
            self.load()

    def test_full_frame_identity_prior_slot_and_reference(self):
        b = self.load(decode_images=True)
        self.assertEqual(b.frame_ids, tuple(self.ids))
        self.assertEqual(b.reference_frame_id, 'actual-4')
        np.testing.assert_array_equal(b.transl, self.pred['incam_transl'][[2, 0, 1]])
        np.testing.assert_array_equal(b.body_pose, self.pred['incam_body_pose'][[2, 0, 1]])
        self.assertFalse(b.initial_shared_betas.any())
        self.assertEqual(b.report()['complete_decode_count'], 6)
        self.assertFalse(b.report()['cached_prior_is_fresh_inference'])

    def test_native_semantics_no_coordinate_or_camera_transform(self):
        b = self.load()
        np.testing.assert_array_equal(b.targets[:, TARGET_CHANNELS], self.xy[:, SOURCE_CHANNELS])
        np.testing.assert_array_equal(b.K_integer_centers[0], self.rows[0]['K_integer_centers'])
        np.testing.assert_array_equal(b.source_w2c, [r['w2c'] for r in self.rows])
        self.assertEqual(b.report()['camera_translation_scale_applications'], 0)
        self.assertEqual(float(b.source_w2c[2, 0, 3]), .4)
        self.assertFalse(b.report()['prior_crop_transform_reapplied'])

    def test_unreviewed_unknown_inactive_and_partial_reviews_preserved(self):
        b = self.load()
        self.assertEqual(b.report()['face_records_missing_inactive'], 153)
        self.assertEqual(b.report()['active_face_fit'], 0)
        self.add_reviews([self.record(0, 30), self.record(1, 40, split='HELD')])
        b = self.load()
        self.assertEqual(b.report()['active_face_fit'], 1)
        self.assertEqual(b.report()['active_face_held'], 1)
        self.assertEqual(b.report()['face_records_missing_inactive'], 151)
        self.assertFalse(b.physical_face_verified.any())
        self.assertFalse(b.report()['fitting_admitted'])

    def test_wrong_review_version_and_family_leak_rejected(self):
        self.add_reviews([self.record(0, 30, uv_native=[0., 0.])])
        with self.assertRaisesRegex(ValueError, 'observation version'):
            self.load()
        self.add_reviews([self.record(0, 30), self.record(1, 30, split='HELD')])
        with self.assertRaisesRegex(ValueError, 'holdout leakage'):
            self.load()

    def test_score_is_not_visibility(self):
        self.add_reviews([self.record(0, 30, visibility_source='detector_score')])
        with self.assertRaisesRegex(ValueError, 'confidence'):
            self.load()
        c, q = relative_detector_confidence(self.obs['scores'], self.obs['boxes'], self.obs['flip_disagreement'])
        old_weight = (np.clip(self.obs['scores']/q, 0, 1)**2 *
            np.exp(-.5*(self.obs['flip_disagreement']/(.012*32))**2))
        self.assertLess(np.max(abs(c.astype(float)**2-old_weight)), 1e-6)

    def test_review_anatomical_and_source_frame_identity(self):
        self.add_reviews([self.record(0, 30, anchor_id='ibug68:31')])
        with self.assertRaisesRegex(ValueError, 'anchor ID'): self.load()
        self.add_reviews([self.record(0, 30, source_frame_index=21)])
        with self.assertRaisesRegex(ValueError, 'source-video frame'): self.load()

    def test_unknown_or_synthetic_units_fail_for_real_before_loading_files(self):
        self.m.pop('camera_length_contract')
        with self.assertRaises(ValueError): self.load()
        self.m['camera_length_contract'] = dict(translation_unit='meters', meters_per_camera_unit=1.,
            basis='synthetic_known', evidence='Fixture')
        self.m['purpose'] = 'real_research_adaptation'
        with self.assertRaisesRegex(ValueError, 'Synthetic scale'): self.load()

    def test_old_optimized_state_cannot_replace_raw_prior(self):
        np.savez(self.root/'prior.npz', **dict(self.pred, latent=np.zeros((3,32))))
        self.m['prior'] = self.desc('prior.npz')
        with self.assertRaisesRegex(ValueError, 'original incam'): self.load()

    def test_prior_cannot_be_joined_by_current_roster_position(self):
        self.rows[0]['official_prediction_slot'] = 0
        self.save('frames.json', dict(records=self.rows)); self.m['frames'] = self.desc('frames.json')
        with self.assertRaisesRegex(ValueError, 'entire prediction roster'): self.load()
        self.rows[0]['official_prediction_slot'] = 2
        self.rows[0]['source_sha256'] = self.rows[1]['source_sha256']
        self.save('frames.json', dict(records=self.rows)); self.m['frames'] = self.desc('frames.json')
        with self.assertRaisesRegex(ValueError, 'slot/crop/image'): self.load()

    def test_observation_order_is_explicit_and_camera_interpolation_refused(self):
        self.obs['frame_ids'] = self.obs['frame_ids'][::-1]
        np.savez(self.root/'obs.npz', **self.obs); self.m['observations'] = self.desc('obs.npz')
        with self.assertRaisesRegex(ValueError, 'frame identities'): self.load()
        self.rows[1]['camera_registration']['interpolation_used'] = True
        self.save('frames.json', dict(records=self.rows)); self.m['frames'] = self.desc('frames.json')
        with self.assertRaisesRegex(ValueError, 'registered'): self.load()

    def test_bound_arrays_and_files_immutable(self):
        b = self.load()
        for x in (b.targets, b.K_integer_centers, b.source_w2c, b.body_pose, b.initial_shared_betas):
            with self.assertRaises(ValueError): x.flat[0] = 9
        with self.assertRaises(TypeError): b.camera_length_contract['basis'] = 'measured'
        Image.new('RGB', (32,32), (1,2,3)).save(self.rows[0]['image'])
        with self.assertRaisesRegex(ValueError, 'source changed'): b.assert_unchanged()


if __name__ == '__main__':
    unittest.main()
