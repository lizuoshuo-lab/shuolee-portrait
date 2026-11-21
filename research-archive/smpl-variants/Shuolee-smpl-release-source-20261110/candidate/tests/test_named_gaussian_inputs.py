"""Named pose/source identity and static export controls, entirely on CPU."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from shuolee_portrait import runtime as rt, training
from shuolee_portrait.export import reference_pose_index
from shuolee_portrait.official_smplifyx import COMMIT


class NamedGaussianInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = ['frame_000003', 'frame_000001', 'frame_000002']
        rows = []
        for i, name in enumerate(self.ids):
            image, mask = self.root/(name+'.png'), self.root/(name+'-mask.png')
            Image.new('RGB', (8, 10), (i*30, 40, 50)).save(image)
            Image.new('L', (8, 10), 255).save(mask)
            K = np.array([[100.+i, 0., 3.2], [0., 97.+i, 4.7], [0., 0., 1.]])
            E = np.eye(4); E[:3, 3] = [i*.2, .1, .7]
            rows.append(dict(frame=name, source_frame_index=i*3, wh=[8, 10],
                K=K.tolist(), w2c=E.tolist(), image=str(image), mask=str(mask),
                source_sha256=rt.sha(image), mask_sha256=rt.sha(mask)))
        K = np.asarray([r['K'] for r in rows]); K[:, :2, 2] -= .5
        vertices = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]*3)
        vertices[1, :, 0] += .2; vertices[2, :, 1] -= .3
        self.fit = dict(vertices_world=vertices, faces=np.array([[0, 1, 2]]),
            frame_ids=np.asarray(self.ids), reference_frame_id=np.asarray(self.ids[1]),
            shared_betas=np.zeros((1, 10)), betas=np.zeros((1, 10)),
            source_frames=np.asarray([r['source_frame_index'] for r in rows]),
            image_sha256=np.asarray([r['source_sha256'] for r in rows]),
            mask_sha256=np.asarray([r['mask_sha256'] for r in rows]),
            K_integer_centers=K, w2c=np.asarray([r['w2c'] for r in rows]),
            root_rotation=np.tile(np.eye(3), (3, 1, 1)), world_translation=np.zeros((3, 3)))
        self.rows = [rows[2], rows[0], rows[1]]
        (self.root/'fit').mkdir()
        np.savez(self.root/'fit/fitted.npz', **self.fit)
        rt.save(self.root/'data/training-inputs.json', dict(records=self.rows))
        results = []
        for row in rows:
            path = self.root/'fit'/(row['frame']+'.json')
            rt.save(path, dict(state='complete', frame_id=row['frame'],
                source_frame_index=row['source_frame_index'], source_image_sha256=row['source_sha256'],
                official_commit=COMMIT, five_stages=True, gvhmr_used=False,
                upstream_loss_monitor_optimizer_unchanged=True,
                translation_applied_once=True, full_K_integer_centers=True))
            results.append(dict(frame_id=row['frame'], path=str(path), sha256=rt.sha(path)))
        source = self.root/'synthetic-source.txt'; source.write_text('explicit synthetic fixture')
        self.receipt = dict(state='complete', official_frame_results=results,
            sources=[dict(path=str(source), sha256=rt.sha(source))],
            fit_sha256=rt.sha(self.root/'fit/fitted.npz'),
            training_inputs_sha256=rt.sha(self.root/'data/training-inputs.json'))
        self.save_receipt()

    def save_receipt(self):
        rt.save(self.root/'fit/official-fit-result.json', self.receipt)

    def test_named_rows_select_their_mesh_not_positional_slot(self):
        layout = training.named_pose_layout(self.fit, self.rows)
        self.assertEqual(layout['pose_indices'], [2, 0, 1])
        self.assertEqual(layout['reference_pose_index'], 1)
        self.assertEqual(layout['reference_frame_id'], 'frame_000001')

    def test_duplicate_missing_or_changed_image_identity_rejected(self):
        cases = []
        rows = copy.deepcopy(self.rows); rows[0]['frame'] = rows[1]['frame']; cases.append(rows)
        cases.append(self.rows[:-1])
        rows = copy.deepcopy(self.rows); rows[0]['mask_sha256'] = 'wrong'; cases.append(rows)
        rows = copy.deepcopy(self.rows); rows[0]['source_frame_index'] += 1; cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                training.named_pose_layout(self.fit, rows)

    def test_camera_and_half_pixel_mismatch_rejected(self):
        for field in ('K', 'w2c'):
            rows = copy.deepcopy(self.rows); rows[0][field][0][-1] += .5
            with self.subTest(field=field), self.assertRaises(ValueError):
                training.named_pose_layout(self.fit, rows)

    def test_identity_drift_and_unnamed_nonstatic_mesh_rejected(self):
        fit = copy.deepcopy(self.fit); fit['betas'][0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'beta'): training.named_pose_layout(fit, self.rows)
        fit = copy.deepcopy(self.fit); del fit['frame_ids']
        with self.assertRaisesRegex(ValueError, 'Nonstatic'): training.named_pose_layout(fit, self.rows)

    def test_actual_complete_receipts_without_quality_approval(self):
        checked = training.verify_official_fit(self.root)
        self.assertEqual(checked['fit_sha256'], self.receipt['fit_sha256'])
        self.assertFalse(checked['quality_accepted'])
        self.assertFalse(checked['formal_training_admitted'])

    def test_stale_fit_or_source_bytes_rejected_before_loop(self):
        source = Path(self.receipt['sources'][0]['path']); source.write_text('changed')
        with patch.object(training, '_train_static') as loop:
            with self.assertRaisesRegex(ValueError, 'Changed or missing'):
                training.verify_official_fit(self.root)
            loop.assert_not_called()
        self.receipt['fit_sha256'] = 'wrong'; self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'completed-fit'):
            training.verify_official_fit(self.root)

    def test_actual_image_change_rejected_even_when_record_is_unchanged(self):
        path = Path(self.rows[0]['image']); Image.new('RGB', (8, 10), 'red').save(path)
        with self.assertRaisesRegex(ValueError, 'image changed'):
            training.verify_official_fit(self.root)

    def test_wrong_solver_and_incomplete_frame_rejected(self):
        descriptor = self.receipt['official_frame_results'][0]
        original = rt.read(descriptor['path'])
        for key, value in [('official_commit', 'custom-solver'), ('five_stages', False),
                           ('gvhmr_used', True), ('translation_applied_once', False)]:
            rt.save(descriptor['path'], dict(original, **{key: value}))
            descriptor['sha256'] = rt.sha(descriptor['path']); self.save_receipt()
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.verify_official_fit(self.root)

    def test_reference_export_must_match_actual_baked_ply_pose(self):
        receipt = dict(reference_frame_id='frame_000001', reference_pose_index=1)
        self.assertEqual(reference_pose_index(self.fit, receipt), (1, 'frame_000001'))
        for bad in [dict(receipt, reference_pose_index=0), dict(receipt, reference_frame_id='frame_000003')]:
            with self.assertRaisesRegex(ValueError, 'reference pose differ'):
                reference_pose_index(self.fit, bad)

    def test_each_render_selects_mesh_and_ply_parity_stays_at_reference(self):
        class Model:
            binding = object()
            timestep = None
            def select_mesh_by_timestep(self, i): self.timestep = i
        model = Model(); static = SimpleNamespace(binding=None, timestep=1)
        render = lambda cam, model, pipe, bg, **kw: dict(render=np.asarray([model.timestep], float))
        for slot in [2, 0, 1, 2]:
            self.assertEqual(training.render_pose(render, None, model, None, None, slot)['render'][0], slot)
        reference, delta = training.reference_ply_comparison(render, None, model, static, None, None, 1)
        np.testing.assert_array_equal(reference, [1.]); np.testing.assert_array_equal(delta, [0.])

    def test_unsupported_skew_fails_without_dropping_k(self):
        row = copy.deepcopy(self.rows[0]); row['K'][0][1] = 2.
        with self.assertRaisesRegex(ValueError, 'skew'): training.camera(row, 0, pose_index=2)

    def test_official_route_dispatches_existing_loop_without_alignment_review(self):
        with patch.object(training, '_train_static') as loop, \
             patch('shuolee_portrait.admission.require_alignment') as legacy_gate:
            training.train_official(self.root, dict(iterations=18000))
        legacy_gate.assert_not_called()
        self.assertEqual(loop.call_args.kwargs, dict(execution_kind='official_smplifyx'))


if __name__ == '__main__': unittest.main()
