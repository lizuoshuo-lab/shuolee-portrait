"""Named-frame I/O and call-routing tests. The solver/model calls are mocked."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from shuolee_portrait import official_sequence as seq, official_smplifyx as api, runtime as rt


class NamedFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); self.rows = []
        self.K = np.array([[10., .2, 3.1], [0., 11., 2.4], [0., 0., 1.]])
        self.contract = dict(translation_unit='sfm_units', meters_per_camera_unit=.25,
                             basis='synthetic_known', evidence='CPU fixture')
        self.basis = (np.array([.1, .2, .3]), np.full((3, 10), .02), np.full((3, 10), .03))
        for i, frame in enumerate(('frame_000007', 'frame_000011')):
            image, mask = self.base/(frame+'.png'), self.base/(frame+'-mask.png')
            Image.new('RGB', (8, 6), (i*50, 30, 60)).save(image)
            Image.new('L', (8, 6), 255).save(mask)
            E = np.eye(4); E[:3, 3] = [1., 2., 3.]
            if i: E[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
            corner = self.K.copy(); corner[:2, 2] += .5
            self.rows.append(dict(frame=frame, image=str(image), mask=str(mask), wh=[8, 6],
                source_sha256=rt.sha(image), mask_sha256=rt.sha(mask), source_frame_index=5+i*3,
                K_integer_centers=self.K.tolist(), K=corner.tolist(), w2c=E.tolist()))
        self.frames = self.base/'inputs.json'; rt.save(self.frames, dict(records=self.rows))
        self.obs = self.base/'observations.npz'
        points = np.arange(2*133*2, dtype=np.float32).reshape(2, 133, 2)
        self.points = points
        np.savez(self.obs, frame_ids=np.array([r['frame'] for r in self.rows][::-1]),
                 keypoints=points[::-1], scores=np.ones((2, 133)), boxes=np.tile([0, 0, 8, 6], (2, 1)),
                 flip_disagreement=np.zeros((2, 133)))
        self.beta = np.full((1, 10), .1, np.float32)
        self.beta_path = self.base/'shape.npz'; np.savez(self.beta_path, betas=self.beta)
        self.policy = dict(mode='supplied', path=str(self.beta_path), sha256=rt.sha(self.beta_path), key='betas')

    def load(self):
        return seq.load_inputs(self.frames, self.obs, camera_length_contract=self.contract,
                               reference_frame=self.rows[1]['frame'])

    def winner(self, beta=None, K=None):
        a = {key: np.zeros((1, 3), np.float32) for key in seq.STACK_FIELDS}
        a.update(betas=self.beta.copy() if beta is None else beta.copy(),
            vertices_camera=np.array([[[1., 0., 4.], [0., 1., 4.], [1., 1., 5.]]], np.float32),
            joints_camera=np.ones((1, 118, 3), np.float32), joints_projected_native=np.ones((1, 118, 2), np.float32),
            expression=np.full((1, 10), .02, np.float32), global_orient=np.array([[.1, 0, .2]], np.float32),
            camera_translation=np.array([[.3, .4, 3.]], np.float32), faces=np.array([[0, 1, 2]]),
            K_integer_centers=self.K.copy() if K is None else K.copy())
        return a

    def report(self):
        return dict(official_commit=api.COMMIT, gvhmr_used=False, five_stages=True,
            upstream_loss_monitor_optimizer_unchanged=True, translation_applied_once=True,
            full_K_integer_centers=True)

    def run_mock(self, root, policy, drift=False):
        calls = []
        def fit(**kwargs):
            calls.append((kwargs['freeze_betas'], kwargs['shared_betas']))
            beta = self.beta if kwargs['shared_betas'] is None else kwargs['shared_betas']
            if drift and kwargs['freeze_betas']: beta = beta + 1
            return self.winner(beta, kwargs['K_integer_centers']), self.report()
        with patch.object(api, 'create_models', return_value=(object(), object(), dict(vposer={}))) as create, \
             patch.object(api, 'adapted_entry_source'), patch.object(api, 'fit_frame', side_effect=fit), \
             patch.object(seq, 'pelvis_basis', return_value=self.basis):
            result = seq.fit_named_sequence(frames_path=self.frames, observations_path=self.obs, root=root,
                official_root=self.base, assets=self.base, camera_length_contract=self.contract,
                reference_frame=self.rows[1]['frame'], shape_policy=policy)
            create.assert_called_once()
        return result, calls

    def test_named_observations_reordered_and_camera_scale_once(self):
        inputs = self.load()
        np.testing.assert_array_equal(inputs['targets'][:, 0, :2], self.points[:, 0])
        winners = {row['frame']: self.winner() for row in self.rows[::-1]}
        arrays, training = seq.assemble(inputs, winners, self.beta, self.basis)
        np.testing.assert_array_equal(arrays['frame_ids'], [row['frame'] for row in self.rows])
        np.testing.assert_array_equal(arrays['w2c'][:, :3, 3], [[.25, .5, .75]]*2)
        for i, E in enumerate(arrays['w2c']):
            np.testing.assert_allclose(arrays['vertices_world'][i] @ E[:3, :3].T + E[:3, 3],
                                       arrays['vertices_camera'][i], atol=1e-12)
            expected = (arrays['root_pivot_model'][i]+arrays['camera_translation'][i]-E[:3, 3]) @ E[:3, :3]
            np.testing.assert_allclose(arrays['world_translation'][i], expected, atol=1e-12)
        self.assertEqual(training['camera_length_contract']['meters_per_camera_unit'], 1.)

    def test_duplicate_observation_names_rejected(self):
        with np.load(self.obs) as z: data = dict(z)
        data['frame_ids'][:] = data['frame_ids'][0]; np.savez(self.obs, **data)
        with self.assertRaisesRegex(ValueError, 'complete unique'): self.load()

    def test_pixel_hash_and_intrinsic_shift_rejected(self):
        self.rows[0]['K'][0][2] += .5; rt.save(self.frames, dict(records=self.rows))
        with self.assertRaisesRegex(ValueError, 'Integer/corner'): self.load()
        self.rows[0]['K'][0][2] = self.K[0, 2] + .5; rt.save(self.frames, dict(records=self.rows))
        Path(self.rows[0]['image']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'image changed'): self.load()

    def test_supplied_shape_one_model_and_exact_roster(self):
        root = self.base/'result'
        report, calls = self.run_mock(root, self.policy)
        self.assertEqual([frozen for frozen, _ in calls], [True, True])
        self.assertEqual(report['frames'], 2)
        self.assertFalse(report['gaussian_training_started'])
        with np.load(root/'fit/fitted.npz') as a:
            np.testing.assert_array_equal(a['shared_betas'], self.beta)
            self.assertEqual(str(a['reference_frame_id']), self.rows[1]['frame'])
        with self.assertRaises(FileExistsError): self.run_mock(root, self.policy)

    def test_reference_fit_is_explicit_then_whole_roster_frozen(self):
        _, calls = self.run_mock(self.base/'ref', dict(mode='fit_reference'))
        self.assertEqual([frozen for frozen, _ in calls], [False, True, True])
        self.assertIsNone(calls[0][1])
        for _, beta in calls[1:]: np.testing.assert_array_equal(beta, self.beta)

    def test_beta_drift_fails_without_completed_receipt(self):
        root = self.base/'drift'
        with self.assertRaisesRegex(ValueError, 'shared frozen shape'):
            self.run_mock(root, self.policy, drift=True)
        self.assertEqual(rt.read(root/'fit/progress.json')['state'], 'failed')
        self.assertFalse((root/'fit/official-fit-result.json').exists())

    def test_shape_hash_and_unknown_policy_rejected(self):
        for policy in (None, dict(mode='automatic'), dict(self.policy, sha256='0'*64)):
            with self.assertRaises(ValueError): seq.read_shape_policy(policy)

    def test_saved_assembly_is_explicitly_historical(self):
        winners = {}
        for row in self.rows:
            p = self.base/(row['frame']+'.npz'); np.savez(p, **self.winner(), frame_id=np.asarray(row['frame']))
            report = self.base/(row['frame']+'.json')
            rt.save(report, dict(self.report(), state='complete', frame_id=row['frame'],
                source_frame_index=row['source_frame_index'], source_image_sha256=row['source_sha256']))
            winners[row['frame']] = dict(npz=str(p), npz_sha256=rt.sha(p), result=str(report), result_sha256=rt.sha(report))
        with patch.object(api, 'fit_frame') as fit, patch.object(api, 'create_models') as create:
            result = seq.assemble_saved_winners(inputs=self.load(), root=self.base/'historical',
                winners=winners, shared_betas=self.beta, basis=self.basis)
        fit.assert_not_called(); create.assert_not_called()
        self.assertFalse(result['new_fitting_started'])
        self.assertEqual(result['execution_kind'], 'historical_readonly_assembly')

    def test_required_assets_do_not_require_prior_weights(self):
        manifest = dict(files=[dict(path='needed', bytes=1, sha256='hash'), dict(path='gvhmr/unused', bytes=1, sha256='bad')])
        (self.base/'needed').write_bytes(b'x')
        with patch.object(rt, 'ASSETS', self.base), patch.object(rt, 'read', return_value=manifest), patch.object(rt, 'sha', return_value='hash'):
            self.assertEqual(rt.verify_assets(['needed'])['files'], 1)
            with self.assertRaises(RuntimeError): rt.verify_assets()

    def test_pipeline_reaches_actual_training_boundary(self):
        from shuolee_portrait import sequence_pipeline as pipeline
        root = self.base/'pipeline'; root.mkdir(); (root/'logs').mkdir()
        rt.save(root/'status.json', dict(state='running', training_started=False))
        cfg = dict(camera_length_contract=dict(self.contract, basis='model_conditioned'),
                   official_smplifyx=dict(source_root=str(self.base), shape_policy=dict(mode='fit_reference')))
        stages = []
        def finish(root, config):
            stages.append('export')
            ply = root/'export/point_cloud.ply'; ply.parent.mkdir(); ply.write_bytes(b'mocked PLY')
            rt.save(root/'result.json', dict(state='completed', format_passed=True,
                reference_frame_id=self.rows[1]['frame'], sha256=rt.sha(ply), quality_accepted=False))
        with patch.object(api, 'adapted_entry_source'), patch('shuolee_portrait.cli.gpu_lock', return_value=nullcontext()), \
             patch('shuolee_portrait.video119.prepare_video') as prepare, patch.object(rt, 'configure'), \
             patch.object(rt, 'verify_assets'), patch.object(seq, 'fit_named_sequence', side_effect=lambda **kw: stages.append('fit')) as fit, \
             patch('shuolee_portrait.training.train_official', side_effect=lambda *args: stages.append('train')) as train, \
             patch('shuolee_portrait.export.export', side_effect=finish):
            result = pipeline.reconstruct(video=self.base/'new.mp4', root=root, assets=self.base,
                                          configuration=cfg, reference_frame=self.rows[1]['frame'])
        self.assertEqual(stages, ['fit', 'train', 'export'])
        self.assertEqual(result['state'], 'completed')
        self.assertFalse(result['quality_accepted'])
        self.assertEqual(train.call_args.args[1]['iterations'], 18000)
        self.assertIsNone(train.call_args.args[1]['training_height'])
        self.assertTrue(rt.read(root/'status.json')['fitting_complete'])
        self.assertEqual(fit.call_args.kwargs['reference_frame'], self.rows[1]['frame'])
        self.assertEqual(prepare.call_args.kwargs['required_assets'], pipeline.OBSERVATION_ASSETS)
        self.assertFalse(any('gvhmr' in p for p in pipeline.OBSERVATION_ASSETS))


if __name__ == '__main__': unittest.main()
