"""Small file/array bridge controls; no model, Torch, renderer or GPU import."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shuolee_portrait import sequence_bundle as bridge
from shuolee_portrait.sequence_input_adapter import COCO133_NAMES


class BridgeControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = ['frame_000001', 'frame_000032', 'frame_000094']
        self.indices = np.array([0, 31, 93])
        self.K = np.array([[40., 1.25, 15.1], [0., 41., 14.4], [0., 0., 1.]])
        self.source_E = np.tile(np.eye(4), (3, 1, 1))
        self.source_E[:, 0, 3] = [0., .2, .4]
        self.source_E[1, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        self.E = self.source_E.copy(); self.E[:, :3, 3] *= 2
        self.rows, crops = [], []
        for i, name in enumerate(self.ids):
            image, mask = self.root/(name+'.png'), self.root/(name+'-mask.png')
            Image.new('RGB', (32, 32), (40+i, 60, 80)).save(image)
            Image.new('L', (32, 32), 255).save(mask)
            self.rows.append(dict(frame=name, source_frame_index=int(self.indices[i]), wh=[32,32],
                image=str(image), source_sha256=bridge.sha(image), mask=str(mask), mask_sha256=bridge.sha(mask),
                K_integer_centers=self.K.tolist(), w2c=self.source_E[i].tolist(), official_prediction_slot=i,
                camera_registration=dict(interpolation_used=False)))
            crops.append(dict(frame=name, source_sha256=bridge.sha(image), source_wh=[32,32],
                source_to_crop=np.eye(3).tolist(), crop_to_source=np.eye(3).tolist(),
                pixel_convention='Integer pixel centres in both source image and crop'))
        self.write('frames.json', dict(records=self.rows)); self.write('crops.json', dict(records=crops))
        np.savez(self.root/'obs.npz', frame_ids=np.array(self.ids), joint_names=np.array(COCO133_NAMES),
            keypoints=np.ones((3,133,2)), scores=np.ones((3,133)),
            boxes=np.tile([0.,0.,31.,31.], (3,1)), flip_disagreement=np.zeros((3,133)))
        np.savez(self.root/'prior.npz', incam_body_pose=np.zeros((3,63)), incam_global_orient=np.zeros((3,3)),
            incam_transl=np.tile([0.,0.,4.],(3,1)), incam_betas=np.zeros((3,10)))
        self.units = dict(translation_unit='sfm_units', meters_per_camera_unit=2., basis='model_conditioned',
            evidence='Declared synthetic fixture gauge; not a measured scale.')
        self.manifest = dict(schema='shuolee-sequence-inputs/1', purpose='synthetic_contract_test',
            pixel_domain='rectified_pinhole', pixel_convention='integer_centers', root_policy='provided_initialization',
            initialization_kind='cached_same_video_gvhmr_incam', initialization_evidence='Known CPU file fixture.',
            reference_frame_id=self.ids[1], camera_length_contract=self.units,
            frames=self.desc('frames.json'), observations=self.desc('obs.npz'), prior=self.desc('prior.npz'),
            crop_mapping=self.desc('crops.json'), reviews=None)
        self.write('inputs.json', self.manifest)
        self.plan = dict(mode='one_full_sequence_research_adaptation', inputs=str(self.root/'inputs.json'),
            source_sha256={str(self.root/'inputs.json'):bridge.sha(self.root/'inputs.json')}, frames=3,
            reference_frame_id=self.ids[1], camera_length_contract=self.units)
        self.write('plan.json', self.plan)
        (self.root/'video.mp4').write_bytes(b'CPU fixture video identity only; not decoded')
        vh = bridge.sha(self.root/'video.mp4')
        self.write('run.json', dict(video=str(self.root/'video.mp4'),video_sha256=vh,
            subject=dict(video_sha256=vh, selected_frames=3,source_wh=[32,32],source_frames=100)))
        world = np.tile(np.array([[0.,0.,4.],[1.,0.,4.],[0.,1.,4.]]), (3,1,1))
        world[:, :, 0] += np.arange(3)[:, None]/4
        camera = np.einsum('nvi,nji->nvj', world, self.E[:,:3,:3]) + self.E[:,None,:3,3]
        self.fit = dict(frame_ids=np.array(self.ids),source_frames=self.indices,
            image_sha256=np.array([r['source_sha256'] for r in self.rows]),
            mask_sha256=np.array([r['mask_sha256'] for r in self.rows]),
            K_integer_centers=np.tile(self.K,(3,1,1)),w2c=self.E,source_w2c=self.source_E,
            camera_length_contract_json=np.asarray(json.dumps(self.units)),reference_frame_id=np.asarray(self.ids[1]),
            betas=np.zeros((1,10)),shared_betas=np.zeros((1,10)),
            vertices_camera=camera,vertices_world=world,faces=np.array([[0,1,2]]))
        np.savez(self.root/'initial.npz',**self.fit)
        # A distinct synthetic final state, without claiming a real optimization.
        self.fit['vertices_world'] = world + [.05, 0., 0.]
        self.fit['vertices_camera'] = np.einsum('nvi,nji->nvj', self.fit['vertices_world'], self.E[:,:3,:3]) + self.E[:,None,:3,3]
        self.finalize()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value),encoding='utf-8')

    def desc(self, name):
        return dict(path=str(self.root/name),sha256=bridge.sha(self.root/name))

    def finalize(self):
        np.savez(self.root/'fit.npz',**self.fit)
        self.receipt = dict(state='budget_ended', all_named_frames_exported=True,fresh_optimizer=True,
            old_optimized_fit_loaded=False,all_K_E_unchanged=True,optimizer=dict(evaluations=5, state='budget_ended', numerical_converged=False, schedule_complete=False, stop='Synthetic budget control'),frames=3,
            reference_frame_id=self.ids[1],camera_length_contract=self.units,
            output_fit_sha256=bridge.sha(self.root/'fit.npz'),plan_sha256=bridge.sha(self.root/'plan.json'))
        self.write('result.json',self.receipt); self.write('progress.json',dict(state='budget_ended',pid=None))

    def assemble(self):
        return bridge.assemble_final_bundle(self.root,self.root/'assembled',source_run=self.root/'run.json',
            sequence_id='fixture-three')

    def test_final_named_reorder_full_K_and_once_normalized_E(self):
        for key in ('frame_ids','source_frames','image_sha256','mask_sha256','K_integer_centers','w2c',
                    'source_w2c','vertices_camera','vertices_world'):
            self.fit[key]=self.fit[key][[2,0,1]]
        self.finalize()
        b,r=self.assemble()
        for i,name in enumerate(self.ids):
            np.testing.assert_array_equal(b.frame(name).w2c,self.E[i])
            np.testing.assert_array_equal(b.frame(name).K_integer_centers,self.K)
        self.assertEqual(b.reference_frame_id,self.ids[1])
        self.assertFalse(r['training_admitted']); self.assertTrue(r['source_video_verified_by_bridge'])
        with np.load(self.root/'assembled'/'poses.npz') as z:
            self.assertNotIn('betas',z.files); self.assertIn('shared_betas',z.files)
        self.assertEqual(r['output_camera_scale_applications'],0)

    def test_initial_inspect_cannot_create_bundle(self):
        report,_,_,_=bridge.inspect_snapshot(self.root/'initial.npz',fit_plan=self.root/'plan.json',
            source_run=self.root/'run.json',sequence_id='fixture-three')
        self.assertEqual(report['snapshot_role'],'initial_not_final')
        self.assertFalse(report['bundle_created']); self.assertFalse((self.root/'assembled').exists())
        self.write('progress.json',dict(state='running',pid=123))
        with self.assertRaisesRegex(ValueError,'terminal'): self.assemble()

    def test_no_fit_evaluations_or_wrong_plan_rejected(self):
        self.receipt['optimizer']['evaluations']=0; self.write('result.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'evaluations'): self.assemble()
        self.receipt['optimizer']['evaluations']=1; self.receipt['plan_sha256']='0'*64
        self.write('result.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'fit plan'): self.assemble()

    def test_failed_optimizer_cannot_hide_behind_nonfailed_outer_receipt(self):
        self.receipt['optimizer']['state'] = 'failed'
        self.write('result.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'terminal states differ'):
            self.assemble()

    def test_source_slot_and_reference_mismatch_rejected(self):
        self.fit['source_frames']=self.indices[::-1]; self.finalize()
        with self.assertRaisesRegex(ValueError,'indices'): self.assemble()
        self.fit['source_frames']=self.indices; self.fit['reference_frame_id']=np.asarray(self.ids[0]); self.finalize()
        with self.assertRaisesRegex(ValueError,'reference'): self.assemble()

    def test_second_scale_application_and_beta_divergence_rejected(self):
        self.fit['w2c']=self.E.copy(); self.fit['w2c'][:,:3,3]*=2; self.finalize()
        with self.assertRaisesRegex(ValueError,'source-unit conversion'): self.assemble()
        self.fit['w2c']=self.E; self.fit['betas']=np.ones((1,10)); self.finalize()
        with self.assertRaisesRegex(ValueError,'beta'): self.assemble()

    def test_image_and_video_mutation_rechecked(self):
        self.assemble()
        (self.root/'video.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Source changed'):
            bridge.load_assembled_bundle(self.root/'assembled'/'manifest.json')

    def test_world_mismatch_and_final_fit_hash_rejected(self):
        self.fit['vertices_world'][0,0,0]+=.01; self.finalize()
        with self.assertRaisesRegex(ValueError,'storage rounding'): self.assemble()
        self.receipt['output_fit_sha256']='0'*64; self.write('result.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'fit.npz'): self.assemble()

    def test_existing_output_not_overwritten(self):
        self.assemble()
        with self.assertRaisesRegex(ValueError,'Output must be new'): self.assemble()

    def test_budget_final_preserves_optimizer_status_and_synthetic_purpose(self):
        bundle, report = self.assemble()
        self.assertEqual(bundle.purpose, 'synthetic_contract_test')
        self.assertFalse(report['numerical_converged'])
        self.assertEqual(report['optimizer'], self.receipt['optimizer'])
        self.assertEqual(report['optimizer_state'], 'budget_ended')

    def test_initial_copied_as_final_is_rejected(self):
        with np.load(self.root/'initial.npz', allow_pickle=False) as archive:
            self.fit = {key: archive[key] for key in archive.files}
        self.finalize()
        with self.assertRaisesRegex(ValueError, 'initial snapshot'):
            self.assemble()
        self.assertFalse((self.root/'assembled').exists())

    def test_unchanged_but_actually_solved_stationary_final_is_allowed(self):
        with np.load(self.root/'initial.npz', allow_pickle=False) as archive:
            self.fit = {key: archive[key] for key in archive.files}
        self.finalize()
        self.receipt['state'] = 'schedule_complete'
        self.receipt['optimizer'].update(state='schedule_complete', schedule_complete=True,
            numerical_converged=True, final_stationary=True, stop=None)
        self.write('result.json', self.receipt)
        self.write('progress.json', dict(state='schedule_complete', pid=None))
        _, report = self.assemble()
        self.assertTrue(report['numerical_converged'])
        self.assertFalse(report['quality_accepted'])

    def use_decoded_acquisition(self):
        self.decoded = dict(video_sha256=bridge.sha(self.root/'video.mp4'), decoded_total=100,
            records=[{key: row[key] for key in ('frame','source_frame_index','wh','source_sha256')}
                     for row in self.rows])
        self.write('decoded.json', self.decoded)
        run = bridge.read(self.root/'run.json')
        run.pop('subject')
        run['decoded_manifest'] = self.desc('decoded.json')
        self.write('run.json', run)

    def test_fresh_decoded_acquisition_is_bound_and_rechecked(self):
        self.use_decoded_acquisition()
        bundle, _ = self.assemble()
        self.assertEqual(bundle.reference_frame_id, self.ids[1])
        self.decoded['records'][1]['source_frame_index'] += 1
        self.write('decoded.json', self.decoded)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            bridge.load_assembled_bundle(self.root/'assembled'/'manifest.json')

    def test_updated_decoded_hash_cannot_rebind_wrong_frame(self):
        self.use_decoded_acquisition()
        self.decoded['records'][1]['source_frame_index'] += 1
        self.write('decoded.json', self.decoded)
        run = bridge.read(self.root/'run.json')
        run['decoded_manifest'] = self.desc('decoded.json')
        self.write('run.json', run)
        with self.assertRaisesRegex(ValueError, 'decoded video source identity'):
            self.assemble()

    def test_updated_output_hash_cannot_substitute_nonfinal_geometry(self):
        self.assemble()
        folder = self.root/'assembled'
        with np.load(folder/'poses.npz', allow_pickle=False) as archive:
            poses = {key: archive[key] for key in archive.files}
        poses['vertices_world'][0,0,0] += .1
        np.savez(folder/'poses.npz', **poses)
        manifest = bridge.read(folder/'manifest.json')
        manifest['poses']['sha256'] = bridge.sha(folder/'poses.npz')
        bridge.write_new(folder/'changed-manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'differs from final fit'):
            bridge.load_assembled_bundle(folder/'changed-manifest.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--report');args=p.parse_args()
    started=time.monotonic()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BridgeControls))
    report=dict(state='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=len(r.failures),
        errors=len(r.errors),seconds=time.monotonic()-started,GPU_tasks=0,body_model_forwards=0,
        source_sha256=bridge.sha(bridge.__file__),tests_sha256=bridge.sha(__file__),
        checks='Synthetic file/identity/unit assembly only; no visual or physical acceptance.',
        failure_details=[str(t)+'\n'+detail for t,detail in r.failures+r.errors])
    if args.report: bridge.write_new(args.report,report)
    print(json.dumps(report,indent=2));sys.exit(not r.wasSuccessful())
