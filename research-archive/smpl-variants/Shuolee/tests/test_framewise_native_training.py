"""CPU training-connection contracts; the injected fixture is not a renderer test."""
import argparse
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from avatar_core.utils.sh_utils import eval_sh
from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle
from shuolee_portrait.free_clothing_transport import FreeClothingTransport
from shuolee_portrait.free_clothing_torch import FreeClothingTorch
from shuolee_portrait.framewise_native_training import (
    FramewiseNativeTrainer, TrainingConfig, foreground_native_loss, fresh_reference_parameters)

CONTROL_ID = 'framewise-native-training-cpu-v1'
FRAME_IDS = ('moved', 'reference')
GAUSSIAN_IDS = ('free-2', 'free-0', 'free-7')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value), encoding='utf-8')


def make_bundle(folder, *, skew=0., change_occluded_pixels=False):
    """Saved PNG/NPZ synthetic fixture with the reference deliberately second."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    K = np.array([[32., skew, 7.2], [0., 34., 5.1], [0., 0., 1.]])
    vertices = np.array([[0., 0., 3.], [.6, 0., 3.], [0., .6, 3.]])
    poses = np.stack([vertices + [.3, -.2, .1], vertices])
    records, specifications = [], {}
    for i, frame in enumerate(FRAME_IDS):
        image, mask, validity = folder/(frame+'.png'), folder/(frame+'-mask.png'), folder/(frame+'-valid.png')
        rgb = np.full((12, 16, 3), [25, 40, 55], dtype=np.uint8)
        rgb[2:10, 3:13] = [110+i*7, 100, 120]
        alpha = np.zeros((12, 16), np.uint8); alpha[2:10, 3:13] = 255
        valid = np.full((12, 16), 255, np.uint8); valid[4:6, 4:6] = 0
        if change_occluded_pixels:
            rgb[4:6, 4:6] = [255, 0, 255]
            alpha[4:6, 4:6] = 0
        Image.fromarray(rgb).save(image); Image.fromarray(alpha).save(mask); Image.fromarray(valid).save(validity)
        E = np.eye(4); E[:3, 3] = [.1*i, -.05*i, .2]
        records.append(dict(frame=frame, source_frame=10*i, sequence_id='test-sequence', video_sha256='e'*64,
            wh=[16, 12], image=image.name, mask=mask.name, source_sha256=sha(image), mask_sha256=sha(mask),
            K_integer_centers=K.tolist(), w2c=E.tolist()))
        specifications[frame] = dict(target_wh=[16, 12], radial=[0.]*6, tangential=[0.]*2, thin_prism=[0.]*4,
            alpha_valid_path=str(validity), alpha_valid_sha256=sha(validity), validity_note='Known synthetic visible mask; 2x2 region explicitly occluded.')
    write(folder/'frames.json', dict(records=records))
    order = [1, 0]
    np.savez(folder/'poses.npz', frame_ids=np.array([FRAME_IDS[i] for i in order]), source_frames=np.array([records[i]['source_frame'] for i in order]),
        image_sha256=np.array([records[i]['source_sha256'] for i in order]), sequence_id=np.array('test-sequence'), video_sha256=np.array('e'*64),
        vertices_world=poses[order], faces=np.array([[0, 1, 2]], dtype=np.int64), shared_betas=np.zeros((1, 10)),
        K_integer_centers=np.stack([K, K]), w2c=np.stack([records[i]['w2c'] for i in order]))
    manifest = dict(schema='shuolee-frame-pose/1', purpose='synthetic_contract_test', sequence_id='test-sequence', video_sha256='e'*64,
        coordinate_system='opencv_world_meters', pixel_domain='rectified_pinhole', pixel_convention='integer_centers', reference_frame_id='reference',
        frames=dict(path='frames.json', rows_key='records', sha256=sha(folder/'frames.json')), poses=dict(path='poses.npz', sha256=sha(folder/'poses.npz')))
    write(folder/'bundle.json', manifest)
    bundle = load_frame_pose_bundle(folder/'bundle.json')
    source = FreeClothingTransport(frame_ids=FRAME_IDS, reference_frame_id='reference', vertices_world=poses, faces=bundle.faces,
        gaussian_ids=GAUSSIAN_IDS, motion_support_triangle_ids=np.zeros(3, dtype=np.int64), translation_unit='meters',
        frame_source_sha256={row['frame']: row['source_sha256'] for row in records})
    length = dict(translation_unit='meters', meters_per_camera_unit=1., basis='synthetic_known', evidence='Saved known synthetic fixture, no physical claim.')
    consumer = FreeClothingTorch.from_cpu(source, camera_length_contract=length,
        frame_source_w2c={f: bundle.frame(f).w2c for f in FRAME_IDS}, dtype=torch.float64)
    initial = fresh_reference_parameters([[.1, .2, 3.2], [.3, .4, 3.3], [.5, .2, 2.9]], initial_scale=.08,
        initial_opacity=.3, initial_rgb=[.42, .38, .46], sh_degree=1, dtype=torch.float64)
    return bundle, consumer, initial, specifications


class CPUContract:
    def manifest(self):
        return dict(pixels='pixel_corners', gaussian_evaluation='ray-3d', input_domain='synthetic_known_camera', camera_optimization=False)


class CPUDifferentiableFixture:
    """Connect parameters/kwargs without representing rasterization physics."""
    def __init__(self): self.calls = []
    def __call__(self, means, quats, scales, opacity, colors, E, K, width, height, **kwargs):
        self.calls.append(dict(means=means.detach().clone(), quats=quats.detach().clone(), colors=colors.detach().clone(),
            E=E.detach().clone(), K=K.detach().clone(), width=width, height=height, kwargs=kwargs))
        color = colors.mean(0) + .01*means.mean(0) + .02*quats[:, 1:].mean(0) + .03*scales.mean(0)
        rgb = color[None, None, None].expand(1, height, width, 3)
        alpha = opacity.mean().reshape(1, 1, 1, 1).expand(1, height, width, 1)
        return rgb, alpha, dict(kind='CPU_differentiable_connection_fixture_not_native_renderer')


def make_trainer(fixture, *, renderer=None, run_id='bounded-cpu-test', specs=None, config=None, ids=GAUSSIAN_IDS, bundle=None, consumer=None):
    b, c, initial, s = fixture
    return FramewiseNativeTrainer(b if bundle is None else bundle, c if consumer is None else consumer, initial,
        gaussian_ids=ids, frame_specs=s if specs is None else specs, render_native=renderer or CPUDifferentiableFixture(),
        render_contract=CPUContract(), renderer_provenance=dict(name='CPU_contract_fixture', revision='1', native_validated=False),
        initialization_provenance=dict(kind='synthetic_known', old_subject_checkpoint_loaded=False), run_id=run_id,
        config=config or TrainingConfig(max_steps=4, initial_points=3, sh_degree=1))


class NativeTrainingControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.fixture = make_bundle(self.path/'inputs')
    def tearDown(self): self.temporary.cleanup()

    def test_foreground_normalization_and_outside_rgb_independence(self):
        rgb = torch.full((3, 4, 3), .6, dtype=torch.float64, requires_grad=True)
        alpha = torch.full((3, 4), .4, dtype=torch.float64, requires_grad=True)
        target = torch.full_like(rgb, .4); mask = torch.zeros_like(alpha); mask[1, 1:3] = 1.
        valid = mask.bool(); alpha_valid = torch.ones_like(valid)
        first = foreground_native_loss(rgb, alpha, target, mask, valid, alpha_valid, alpha_weight=.1)
        changed = target.clone(); changed[~valid] = 100.
        second = foreground_native_loss(rgb, alpha, changed, mask, valid, alpha_valid, alpha_weight=.1)
        self.assertEqual(float(first['rgb_l1']), .6-.4)
        self.assertTrue(torch.equal(first['loss'], second['loss']))
        gradient = torch.autograd.grad(first['loss'], rgb)[0]
        self.assertTrue(torch.equal(gradient[~valid], torch.zeros_like(gradient[~valid])))
        torch.testing.assert_close(gradient[valid], torch.full_like(gradient[valid], 1/6), atol=0, rtol=0)

    def test_alpha_valid_background_and_occluded_independence(self):
        rgb = torch.full((2, 2, 3), .4, dtype=torch.float64)
        target = rgb.clone(); mask = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        rv = mask.bool(); av = torch.tensor([[True, True], [False, False]])
        alpha = torch.tensor([[.8, .5], [.9, .7]], dtype=torch.float64, requires_grad=True)
        first = foreground_native_loss(rgb, alpha, target, mask, rv, av, alpha_weight=1.)
        self.assertAlmostEqual(float(first['alpha_l1']), .35)
        changed = mask.clone(); changed[~av] = 1.
        second = foreground_native_loss(rgb, alpha, target, changed, rv, av, alpha_weight=1.)
        self.assertTrue(torch.equal(first['loss'], second['loss']))
        grad = torch.autograd.grad(first['loss'], alpha)[0]
        torch.testing.assert_close(grad, torch.tensor([[-.5, .5], [0., 0.]], dtype=torch.float64), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            foreground_native_loss(rgb, alpha, target, mask, torch.zeros_like(rv), av, alpha_weight=1.)

    def test_full_camera_resize_plus_half_pixel_once(self):
        specs = copy.deepcopy(self.fixture[3]); specs['moved']['target_wh'] = [8, 6]
        renderer = CPUDifferentiableFixture(); trainer = make_trainer(self.fixture, renderer=renderer, specs=specs)
        observed = trainer.load_observation('moved')
        self.assertEqual(tuple(observed.rgb.shape), (6, 8, 3))
        trainer.render('moved')
        called = renderer.calls[0]
        # Integer center: (u+.5)*.5-.5; renderer corner adds .5 once.
        expected = torch.tensor([[16., 0., 3.85], [0., 17., 2.8], [0., 0., 1.]], dtype=torch.float64)
        torch.testing.assert_close(called['K'][0], expected, atol=1e-14, rtol=0)
        torch.testing.assert_close(called['E'][0], torch.tensor(self.fixture[0].frame('moved').w2c.copy()), atol=0, rtol=0)
        self.assertEqual((called['width'], called['height']), (8, 6))
        self.assertIsNone(called['kwargs']['sh_degree'])

    def test_named_frame_and_gaussian_binding(self):
        trainer = make_trainer(self.fixture); observation = trainer.load_observation('reference')
        with self.assertRaises(ValueError): trainer.step('moved', observation)
        with self.assertRaises(ValueError): trainer.render('missing')
        with self.assertRaises(ValueError): make_trainer(self.fixture, ids=GAUSSIAN_IDS[::-1])
        other = make_trainer(self.fixture, run_id='another-run')
        with self.assertRaises(ValueError): other.step('reference', observation)
        self.assertEqual(trainer.render_calls, 0)

    def test_actual_mask_read_and_observation_mutation(self):
        trainer = make_trainer(self.fixture); observation = trainer.load_observation('reference')
        self.assertFalse(bool(observation.rgb_valid[4:6, 4:6].any()))
        self.assertTrue(bool(observation.alpha_valid[0, 0]))
        self.assertFalse(bool(observation.rgb_valid[0, 0]))
        observation.rgb.add_(.1)
        with self.assertRaisesRegex(ValueError, 'mutated'):
            trainer.step('reference', observation)
        self.assertEqual(trainer.render_calls, 0)

    def test_resized_valid_pixels_exclude_entire_occluded_filter_footprint(self):
        altered = make_bundle(self.path/'changed-occlusion', change_occluded_pixels=True)
        observed = []
        for fixture in (self.fixture, altered):
            specs = copy.deepcopy(fixture[3])
            for spec in specs.values(): spec['target_wh'] = [8, 6]
            observed.append(make_trainer(fixture, specs=specs).load_observation('reference'))
        first, second = observed
        self.assertTrue(torch.equal(first.alpha_valid, second.alpha_valid))
        self.assertTrue(torch.equal(first.rgb_valid, second.rgb_valid))
        self.assertTrue(torch.equal(first.rgb[first.alpha_valid], second.rgb[second.alpha_valid]))
        self.assertTrue(torch.equal(first.foreground[first.alpha_valid], second.foreground[second.alpha_valid]))
        self.assertGreater(int((~first.alpha_valid).sum()), 1)
        prediction_rgb = torch.full_like(first.rgb, .6)
        prediction_alpha = torch.full_like(first.foreground, .4)
        losses = [foreground_native_loss(prediction_rgb, prediction_alpha, obs.rgb, obs.foreground,
            obs.rgb_valid, obs.alpha_valid, alpha_weight=.1)['loss'] for obs in observed]
        self.assertTrue(torch.equal(*losses))

    def test_reference_direction_SH_and_single_native_call(self):
        renderer = CPUDifferentiableFixture(); trainer = make_trainer(self.fixture, renderer=renderer)
        with torch.no_grad(): trainer.field.sh[:, :, 1:] = .1
        posed = trainer.consumer('moved', trainer.field.xyz, trainer.field.quats, trainer.field.log_scales, gaussian_ids=GAUSSIAN_IDS)
        expected = (eval_sh(1, trainer.field.sh, posed['reference_view_directions'])+.5).clamp_min(0)
        trainer.step('moved', trainer.load_observation('moved'))
        self.assertEqual(len(renderer.calls), 1)
        torch.testing.assert_close(renderer.calls[0]['colors'], expected, atol=0, rtol=0)
        torch.testing.assert_close(renderer.calls[0]['means'], posed['centers'], atol=0, rtol=0)
        self.assertEqual(trainer.step_count, 1)

    def test_empty_adam_fixed_layout_and_budget(self):
        trainer = make_trainer(self.fixture, config=TrainingConfig(max_steps=1, initial_points=3, sh_degree=1))
        self.assertEqual(len(trainer.optimizer.state), 0)
        identities = [id(p) for p in trainer.field.parameters()]
        trainer.step('moved')
        self.assertEqual(len(trainer.optimizer.state), 5)
        self.assertEqual([id(p) for p in trainer.field.parameters()], identities)
        with self.assertRaisesRegex(ValueError, 'budget'): trainer.step('reference')
        trainer.optimizer.param_groups[0]['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'options changed'): trainer.render('reference')

    def test_camera_mismatch_and_skew_rejected(self):
        b, c, initial, specs = self.fixture
        changed = copy.deepcopy(c.report()['camera_length_contract'])
        raw = {f: b.frame(f).w2c.copy() for f in FRAME_IDS}; raw['moved'][0, 3] += .1
        src = FreeClothingTransport(frame_ids=FRAME_IDS, reference_frame_id='reference',
            vertices_world=np.stack([b.frame(f).vertices_world for f in FRAME_IDS]), faces=b.faces,
            gaussian_ids=GAUSSIAN_IDS, motion_support_triangle_ids=np.zeros(3, dtype=np.int64), translation_unit='meters',
            frame_source_sha256={f: 'f'*64 for f in FRAME_IDS})
        wrong = FreeClothingTorch.from_cpu(src, camera_length_contract=changed, frame_source_w2c=raw, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'camera differ'): make_trainer(self.fixture, consumer=wrong)
        with self.assertRaisesRegex(ValueError, 'zero skew'): make_trainer(make_bundle(self.path/'skew', skew=.2))

    def test_real_count_and_missing_validity_cannot_silently_fallback(self):
        b, _, _, specs = self.fixture
        with self.assertRaisesRegex(ValueError, '200000'):
            make_trainer(self.fixture, bundle=replace(b, purpose='real'))
        incomplete = copy.deepcopy(specs); del incomplete['reference']['alpha_valid_path']
        with self.assertRaisesRegex(ValueError, 'Frame spec'): make_trainer(self.fixture, specs=incomplete)
        with self.assertRaisesRegex(ValueError, 'budget|bounded'): make_trainer(self.fixture, config=TrainingConfig(max_steps=18001, initial_points=3))

    def test_nonzero_residual_cannot_change_pinhole_bundle_projection(self):
        for name in ('radial', 'tangential', 'thin_prism'):
            for value in (.01, 1e-300):
                specs = copy.deepcopy(self.fixture[3]); specs['moved'][name][0] = value
                with self.assertRaisesRegex(ValueError, 'zero residual'):
                    make_trainer(self.fixture, specs=specs)

    def test_reference_and_checkpoint_reject_invalid_final_representation(self):
        for parameter, value in (('quats', 1e308), ('quats', 0.), ('log_scales', -1000.), ('log_scales', 1000.)):
            trainer = make_trainer(self.fixture)
            with torch.no_grad(): getattr(trainer.field, parameter).fill_(value)
            with self.assertRaisesRegex(ValueError, 'norm|representation'):
                trainer.reference_state()
            with self.assertRaisesRegex(ValueError, 'norm|representation'):
                trainer.save_checkpoint(self.path/(parameter+str(value)+'.pt'))

    def test_checkpoint_roundtrip_optimizer_and_named_reference(self):
        trainer = make_trainer(self.fixture)
        trainer.step('moved'); trainer.step('reference')
        checkpoint = self.path/'state.pt'; trainer.save_checkpoint(checkpoint)
        restored = make_trainer(self.fixture)
        restored.load_checkpoint(checkpoint, resume_same_run=True)
        self.assertEqual(restored.step_count, 2)
        for name, value in trainer.field.state_dict().items():
            torch.testing.assert_close(value, restored.field.state_dict()[name], atol=0, rtol=0)
        # Exact next Adam step is a stronger check than matching parameter bytes.
        trainer.step('moved'); restored.step('moved')
        for name, value in trainer.field.state_dict().items():
            torch.testing.assert_close(value, restored.field.state_dict()[name], atol=0, rtol=0)
        reference = restored.reference_state()
        self.assertEqual(reference['frame_id'], 'reference')
        torch.testing.assert_close(reference['parameters']['xyz'], restored.field.xyz, atol=0, rtol=0)
        self.assertFalse(reference['posed_clouds_averaged'])
        self.assertFalse(reference['training_quality_accepted'])
        restored.save_reference_state(self.path/'reference.pt')
        with self.assertRaises(ValueError): restored.save_reference_state(self.path/'reference.pt')

    def test_checkpoint_metadata_and_optimizer_corruption_fail_before_mutation(self):
        trainer = make_trainer(self.fixture); trainer.step('reference')
        path = self.path/'checkpoint.pt'; trainer.save_checkpoint(path)
        saved = torch.load(path, weights_only=True)
        cases = []
        no_meta = copy.deepcopy(saved); del no_meta['binding_json']; cases.append(no_meta)
        wrong_count = copy.deepcopy(saved); wrong_count['step'] = 2; cases.append(wrong_count)
        wrong_state = copy.deepcopy(saved); wrong_state['optimizer']['state'][0]['step'] += 1; cases.append(wrong_state)
        wrong_layout = copy.deepcopy(saved); wrong_layout['optimizer']['param_groups'][0]['params'] = [1]; cases.append(wrong_layout)
        for i, corrupted in enumerate(cases):
            target = make_trainer(self.fixture); before = target.field.xyz.detach().clone()
            bad = self.path/('bad-%d.pt' % i); torch.save(corrupted, bad)
            with self.assertRaises(ValueError): target.load_checkpoint(bad, resume_same_run=True)
            torch.testing.assert_close(target.field.xyz, before, atol=0, rtol=0)
            self.assertEqual(len(target.optimizer.state), 0)
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            make_trainer(self.fixture, run_id='other-run').load_checkpoint(path, resume_same_run=True)

    def test_checkpoint_unknown_metadata_and_changed_validity_file(self):
        trainer = make_trainer(self.fixture); path = self.path/'zero.pt'; trainer.save_checkpoint(path)
        with self.assertRaises(ValueError): make_trainer(self.fixture).load_checkpoint(path)
        changed = Path(self.fixture[3]['reference']['alpha_valid_path'])
        target = make_trainer(self.fixture)
        with changed.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            target.load_checkpoint(path, resume_same_run=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--report', type=Path, required=True); args = parser.parse_args()
    if args.report.exists(): raise SystemExit('Refusing to overwrite existing CPU-control evidence.')
    torch.set_num_threads(2); start = time.monotonic(); transcript = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeTrainingControls)
    names = [t.id().split('.')[-1] for t in suite]
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    elapsed = time.monotonic()-start
    files = [Path(__file__), ROOT/'src/shuolee_portrait/framewise_native_training.py']
    report = dict(control_id=CONTROL_ID, outcome='passed' if result.wasSuccessful() and elapsed < 120 else 'failed',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), elapsed_seconds=elapsed,
        named_controls=names, CPU_budget_seconds=120, GPU_executions=0, native_raster_calls=0, body_model_forwards=0,
        real_training_steps=0, old_test_suites_run=0, CPU_fixture_optimizer_steps_only=True,
        native_training_validated=False, quality_accepted=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}, transcript=transcript.getvalue())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(transcript.getvalue()); print(json.dumps({k:report[k] for k in ('outcome', 'tests_run', 'failures', 'errors', 'elapsed_seconds')}))
    raise SystemExit(0 if report['outcome'] == 'passed' else 1)
