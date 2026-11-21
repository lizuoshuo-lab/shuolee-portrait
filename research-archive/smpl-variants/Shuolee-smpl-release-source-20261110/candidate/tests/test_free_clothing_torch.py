"""New fixed-table consumer controls only; no body model, GPU or old test run."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait.free_clothing_transport import FreeClothingTransport
from shuolee_portrait.free_clothing_torch import FreeClothingTorch

CONTROL_ID = 'torch-clothing-consumer123-v1'
FRAMES = ('sheared', 'orbit', 'reference', 'half_turn')
IDS = ('free-c', 'free-z', 'free-a', 'free-m', 'free-x')
V = np.array([[0., 0., 4.], [1.1, .1, 4.1], [.2, 1., 4.2],
              [1.8, -.5, 3.7], [2., .6, 3.9], [2.2, -.3, 4.9]])
FACES = np.array([[0, 1, 2], [3, 4, 5]])
SUPPORT = np.array([0, 1, 0, 1, 0])
XYZ = np.array([[.1, .25, 4.6], [2.3, .4, 4.2], [1.3, -.7, 4.8],
                [1.5, .1, 3.6], [-.3, .9, 3.7]])
QUATS = np.array([[.9, .1, -.2, .3], [.7, -.3, .4, .2], [.8, .2, .15, -.1],
                  [.6, .4, .1, -.2], [.9, .05, .2, .1]])
SCALES = np.array([[.13, .08, .04], [.11, .07, .03], [.09, .05, .12],
                   [.14, .07, .05], [.08, .04, .06]])
LOGS = np.log(SCALES)
SHIFT = np.array([.4, -.6, .3])
METRICS = {}


def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


Q = rodrigues([.3, -.6, .7], .63)
HALF = np.diag([1., -1., -1.])


def mesh_frames():
    shear = V.copy()
    shear[1] += [.45, .1, .12]
    shear[5] += [.15, -.22, .1]
    return np.stack([shear, 1.4 * V @ Q.T + SHIFT, V, V @ HALF.T + [0., 0., 9.]])


def length_contract(factor=1., unit='meters'):
    return dict(translation_unit=unit, meters_per_camera_unit=factor,
                basis='synthetic_known', evidence='Known synthetic scene; not a physical camera calibration.')


def source(vertices=None, frames=FRAMES):
    vertices = mesh_frames() if vertices is None else vertices
    return FreeClothingTransport(frame_ids=frames, reference_frame_id='reference',
        gaussian_ids=IDS, vertices_world=vertices, faces=FACES,
        motion_support_triangle_ids=SUPPORT, translation_unit='meters',
        frame_source_sha256={f: hashlib.sha256(v.tobytes()).hexdigest() for f, v in zip(frames, vertices)})


def cameras():
    result = {}
    for i, frame in enumerate(FRAMES):
        R = rodrigues([.1, .8, -.2], .04 * i)
        center = np.array([.2 + .15 * i, -1.1, -.8 + .1 * i])
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, -R @ center
        result[frame] = E
    return result


def prepare(obj=None, dtype=torch.float64, camera_map=None, contract=None):
    return FreeClothingTorch.from_cpu(source() if obj is None else obj,
        camera_length_contract=length_contract() if contract is None else contract,
        frame_source_w2c=cameras() if camera_map is None else camera_map, dtype=dtype, device='cpu')


def params(dtype=torch.float64, grad=False, scale=1.):
    return (torch.tensor(XYZ * scale, dtype=dtype, requires_grad=grad),
            torch.tensor(QUATS, dtype=dtype, requires_grad=grad),
            torch.tensor(LOGS + np.log(scale), dtype=dtype, requires_grad=grad))


def qmatrix_independent(q):
    """Independent skew-matrix construction, not the consumer's element formula."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    zeros = torch.zeros_like(w)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), -1).reshape(-1, 3, 3)
    return torch.eye(3, dtype=q.dtype, device=q.device)[None] + 2 * w[:, None, None] * skew + 2 * skew @ skew


def covariance_independent(q, logs):
    R = qmatrix_independent(q)
    L = R * logs.exp()[:, None]
    return L @ L.transpose(-1, -2)


def actual(consumer=None, frame='orbit', tensors=None, covariance=True, ids=IDS):
    consumer = prepare() if consumer is None else consumer
    return consumer(frame, *(params(consumer.dtype) if tensors is None else tensors),
                    gaussian_ids=ids, return_covariance=covariance)


def sh_evaluator():
    spec = importlib.util.spec_from_file_location('consumer_existing_sh', ROOT / 'src/avatar_core/utils/sh_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_sh


class TorchConsumerControls(unittest.TestCase):
    def test_torch_tables_match_cpu_oracle(self):
        obj = source()
        consumer = prepare(obj)
        cov = covariance_independent(*params()[1:]).numpy()
        camera_map = cameras()
        largest = 0.
        for frame in FRAMES:
            E = camera_map[frame]
            oracle = obj.transport_numpy(frame, XYZ, cov, -E[:3, :3].T @ E[:3, 3], gaussian_ids=IDS)
            out = actual(consumer, frame)
            for key in ('centers', 'covariance', 'reference_view_directions', 'rotations', 'support_vertex_residual'):
                delta = np.max(np.abs(out[key].numpy() - oracle[key]))
                largest = max(largest, float(delta))
                np.testing.assert_allclose(out[key].numpy(), oracle[key], atol=4e-14, rtol=0)
        METRICS['CPU_oracle_max_difference'] = largest

    def test_exact_reference(self):
        tensors = params()
        out = actual(frame='reference', tensors=tensors)
        self.assertTrue(torch.equal(out['centers'], tensors[0]))
        self.assertTrue(torch.equal(out['log_scales'], tensors[2]))
        torch.testing.assert_close(out['quats'], tensors[1] / torch.linalg.vector_norm(tensors[1], dim=-1, keepdim=True),
                                   atol=4*torch.finfo(tensors[1].dtype).eps, rtol=0)
        self.assertTrue(torch.equal(out['motion_quats'], torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).expand(5, -1)))
        self.assertGreater(float(torch.linalg.vector_norm(out['centers'][2] - torch.tensor(V[0]))), 1.)

    def test_half_turn(self):
        out = actual(frame='half_turn')
        np.testing.assert_allclose(out['centers'].numpy(), XYZ @ HALF.T + [0., 0., 9.], atol=3e-14, rtol=0)
        expected = torch.tensor(HALF)[None] @ qmatrix_independent(params()[1])
        torch.testing.assert_close(qmatrix_independent(out['quats']), expected, atol=4e-14, rtol=0)

    def test_shear_residual(self):
        obj = source()
        out = actual(prepare(obj), 'sheared')
        self.assertGreater(float(out['support_vertex_residual'].min()), .01)
        self.assertGreater(float((out['support_vertex_residual'][0] - out['support_vertex_residual'][1]).abs()), .001)
        np.testing.assert_allclose(out['support_vertex_residual'].numpy(), obj._residuals[0, SUPPORT], atol=1e-14, rtol=0)
        self.assertFalse(out['training_admitted'])

    def test_two_scale_projection(self):
        values = []
        K = np.array([[810., 4., 337.2], [0., 790., 211.8], [0., 0., 1.]])
        for factor in (1., 2.):
            consumer = prepare(source(mesh_frames() * factor), contract=length_contract(factor, 'sfm_units'))
            out = actual(consumer, tensors=params(scale=factor))
            E = cameras()['orbit'].copy()
            E[:3, 3] *= factor
            points = out['centers'].numpy() @ E[:3, :3].T + E[:3, 3]
            uvw = points @ K.T
            values.append((uvw[:, :2] / uvw[:, 2:], out['reference_view_directions'].numpy(), out['covariance'].numpy() / factor**2))
            metadata = consumer.report()
            self.assertEqual(metadata['camera_translation_scale_applications'], 1)
            self.assertFalse(metadata['physical_camera_accepted'])
            self.assertFalse(metadata['source_camera_metric_relationship_independently_verified'])
        for a, b in zip(*values):
            np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
        METRICS['synthetic_gauge_pixel_max_difference'] = float(np.max(abs(values[0][0] - values[1][0])))

    def test_trainable_gradient_chain(self):
        consumer = prepare()
        xyz, q, logs = params(grad=True)
        coefficients = torch.tensor(np.random.default_rng(237).normal(0, .1, (5, 3, 16)), requires_grad=True)
        eval_sh = sh_evaluator()
        out = actual(consumer, tensors=(xyz, q, logs))
        weights = torch.arange(1, 16, dtype=torch.float64).reshape(5, 3) / 10
        # Frobenius norm alone is rotation-invariant and cannot test q gradients.
        covariance_weight = torch.tensor([[.3, .12, -.08], [.12, .5, .04], [-.08, .04, .9]], dtype=torch.float64)
        loss = (out['centers'] * weights).sum() + 7 * (out['covariance'] * covariance_weight).sum() + (eval_sh(3, coefficients, out['reference_view_directions']) * weights).sum()
        gradients = torch.autograd.grad(loss, (xyz, q, logs, coefficients))
        x2, q2, l2 = params(grad=True)
        sh2 = coefficients.detach().clone().requires_grad_(True)
        direct_x = 1.4 * x2 @ torch.tensor(Q).T + torch.tensor(SHIFT)
        direct_cov = 1.4**2 * torch.tensor(Q)[None] @ covariance_independent(q2, l2) @ torch.tensor(Q).T[None]
        E = cameras()['orbit']
        cr = ((-E[:3, :3].T @ E[:3, 3]) - SHIFT) @ Q / 1.4
        dirs = x2 - torch.tensor(cr)
        dirs = dirs / torch.linalg.vector_norm(dirs, dim=-1, keepdim=True)
        direct_loss = (direct_x * weights).sum() + 7 * (direct_cov * covariance_weight).sum() + (eval_sh(3, sh2, dirs) * weights).sum()
        expected = torch.autograd.grad(direct_loss, (x2, q2, l2, sh2))
        METRICS['gradient_max_difference'] = {}
        for key, g, ref in zip(('XYZ', 'quaternion', 'log_scale', 'SH'), gradients, expected):
            self.assertTrue(bool(torch.isfinite(g).all()))
            self.assertGreater(float(g.abs().max()), 1e-7)
            torch.testing.assert_close(g, ref, atol=2e-12, rtol=0)
            METRICS['gradient_max_difference'][key] = float((g-ref).abs().max())

    def test_fixed_frame_gaussian_identity(self):
        consumer = prepare()
        with self.assertRaises(ValueError): actual(consumer, 'missing')
        with self.assertRaises(ValueError): actual(consumer, ids=IDS[::-1])
        with self.assertRaises(ValueError): actual(consumer, tensors=tuple(t[:4] for t in params()))
        order = [3, 1, 0, 2]
        reordered = prepare(source(mesh_frames()[order], tuple(FRAMES[i] for i in order)))
        torch.testing.assert_close(actual(consumer)['centers'], actual(reordered)['centers'], atol=0, rtol=0)

    def test_frozen_camera_contract(self):
        obj = source()
        with self.assertRaises((ValueError, TypeError)):
            FreeClothingTorch.from_cpu(obj, camera_length_contract=None, frame_source_w2c=cameras())
        bad = length_contract(); bad['meters_per_camera_unit'] = [1., 2.]
        with self.assertRaises(ValueError): prepare(obj, contract=bad)
        with self.assertRaises(ValueError): prepare(obj, contract=length_contract(2., 'meters'))
        bad_cameras = cameras(); del bad_cameras['orbit']
        with self.assertRaises(ValueError): prepare(obj, camera_map=bad_cameras)
        bad_cameras = cameras(); bad_cameras['orbit'][0, 0] = 3.
        with self.assertRaises(ValueError): prepare(obj, camera_map=bad_cameras)
        metadata = prepare(obj).report()
        self.assertEqual(metadata['source_report']['frame_source_sha256'], obj.frame_source_sha256)
        self.assertEqual(metadata['motion_support_triangle_ids'], SUPPORT.tolist())
        self.assertFalse(metadata['source_frame_files_independently_verified'])
        self.assertFalse(metadata['training_admitted'])

    def test_each_named_camera_must_be_one_matrix(self):
        for count in (1, 2):
            camera_map = cameras()
            camera_map['orbit'] = np.repeat(camera_map['orbit'][None], count, axis=0)
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'exactly one 4x4'):
                prepare(camera_map=camera_map)

    def test_source_copy_isolation(self):
        vertices, frame_cameras = mesh_frames(), cameras()
        consumer = prepare(source(vertices), camera_map=frame_cameras)
        first = actual(consumer)
        vertices[:] = -999.
        frame_cameras['orbit'][:] = 0.
        metadata = consumer.report(); metadata['normalized_w2c'][1][0][0] = -999.
        first['rotations'][:] = 0.
        first['motion_quats'][:] = 0.
        first['motion_support_triangle_ids'][:] = 1
        second = actual(consumer)
        torch.testing.assert_close(second['centers'], first['centers'], atol=0, rtol=0)
        self.assertGreater(float(second['rotations'].abs().max()), .5)
        np.testing.assert_array_equal(second['motion_support_triangle_ids'].numpy(), SUPPORT)
        self.assertNotEqual(consumer.report()['normalized_w2c'][1][0][0], -999.)

    def test_hotpath_no_cpu_numpy_or_eigh(self):
        consumer = prepare()
        tensors = params()
        expected = actual(consumer, tensors=tensors, covariance=False)
        def forbidden(*args, **kwargs):
            raise AssertionError('Forbidden CPU copy, parameter detach, per-step preparation or decomposition')
        with ExitStack() as stack:
            for name in ('cpu', 'numpy', 'detach', 'to'):
                stack.enter_context(mock.patch.object(torch.Tensor, name, forbidden))
            for owner, name in ((torch, 'tensor'), (np, 'asarray'), (torch.linalg, 'eigh'),
                                (torch.linalg, 'eigvalsh'), (torch.linalg, 'cholesky_ex'),
                                (FreeClothingTransport, 'assert_unchanged')):
                stack.enter_context(mock.patch.object(owner, name, forbidden))
            output = actual(consumer, tensors=tensors, covariance=False)
        self.assertIsNone(output['covariance'])
        torch.testing.assert_close(output['centers'], expected['centers'], atol=0, rtol=0)

    def test_fixed_table_mutation_guard(self):
        for name in ('_rotations', '_quaternions', '_scales', '_centroids', '_residuals', '_support', '_camera_centers'):
            consumer = prepare()
            getattr(consumer, name).add_(1)
            with self.assertRaisesRegex(ValueError, 'mutated or replaced'):
                actual(consumer, covariance=False)
        consumer = prepare()
        object.__setattr__(consumer, '_camera_centers', consumer._camera_centers.clone())
        with self.assertRaisesRegex(ValueError, 'mutated or replaced'):
            actual(consumer, covariance=False)

    def test_frozen_tables_cannot_be_made_trainable(self):
        for name in ('_rotations', '_quaternions', '_scales', '_centroids', '_residuals', '_camera_centers'):
            consumer = prepare()
            table = getattr(consumer, name)
            version = table._version
            table.requires_grad_(True)
            self.assertEqual(table._version, version)  # Version alone misses this mutation.
            with self.subTest(table=name), self.assertRaisesRegex(ValueError, 'made trainable'):
                actual(consumer, covariance=False)

    def test_invalid_values_fail_closed(self):
        consumer = prepare(dtype=torch.float32)
        for target, value in ((0, float('nan')), (1, 0.), (2, -1000.), (2, 1000.)):
            values = list(params(torch.float32)); values[target][:] = value
            with self.assertRaises((ValueError, RuntimeError)):
                actual(consumer, tensors=tuple(values))
        with self.assertRaises(ValueError): actual(consumer, tensors=params(torch.float64))
        with self.assertRaises(ValueError): prepare(dtype=torch.float16)
        # Positive radius alone is insufficient if its squared covariance underflows.
        values = list(params(torch.float32)); values[2][:] = -60.
        for full_cov in (False, True):
            with self.assertRaisesRegex(RuntimeError, 'covariance.*underflow|positive definite'):
                actual(consumer, tensors=tuple(values), covariance=full_cov)
        coincident = cameras()
        coincident['reference'][:3, :3] = np.eye(3)
        coincident['reference'][:3, 3] = -XYZ[0]
        with self.assertRaisesRegex(RuntimeError, 'coincides'):
            actual(prepare(camera_map=coincident), frame='reference')

    def test_precision_autocast(self):
        consumer = prepare(dtype=torch.float32)
        tensors = params(torch.float32, grad=True)
        expected = actual(consumer, tensors=tensors)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = actual(consumer, tensors=tensors)
        for field in ('centers', 'quats', 'scales', 'log_scales', 'covariance', 'reference_view_directions'):
            self.assertEqual(output[field].dtype, torch.float32)
            torch.testing.assert_close(output[field], expected[field], atol=0, rtol=0)
        loss = output['centers'].square().sum() + output['covariance'].square().sum()
        for grad in torch.autograd.grad(loss, tensors):
            self.assertEqual(grad.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_quaternion_amplitude_invariance_and_parameter_gradients(self):
        norm_error, gradient_error = 0., 0.
        for dtype, amplitudes in ((torch.float32, (1e-23, 1e-22, 1e20, 1e30)),
                                   (torch.float64, (1e-162, 1e-200, 1e200))):
            consumer = prepare(dtype=dtype)
            tolerance = 2e-6 if dtype == torch.float32 else 2e-13
            covariance_weight = torch.tensor([[.3, .12, -.08], [.12, .5, .04], [-.08, .04, .9]], dtype=dtype)
            for amplitude in amplitudes:
                for frame in ('reference', 'orbit'):
                    with self.subTest(dtype=dtype, amplitude=amplitude, frame=frame):
                        xyz, _, logs = params(dtype)
                        base = torch.tensor([[1., 2., 3., 4.]], dtype=dtype).repeat(len(IDS), 1)
                        q = (base * amplitude).requires_grad_(True)
                        out = actual(consumer, frame, tensors=(xyz, q, logs))
                        # The independent oracle operates on moderate values;
                        # its normalization never squares extreme amplitudes.
                        oracle_q = (q.detach() / amplitude).requires_grad_(True)
                        expected_covariance = covariance_independent(oracle_q, logs)
                        if frame == 'orbit':
                            rotation = torch.tensor(Q, dtype=dtype)
                            expected_covariance = 1.4**2 * rotation[None] @ expected_covariance @ rotation.T[None]
                        torch.testing.assert_close(out['covariance'], expected_covariance, atol=tolerance*.02, rtol=tolerance)
                        norm_delta = float((torch.linalg.vector_norm(out['quats'].double(), dim=-1)-1).abs().max())
                        norm_error = max(norm_error, norm_delta)
                        self.assertLessEqual(norm_delta, 8*torch.finfo(dtype).eps)
                        gradient = torch.autograd.grad((out['covariance'] * covariance_weight).sum(), q)[0]
                        expected_gradient = torch.autograd.grad((expected_covariance * covariance_weight).sum(), oracle_q)[0]
                        self.assertTrue(bool(torch.isfinite(gradient).all()))
                        self.assertGreater(float(gradient.abs().max()), 0.)
                        torch.testing.assert_close(gradient * amplitude, expected_gradient, atol=tolerance*.02, rtol=tolerance)
                        gradient_error = max(gradient_error, float((gradient * amplitude-expected_gradient).abs().max()))
        METRICS['extreme_quaternion_unit_norm_max_error'] = norm_error
        METRICS['amplitude_scaled_quaternion_gradient_max_error'] = gradient_error

    def test_quaternion_with_unrepresentable_inverse_amplitude_is_rejected(self):
        for dtype, amplitude in ((torch.float32, 1e-40), (torch.float64, 1e-320)):
            consumer = prepare(dtype=dtype)
            xyz, _, logs = params(dtype)
            q = torch.tensor([[1., 2., 3., 4.]], dtype=dtype).repeat(len(IDS), 1) * amplitude
            self.assertTrue(bool(torch.isfinite(q).all() and (q > 0).all()))
            for covariance in (False, True):
                with self.subTest(dtype=dtype, covariance=covariance), self.assertRaisesRegex(RuntimeError, 'finite parameter gradients'):
                    actual(consumer, tensors=(xyz, q, logs), covariance=covariance)

    def test_preparation_under_inference_mode_keeps_guarded_training_tables(self):
        previous = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            consumer = prepare()
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertEqual(torch.is_inference_mode_enabled(), previous)
        for name, _, version in consumer._buffer_guards:
            value = getattr(consumer, name)
            self.assertFalse(torch.is_inference(value), name)
            self.assertFalse(value.requires_grad, name)
            self.assertEqual(value._version, version)
        tensors = params(grad=True)
        out = actual(consumer, tensors=tensors)
        covariance_weight = torch.tensor([[.3, .12, -.08], [.12, .5, .04], [-.08, .04, .9]], dtype=torch.float64)
        loss = out['centers'].square().sum() + (out['covariance'] * covariance_weight).sum()
        for gradient in torch.autograd.grad(loss, tensors):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().max()), 0.)
        consumer._scales.add_(1.)
        with self.assertRaisesRegex(ValueError, 'mutated or replaced'):
            actual(consumer, covariance=False)

    def test_preparation_ignores_ambient_default_device(self):
        # A non-CPU default is simulated without initializing or using CUDA.
        obj = source()
        with torch.device('meta'):
            consumer = prepare(obj, dtype=torch.float32)
        self.assertEqual(consumer.device, torch.device('cpu'))
        expected = actual(prepare(obj, dtype=torch.float32))
        torch.testing.assert_close(actual(consumer)['centers'], expected['centers'], atol=0, rtol=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit('Refusing to overwrite an existing control report.')
    torch.set_num_threads(2)
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TorchConsumerControls)
    names = [test.id().split('.')[-1] for test in suite]
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    elapsed = time.monotonic() - started
    files = [Path(__file__), ROOT / 'src/shuolee_portrait/free_clothing_torch.py',
             ROOT / 'src/shuolee_portrait/free_clothing_transport.py',
             ROOT / 'src/shuolee_portrait/camera_world_units.py']
    report = dict(control_id=CONTROL_ID, outcome='passed' if result.wasSuccessful() and elapsed <= 120 else 'failed',
                  named_controls=names, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  elapsed_seconds=elapsed, CPU_budget_seconds=120, device='cpu', torch_version=torch.__version__,
                  GPU_executions=0, body_model_forwards=0, training_steps=0, old_test_suites_run=0,
                  CUDA_parity_validated=False, physical_camera_accepted=False, training_admitted=False,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  metrics=METRICS, transcript=transcript.getvalue())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(transcript.getvalue())
    print(json.dumps({k:report[k] for k in ('control_id', 'outcome', 'tests_run', 'failures', 'errors', 'elapsed_seconds')}))
    raise SystemExit(0 if report['outcome'] == 'passed' else 1)
