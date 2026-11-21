"""Exact streamed-objective CPU controls. No human model or GPU inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait import sequence_fitting as fitting
from shuolee_portrait import sequence_optimizer as solver


class ToyVPoser(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('matrix', torch.sin(torch.arange(32 * 63, dtype=torch.float64)).reshape(32, 63) * .02)

    def decode(self, latent):
        return {'pose_body': torch.tanh(latent @ self.matrix)}


class ToyBody(nn.Module):
    num_pca_comps = 12

    def __init__(self):
        super().__init__()
        self.register_buffer('base', torch.stack([
            torch.linspace(-.4, .4, 118), torch.sin(torch.arange(118)) * .3,
            torch.cos(torch.arange(118)) * .1], -1).to(torch.float64))
        self.calls = []

    def forward(self, **state):
        n = state['betas'].shape[0]
        self.calls.append(n)
        # Small nonlinear, frame-separable model with dependence on every
        # learned block. It is deliberately not a SMPL-X implementation.
        b = state['betas']
        pose = state['body_pose']
        shift = torch.stack([b.sum(1), b.square().sum(1), torch.sin(b).sum(1)], -1) * .025
        articulation = torch.stack([pose[:, ::3].sum(1), pose[:, 1::3].sum(1), pose[:, 2::3].sum(1)], -1) * .03
        expression = state['expression'].sum(1)[:, None] * self.base[:, :].square().sum(0)[None, :] * .0001
        joints = (self.base[None] + shift[:, None] + articulation[:, None] + expression[:, None]
                  + .05 * torch.sin(state['global_orient'])[:, None]
                  + .02 * state['jaw_pose'][:, None] + state['transl'][:, None])
        full_pose = torch.cat([state['global_orient'], pose, state['jaw_pose']], -1)
        hand = state['left_hand_pose'].repeat(1, 4)[:, :45]
        return SimpleNamespace(joints=joints, vertices=joints[:, :17], full_pose=full_pose,
                               left_hand_pose=hand, right_hand_pose=hand)


def make_case(n=2, factor=1.):
    K = np.repeat(np.array([[800., 7., 353.4], [0., 790., 241.8], [0., 0., 1.]])[None], n, 0)
    E = np.repeat(np.eye(4)[None], n, 0)
    E[:, :3, 3] = np.arange(n * 3).reshape(n, 3) * .17
    initial = dict(global_orient=np.arange(n * 3).reshape(n, 3) * .002,
                   transl=np.tile([.02, -.01, 3.2], (n, 1)),
                   latent=np.sin(np.arange(n * 32)).reshape(n, 32) * .002,
                   jaw_pose=np.ones((n, 3)) * .00002,
                   expression=np.ones((n, 10)) * .002)
    contract = dict(translation_unit='sfm_units', meters_per_camera_unit=factor,
                    basis='model_conditioned', evidence='Explicit CPU common arbitrary gauge; not measured.')
    model = fitting.SharedIdentitySequence(ToyBody(), ToyVPoser(), tuple(f'frame:{i}' for i in range(n)),
                                          K, E, initial, camera_length_contract=contract)
    with torch.no_grad():
        model.state['betas'].copy_(torch.linspace(-.01, .01, 10).reshape(1, 10))
        targets = model.project(model.forward_state()).detach() + .5
    shape = (n, 51)
    visibility = np.full(shape, 'unknown', dtype='<U10')
    split = np.full(shape, 'UNASSIGNED', dtype='<U10')
    visibility[:, :3] = 'observed'
    split[:, :2] = 'FIT'
    split[:, 2] = 'HELD'
    family = np.array([[f'unique:{i}:{j}' for j in range(51)] for i in range(n)])
    obs = fitting.SequenceObservations(model.frame_ids, targets,
                torch.linspace(.1, 1., n * 118).reshape(n, 118), visibility, split,
                np.ones(shape, bool), np.zeros(shape, bool), family)
    return model, obs, [1520 + i for i in range(n)]


class NoStepLBFGS:
    records = []
    def __init__(self, params, **kwargs):
        self.parameter = params[0]
        self.kwargs = kwargs

    def zero_grad(self):
        self.parameter.grad = None

    def step(self, closure):
        loss = closure()
        self.records.append((self.kwargs, float(loss.detach()), self.parameter.grad.detach().clone()))
        return loss


class StreamedControls(unittest.TestCase):
    def assert_close(self, actual, expected, rtol=1e-11, atol=1e-9):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def dense(self, model, vector, obs, stage, heights, jacobian=False):
        probe = vector.detach().clone().requires_grad_(True)
        residual, _ = fitting.sequence_residuals(model, probe, obs, stage, heights)
        value = residual.square().sum()
        gradient = torch.autograd.grad(value, probe)[0]
        if not jacobian:
            return value.detach(), gradient
        J = torch.autograd.functional.jacobian(
            lambda x: fitting.sequence_residuals(model, x, obs, stage, heights)[0], probe, vectorize=True)
        return value.detach(), gradient, J.square().sum(0), J.T @ residual.detach()

    def test_all_five_stages_loss_gradient_columns_and_jtr_equal_dense(self):
        model, obs, heights = make_case()
        for stage in range(5):
            with self.subTest(stage=stage):
                expected = self.dense(model, model.flatten(), obs, stage, heights, True)
                model.body.calls.clear()
                actual = fitting.sequence_objective_streamed(model, model.flatten(), obs, stage, heights,
                                                              with_scaling=True)
                for key, value in zip(('loss', 'gradient', 'column_square', 'jtr'), expected):
                    self.assert_close(actual[key], value)
                self.assertEqual(model.body.calls, [1, 1])
                self.assertEqual(actual['peak_jacobian_rows'], 8)
                self.assertEqual(actual['peak_local_parameters'], 61)
                self.assertEqual(actual['frames_completed'], 2)
                self.assert_close(torch.rsqrt((2 * actual['column_square']).clamp_min(1)),
                                  torch.rsqrt((2 * expected[2]).clamp_min(1)), atol=1e-13)

    def test_selected_projection_and_world_use_original_frame_indices(self):
        model, obs, heights = make_case(3)
        out = model.forward_state()
        subset = model.forward_state(frame_indices=[2, 0])
        self.assert_close(model.project(subset, frame_indices=[2, 0]), model.project(out)[[2, 0]])
        self.assert_close(model.world_vertices(subset, frame_indices=[2, 0]), model.world_vertices(out)[[2, 0]])
        for invalid in [[], [1, 1], [-1], [3], [1.5], [True]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                model.forward_state(frame_indices=invalid)

    def test_real94_shape_keeps_one_frame_graph_and_one_identity_prior(self):
        model, obs, heights = make_case(94)
        expected = self.dense(model, model.flatten(), obs, 4, heights)
        model.body.calls.clear()
        actual = fitting.sequence_objective_streamed(model, model.flatten(), obs, 4, heights)
        self.assertEqual(model.flatten().numel(), 4804)
        self.assertEqual(model.body.calls, [1] * 94)
        self.assertEqual(actual['peak_local_parameters'], 61)
        self.assert_close(actual['loss'], expected[0])
        self.assert_close(actual['gradient'], expected[1])

    def test_shared_shape_prior_is_added_once_and_perframe_normalization_unchanged(self):
        model, obs, heights = make_case(3)
        vector = model.flatten()
        terms = [fitting.sequence_residuals(model, vector, obs, 4, heights,
                            frame_indices=[i], include_shared_shape_prior=False)[0] for i in range(3)]
        residual, _ = fitting.sequence_residuals(model, vector, obs, 4, heights)
        prior = (model.state['betas'] * 5).square().sum()
        self.assert_close(sum(r.square().sum() for r in terms) + prior, residual.square().sum())

    def test_frozen_review_and_ignored_observations_stay_out_of_streamed_loss(self):
        model, obs, heights = make_case()
        actual = fitting.sequence_objective_streamed(model, model.flatten(), obs, 4, heights, with_scaling=True)
        changed = copy.deepcopy(obs)
        inactive = changed.fitting_confidence() == 0
        changed.targets[inactive] += 10000.
        altered = fitting.sequence_objective_streamed(model, model.flatten(), changed, 4, heights, with_scaling=True)
        for key in ('loss', 'gradient', 'column_square', 'jtr'):
            self.assertTrue(torch.equal(actual[key], altered[key]), key)

    def test_declared_scene_scale_affects_world_export_not_camera_fit_objective_or_gradient(self):
        first, obs, heights = make_case(3, factor=1.)
        second, _, _ = make_case(3, factor=2.)
        self.assertTrue(torch.equal(first.source_w2c, second.source_w2c))
        self.assertTrue(torch.equal(first.K, second.K))
        for stage in range(5):
            a = fitting.sequence_objective_streamed(first, first.flatten(), obs, stage, heights)
            b = fitting.sequence_objective_streamed(second, second.flatten(), obs, stage, heights)
            self.assertTrue(torch.equal(a['loss'], b['loss']))
            self.assertTrue(torch.equal(a['gradient'], b['gradient']))
        expected = first.source_w2c[:, None, :3, 3].expand(-1, 17, -1)
        difference = first.world_vertices(first.forward_state()) - second.world_vertices(second.forward_state())
        self.assert_close(difference, expected)
        self.assertEqual(second.camera_length_contract['basis'], 'model_conditioned')

    def test_partial_frame_sweep_raises_and_does_not_mutate_parameters(self):
        model, obs, heights = make_case(3)
        original = model.flatten()
        count = 0
        def stop():
            nonlocal count
            count += 1
            if count == 2:
                raise TimeoutError('CPU stop before second frame')
        with self.assertRaises(TimeoutError):
            fitting.sequence_objective_streamed(model, original, obs, 4, heights, boundary=stop)
        self.assertTrue(torch.equal(original, model.flatten()))

    def test_invalid_later_frame_prevents_complete_objective(self):
        model, obs, heights = make_case(2)
        vector = model.flatten()
        a, _, _ = model.layout()['transl']
        vector[a + 5] = -10.
        with self.assertRaises(RuntimeError):
            fitting.sequence_objective_streamed(model, vector, obs, 4, heights)

    def test_dense_and_streamed_optimizer_keep_all_stages_lbfgs_options_and_closure_gradient(self):
        records = []
        reports = []
        for mode in ('dense', 'streamed'):
            model, obs, heights = make_case()
            NoStepLBFGS.records = []
            with patch.object(solver.torch.optim, 'LBFGS', NoStepLBFGS):
                report, theta = solver.optimize_sequence(model, obs, heights, evaluation_mode=mode,
                    root_policy='provided_initialization', initialization_source='CPU toy original roots', seconds=60)
            self.assertEqual(report['state'], 'schedule_complete')
            self.assertEqual([r['stage'] for r in report['stages']], list(range(5)))
            self.assertFalse(report['physical_accuracy_passed'])
            self.assertFalse(report['numerical_converged'])
            records.append(NoStepLBFGS.records)
            reports.append(report)
            if mode == 'streamed':
                work = report['streamed_work']
                self.assertEqual(work['sweeps_started'], report['evaluations'])
                self.assertEqual(work['sweeps_completed'], report['evaluations'])
                self.assertEqual(work['frame_forwards_completed'], 2 * report['evaluations'])
                self.assertEqual(work['objective_backward_calls_completed'], work['frame_forwards_completed'])
                self.assertEqual(work['jacobian_vjp_seed_rows_completed'], 5 * 2 * 375)
        self.assertEqual(len(records[0]), len(records[1]))
        for a, b in zip(*records):
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[0]['line_search_fn'], 'strong_wolfe')
            self.assertAlmostEqual(a[1], b[1], places=8)
            self.assert_close(a[2], b[2])
        self.assertAlmostEqual(reports[0]['final_diagnostic']['loss'], reports[1]['final_diagnostic']['loss'], places=8)

    def test_real_lbfgs_bounded_trajectory_matches_reference(self):
        results = []
        for mode in ('dense', 'streamed'):
            model, obs, heights = make_case()
            report, theta = solver.optimize_sequence(model, obs, heights, evaluation_mode=mode,
                root_policy='provided_initialization', initialization_source='CPU toy roots',
                seconds=60, max_evaluations=8)
            self.assertEqual(report['state'], 'budget_ended')
            self.assertEqual(report['evaluations'], 8)
            self.assertFalse(report['numerical_converged'])
            results.append(theta)
        self.assert_close(*results, rtol=1e-9, atol=1e-10)

    def test_zero_budget_and_default_torso_refusal_do_not_forward(self):
        model, obs, heights = make_case()
        model.body.calls.clear()
        report, _ = solver.optimize_sequence(model, obs, heights, evaluation_mode='streamed')
        self.assertEqual(report['state'], 'root_support_insufficient')
        self.assertEqual(model.body.calls, [])
        report, _ = solver.optimize_sequence(model, obs, heights, evaluation_mode='streamed',
            root_policy='provided_initialization', initialization_source='CPU toy roots', seconds=0)
        self.assertEqual(report['state'], 'budget_ended')
        self.assertEqual(model.body.calls, [])

    def test_partial_scaling_sweep_is_counted_but_never_accepted_as_best(self):
        model, obs, heights = make_case()
        original = model.flatten()
        actual_streamer = solver.sequence_objective_streamed
        def interrupted(*args, **kwargs):
            completed = 0
            original_work = kwargs['work_callback']
            def work(event, count):
                nonlocal completed
                original_work(event, count)
                if event == 'frame_forwards_completed':
                    completed += count
            def stop():
                if completed:
                    raise TimeoutError('Deterministic CPU partial-sweep interruption')
            kwargs['boundary'] = stop
            kwargs['work_callback'] = work
            return actual_streamer(*args, **kwargs)
        with patch.object(solver, 'sequence_objective_streamed', interrupted):
            report, theta = solver.optimize_sequence(model, obs, heights, evaluation_mode='streamed',
                root_policy='provided_initialization', initialization_source='CPU toy roots', max_evaluations=1)
        self.assertEqual(report['state'], 'budget_ended')
        self.assertEqual(report['streamed_work']['sweeps_started'], 1)
        self.assertEqual(report['streamed_work']['sweeps_completed'], 0)
        self.assertEqual(report['streamed_work']['frame_forwards_completed'], 1)
        self.assertEqual(report['streamed_work']['objective_backward_calls_completed'], 1)
        self.assertTrue(torch.equal(theta, original))
        self.assertTrue(torch.equal(model.flatten(), original))

    def test_streamed_root_branch_equals_dense_for_explicitly_eligible_toy_support(self):
        results = []
        for mode in ('dense', 'streamed'):
            model, obs, heights = make_case()
            # Analytic synthetic support only; production observations still
            # remove hips and default torso count refusal is tested above.
            obs.fitting_confidence = lambda: obs.confidence.clone()
            NoStepLBFGS.records = []
            with patch.object(solver.torch.optim, 'LBFGS', NoStepLBFGS):
                report, theta = solver.optimize_sequence(model, obs, heights, evaluation_mode=mode, seconds=60)
            self.assertEqual(report['stages'][0]['stage'], 'root')
            results.append(NoStepLBFGS.records[0])
        self.assertAlmostEqual(results[0][1], results[1][1], places=10)
        self.assert_close(results[0][2], results[1][2])

    def test_invalid_mode_rejected_before_model_access(self):
        with self.assertRaises(ValueError):
            solver.optimize_sequence(object(), object(), [], evaluation_mode='approximate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StreamedControls))
    record = dict(state='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic()-start,
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (Path(fitting.__file__), Path(solver.__file__), Path(__file__))},
        execution_device='cpu', cuda_initialized=torch.cuda.is_initialized(), real_model_forwards=0,
        new_person_fits=0, Gaussian_steps=0,
        scope='Exact nonlinear CPU toy objective, derivatives, scaling and LBFGS integration. No real-person quality/observability claim.')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
