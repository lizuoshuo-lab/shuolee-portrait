"""CPU-only regressions for sequence optimizer control flow, never SMPL fitting."""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait import sequence_optimizer as solver
from shuolee_portrait.sequence_fitting import SequenceObservations


class TinyModel:
    def __init__(self):
        self.frame_ids = ('cpu:0',)
        self.state = {'betas': torch.zeros((1, 1), dtype=torch.float64),
                      'global_orient': torch.zeros((1, 3), dtype=torch.float64),
                      'transl': torch.tensor([[0., 0., 4.]], dtype=torch.float64)}
        self.target = self.flatten()
        self.target[0] = 1.

    def layout(self):
        return {'betas': (0, 1, (1, 1)), 'global_orient': (1, 4, (1, 3)),
                'transl': (4, 7, (1, 3))}

    def flatten(self):
        return torch.cat([value.detach().reshape(-1) for value in self.state.values()])

    def unpack(self, vector):
        return {name: vector[a:b].reshape(shape) for name, (a, b, shape) in self.layout().items()}

    def restore(self, vector):
        for key, value in self.unpack(vector).items():
            self.state[key] = value.detach().clone()

    def forward_state(self, state=None):
        state = self.state if state is None else state
        base = torch.arange(118, dtype=torch.float64).reshape(1, 118, 1) / 100.
        xy = base.expand(1, 118, 2) + state['global_orient'][:, None, :2] + state['transl'][:, None, :2]
        z = state['transl'][:, None, 2:3].expand(1, 118, 1)
        joints = torch.cat([xy, z], dim=-1)
        return SimpleNamespace(vertices=joints, joints=joints)

    def project(self, out):
        return out.joints[..., :2]


class SyntheticObservations:
    """Injected eligible torso support solely to exercise optimizer numerics."""
    def __init__(self, model):
        self.targets = model.project(model.forward_state()).detach().clone()
        self.confidence = torch.ones((1, 118), dtype=torch.float64)

    def fitting_confidence(self):
        return self.confidence.clone()


def quadratic_residuals(model, vector, observations, stage, heights):
    return vector - model.target, model.forward_state(model.unpack(vector))


class OldLossStep:
    """Tiny steps but deliberately constant/stale return; endpoint keeps moving."""
    def __init__(self, params, **kwargs):
        self.parameter = params[0]
        self.root = kwargs['max_iter'] == 60

    def zero_grad(self):
        self.parameter.grad = None

    def step(self, closure):
        closure()
        if not self.root:
            with torch.no_grad():
                self.parameter.sub_(.001 * self.parameter.grad)
        return torch.tensor(123456., dtype=torch.float64)


class TrialBestStep(OldLossStep):
    """A lower-loss trial is observed before restoring a worse accepted endpoint."""
    def step(self, closure):
        closure()
        if not self.root:
            with torch.no_grad():
                self.parameter[0] = .2
            closure()
            with torch.no_grad():
                self.parameter.zero_()
        return torch.tensor(123456., dtype=torch.float64)


class StationaryStep(OldLossStep):
    def step(self, closure):
        loss = closure()
        if not self.root:
            with torch.no_grad():
                self.parameter.sub_(self.parameter.grad)
        return loss


class OptimizerRegression(unittest.TestCase):
    def run_solver(self, optimizer=OldLossStep, **budget):
        model = TinyModel()
        observations = SyntheticObservations(model)
        with patch.object(solver, 'sequence_residuals', quadratic_residuals), patch.object(solver.torch.optim, 'LBFGS', optimizer):
            report, theta = solver.optimize_sequence(model, observations, [1000], **budget)
        return model, report, theta

    def test_real_torch_lbfgs_returns_entry_loss(self):
        x = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        opt = torch.optim.LBFGS([x], lr=1, max_iter=5, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            loss = x.square().sum()
            loss.backward()
            return loss
        returned = float(opt.step(closure))
        self.assertEqual(returned, 4.)
        self.assertLess(float(x.square().sum()), 1e-12)

    def test_actual_endpoints_ignore_stale_step_return_and_outer_budget_is_not_convergence(self):
        model, report, theta = self.run_solver()
        self.assertEqual(report['state'], 'schedule_complete')
        self.assertTrue(report['schedule_complete'])
        self.assertFalse(report['numerical_converged'])
        self.assertFalse(report['final_stationary'])
        self.assertTrue(all(stage['outer_calls'] == 30 and stage['stop'] == 'outer_budget'
                            for stage in report['stages'][1:]))
        expected = float((theta - model.target).square().sum())
        self.assertAlmostEqual(report['final_diagnostic']['loss'], expected, places=14)
        self.assertLess(expected, 1.)
        self.assertTrue(torch.equal(theta, model.flatten()))

    def test_restored_best_diagnostics_are_not_endpoint_or_last_closure_diagnostics(self):
        model, report, theta = self.run_solver(TrialBestStep)
        last = report['stages'][-1]
        expected = float((theta - model.target).square().sum())
        expected_raw_grad = float((2 * (theta - model.target)).abs().max())
        self.assertEqual(last['stop'], 'objective_stagnation')
        self.assertFalse(report['numerical_converged'])
        self.assertLess(expected, last['endpoint']['loss'])
        self.assertAlmostEqual(last['restored_best']['loss'], expected, places=14)
        self.assertAlmostEqual(report['final_diagnostic']['loss'], expected, places=14)
        self.assertAlmostEqual(report['final_diagnostic']['raw_gradient_max'], expected_raw_grad, places=14)
        self.assertTrue(torch.equal(theta, model.flatten()))

    def test_stationarity_requires_a_fresh_final_gradient(self):
        _, report, _ = self.run_solver(StationaryStep)
        self.assertTrue(report['schedule_complete'])
        self.assertTrue(report['numerical_converged'])
        self.assertTrue(report['final_stationary'])
        self.assertLessEqual(report['final_diagnostic']['optimizer_gradient_max'], 1e-9)
        self.assertTrue(all(stage['stop'] == 'stationary_gradient' for stage in report['stages'][1:]))

    def test_evaluation_budget_stops_without_convergence_or_unbudgeted_final_probe(self):
        model, report, theta = self.run_solver(max_evaluations=5)
        self.assertEqual(report['state'], 'budget_ended')
        self.assertEqual(report['evaluations'], 5)
        self.assertFalse(report['schedule_complete'])
        self.assertFalse(report['numerical_converged'])
        self.assertIsNone(report['final_stationary'])
        self.assertEqual(report['final_diagnostic']['status'], 'unavailable')
        self.assertTrue(torch.equal(theta, model.flatten()))

    def test_time_budget_never_implies_convergence(self):
        _, report, _ = self.run_solver(seconds=0)
        self.assertEqual(report['state'], 'budget_ended')
        self.assertEqual(report['evaluations'], 0)
        self.assertFalse(report['numerical_converged'])

    def make_official_observations(self, model):
        shape = (1, 51)
        return SequenceObservations(model.frame_ids, model.project(model.forward_state()).detach().clone(),
            torch.ones((1, 118), dtype=torch.float64),
            np.full(shape, 'observed'), np.full(shape, 'FIT'), np.ones(shape, bool),
            np.zeros(shape, bool), np.array([[f'semantic:{i}' for i in range(51)]]))

    def test_two_shoulders_explicitly_stop_and_preserve_initialization(self):
        model = TinyModel()
        before = model.flatten()
        observations = self.make_official_observations(model)
        report, theta = solver.optimize_sequence(model, observations, [1000])
        self.assertEqual(report['state'], 'root_support_insufficient')
        self.assertEqual(report['root_support']['eligible_counts'], [2])
        self.assertEqual(report['evaluations'], 0)
        self.assertFalse(report['schedule_complete'])
        self.assertFalse(report['numerical_converged'])
        self.assertTrue(torch.equal(theta, before))
        self.assertTrue(torch.equal(model.flatten(), before))

    def test_ignored_hips_and_zero_confidence_points_do_not_change_root_value_or_gradient(self):
        model = TinyModel()
        observations = self.make_official_observations(model)
        observations.confidence[:, 2] = 0.
        observations.confidence[:, 5] = .5
        observations.targets[:, 5] += 2.
        data, _ = solver._root_data(model, observations, [1000])
        initial_t = model.state['transl'].clone()
        probe = model.flatten().requires_grad_(True)
        loss = solver._root_objective(model, probe, initial_t, data)
        grad = torch.autograd.grad(loss, probe)[0]
        # One active shoulder: 2 coordinates * displacement^2 * confidence^2.
        self.assertAlmostEqual(float(loss), 2., places=14)
        observations.targets[:, [9, 12, 2]] += 100000.
        observations.targets[:, 67:] += 100000.
        changed, _ = solver._root_data(model, observations, [1000])
        loss_changed = solver._root_objective(model, probe, initial_t, changed)
        grad_changed = torch.autograd.grad(loss_changed, probe)[0]
        self.assertTrue(torch.equal(loss, loss_changed))
        self.assertTrue(torch.equal(grad, grad_changed))
        observations.targets[:, 5] += 1000.
        observations.confidence.zero_()
        self.assertEqual(float(solver._root_objective(model, probe, initial_t, data)), float(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OptimizerRegression)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(state='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic()-start,
                  torch_version=torch.__version__, execution_device='cpu', cuda_initialized=torch.cuda.is_initialized(),
                  optimizer_sha256=hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
                  test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations=['Synthetic CPU optimizer-control tests, not a real SMPL-X or sequence fit.',
                               'No real-data quality, speed, full root observability or Gaussian acceptance established.',
                               'Current official torso eligibility leaves only two shoulders and intentionally stops.',
                               'Three eligible torso anchors are only a necessary count check.',
                               'Budget exhaustion can leave final diagnostics unavailable; no unbudgeted probe is made.',
                               'Stationarity uses the original frozen per-stage optimizer scaling; raw gradients are also reported.'])
    args.result.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
