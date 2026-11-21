"""CPU toy integration of explicit sequence root policy; no human model run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from shuolee_portrait import sequence_optimizer as solver
import test_sequence_optimizer_cpu as optimizer_fixtures
from test_sequence_optimizer_cpu import (
    TinyModel, SyntheticObservations,
    StationaryStep, quadratic_residuals,
)


SOURCE = 'fixed CPU toy initialization; no physical root or camera claim'


def observed_residuals(model, vector, observations, stage, heights):
    """Small analytic consumer of the production observation eligibility API."""
    out = model.forward_state(model.unpack(vector))
    weight = observations.fitting_confidence()
    delta = model.project(out) - observations.targets
    eligible = torch.where(weight[..., None] > 0, delta, torch.zeros_like(delta))
    return torch.cat([(eligible * weight[..., None]).reshape(-1),
                      vector - model.target]), out


class RootPolicyControls(unittest.TestCase):
    def observations(self, model):
        return optimizer_fixtures.OptimizerRegression().make_official_observations(model)

    def provided(self, model, observations, residual=quadratic_residuals, **kwargs):
        with patch.object(solver, 'sequence_residuals', residual), \
                patch.object(solver.torch.optim, 'LBFGS', StationaryStep), \
                patch.object(solver, '_root_objective', side_effect=AssertionError('No torso pre-alignment')):
            return solver.optimize_sequence(
                model, observations, [1000], root_policy='provided_initialization',
                initialization_source=SOURCE, **kwargs)

    def test_default_policy_still_refuses_two_shoulders_without_forward(self):
        model = TinyModel()
        observations = self.observations(model)
        original = model.flatten()
        with patch.object(model, 'forward_state', side_effect=AssertionError('No model forward')):
            report, theta = solver.optimize_sequence(model, observations, [1000])
        self.assertEqual(report['state'], 'root_support_insufficient')
        self.assertEqual(report['root_support']['eligible_counts'], [2])
        self.assertEqual(report['root_support']['minimum_required'], 3)
        self.assertEqual(report['root_initialization']['policy'], 'eligible_torso')
        self.assertFalse(report['root_initialization']['prealignment_performed'])
        self.assertEqual(report['evaluations'], 0)
        self.assertTrue(torch.equal(theta, original))

    def test_provided_policy_requires_source_and_rejects_unknown_policy_before_model_access(self):
        for arguments in [dict(root_policy='implicit'),
                          dict(root_policy='provided_initialization'),
                          dict(root_policy='provided_initialization', initialization_source='  '),
                          dict(root_policy='provided_initialization', initialization_source={})]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                solver.optimize_sequence(object(), object(), [1000], **arguments)

    def test_provided_root_enters_exactly_five_joint_stages_unchanged_and_can_then_optimize(self):
        model = TinyModel()
        observations = self.observations(model)
        initial = model.flatten()
        model.target[1] = .15
        model.target[4] = .25
        entered = []
        def capture(model, vector, obs, stage, heights):
            entered.append((stage, vector.detach().clone()))
            return quadratic_residuals(model, vector, obs, stage, heights)
        report, theta = self.provided(model, observations, capture)
        self.assertTrue(torch.equal(entered[0][1], initial))
        self.assertEqual([stage['stage'] for stage in report['stages']], list(range(5)))
        self.assertEqual(set(stage for stage, _ in entered), set(range(5)))
        self.assertTrue(report['schedule_complete'])
        self.assertTrue(report['numerical_converged'])
        self.assertGreater(float((theta[1:] - initial[1:]).abs().max()), .1)
        self.assertEqual(report['root_support']['eligible_counts'], [2])
        self.assertFalse(report['root_support']['minimum_count_passed'])
        self.assertEqual(report['root_initialization']['source'], SOURCE)
        self.assertTrue(report['root_initialization']['provided_root_preserved_before_joint'])
        self.assertFalse(report['root_initialization']['prealignment_performed'])
        self.assertFalse(report['root_initialization']['sufficient_observability_established'])
        self.assertFalse(report['physical_accuracy_passed'])

    def test_ignored_hips_duplicate_face_and_held_targets_cannot_enter_new_policy(self):
        baseline = TinyModel()
        observations = self.observations(baseline)
        observations.face_split[:, 0] = 'HELD'
        altered = copy.deepcopy(observations)
        ignored = [0, 1, 8, 9, 12, 15, 16, 17, 18, 25, 46, 67]
        altered.targets[:, ignored] += 100000.
        before_conf = altered.confidence.clone()
        before_targets = altered.targets.clone()
        self.assertFalse(altered.fitting_confidence()[:, ignored].any())
        first, theta = self.provided(baseline, observations, observed_residuals)
        second, changed = self.provided(TinyModel(), altered, observed_residuals)
        self.assertTrue(torch.equal(theta, changed))
        self.assertEqual(first['final_diagnostic'], second['final_diagnostic'])
        self.assertTrue(torch.equal(altered.confidence, before_conf))
        self.assertTrue(torch.equal(altered.targets, before_targets))

    def test_provided_policy_zero_time_budget_preserves_root_and_no_unbudgeted_forward(self):
        model = TinyModel()
        observations = self.observations(model)
        original = model.flatten()
        with patch.object(model, 'forward_state', side_effect=AssertionError('No unbudgeted forward')):
            report, theta = self.provided(model, observations, seconds=0)
        self.assertEqual(report['state'], 'budget_ended')
        self.assertEqual(report['evaluations'], 0)
        self.assertFalse(report['schedule_complete'])
        self.assertFalse(report['numerical_converged'])
        self.assertTrue(torch.equal(theta, original))
        self.assertEqual(report['root_initialization']['source'], SOURCE)

    def test_provided_policy_evaluation_budget_remains_effective(self):
        model = TinyModel()
        report, _ = self.provided(model, self.observations(model), max_evaluations=3)
        self.assertEqual(report['state'], 'budget_ended')
        self.assertEqual(report['evaluations'], 3)
        self.assertFalse(report['numerical_converged'])
        self.assertEqual(report['final_diagnostic']['status'], 'unavailable')

    def test_provided_policy_does_not_bypass_observation_height_contract(self):
        model = TinyModel()
        observations = self.observations(model)
        with self.assertRaises(ValueError):
            solver.optimize_sequence(model, observations, [0],
                                     root_policy='provided_initialization', initialization_source=SOURCE)

    def test_explicit_default_policy_still_executes_root_for_sufficient_toy_support(self):
        model = TinyModel()
        observations = SyntheticObservations(model)
        with patch.object(solver, 'sequence_residuals', quadratic_residuals), \
                patch.object(solver.torch.optim, 'LBFGS', StationaryStep):
            report, _ = solver.optimize_sequence(model, observations, [1000],
                                                 root_policy='eligible_torso', initialization_source=SOURCE)
        self.assertEqual([x['stage'] for x in report['stages']], ['root', 0, 1, 2, 3, 4])
        self.assertTrue(report['root_initialization']['prealignment_performed'])
        self.assertFalse(report['root_initialization']['provided_root_preserved_before_joint'])
        self.assertFalse(report['root_initialization']['sufficient_observability_established'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RootPolicyControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(state='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic()-start,
                  optimizer_sha256=hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
                  test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cuda_initialized=torch.cuda.is_initialized(), execution_device='cpu',
                  real_model_forwards=0, new_person_fits=0, Gaussian_steps=0,
                  scope='CPU toy policy integration. No new human/GPU/physical-observability claim.')
    args.result.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
