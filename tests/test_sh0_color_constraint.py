from types import SimpleNamespace
import unittest

import torch

from shuolee_portrait.sh0_main import DCColorProjection


C0 = .28209479177387814


class SH0ColorConstraintTests(unittest.TestCase):
    def test_adam_projection_clears_only_outward_first_momentum(self):
        dc = torch.nn.Parameter(torch.zeros(2, 1, 3))
        other = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([dc, other], lr=.1)
        model = SimpleNamespace(_features_dc=dc, optimizer=optimizer, get_opacity=torch.ones(2, 1))
        rule = DCColorProjection(dc, C0)
        initial = rule.measure(model)
        dc.grad = torch.tensor([[[-1., 1., -.2]], [[1., -1., .2]]])
        other.grad = torch.ones_like(other)
        with torch.no_grad():
            dc[0, 0, :2] = torch.tensor([rule.bound-.01, -rule.bound+.01])
            dc[1, 0, :2] = torch.tensor([rule.bound, -rule.bound])
        optimizer.step()
        variance = optimizer.state[dc]['exp_avg_sq'].clone()
        other_state = optimizer.state[other]['exp_avg'].clone()
        rule.apply(model, after_step=True)
        moment = optimizer.state[dc]['exp_avg']
        self.assertEqual(moment[0, 0, :2].count_nonzero().item(), 0)
        self.assertGreater(moment[1, 0, 0].item(), 0)
        self.assertLess(moment[1, 0, 1].item(), 0)
        torch.testing.assert_close(optimizer.state[dc]['exp_avg_sq'], variance)
        torch.testing.assert_close(optimizer.state[other]['exp_avg'], other_state)
        report = rule.report(model, initial)
        self.assertEqual(report['projected_channels'], 2)
        self.assertEqual(report['cleared_outward_first_moments'], 2)
        self.assertEqual(report['terminal']['channels_outside_bounds'], 0)

        dc.grad = -dc.grad
        other.grad = None
        optimizer.step()
        rule.apply(model, after_step=True)
        self.assertLess(dc[0, 0, 0].item(), rule.bound)
        self.assertGreater(dc[0, 0, 1].item(), -rule.bound)

    def test_replaced_color_parameter_is_bounded_without_initializing_adam_state(self):
        old = torch.nn.Parameter(torch.zeros(1, 1, 3))
        optimizer = torch.optim.Adam([old])
        rule = DCColorProjection(old, C0)
        new = torch.nn.Parameter(torch.tensor([[[3., -3., 0.]], [[0., 0., 0.]]]))
        optimizer.param_groups[0]['params'] = [new]
        model = SimpleNamespace(_features_dc=new, optimizer=optimizer, get_opacity=torch.ones(2, 1))
        rule.apply(model, after_step=False)
        report = rule.report(model, rule.measure(model))
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(old.count_nonzero().item(), 0)
        self.assertEqual(report['topology_projections'], 1)
        self.assertEqual(report['projected_channels'], 2)
        decoded = .5+new.detach()*C0
        self.assertGreaterEqual(decoded.min().item(), 0.)
        self.assertLessEqual(decoded.max().item(), 1.)
