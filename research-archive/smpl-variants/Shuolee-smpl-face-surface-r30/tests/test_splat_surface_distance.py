"""Independent numerical controls for fixed-plane Gaussian distribution distance."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import torch
from shuolee_portrait.splat_surface_distance import (
    choose_world_split_rows, folded_normal_distance, plane_distribution, tangent_split_seeds,
)


class SplatSurfaceDistance(unittest.TestCase):
    def test_analytic_absolute_distance_matches_sampling(self):
        mu = torch.tensor([-0.005, 0., .002, .001],dtype=torch.float64)
        sigma = torch.tensor([.002, .001, .003, 0.],dtype=torch.float64)
        draws = torch.randn((200000,1), dtype=mu.dtype, generator=torch.Generator().manual_seed(937))
        measured = (mu+draws*sigma).abs().mean(0)
        torch.testing.assert_close(folded_normal_distance(mu,sigma), measured, rtol=.008, atol=2e-6)
        self.assertAlmostEqual(float(folded_normal_distance(mu,sigma)[1]), .001*math.sqrt(2/math.pi), 13)

    def test_isotropic_thickness_participates_and_rotation_changes_normal_extent(self):
        center = torch.tensor([[0.,0.,0.],[0.,0.,0.]],dtype=torch.float64,requires_grad=True)
        scale = torch.tensor([[.004,.004,.004],[.008,.001,.008]],dtype=torch.float64).log().requires_grad_()
        q = torch.tensor([[1.,0.,0.,0.],[math.cos(math.pi/8),math.sin(math.pi/8),0.,0.]],
                         dtype=torch.float64,requires_grad=True)
        mean, normal_sigma, distance = plane_distribution(center,scale,q,torch.ones(2,dtype=q.dtype))
        expected = torch.tensor([.004,math.sqrt((.001**2+.008**2)/2)],dtype=q.dtype)
        torch.testing.assert_close(normal_sigma, expected, rtol=1e-12, atol=1e-14)
        self.assertTrue(torch.equal(mean, torch.zeros_like(mean)))
        distance.sum().backward()
        self.assertGreater(float(scale.grad[0,1]), 0.)
        self.assertEqual(float(scale.grad[0,0]), 0.)
        self.assertGreater(float(q.grad[1].abs().sum()), 0.)
        self.assertEqual(float(q.grad[0].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(center.grad).all())

    def test_tangent_split_preserves_tangent_moment_but_removes_normal_spike(self):
        from roma import quat_wxyz_to_xyzw, unitquat_to_rotmat
        center = torch.tensor([[.001,.007,-.004]],dtype=torch.float64)
        log_axes = torch.tensor([[.012,.002,.007]],dtype=center.dtype).log()
        q = torch.tensor([[math.cos(math.pi/7),math.sin(math.pi/7),0.,0.]],dtype=center.dtype)
        scale = torch.tensor([1.],dtype=center.dtype)
        children, logs, rotations = tangent_split_seeds(
            center,log_axes,q,scale,center_clearance_m=.002,normal_sigma_cap_m=.0015)
        old_R = unitquat_to_rotmat(quat_wxyz_to_xyzw(q))[..., [0,2],:]
        old_C = (old_R*log_axes.exp()[...,None,:]) @ (old_R*log_axes.exp()[...,None,:]).transpose(-1,-2)
        new_R = unitquat_to_rotmat(quat_wxyz_to_xyzw(rotations))
        new_axes = (new_R*logs.exp()[...,None,:])[..., [0,2],:]
        delta = (children-center[...,None,:])[...,[0,2]]
        moment = (new_axes@new_axes.transpose(-1,-2)+delta[...,None]*delta[...,None,:]).mean(1)
        torch.testing.assert_close(old_C, moment, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(new_R[...,1,1], torch.ones_like(new_R[...,1,1]))
        _, normal, distance = plane_distribution(children,logs,rotations,scale[...,None])
        self.assertLess(float(distance.max()), float(plane_distribution(center,log_axes,q,scale)[2]))
        self.assertAlmostEqual(float(normal.max()), .0015, 13)

    def test_world_candidates_use_repeat_support_and_metric_residual(self):
        d = torch.tensor([.008,.011,.008,.002,.005])
        bins = torch.tensor([3,1,2,8,4])
        ids = torch.tensor([12,7,2,6,5])
        selected = choose_world_split_rows(d,bins,ids,threshold=.004,budget=2)
        self.assertEqual(selected.tolist(),[2,0])


if __name__ == '__main__':
    unittest.main()
