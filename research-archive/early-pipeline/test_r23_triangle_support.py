"""Run actual R23 free-clothing transport and stock topology on CPU."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from plyfile import PlyData

sys.path[:0] = ['src', 'src/avatar_core']
from avatar_core.scene import gaussian_model
from avatar_core.scene.clothing_gaussian_model import ClothingGaussianModel
from avatar_core.scene.smplx_gaussian_model import SMPLXGaussianModel
from shuolee_portrait.drivable_clothing_training import _split
from shuolee_portrait.free_clothing_torch import normalized_quaternion_matrix
from utils import general_utils


DTYPE = torch.float64


def tensor(values):
    return torch.tensor(values, dtype=DTYPE)


class CpuOnlyAllocations:
    cuda = SimpleNamespace(empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, *args, device='cpu', **kwargs):
        return torch.zeros(*args, device='cpu', **kwargs)


class TriangleSupportTest(unittest.TestCase):
    def setUp(self):
        self.reference = tensor([[0, 0, 2], [1, 0, 2], [0, 1, 2],
                                 [3, 0, 2], [3, 1, 2], [3, 0, 3]])
        self.faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        self.rotations = tensor([[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                 [[1, 0, 0], [0, -1, 0], [0, 0, -1]]])
        self.scales = tensor([2, .5])
        self.translations = tensor([[1, 5, 0], [3, 1, 7]])
        posed = (self.scales[:, None, None] * self.reference.reshape(2, 3, 3)
                 @ self.rotations.transpose(-1, -2)) + self.translations[:, None]
        self.vertices = torch.stack((posed.reshape(6, 3), self.reference))
        self.support = torch.tensor([1, 0])
        self.xyz = tensor([[.8, 4, 5], [3.1, .3, 6.4]])
        self.sigma = tensor([[.2, .15, .09], [.1, .07, .03]])
        self.quat = tensor([[.6, .4, .1, -.2], [.9, .05, .2, .1]])
        self.model = ClothingGaussianModel(0)
        self.model._xyz = torch.nn.Parameter(self.xyz.clone())
        self.model._scaling = torch.nn.Parameter(self.sigma.log())
        self.model._rotation = torch.nn.Parameter(self.quat.clone())
        self.model._features_dc = torch.nn.Parameter(tensor([[[.3, .2, .1]], [[.6, .5, .2]]]))
        self.model._features_rest = torch.nn.Parameter(torch.zeros(2, 0, 3, dtype=DTYPE))
        self.model._opacity = torch.nn.Parameter(tensor([[1], [2]]))
        self.model.configure_triangle_support(self.vertices, self.faces, 1, self.support,
                                              stable_ids=torch.tensor([17, 20]), next_stable_id=31)

    def test_pose_covariance_gradients_and_ply(self):
        model, owners = self.model, self.support
        assert model.get_xyz is model._xyz
        torch.testing.assert_close(model.get_scaling, self.sigma)
        torch.testing.assert_close(model.get_rotation, torch.nn.functional.normalize(self.quat))
        model.select_mesh_by_timestep(0)
        transform = self.rotations[owners]*self.scales[owners, None, None]
        expected_xyz = torch.einsum('nij,nj->ni', transform, self.xyz)+self.translations[owners]
        torch.testing.assert_close(model.get_xyz, expected_xyz)
        linear, translation = model.deformation()
        torch.testing.assert_close(linear, transform)
        torch.testing.assert_close(translation, self.translations[owners])
        ref_rotation = normalized_quaternion_matrix(torch.nn.functional.normalize(self.quat))
        axes = transform @ ref_rotation * self.sigma[:, None]
        covariance = axes @ axes.transpose(-1, -2)
        rows, cols = (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)
        torch.testing.assert_close(model.get_covariance(), covariance[:, rows, cols])
        renderer_axes = normalized_quaternion_matrix(model.get_rotation)*model.get_scaling[:, None]
        torch.testing.assert_close(renderer_axes @ renderer_axes.transpose(-1, -2), covariance)
        loss = model.get_xyz.square().sum() + (model.get_covariance()*tensor([1, 2, 3, 4, 5, 6])).sum()
        for gradient in torch.autograd.grad(loss, (model._xyz, model._scaling, model._rotation)):
            assert torch.isfinite(gradient).all() and (gradient.abs().sum(dim=1) > 0).all()
        with tempfile.TemporaryDirectory() as temp:
            posed, canonical = Path(temp)/'posed.ply', Path(temp)/'canonical.ply'
            model.select_mesh_by_timestep(1)
            model.export_world_ply(posed, timestep=0)
            assert model.timestep == 1
            model.save_ply(str(canonical))
            for path, expected in ((posed, expected_xyz), (canonical, self.xyz)):
                row = PlyData.read(path)['vertex']
                xyz = tensor([[float(row[k][i]) for k in ('x', 'y', 'z')] for i in range(2)])
                torch.testing.assert_close(xyz, expected, atol=2e-6, rtol=1e-6)
            row = PlyData.read(posed)['vertex']
            q = tensor([[float(row[f'rot_{k}'][i]) for k in range(4)] for i in range(2)])
            sigma = tensor([[float(row[f'scale_{k}'][i]) for k in range(3)] for i in range(2)]).exp()
            axes = normalized_quaternion_matrix(torch.nn.functional.normalize(q))*sigma[:, None]
            torch.testing.assert_close(axes @ axes.transpose(-1, -2), covariance, atol=2e-7, rtol=2e-6)
        with self.assertRaises(RuntimeError):
            model.refresh_anchors()
        with self.assertRaises(ValueError):
            model.select_mesh_by_timestep(0.5)

    def test_native_clone_split_and_prune_keep_owner_and_identity(self):
        model = self.model
        fields = {'xyz': '_xyz', 'f_dc': '_features_dc', 'f_rest': '_features_rest',
                  'opacity': '_opacity', 'scaling': '_scaling', 'rotation': '_rotation'}
        model.optimizer = torch.optim.Adam([{'name': name, 'params': [getattr(model, field)]}
                                            for name, field in fields.items()], lr=.001)
        sum(getattr(model, field).square().sum() for field in fields.values()).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.percent_dense = .15
        model.xyz_gradient_accum = torch.ones(2, 1)
        model.denom = torch.ones(2, 1)
        model.max_radii2D = torch.ones(2)
        model.select_mesh_by_timestep(0)
        # The posed scales would reverse which row is split and which is cloned.
        # Redirect only CUDA-only stock allocations and rotation output to CPU.
        def rotation(q):
            return normalized_quaternion_matrix(torch.nn.functional.normalize(q)).float()
        sigma = model._scaling.detach().exp()
        with torch.no_grad(), patch.object(gaussian_model, 'torch', CpuOnlyAllocations()), \
                patch.object(gaussian_model, 'build_rotation', rotation):
            model.densify_and_prune(.5, .005, 1., None)
        assert model.timestep == 0
        assert model.densification_ids.tolist() == [20, 31, 32, 33]
        assert model.support_face_ids.tolist() == [0, 0, 1, 1]
        assert model.next_densification_id == 34
        expected = torch.stack((sigma[1], sigma[1], sigma[0]/1.6, sigma[0]/1.6))
        torch.testing.assert_close(model._scaling.exp(), expected)
        torch.testing.assert_close(model.get_scaling, expected*self.scales[model.support_face_ids, None])
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            assert len(parameter) == len(model.densification_ids)
            assert model.optimizer.state[parameter]['exp_avg'].shape == parameter.shape
        model.prune_points(torch.tensor([False, False, False, True]))
        model.configure_triangle_support(self.vertices, self.faces, 1, model.support_face_ids)
        assert model.densification_ids.tolist() == [20, 31, 32]
        assert model.next_densification_id == 34

    def test_nonuniform_triangle_change_matches_stock_bound_render(self):
        old = SMPLXGaussianModel(0)
        vertices = self.vertices.clone()
        vertices[0, 1] += tensor([-.3, 2, .3])
        vertices[0, 5] += tensor([.9, .4, -.5])
        old.vertices_sequence, old.faces, old.binding = vertices, self.faces, self.support
        old._xyz, old._scaling, old._rotation = self.xyz, self.sigma.log(), self.quat
        old.select_mesh_by_timestep(1)
        cloth = ClothingGaussianModel(0)
        cloth._xyz, cloth._scaling, cloth._rotation = old.get_xyz, old.get_scaling.log(), old.get_rotation
        cloth.configure_triangle_support(vertices, self.faces, 1, self.support)
        old.select_mesh_by_timestep(0)
        cloth.select_mesh_by_timestep(0)
        torch.testing.assert_close(cloth.get_xyz, old.get_xyz, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cloth.get_scaling, old.get_scaling, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(normalized_quaternion_matrix(cloth.get_rotation),
                                   normalized_quaternion_matrix(old.get_rotation), rtol=1e-12, atol=1e-12)

    def test_r23_trainer_tangent_split_after_adam(self):
        model = self.model
        model._rotation = torch.nn.Parameter(tensor([[1, 0, 0, 0], [1, 0, 0, 0]]))
        fields = {'xyz': '_xyz', 'f_dc': '_features_dc', 'f_rest': '_features_rest',
                  'opacity': '_opacity', 'scaling': '_scaling', 'rotation': '_rotation'}
        model.optimizer = torch.optim.Adam([{'name': name, 'params': [getattr(model, field)]}
                                            for name, field in fields.items()], lr=.001)
        sum(getattr(model, field).square().sum() for field in fields.values()).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model._opacity.requires_grad_(False)
        model.xyz_gradient_accum = torch.ones(2, 1)
        model.denom = torch.ones(2, 1)
        model.max_radii2D = torch.ones(2)
        model.select_mesh_by_timestep(0)
        parent_center, parent_sigma, parent_alpha = model._xyz[0].detach().clone(), model._scaling[0].detach().exp(), model.get_opacity[0].clone()
        normals = torch.nn.functional.normalize(tensor([[1, 1, 1], [1, 0, 0]]))

        def rotation(q):
            return normalized_quaternion_matrix(torch.nn.functional.normalize(q))

        with patch.object(gaussian_model, 'torch', CpuOnlyAllocations()), \
                patch.object(general_utils, 'build_rotation', rotation):
            receipt = _split(model, .5, normals)
        assert receipt == dict(eligible=2, tangent_eligible=1, parents=1, children=2,
                               parent_ids=[17], first_child_id=31)
        assert model.timestep == 0
        assert model.densification_ids.tolist() == [20, 31, 32]
        assert model.next_densification_id == 33
        assert model.support_face_ids.tolist() == [0, 1, 1]
        offsets = model._xyz[1:]-parent_center
        assert not (offsets@normals[1]).count_nonzero()
        displacement = parent_sigma*tensor([0, .5, 0])
        torch.testing.assert_close(offsets, torch.stack((-displacement, displacement)))
        sigma = model._scaling[1:].exp()
        torch.testing.assert_close(sigma.square() + offsets.square(), parent_sigma.square().expand_as(sigma))
        torch.testing.assert_close(1-(1-model.get_opacity[1:]).prod(), parent_alpha[0])
        assert model._opacity.requires_grad is False
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            assert model.optimizer.state[parameter]['exp_avg'].shape == parameter.shape
            assert not model.optimizer.state[parameter]['exp_avg'][1:].count_nonzero()


if __name__ == '__main__':
    unittest.main()
