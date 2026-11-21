"""Exercise the production base Gaussian split without CUDA or the optimizer."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


CORE = Path(__file__).resolve().parents[1] / 'src/avatar_core'
MODEL_FILE = CORE / 'scene/gaussian_model.py'
MODEL_SOURCE = MODEL_FILE.read_text(encoding='utf-8')
CHILDREN = 6
LOCAL_STD = torch.tensor([[.4, .2, .1], [.6, .2, .15], [1.4, .8, .5], [1., .4, .6]])
LOCAL_CENTER = torch.tensor([[.1, .2, .3], [.2, -.1, .4], [-.2, .7, .1], [.4, -.2, .6]])
LOCAL_QUAT = torch.tensor([[1., 0, 0, 1], [1., 0, 0, 0], [1., 1, 0, 0], [1., 0, 1, 0]])
LOCAL_ROT = torch.tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]],
                          [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
                          [[1., 0, 0], [0, 0, -1], [0, 1, 0]],
                          [[0., 0, 1], [0, 1, 0], [-1, 0, 0]]])
FACE_ROT = LOCAL_ROT[[3, 0, 2]]
FACE_SCALE = torch.tensor([[2.], [.5], [3.]])
FACE_CENTER = torch.tensor([[3., -2., 5.], [-1., 7., 2.], [.5, 6., -1.]])
BINDING = torch.tensor([0, 1, 1, 2])
BOUND_SELECTED = torch.tensor([0, 2])
UNBOUND_SELECTED = torch.tensor([1, 2])


class CpuSplitTorch:
    """Redirect allocations and supply exact deterministic covariance samples."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, *args, device, **kwargs):
        return torch.zeros(*args, device='cpu', **kwargs)

    def ones_like(self, *args, device, **kwargs):
        return torch.ones_like(*args, device='cpu', **kwargs)

    def normal(self, *, mean, std):
        axes = torch.eye(3, dtype=std.dtype) * (3 ** .5)
        sigma_points = torch.cat((axes, -axes))
        return mean + std * sigma_points.repeat_interleave(len(std) // CHILDREN, dim=0)


def cpu_model(source):
    # Load exact method bodies. Full module import requires simple_knn CUDA.
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
    methods = {'get_scaling', 'get_xyz', 'densify_and_split'}
    model.body = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    utility = CORE / 'utils/general_utils.py'
    rotation = next(node for node in ast.parse(utility.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'build_rotation')
    module = ast.Module(body=[rotation, model], type_ignores=[])
    namespace = {'torch': CpuSplitTorch()}
    exec(compile(module, str(MODEL_FILE), 'exec'), namespace)
    return namespace['GaussianModel']


def split(model_type, *, bound, world_factor=1.):
    model = model_type()
    model._xyz = LOCAL_CENTER.clone()
    model._scaling = LOCAL_STD.log()
    model._rotation = LOCAL_QUAT.clone()
    model._features_dc = torch.zeros(4, 1, 3)
    model._features_rest = torch.zeros(4, 0, 3)
    model._opacity = torch.zeros(4, 1)
    model.scaling_activation = torch.exp
    model.scaling_inverse_activation = torch.log
    model.percent_dense = .5
    model.binding = BINDING.clone() if bound else None
    model.binding_counter = BINDING.bincount().to(torch.int32)
    model.face_orien_mat = FACE_ROT
    model.face_scaling = FACE_SCALE * world_factor
    model.face_center = FACE_CENTER * world_factor
    captured = SimpleNamespace()

    def postfix(xyz, _dc, _rest, _opacity, scaling, _rotation):
        captured.local = xyz
        captured.local_std = scaling.exp()

    def prune(mask):
        captured.prune = mask

    model.densification_postfix = postfix
    model.prune_points = prune
    model.densify_and_split(torch.tensor([[1.], [1.], [1.], [.2]]), .5, world_factor, N=CHILDREN)
    model._xyz = captured.local
    if bound:
        model.binding = model.binding[4:]
    captured.binding = model.binding
    captured.world = model.get_xyz
    return captured


def population_covariance(positions):
    offsets = positions - positions.mean(dim=0)
    return torch.einsum('npi,npj->pij', offsets, offsets) / len(positions)


def covariance_for(rotation, std):
    axes = rotation * std[:, None, :]
    return axes @ axes.transpose(-1, -2)


class BoundSplitUnitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = cpu_model(MODEL_SOURCE)

    def test_bound_split_uses_local_anisotropic_covariance(self):
        children = split(self.model, bound=True)
        chosen = BOUND_SELECTED
        self.assertTrue(torch.equal(children.binding, BINDING[chosen].repeat(CHILDREN)))
        local = children.local.reshape(CHILDREN, len(chosen), 3)
        world = children.world.reshape(CHILDREN, len(chosen), 3)
        face = BINDING[chosen]
        R = FACE_ROT[face]
        local_rotation = LOCAL_ROT[chosen]
        std = LOCAL_STD[chosen]
        torch.testing.assert_close(local.mean(0), LOCAL_CENTER[chosen])
        torch.testing.assert_close(population_covariance(local), covariance_for(local_rotation, std))
        torch.testing.assert_close(population_covariance(world), covariance_for(R @ local_rotation, FACE_SCALE[face]*std))
        torch.testing.assert_close(children.local_std, std.repeat(CHILDREN, 1) / (.8*CHILDREN))

    def test_bound_split_is_equivariant_under_world_scaling(self):
        original = split(self.model, bound=True)
        scaled = split(self.model, bound=True, world_factor=3.)
        torch.testing.assert_close(scaled.local, original.local)
        torch.testing.assert_close(scaled.world, original.world * 3.)
        self.assertTrue(torch.equal(scaled.prune, original.prune))

    def test_split_keeps_world_admission_and_unbound_offsets(self):
        bound = split(self.model, bound=True)
        unbound = split(self.model, bound=False)
        self.assertEqual(bound.prune.tolist(), [True, False, True, False] + [False] * (2*CHILDREN))
        self.assertEqual(unbound.prune.tolist(), [False, True, True, False] + [False] * (2*CHILDREN))
        chosen = UNBOUND_SELECTED
        world = unbound.world.reshape(CHILDREN, len(chosen), 3)
        torch.testing.assert_close(world.mean(0), LOCAL_CENTER[chosen])
        torch.testing.assert_close(population_covariance(world), covariance_for(LOCAL_ROT[chosen], LOCAL_STD[chosen]))
        torch.testing.assert_close(unbound.local_std, LOCAL_STD[chosen].repeat(CHILDREN, 1) / (.8*CHILDREN))


if __name__ == '__main__':
    unittest.main()
