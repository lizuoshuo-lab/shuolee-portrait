"""CPU interface checks; FakeTensor CUDA devices do not execute GPU kernels."""
from pathlib import Path
import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.smplifyx_runtime import cache_static_topology


def body_fixture():
    body = torch.nn.Module()
    body.register_buffer('parents', torch.arange(-1, 54, dtype=torch.int64))
    with FakeTensorMode():
        body.register_parameter('betas', torch.nn.Parameter(torch.zeros(1, 10, device='cuda')))
        body.register_buffer('template', torch.zeros(1, 3, device='cuda'))
    return body


class StaticTopologyTests(unittest.TestCase):
    def test_idempotent_and_only_parent_buffer_can_change(self):
        body = body_fixture()
        original = body.parents.clone()
        betas, template = body.betas, body.template
        first = cache_static_topology(body)
        second = cache_static_topology(body)
        self.assertTrue(torch.equal(body.parents, original))
        self.assertEqual(body.parents.device.type, 'cpu')
        self.assertIs(body.betas, betas)
        self.assertIs(body.template, template)
        self.assertTrue(body.betas.requires_grad)
        self.assertIs(body.get_buffer('parents'), body.parents)
        self.assertEqual(first, second)

    def test_wrong_shape_and_dtype_leave_model_unchanged(self):
        for parents in (torch.arange(-1, 53), torch.arange(-1, 54).float()):
            with self.subTest(shape=parents.shape, dtype=parents.dtype):
                body = body_fixture()
                body.parents = parents
                with self.assertRaisesRegex(ValueError, '55-node SMPL-X int64'):
                    cache_static_topology(body)
                self.assertIs(body.parents, parents)

    def test_invalid_hierarchy_is_not_repaired(self):
        for index, value in ((0, 0), (2, -1), (2, 2), (2, 54)):
            with self.subTest(index=index, value=value):
                body = body_fixture()
                body.parents[index] = value
                original = body.parents
                with self.assertRaisesRegex(ValueError, 'Invalid parent topology'):
                    cache_static_topology(body)
                self.assertIs(body.parents, original)

    def test_cpu_parameter_rejected_without_mutation(self):
        body = body_fixture()
        body.betas = torch.nn.Parameter(torch.zeros(1, 10))
        parents, betas = body.parents, body.betas
        with self.assertRaisesRegex(ValueError, 'must remain on CUDA'):
            cache_static_topology(body)
        self.assertIs(body.parents, parents)
        self.assertIs(body.betas, betas)

    def test_cpu_float_buffer_rejected_without_mutation(self):
        body = body_fixture()
        body.template = torch.zeros(1, 3)
        parents, template = body.parents, body.template
        with self.assertRaisesRegex(ValueError, 'must remain on CUDA'):
            cache_static_topology(body)
        self.assertIs(body.parents, parents)
        self.assertIs(body.template, template)


if __name__ == '__main__':
    unittest.main()
