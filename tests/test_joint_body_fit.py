import unittest

import torch

from shuolee_portrait.joint_body_fit import _project_feasible_step


class JointBlockProjectionTests(unittest.TestCase):
    def test_projection_restricts_only_the_threatened_rows_and_directions(self):
        update = torch.tensor([[3., 4., 2.], [3., 4., 2.]])
        normals = torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                                [[0., 0., 0.], [0., -1., 0.]]])
        slack = torch.tensor([[1., 0.], [0., 1.]])
        projected = _project_feasible_step(update, normals, slack, sweeps=4)
        torch.testing.assert_close(projected, torch.tensor([[1., 0., 2.], [3., 4., 2.]]))

    def test_multiple_face_constraints_preserve_a_feasible_tangent_update(self):
        update = torch.tensor([[4., 2., 1.]])
        normals = torch.tensor([[[1., 1., 0.], [1., 0., 0.]]])
        slack = torch.zeros((1, 2))
        projected = _project_feasible_step(update, normals, slack, sweeps=4)
        self.assertTrue(torch.all(torch.einsum('bid,bd->bi', normals, projected) <= slack))
        torch.testing.assert_close(projected, torch.tensor([[0., -1., 1.]]))
