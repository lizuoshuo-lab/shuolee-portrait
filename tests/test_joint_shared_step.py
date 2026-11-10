import unittest

import numpy as np

from shuolee_portrait.joint_shared_step import project_halfspaces, row_constraints


class SharedStepTests(unittest.TestCase):
    def test_active_constraint_keeps_a_coupled_descent_direction(self):
        normals = np.array([[[1., 1.]], [[1., 1.]]])
        matrix, owners, metrics, norm = row_constraints(normals, np.ones((2, 1), bool), 1)
        np.testing.assert_array_equal(owners, [0, 1])
        np.testing.assert_array_equal(metrics, [0, 0])
        matrix = matrix.multiply(1/norm[:, None]).tocsr()
        direction, proof = project_halfspaces(np.array([1., 0., 0.]), matrix, np.zeros(2))
        np.testing.assert_allclose(direction, [1/3, -1/3, -1/3], atol=2e-5)
        self.assertTrue(proof['certified'])

    def test_unobserved_frame_has_no_constraint_on_its_compensation(self):
        normals = np.array([[[1., 0.]], [[1., 1.]]])
        matrix, _, _, _ = row_constraints(normals, np.array([[True], [False]]), 1)
        direction, proof = project_halfspaces(np.array([.7, .3, .5]), matrix, np.array([.2]))
        np.testing.assert_allclose(direction, [.2, .3, .5], atol=2e-5)
        self.assertTrue(proof['certified'])
