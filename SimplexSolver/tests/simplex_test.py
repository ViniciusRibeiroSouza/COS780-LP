import unittest

import numpy as np


from SimplexSolver.simplex import SimplexPrimal


class SimplexTest(unittest.TestCase):

    def test_check_linear_constraints_vector_values_grater_than_zero(self):
        simplex_solver = simplex_solver_base()
        self.assertEqual(simplex_solver.check(), None)

    def test_check_linear_constraints_vector_values_less_than_zero(self):
        simplex_solver = simplex_solver_base()
        simplex_solver.linear_constraints_vector = np.array([-1, 0, 1])
        with self.assertRaises(ValueError):
            simplex_solver.check()


def simplex_solver_base():
    linear_coefficients_matrix = np.ones(shape=(2, 2))
    linear_constraints_vector = np.ones(shape=(2, 1))
    linear_cost_vector = np.ones(shape=(1, 2))
    simplex_solver = SimplexPrimal(linear_coefficients_matrix,
                                   linear_constraints_vector,
                                   linear_cost_vector)
    return simplex_solver


if __name__ == '__main__':
    unittest.main()
