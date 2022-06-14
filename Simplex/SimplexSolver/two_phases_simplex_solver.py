import pprint

import numpy as np
# from numba import jit
from Simplex.SimplexSolver.standard_linear_programming_form import StandardFormLP


class TwoPhaseSimplexSolver:
    # BFS = Base Feasible Set
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.vars = None
        self.current_b = None
        self.A_expanded = None
        self.non_base_vars = None
        self.artificial_vars = None
        self.base_feasible_set = None

    def _phase1_simplex(self, artificial_vars_index, base_index, verbose=False):
        # Phase 1 of the Two-Phase SimplexSolver Algorithm
        n_variables = matrix_A.shape[1]
        cost_vector = _create_aux_slack_cost_function(n_variables, artificial_vars_index)
        if verbose:
            pprint.pprint(f"Phase 1: Cost Vector: {cost_vector}")
        cost_vector, base_index, non_base_index = _simplex(matrix_A, matrix_b, cost_vector,
                                                           base_index, verbose)
        # If the objective value is close enough to 0, it is feasible.
        if np.min(cost_vector) >= 0:
            return True, (base_index, non_base_index)
        else:
            return False, (None, None)

    def solve(self, model: StandardFormLP):
        """ Solves a simplex instance. """
        self.A_expanded, self.current_b, self.artificial_vars, self.base_feasible_set = model.get_standard_lp_form()
        rows, columns = self.A_expanded.shape
        self.vars = {i: str(i) for i in range(self.A_expanded.shape[1])}
        # Phase 1 (Only if we have artificial variables)
        if len(self.artificial_vars) != 0:
            self._phase1_simplex()
            if not is_feasible:
                return 'Infeasible', None, None
            # Remove artificial variables
            self.A_expanded = np.column_stack((self.A_expanded[:, 0:columns - len(self.artificial_vars) - 1],
                                               self.A_expanded[:, -1]))
        # Phase 2
        self.A_expanded, obj, self.base_feasible_set, bounded = _phase2_simplex(self.A_expanded, model.c,
                                                                                self.base_feasible_set, self.verbose)
        if bounded:
            num_vars = self.A_expanded.shape[1] - 1
            x = np.zeros(num_vars)
            for i, coord in enumerate(self.base_feasible_set):
                row, col = coord
                x[col] = self.A_expanded[row, -1]
            return 'Solved', obj[-1], x
        else:
            return 'Unbounded', None, None


# # @jit(nopython=True)
# def _phase2_simplex(matrix_a, cost_vector, base_feasible_set, verbose):
#     # Phase 2 of the Two-Phase simplex algorithm. Assumes the table is starting at a base_feasible_set.
#     if verbose:  # todo change this to print matrix
#         pass
#     obj = _calc_objective_function_value(matrix_a, cost_vector, base_feasible_set)
#     matrix_a, obj, base_feasible_set, bounded = _simplex(matrix_a, obj, base_feasible_set, True)
#     return matrix_a, obj, base_feasible_set, bounded


# @jit(nopython=True)



# @jit(nopython=True)
def _create_aux_slack_cost_function(n_variables, artificial_vars, high_cost_const=10e3):
    # Create the objective function
    aux_slack_cost_vector = np.zeros(n_variables)
    artificial_cols = list(map(lambda x: x[1], artificial_vars))
    aux_slack_cost_vector[artificial_cols] = 1 * high_cost_const
    return np.array(aux_slack_cost_vector)


def get_array_from_index(array, list_of_index):
    return np.take(array.copy(), list_of_index, axis=1)


def get_reduced_cost_vector(matrix_A, vector_Cost, non_base_index, base_index, inv_matrix_base):
    vector_cost_non_base_vars = get_array_from_index(vector_Cost, non_base_index)
    vector_cost_base_vars = get_array_from_index(vector_Cost, base_index)
    matrix_a_non_base_vars = get_array_from_index(matrix_A, non_base_index)
    first_mat_multiplication = np.matmul(vector_cost_base_vars, inv_matrix_base)
    second_mat_multiplication = np.matmul(first_mat_multiplication, matrix_a_non_base_vars)
    reduce_cost_transposed = vector_cost_non_base_vars - second_mat_multiplication
    return reduce_cost_transposed


# @jit(nopython=True)
def _simplex(matrix_A, vector_b, vector_Cost, base_feasible_set, verbose):
    # The simplex algorithm. Uses Bland's rule to avoid cycling.
    reduced_cost = None
    inv_matrix_base = None
    n_cols_matrix_a = matrix_A.shape[1]
    base_index = get_values_from_set(base_feasible_set)
    non_base_index = get_non_base_index(base_index, n_cols_matrix_a)
    while True:
        # Find the variable to enter the basis. Using Bland's Rule (select the first)
        inverse_base_matrix, inverse_base_vector = get_inverse_base_vector(base_index, matrix_A)
        current_base_solution = np.matmul(inverse_base_vector, vector_b)
        reduced_cost = get_reduced_cost_vector(matrix_A, vector_Cost,
                                               non_base_index, base_index, inverse_base_matrix)
        interrupt, arg = get_leaving_base_index(reduced_cost)
        if interrupt:
            break
        index_comparison = get_entering_base_index(arg, current_base_solution, inverse_base_vector, matrix_A)
        update_index_base_and_non_base(arg, base_index, index_comparison, non_base_index)
        print("OK")
    return reduced_cost, base_index, non_base_index


def get_non_base_index(base_index, n_cols_matrix_a):
    return [i for i in range(n_cols_matrix_a) if i not in base_index]


def get_values_from_set(base_feasible_set):
    return list(map(lambda x: x[1], base_feasible_set))


def update_index_base_and_non_base(arg, base_index, index_comparison, non_base_index):
    leave_base = base_index[index_comparison]
    entering_base = non_base_index[arg]
    base_index.remove(leave_base)
    non_base_index.remove(entering_base)
    base_index.append(entering_base)
    non_base_index.append(leave_base)


def get_entering_base_index(arg, current_base_solution, inverse_base_vector, matrix_a):
    a_i = get_array_from_index(matrix_a, arg)
    a_i = np.array([a_i]).transpose()
    variable_cost = np.matmul(inverse_base_vector, a_i)
    # teste da razao
    ratio = np.divide(current_base_solution, a_i)
    index_comparison = np.argmin(ratio)
    return index_comparison


def get_leaving_base_index(reduced_cost):
    arg = np.argmax(reduced_cost * -1)
    if reduced_cost[0][arg] >= 0:
        pass
    return True, arg


def get_inverse_base_vector(base_index, matrix_a):
    inverse_base_vector = get_array_from_index(matrix_a, base_index)
    inverse_base_matrix = np.linalg.inv(inverse_base_vector)
    return inverse_base_matrix, inverse_base_vector