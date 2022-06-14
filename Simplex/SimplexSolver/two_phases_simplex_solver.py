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
        self.vector_cost = None
        self.artificial_vars = None
        self.base_feasible_set = None
        self.current_base_index = None
        self.inverse_base_matrix = None
        self.artificial_vars_index = None
        self.current_non_base_index = None

    def index_update(self, index_to_leave_base, index_entering_base):
        leave_base = self.current_base_index[index_to_leave_base]
        entering_base = self.current_non_base_index[index_entering_base]
        replace_item_array(self.current_base_index, leave_base, entering_base)
        replace_item_array(self.current_non_base_index, entering_base, leave_base)

    def _phase1_simplex(self, verbose=False):
        # Phase 1 of the Two-Phase SimplexSolver Algorithm
        n_variables = self.A_expanded.shape[1]
        cost_vector = _create_aux_slack_cost_function(n_variables, self.artificial_vars_index)
        if verbose:
            pprint.pprint(f"Phase 1: Cost Vector: {cost_vector}")
        cost_vector, self.current_base_index, self.current_non_base_index, is_feasible = self._simplex(cost_vector,
                                                                                                       verbose)
        # If the objective value is close enough to 0, it is feasible.
        if np.min(cost_vector) >= 0 and is_feasible:
            return True
        else:
            return False

    def _phase2_simplex(self, cost_vector, verbose=False):
        # Phase 2 of the Two-Phase simplex algorithm. Assumes the table is starting at a base_feasible_set.
        if verbose:
            pprint.pprint(f"Phase 2: Cost Vector: {cost_vector}")
        result_cost_vector, base_index, non_base_index, is_feasible = self._simplex(cost_vector, True)
        return result_cost_vector, base_index, non_base_index, is_feasible

    def _simplex(self, vector_Cost, verbose):
        # The simplex algorithm. Uses Bland's rule to avoid cycling.
        is_feasible = True
        iterations = 10e3
        reduced_cost_vector = None
        self.inverse_base_matrix_update()
        while True:
            iterations -= 1
            if iterations <= 0:
                is_feasible = False
                break
            if verbose:
                self.print_simplex_step()
            current_base_solution = np.matmul(self.inverse_base_matrix, self.current_b)
            reduced_cost_vector = _get_reduced_cost_vector(self.A_expanded, vector_Cost, self.current_non_base_index,
                                                           self.current_base_index, self.inverse_base_matrix)
            interrupt, index_entering_base = get_base_leaving_index(reduced_cost_vector)
            if interrupt:
                break
            index_to_leave_base = get_entering_base_index(index_entering_base, current_base_solution,
                                                          self.inverse_base_matrix, self.A_expanded)
            self.index_update(index_to_leave_base, index_entering_base)
            self.inverse_base_matrix_update(index_to_leave_base, index_entering_base)
        return reduced_cost_vector, self.current_base_index, self.current_non_base_index, is_feasible

    def print_simplex_step(self):
        pprint.pprint(f"Cost Vector: {self.vector_cost}")
        pprint.pprint(f"Matrix A: {self.A_expanded}")
        pprint.pprint(f"Base index: {self.current_base_index} | Non Base index: {self.current_non_base_index}")

    def solve(self, model: StandardFormLP):
        """ Solves a simplex instance. """
        self.A_expanded, self.current_b, self.artificial_vars, self.base_feasible_set = model.get_standard_lp_form()
        rows, columns = self.A_expanded.shape
        self.vars = {i: str(i) for i in range(self.A_expanded.shape[1])}
        self.artificial_vars_index = _get_values_from_set(self.artificial_vars)
        self.current_base_index = _get_values_from_set(self.base_feasible_set)
        self.current_non_base_index = _get_non_base_index(self.current_base_index, columns)
        # Phase 1 (Only if we have artificial variables)
        if len(self.artificial_vars) != 0:
            is_feasible = self._phase1_simplex()
            if not is_feasible:
                return 'Infeasible', None, None
            # Remove artificial variables
            self.A_expanded = self.A_expanded[:, 0:columns - len(self.artificial_vars)]
            self.current_non_base_index = _get_non_base_index(self.current_base_index, self.A_expanded.shape[1])
        # Phase 2
        result_cost_vector, self.current_base_index, \
            self.current_non_base_index, is_feasible = self._phase2_simplex(model.c, self.verbose)
        if is_feasible:
            num_vars = self.A_expanded.shape[1] - 1
            x = np.zeros(num_vars)
            for i, coord in enumerate(self.base_feasible_set):
                row, col = coord
                x[col] = self.A_expanded[row, -1]
            return 'Solved'
        else:
            return 'Unbounded'

    def inverse_base_matrix_update(self, index_to_leave_base=None, index_entering_base=None):
        matrix_a_base_cols = _get_array_from_index(self.A_expanded, self.current_base_index)
        self.inverse_base_matrix = np.linalg.inv(matrix_a_base_cols)
        # if self.inverse_base_matrix is None:
        #     matrix_a_base_cols = _get_array_from_index(self.A_expanded, self.current_base_index)
        #     self.inverse_base_matrix = np.linalg.inv(matrix_a_base_cols)
        # else:
        #     pass


# @jit(nopython=True)
def _create_aux_slack_cost_function(n_variables, artificial_cols, high_cost_const=10e3):
    # Create the objective function
    aux_slack_cost_vector = np.zeros(n_variables)
    aux_slack_cost_vector[artificial_cols] = 1 * high_cost_const
    return np.array([aux_slack_cost_vector])


# @jit(nopython=True)
def _get_array_from_index(array, list_of_index):
    return np.take(array.copy(), list_of_index, axis=1)


# @jit(nopython=True)
def _get_reduced_cost_vector(matrix_A, vector_Cost, non_base_index, base_index, inv_matrix_base):
    vector_cost_non_base_vars = _get_array_from_index(vector_Cost, non_base_index)
    vector_cost_base_vars = _get_array_from_index(vector_Cost, base_index)
    matrix_a_non_base_vars = _get_array_from_index(matrix_A, non_base_index)
    first_mat_multiplication = np.matmul(vector_cost_base_vars, inv_matrix_base)
    second_mat_multiplication = np.matmul(first_mat_multiplication, matrix_a_non_base_vars)
    reduce_cost_transposed = vector_cost_non_base_vars - second_mat_multiplication
    return reduce_cost_transposed


# @jit(nopython=True)
def _get_non_base_index(base_index, n_cols_matrix_a):
    return [i for i in range(n_cols_matrix_a) if i not in base_index]


# @jit(nopython=True)
def _get_values_from_set(base_feasible_set):
    return list(map(lambda x: x[1], base_feasible_set))


# @jit(nopython=True)
def get_entering_base_index(index_to_leave_base, current_base_solution, inverse_base_vector, matrix_a):
    # Implements ration test to get the index og the column entering the base
    leaving_base_column = _get_array_from_index(matrix_a, index_to_leave_base)
    leaving_base_column = np.array([leaving_base_column]).transpose()
    variable_cost_column = np.matmul(inverse_base_vector, leaving_base_column)
    ratio = np.divide(current_base_solution, variable_cost_column)
    index_entering_base = np.argmin(np.abs(ratio))
    return index_entering_base


# @jit(nopython=True)
def get_base_leaving_index(reduced_cost):
    cost_negative = reduced_cost * -1
    index_colum_to_leave = np.argmax(cost_negative)
    if cost_negative[0][index_colum_to_leave] <= 0:
        return True, None
    return False, index_colum_to_leave


# @jit(nopython=True)
def replace_item_array(array, item_out, item_in):
    for index, val in enumerate(array):
        if val == item_out:
            array[index] = item_in
