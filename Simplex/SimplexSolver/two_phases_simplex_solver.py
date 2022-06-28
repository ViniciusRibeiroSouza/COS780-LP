import pprint
import numpy as np
from Simplex.SimplexSolver.standard_linear_programming_form import StandardFormLP


class TwoPhaseSimplexSolver:
    # BFS = Base Feasible Set
    def __init__(self, verbose=True):
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

    def _phase1_simplex(self, verbose=True):
        # Phase 1 of the Two-Phase SimplexSolver Algorithm
        n_variables = self.A_expanded.shape[1]
        cost_vector = _create_aux_slack_cost_function(n_variables, self.artificial_vars_index)
        if verbose:
            pprint.pprint(f"Phase 1: Cost Vector: {cost_vector}")
        cost_vector, current_solution, self.current_base_index, self.current_non_base_index, is_feasible = self._simplex(cost_vector, verbose)
        # If the objective value is close enough to 0, it is feasible.
        if np.min(cost_vector) >= 0 and is_feasible:
            return True
        else:
            return False

    def _phase2_simplex(self, cost_vector, verbose=True):
        # Phase 2 of the Two-Phase simplex algorithm. Assumes the table is starting at a base_feasible_set.
        if verbose:
            phase2_print(cost_vector)
        _, current_solution, base_index, non_base_index, is_feasible = self._simplex(cost_vector, verbose)
        return current_solution, base_index, non_base_index, is_feasible

    def _simplex(self, current_cost_vector, verbose):
        # The simplex algorithm. Uses Bland rule to avoid cycling.
        is_feasible = True
        iterations = 10e3
        reduced_cost_vector = None
        current_base_solution = None
        self.inverse_base_matrix_update()
        while True:
            iterations -= 1
            if iterations <= 0:
                is_feasible = False
                break
            if verbose:
                self.print_simplex_step(current_cost_vector=current_cost_vector)
            current_base_solution = np.matmul(self.inverse_base_matrix, self.current_b)
            _check_current_base_solution_is_valid(current_base_solution)
            reduced_cost_vector = _get_reduced_cost_vector(self.A_expanded, current_cost_vector,
                                                           self.current_non_base_index, self.current_base_index,
                                                           self.inverse_base_matrix)
            interrupt, index_entering_base = get_base_leaving_index(reduced_cost_vector)
            if interrupt:
                break
            index_to_leave_base = get_entering_base_index(index_entering_base, current_base_solution,
                                                          self.inverse_base_matrix, self.A_expanded)
            self.index_update(index_to_leave_base, index_entering_base)
            self.inverse_base_matrix_update(index_to_leave_base, index_entering_base)
        return reduced_cost_vector, current_base_solution, self.current_base_index, self.current_non_base_index, is_feasible

    def print_simplex_step(self, current_cost_vector):
        pprint.pprint(f" ------------- Simplex Step ---------------")
        pprint.pprint(f"Cost Vector: {current_cost_vector}")
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
        try:
            # Phase 1 (Only if we have artificial variables)
            if len(self.artificial_vars) != 0:
                is_feasible = self._phase1_simplex(verbose=self.verbose)
                if not is_feasible:
                    return 'Infeasible', None, None
                # Remove artificial variables
                self.A_expanded = self.A_expanded[:, 0:columns - len(self.artificial_vars)]
                self.current_non_base_index = _get_non_base_index(self.current_base_index, self.A_expanded.shape[1])
            # Phase 2
            cost_vector = np.asarray([model.c])
            if cost_vector.shape[1] == 1:
                cost_vector = cost_vector.transpose()
            current_solution, self.current_base_index, self.current_non_base_index, is_feasible = \
                self._phase2_simplex(cost_vector, self.verbose)
            if is_feasible:
                final_cost_value = _get_final_cost_value(current_solution, cost_vector, self.current_base_index, model.original_objective)
                if self.verbose:
                    print_final_cost(self.current_base_index, current_solution, final_cost_value)
                return 'Solved', final_cost_value, current_solution
            else:
                return 'Unbounded', None, None
        except Exception as error:
            if type(error).__name__ == ValueError.__name__:
                pprint.pprint(error)
                return 'Unbounded', None, None
            else:
                pprint.pprint(f"Got exception error: {error}")

    def inverse_base_matrix_update(self, index_to_leave_base=None, index_entering_base=None):
        matrix_a_base_cols = _get_array_from_index(self.A_expanded, self.current_base_index)
        self.inverse_base_matrix = np.linalg.inv(matrix_a_base_cols)


def print_final_cost(current_base_index, current_solution, final_cost_value):
    pprint.pprint(f" ------------- Final Step ---------------")
    pprint.pprint(f"Final Cost: {final_cost_value}")
    pprint.pprint(f"Final Solution: {current_solution}")
    pprint.pprint(f"Final base index: {current_base_index}")


def phase2_print(cost_vector):
    pprint.pprint(f"-----------------------------------")
    pprint.pprint(f"Phase 2: Cost Vector: {cost_vector}")


def _check_current_base_solution_is_valid(current_base_solution):
    if np.min(current_base_solution) < 0:
        raise ValueError("Solution must be grater than or equal 0.")


def _get_final_cost_value(current_solution, cost_vector, current_base_index, max_or_min):
    current_cost_base_index = _get_array_from_index(cost_vector, current_base_index)
    partial_result = np.matmul(current_cost_base_index, current_solution)
    if max_or_min == "max":
        partial_result *= -1
    return partial_result


def _create_aux_slack_cost_function(n_variables, artificial_cols, high_cost_const=10e3):
    # Create the objective function
    aux_slack_cost_vector = np.zeros(n_variables)
    aux_slack_cost_vector[artificial_cols] = 1 * high_cost_const
    return np.array([aux_slack_cost_vector])


def _get_array_from_index(array, list_of_index):
    return np.take(array.copy(), list_of_index, axis=1)


def _get_reduced_cost_vector(matrix_A, cost_vector, non_base_index, base_index, inv_matrix_base):
    vector_cost_non_base_vars = _get_array_from_index(cost_vector, non_base_index)
    vector_cost_base_vars = _get_array_from_index(cost_vector, base_index)
    matrix_a_non_base_vars = _get_array_from_index(matrix_A, non_base_index)
    first_mat_multiplication = np.matmul(vector_cost_base_vars, inv_matrix_base)
    second_mat_multiplication = np.matmul(first_mat_multiplication, matrix_a_non_base_vars)
    reduce_cost_transposed = vector_cost_non_base_vars - second_mat_multiplication
    return reduce_cost_transposed


def _get_non_base_index(base_index, n_cols_matrix_a):
    return [i for i in range(n_cols_matrix_a) if i not in base_index]


def _get_values_from_set(base_feasible_set):
    return list(map(lambda x: x[1], base_feasible_set))


def filter_ratio_test_for_non_negative_variables(vector_1, vector_2):
    removed_index = 0
    for index, item in enumerate(vector_2):
        if item[0] <= 0:
            vector_2 = np.delete(vector_2, index)
            vector_1 = np.delete(vector_1, index)
            removed_index += 1
    return vector_1, vector_2, removed_index


def get_entering_base_index(index_to_leave_base, current_base_solution, inverse_base_vector, matrix_a):
    # Implements ration test to get the index og the column entering the base
    leaving_base_column = _get_array_from_index(matrix_a, index_to_leave_base)
    leaving_base_column = np.array([leaving_base_column]).transpose()
    variable_cost_column = np.matmul(inverse_base_vector, leaving_base_column)
    # ratio test
    ratio_list = []
    index_list = []
    for index, item in enumerate(variable_cost_column):
        if item > 0:
            ratio = current_base_solution[index] / item
            ratio_list.append(ratio)
            index_list.append(index)
    min_ration_index = np.argmin(ratio_list)
    index_entering_base = index_list[min_ration_index]
    return index_entering_base


def get_base_leaving_index(reduced_cost):
    cost_negative = reduced_cost * -1
    index_colum_to_leave = np.argmax(cost_negative)
    if cost_negative[0][index_colum_to_leave] <= 0:
        return True, None
    return False, index_colum_to_leave


def replace_item_array(array, item_out, item_in):
    for index, val in enumerate(array):
        if val == item_out:
            array[index] = item_in
