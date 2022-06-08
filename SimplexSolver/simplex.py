import numpy as np

from SimplexSolver.utils import get_matrix_by_list_index


class SimplexPrimal:
    def __init__(self, linear_coefficients_matrix,
                 linear_constraints_vector,
                 linear_cost_vector):
        self.reached_solution = False
        self.slack_variables = None
        self.current_cost_vector = None
        self.current_base_matrix = None
        self.current_basic_solution = None
        self.current_non_base_matrix = None
        self.current_inv_base_matrix = None
        self.current_base_cost_vector = None
        self.current_base_index_vector = None
        self.current_constraints_vector = None
        self.linear_cost_vector_expanded = None
        self.current_coefficients_matrix = None
        self.current_non_base_index_vector = None
        self.linear_coefficients_matrix_expanded = None
        self.linear_coefficients_matrix = np.array(linear_coefficients_matrix)
        self.linear_constraints_vector = np.array(linear_constraints_vector)
        self.linear_cost_vector = np.array(linear_cost_vector)
        self.check()

    def check(self):
        invalid_vector_index, _ = np.where(self.linear_constraints_vector < 0)
        if len(invalid_vector_index) > 0:
            raise ValueError("All values in the constraints vector must be grater than or equal to 0.")

    def frame_side_problem(self):
        self.slack_variables = np.eye(self.linear_constraints_vector.shape[0])
        self.linear_coefficients_matrix_expanded = np.column_stack(self.linear_coefficients_matrix,
                                                                   self.slack_variables)
        cost_vector_expanded = np.zeros(shape=(1, self.linear_coefficients_matrix.shape[1]))
        slack_vector_expanded = np.ones(shape=(1, self.slack_variables.shape[1]))
        self.linear_cost_vector_expanded = np.concatenate((cost_vector_expanded, slack_vector_expanded))

    def init_solver(self):
        self.reached_solution = False
        self.current_base_matrix = None
        self.current_basic_solution = None
        self.current_non_base_matrix = None
        self.current_base_cost_vector = None
        self.current_base_index_vector = None
        self.current_non_base_index_vector = None
        self.current_cost_vector = self.linear_cost_vector
        self.current_constraints_vector = self.linear_constraints_vector
        self.current_coefficients_matrix = self.linear_coefficients_matrix

    def solve(self, side_problem=False):
        self.init_solver()
        if side_problem:
            self.current_cost_vector = self.linear_cost_vector_expanded
            self.current_coefficients_matrix = self.linear_coefficients_matrix_expanded
        while self.reached_solution:
            self.current_base_matrix = self.get_current_base_matrix()
            self.current_non_base_matrix = self.get_current_non_base_matrix()
            self.current_basic_solution = self.calculate_current_basic_solution()
            self.check_viability()
            index_base_in, index_base_out, viability = self.get_base_change_indexes()
            if not viability:
                pass
            self.update_current_inv_base_matrix(base_change_in=index_base_in,
                                                base_change_out=index_base_out)
            self.update_base_and_non_base_index_vector()
            self.check_is_optimal()

    def get_current_base_matrix(self):
        return get_matrix_by_list_index(self.current_coefficients_matrix, self.current_base_index_vector)

    def get_current_non_base_matrix(self):
        return get_matrix_by_list_index(self.current_coefficients_matrix, self.current_non_base_index_vector)

    def calculate_current_basic_solution(self):
        self.get_current_inv_base_matrix(base_change=None)
        return np.dot(self.current_inv_base_matrix, self.current_base_matrix)

    def get_current_inv_base_matrix(self, base_change):
        pass

    def check_viability(self):
        pass

    def get_reduced_cost(self):
        return np.array([])

    def get_base_change_indexes(self):
        return _, _, _

    def update_base_and_non_base_index_vector(self):
        pass

    def update_current_inv_base_matrix(self, base_change_in, base_change_out):
        pass

    def check_is_optimal(self):
        pass
