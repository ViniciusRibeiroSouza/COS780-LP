import numpy as np


class StandardFormLP:
    # Todo adjust documentation and variables behavior.
    def __init__(self, a_matrix, b_matrix, c_matrix, signs=None, directions=None, max_or_min=None, standard_obj='max'):
        """
        Constructs a StandardFormLP from a Generic LP.
            max c^T x (can also be min, if standard_obj='min')
            s.t. Ax = b
            x >= 0
         """
        assert max_or_min == 'max' or max_or_min == 'min'
        self.original_objective = max_or_min
        self.vars = {i: str(i) for i in range(a_matrix.shape[1])}
        self.standard_form_obj = standard_obj
        self.slacks_variables = []

        # Variables of a Standard Form LP.
        self.A = a_matrix
        self.b = b_matrix
        self.c = c_matrix if max_or_min == standard_obj else -1 * c_matrix
        self._fix_var_domains(signs)
        self._add_slack_vars(directions)

    def _fix_var_domains(self, domain_restrictions):
        """
        Alters the A and c matrix so that all variables have domain x >= 0.
        """
        # Todo: Adjust problem if variables are free
        for i, variable_restriction in enumerate(domain_restrictions):
            if variable_restriction == '<':
                self.c[i] *= -1
                self.A[:, i] *= -1
                self.vars[i] = '-' + self.vars[i]
            elif variable_restriction == '=':
                # We must cancel column value if Xi=0
                self.c = np.append(self.c, -1 * self.c[i])
                transposed_col = np.array([-1 * self.A[:, i]])
                self.A = np.append(self.A, transposed_col.T, 1)
                self.vars[i] = self.vars[i] + '-' + str(self.A.shape[1] - 1)

    def _add_slack_vars(self, restriction_symbols):
        """
        Adds slack variables to transform inequalities to equalities.
        """
        for i, restriction_symbol in enumerate(restriction_symbols):
            rows = self.A.shape[0]
            if restriction_symbol == '>':
                self.A = np.append(self.A, -1 * np.zeros(shape=[rows, 1]), 1)
                self.A[i, -1] = -1
                self.c = np.append(self.c, 0)
                self.slacks_variables.append(self.A.shape[1] - 1)
            elif restriction_symbol == '<':
                self.A = np.append(self.A, 1 * np.zeros(shape=[rows, 1]), 1)
                self.A[i, -1] = 1
                self.c = np.append(self.c, 0)
                self.slacks_variables.append(self.A.shape[1] - 1)

    def get_standard_lp_form(self):
        """
        Generates a table for Phase 1 of the simplex algorithm.
        Adds artificial variables as needed.
        """
        artificial_variables = []
        bfs = []
        np_array_b = np.array(self.b)
        rows, cols = self.A.shape
        num_artificial = min(rows, cols)
        np_array_a = np.array(self.A)
        for row, col in enumerate(self.slacks_variables):
            if np_array_a[row, col] == -1 and np_array_b[row] < 0:
                np_array_a[row] = -1 * np_array_a[row]
                np_array_b[row] = -1 * np_array_b[row]
                num_artificial -= 1
                bfs.append((row, col))
            elif np_array_a[row, col] == 1 and np_array_b[row] > 0:
                bfs.append((row, col))
                num_artificial -= 1
        np_array_a = np.append(np_array_a, np.zeros(shape=[rows, num_artificial]), 1)
        # Add artificial variables
        rows, cols = np_array_a.shape
        bfs_rows = set(map(lambda x: x[0], bfs))
        artificial_val = 0
        for row_index in range(rows):
            if row_index in bfs_rows:
                continue
            column_index = cols - num_artificial + artificial_val
            artificial_variables.append((row_index, column_index))
            bfs.append((row_index, column_index))
            if np_array_a[row_index, -1] < 0:
                np_array_a[row_index, :] = -1 * np_array_a[row_index, :]
            np_array_a[row_index, column_index] = 1
            artificial_val += 1
        return np.column_stack((np_array_a, np_array_b)), artificial_variables, bfs

    def are_dependent_constraints(self):
        number_of_rows, number_of_columns = self.A.shape
        if number_of_rows <= number_of_columns and number_of_rows != np.linalg.matrix_rank(self.A):
            return True
        return False

    def convert_variables(self, x):
        """Given a solution to the standard LP x, return the values of the original variables"""
        assert len(x) == self.A.shape[1]
        ans = np.zeros(len(self.vars))
        for i in range(len(self.vars)):
            standard_vars = self.vars[i].split('-')
            if standard_vars[0] != '':
                ans[i] += x[int(standard_vars[0])]
            if len(standard_vars) > 1:
                ans[i] -= x[int(standard_vars[1])]
        assert len(ans) == len(self.vars)
        return ans

    def convert_objective(self, obj):
        """Given an objective value to the standard LP obj, return the values"""
        return obj if self.original_objective == self.standard_form_obj else -1 * obj
