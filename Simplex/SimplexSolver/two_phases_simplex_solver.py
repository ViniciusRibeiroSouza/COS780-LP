import numpy as np
# from numba import jit
from Simplex.SimplexSolver.standard_linear_programming_form import StandardFormLP


class TwoPhaseSimplexSolver:
    # BFS = Base Feasible Set
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.base_feasible_set = None
        self.A_expanded = None
        self.artificial_vars = None

    def solve(self, model: StandardFormLP):
        """ Solves a simplex instance. """
        self.A_expanded, self.artificial_vars, self.base_feasible_set = model.get_standard_lp_form()
        rows, columns = self.A_expanded.shape
        # Phase 1 (Only if we have artificial variables)
        if len(self.artificial_vars) != 0:
            is_feasible, table_vals = _phase1_simplex(self.A_expanded, self.artificial_vars,
                                                      self.base_feasible_set, self.verbose)
            self.A_expanded, self.base_feasible_set = table_vals
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


# @jit(nopython=True)
def _phase2_simplex(matrix_a, cost_vector, base_feasible_set, verbose):
    # Phase 2 of the Two-Phase simplex algorithm. Assumes the table is starting at a base_feasible_set.
    if verbose:  # todo change this to print matrix
        pass
    obj = _calc_objective_function_value(matrix_a, cost_vector, base_feasible_set)
    matrix_a, obj, base_feasible_set, bounded = _simplex(matrix_a, obj, base_feasible_set, True)
    return matrix_a, obj, base_feasible_set, bounded


# @jit(nopython=True)
def _phase1_simplex(matrix_a, artificial_vars, base_feasible_set, verbose=False):
    # Phase 1 of the Two-Phase SimplexSolver Algorithm
    if verbose:  # todo change this to print matrix
        pass
    cost_vector = _create_aux_slack_cost_function(matrix_a, artificial_vars, base_feasible_set)
    matrix_a, cost_vector, base_feasible_set, _ = _simplex(matrix_a, cost_vector, base_feasible_set, verbose)
    # If the objective value is close enough to 0, it is feasible.
    if np.isclose(cost_vector[-1], 0):
        return True, (matrix_a, base_feasible_set)
    else:
        return False, (None, None)


# @jit(nopython=True)
def _create_aux_slack_cost_function(matrix_a, artificial_vars, base_feasible_set):
    # Create the objective function
    n_cols = matrix_a.shape[1]
    aux_slack_cost_vector = np.zeros(n_cols - 1)
    artificial_cols = list(map(lambda x: x[1], artificial_vars))
    aux_slack_cost_vector[artificial_cols] = -1
    return _calc_objective_function_value(matrix_a, aux_slack_cost_vector, base_feasible_set)


# @jit(nopython=True)
def _simplex(matrix_a, objective_vector, base_feasible_set, verbose):
    # The simplex algorithm. Takes a bfs as input. Uses Bland's rule to avoid cycling.
    # Should only take in feasible problems.
    while True:
        if verbose:  # todo change this to print matrix
            pass
        # Find the variable to enter the basis. Using Bland's Rule (select the first)
        negatives = np.where(objective_vector[:-1] < 0)[0]
        if len(negatives) == 0:
            break
        new_basis = negatives[0]

        # Find the variable to leave the basis. Bland's Rule.
        row = -1
        min_cost = float('Inf')
        for i in range(matrix_a.shape[0]):
            if matrix_a[i, new_basis] > 0:
                cost = matrix_a[i, -1] / matrix_a[i, new_basis]
                if cost < min_cost:
                    row = i
                    min_cost = cost
        if row == -1:
            return matrix_a, objective_vector, base_feasible_set, False
        to_leave = list(filter(lambda x: x[0] == row, base_feasible_set))
        matrix_a, objective_vector = _pivot(matrix_a, objective_vector, row, new_basis)
        assert len(to_leave) == 1
        base_feasible_set.remove(to_leave[0])
        base_feasible_set.append((row, new_basis))
        if verbose:
            print('Removing', to_leave[0], 'Adding', new_basis)
    return matrix_a, objective_vector, base_feasible_set, True


# @jit(nopython=True)
def _calc_objective_function_value(table, c, base_feasible_set):
    n, m = table.shape
    obj = np.append(c, 0)
    for coord in base_feasible_set:
        row, col = coord
        obj = obj - obj[col] * table[row, :]
    obj = -1 * obj  # TODO
    return obj


# @jit(nopython=True)
def _pivot(table, obj, row, column):
    # Row Reduction
    table[row, :] = table[row, :] / table[row, column]
    rows, cols = table.shape
    for r in range(rows):
        if r != row:
            table[r, :] = table[r, :] - table[r, column] * table[row, :]
            obj = obj - obj[column] * table[row, :]
    return table, obj
