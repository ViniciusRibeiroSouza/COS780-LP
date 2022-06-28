import numpy as np

from Simplex.SimplexSolver import StandardFormLP, TwoPhaseSimplexSolver


class LPModel:
    def __init__(self):
        self.A = []
        self.B = []
        self.C = []
        self.model = None
        self.signs = []
        self.max_or_min = None
        self.directions = []

    def build(self):

        self.model = StandardFormLP(a_matrix=np.array(self.A), b_matrix=np.array(self.B), c_matrix=np.array(self.C),
                                    max_or_min=self.max_or_min, signs=self.signs, directions=self.directions)

    def set_domain_variable(self, signs):
        valid_sign_list = ['>', '=', '<']
        for sign in signs:
            if sign not in valid_sign_list:
                raise ValueError(f"Sign must be of type: {valid_sign_list}")
            self.signs.append(sign)

    def set_constraint(self, vector, direction):
        valid_sign_list = ['>', '=', '<']
        if direction not in valid_sign_list:
            raise ValueError(f"Direction must be of type: {valid_sign_list}")
        self.A.append(vector)
        self.directions.append(direction)

    def set_cost_vector(self, vector):
        for item in vector:
            self.C.append([item])

    def set_bound_vector(self, vector):
        for item in vector:
            self.B.append([item])

    def set_objective(self, current_objective):
        valid_obj_list = ['min', 'max']
        if current_objective not in valid_obj_list:
            raise ValueError(f"Direction must be of type: {valid_obj_list}")
        self.max_or_min = current_objective

    def solve(self):
        solver = TwoPhaseSimplexSolver(verbose=True)
        solver.solve(model=self.model)
