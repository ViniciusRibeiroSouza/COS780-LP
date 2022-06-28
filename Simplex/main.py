import pprint
from SimplexSolver.simplex import LPModel

if __name__ == '__main__':
    linear_model = LPModel()
    linear_model.set_objective('min')
    linear_model.set_domain_variable(['>', '>'])
    linear_model.set_constraint([-2, 3], '>')
    linear_model.set_constraint([3, 2], '>')
    linear_model.set_bound_vector([9, 12])
    linear_model.set_cost_vector([2, 1, 0, 0])
    linear_model.build()

    modeling_problem = linear_model.model
    pprint.pprint(f"Matrix A: {modeling_problem.A}")
    pprint.pprint(f"Matrix B: {modeling_problem.b}")
    pprint.pprint(f"Matrix C: {modeling_problem.c}")
    linear_model.solve()
