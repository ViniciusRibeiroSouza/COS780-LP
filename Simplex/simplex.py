import numpy as np

from Simplex.SimplexSolver import StandardFormLP, TwoPhaseSimplexSolver

A = [[-2, 3, -1, 0],
     [3, 2, 0, -1]]
B = [[9],
     [12]]
C = [[2],
     [1],
     [0],
     [0]]
signs = ['>', '>', '>']
directions = ['=', '>']
model = StandardFormLP(a_matrix=np.array(A),
                       b_matrix=np.array(B),
                       c_matrix=np.array(C),
                       max_or_min='min',
                       signs=signs,
                       directions=directions)
solver = TwoPhaseSimplexSolver(verbose=False)
var_1, var_2, var_3 = solver.solve(model=model)
