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
signs = ['=', '<', '<']
directions = ['=', '=']

model = StandardFormLP(a_matrix=np.array(A),
                       b_matrix=np.array(B),
                       c_matrix=np.array(C),
                       max_or_min='max',
                       signs=signs,
                       directions=directions)
solver = TwoPhaseSimplexSolver(verbose=True)
solver.solve(model=model)

# Todo: 1. Finish phase 2 simplex
# Todo: 2. Finish inverse matrix update
# Todo: 3. Add free variable mode inverse matrix update
# Todo: 4. Finish code to work with max/min
# Todo: 5. Add Big M method
