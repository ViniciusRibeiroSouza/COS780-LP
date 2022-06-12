import numpy as np

from SimplexSolver.statandardFormLP import StandardFormLP

A = [[-2, 3, -10, 0],
     [3, 2, 0, 15]]
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
var = model.get_standard_lp_form()
print(var)