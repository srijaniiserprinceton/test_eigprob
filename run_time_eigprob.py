import numpy as np
import sys
sys.path.append('./src')
import main as time_eig

# defining the sizes of the matrix
N_arr = np.logspace(1,3,3,dtype='int')

# defining the solvers to be used
solver_arr = np.array(['numpy','scipy','scipy_sparse','tensorflow'])

# solving and timing eigenvalue problem using different solvers
time_arr = time_eig.time_eigpob_solver(N_arr=N_arr, solvers=solver_arr)

for N_arr_ind, n in enumerate(N_arr):
    print('{n:>4}x{n:<4} matrix:')
    for solve_arr_ind, solver in enumerate(solver_arr):
        print('{solver:<12}: {time_arr[i]:7.4f} seconds.')
    print("{' ':*<30}")
