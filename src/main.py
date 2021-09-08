import numpy as np
import time
import functions as fn

def time_eigprob_solver(N_arr=np.array([10], dtype='int'),
                        solvers=np.array(['numpy']),
                        diag_shift=5, tot_iter=10):
    # creating matrix to store the timings
    time_arr = np.zeros((len(solvers), len(N_arr)), dtype='float32')

    # looping over different solvers
    for solver_ind, solver in enumerate(solvers):
        # initializing the class for solving using specified solvers
        time_eigprob_solver = fn.eigprob_solver(solver_name=solver, diag_shift=diag_shift)
        # looping over different matrix sizes 
        for size_ind, size in enumerate(N_arr):
            # specifying the size of matrix and buliding it
            time_eigprob_solver.mat = time_eigprob_solver.build_sparse_matrix(size)
            
            # timing the solver
            T1 = time.time()
            for Niter in range(tot_iter):
                time_eigprob_solver.solver(time_eigprob_solver.mat)
            T2 = time.time()

            time_arr[solver_ind, size_ind] = (T2 - T1)/tot_iter

    # returning the timed array
    # size = (solver_ind, size_ind)
    return time_arr
