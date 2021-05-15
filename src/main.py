import numpy as np
import time

def time_eigpob_solver(N_arr=np.array([10],dtype='int'),solvers=np.array(['numpy']),tot_iter=10):
    # creating matrix to store the timings
    time_arr = np.zeros((len(solvers),len(N_arr)),dtype='float32')

    # looping over different solvers
    for solver_ind, solver in solvers:
        # initializing the class for solving using specified solvers
        time_eigprob_solver = eigprob_solver(solver)
        # looping over different matrix sizes 
        for size_ind, size in N_arr:
            # specifying the size of matrix and buliding it
            time_eigprob_solver.mat = build_sparse_matrix(size)
            
            # timing the solver
            T1 = time.time()
            for Niter in range(tot_iter): time_eigprob_solver.solve_eigprob()
            T2 = time.time()

            time_arr[solver_ind, size_ind] = (T2 - T1)/tot_iter

    # returning the timed array
    return time_arr
            
