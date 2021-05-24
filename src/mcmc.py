import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys
sys.path.append('./src')
from functions import eigprob_solver as EGS
from schwimmbad import MPIPool
from mpi4py import MPI


NDIM = 2
NWALKERS = 24
MAXITER = 4
MATSIZE = 2048
SOLVERS = np.array(['numpy', 'scipy', 'scipy_sparse']) #, 'tensorflow'])
GLOBAL_COUNTER = 0



def log_likelihood(theta, x, y, yerr):
    eigSolver = EGS(solver_name=SOLVERS[0])
    m, b = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * np.log(f_true))
    eigSolver.mat = eigSolver.build_sparse_matrix(MATSIZE)
    eigSolver.solver(eigSolver.mat)
    # GLOBAL_COUNTER = GLOBAL_COUNTER + 1
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)



if __name__ == "__main__":
    np.random.seed(123)
    GLOBAL_COUNTER = 0

    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true
    y += np.abs(y * f_true) * np.random.randn(N)
    y += yerr * np.random.randn(N)

    x0 = np.linspace(0, 10, 500)


    theta_init = np.array([m_true, b_true]).reshape(1, NDIM)
    pos = theta_init + 1e-4 * np.random.randn(NWALKERS, NDIM)

    with MPIPool() as pool:
        GLOBAL_COUNTER = 0
        comm = MPI.COMM_WORLD
        mpirank = comm.Get_rank()
        print(f"Process {mpirank:3d}: Running MPI")
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(NWALKERS, NDIM,
                                        log_probability,
                                        args=(x, y, yerr),
                                        pool=pool)
        sampler.run_mcmc(pos, MAXITER, progress=True)
    print(f"Total count = {GLOBAL_COUNTER}")

