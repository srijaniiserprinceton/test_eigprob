import scipy.sparse
import scipy.sparse.linalg
import numpy as np
# import tensorflow as tf

class eigprob_solver:
    def __init__(self, solver_name='numpy', diag_shift=5):
        self.n = None      # size of each matrix
        self.mat = None    # the matrix 
        self.solver_name = solver_name # the solver to use
        self.solver = None  # solver function callable
        self.solve_eigprob() #assigning solver
        self.diag_shift = diag_shift


    def build_sparse_matrix(self, size):
        '''creating the sparse matrix of a desired size'''
        self.n = size
        # generating a completely filled random matrix
        sparse_mat = scipy.sparse.random(self.n, self.n, density=1)
        # converting to dense representation
        dense_mat = sparse_mat.todense()
        # making it symmetric (hermitian)
        dense_mat = 0.5 * (dense_mat + dense_mat.T)

        # making the same kind of sparse structure as supermatrix
        supmat = np.zeros_like(dense_mat)
        supmat += np.diag(np.diag(dense_mat))
        supmat += np.diag(np.diag(dense_mat,k=self.diag_shift),k=self.diag_shift)
        supmat += np.diag(np.diag(dense_mat,k=-self.diag_shift),k=-self.diag_shift)

        return supmat


    def solve_eigprob(self):
        '''choosing the solver and calling respective function'''
        # setting solver to numpy.linalg.eigh
        if(self.solver_name == 'numpy'):
            self.solver = np.linalg.eigh

        # setting solver to scipy.linalg.eigh
        elif(self.solver_name == 'scipy'):
            self.solver = scipy.linalg.eigh

        # setting solver to scipy.sparse.linalg.eigsh
        elif(self.solver_name == 'scipy_sparse'):
            self.solver = scipy.sparse.linalg.eigsh

        else:
            self.solver = None
