import scipy.sparse
import scipy.sparse.linalg
import numpy as np
# import tensorflow as tf

class eigprob_solver:
    def __init__(self, solver_name='numpy'):
        self.n = None      # size of each matrix
        self.mat = None    # the matrix 
        self.solver_name = solver_name # the solver to use
        self.solver = None  # solver function callable
        self.solve_eigprob() #assigning solver


    def build_sparse_matrix(self, size):
        '''creating the sparse matrix of a desired size'''
        self.n = size
        sparse_mat = scipy.sparse.random(self.n, self.n)
        dense_mat = sparse_mat.todense()
        return dense_mat


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

        # for tensorflow.linalg.eigh
        # elif(self.solver_name == 'tensorflow'):
        #     self.solver = tf.linalg.eigh

        # for cupy.linalg.eigh (not yet available)
        else:
            self.solver = None
