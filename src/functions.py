import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import tensorflow as tf

class eigprob_solver:
    def __init__(self, solver='numpy'):
        self.n = None      # size of each matrix
        self.mat = None    # the matrix 
        self.solver = solver # the solver to use

    def build_sparse_matrix(self,size):
        '''creating the sparse matrix of a desired size
        '''
        self.n = size
        sparse_mat = scipy.sparse.random(self.n,self.n)
        dense_mat = sparse_mat.todense()
        return dense_mat


        def solve_eigprob(self):
            '''choosing the solver and calling respective function
            '''

            if(self.solver == 'numpy'):          # for numpy.linalg.eigh
                self.solver = np.linalg.eigh

            elif(self.solver == 'scipy'):        # for scipy.linalg.eigh
                self.solver = scipy.linalg.eigh

            elif(self.solver == 'scipy_sparse'): # for scipy.sparse.linalg.eigsh
                self.solver = scipy.sparse.linalg.eigsh

            elif(self.solver == 'tensorflow'):   # for tensorflow.linalg.eigh
                self.solver = tf.linalg.eigh

            else:                                # for cupy.linalg.eigh (not yet available)
                self.solver = None
