import numpy as np
import scipy.linalg
import  scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf
import time

# creating the sparse matrix of a desired size
def build_sparse_matrix(size):
    sparse_mat = scipy.sparse.random(size,size)
    dense_mat = sparse_mat.todense()
    return dense_mat

# solving the eigenvalue problem
def solve_eigprob(n):
    # getting the sparse matrix
    mat = build_sparse_matrix(n)
    # the total number of iterations to average the time over
    N_iter = 10

    # timing scipy.linalg.eigh
    t1a = time.time()
    for j in range(N_iter): evals, evecs = scipy.linalg.eigh(mat)
    t1b = time.time()

    # timing scipy.sparse.linalg.eigsh                                                           
    t2a = time.time()
    for j in range(N_iter): evals, evecs = scipy.sparse.linalg.eigsh(mat)
    t2b = time.time()

    t3a = time.time()
    mat = tf.convert_to_tensor(mat)
    for j in range(N_iter): evals, evecs = tf.linalg.eigh(mat)
    t3b = time.time()

    return ((t1b-t1a)/N_iter, (t2b-t2a)/N_iter, (t3b-t3a)/N_iter)

# timing computation of different size matrices                                                   
N = np.logspace(1,3.5,6,dtype='int')
print('--Martix size------non-sparse solver-------sparse solver---------tensorflow-----')
for i, n in enumerate(N):
    time_eig, time_sparse_eigsh, time_tf = solve_eigprob(n)
    print(f'{n}x{n} matrix: {time_eig:.3f} seconds and {time_sparse_eigsh:.3f} seconds and {time_tf:.3f} seconds')
