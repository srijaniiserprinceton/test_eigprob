import cupy as cp
import  scipy.sparse
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
    t1 = time.time()
    for j in range(N_iter): evals, evecs = cp.linalg.eigh(mat)
    t2 = time.time()

    return ((t2-t1)/N_iter)

# timing computation of different size matrices                                                  \
N = cp.logspace(1,3.5,6,dtype='int')
print('--Martix size------non-sparse solver-------sparse solver---------')
for i, n in enumerate(N):
    time_eig = solve_eigprob(n)
    print(f'{n}x{n} matrix: {time_eig:.3f} seconds.')
