import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg

N = 10

# initializing a completely filled matrix with random values
A_sparse = random(N,N, density = 1)
# converting to its dense representation
A_dense = A_sparse.todense()
# making a symmetric matrix
A_dense = 0.5 * (A_dense + A_dense.T)

# choosing the main and the fifth diagonal (upper and lower)
# when creating the sparse matrix
diag_shift = 5

# creating the linear operator
def LinOp_func(v):
    Av = np.zeros((N,))
    
    # constructing the product rule for the three diagonals
    main_diag_vec = np.diag(A_dense) * v
    upper_diag_vec = np.diag(A_dense,k=diag_shift) * v[diag_shift:]
    lower_diag_vec = np.diag(A_dense,k=-diag_shift) * v[:diag_shift]

    # tiling it accordingly
    Av += main_diag_vec
    Av[:diag_shift] += upper_diag_vec
    Av[diag_shift:] += lower_diag_vec

    return Av

L_op = LinearOperator((N,N), matvec=LinOp_func)

# creating the dense version of A as above
A = np.zeros_like(A_dense)
A += np.diag(np.diag(A_dense))
A += np.diag(np.diag(A_dense,k=diag_shift),k=diag_shift)
A += np.diag(np.diag(A_dense,k=-diag_shift),k=-diag_shift)

print('A:\n', A)

# calculating the product using LinearOperator
prod_linOp = L_op.matvec(np.ones(N))
# crudely calculating the product
prod_crude = A * np.ones((N,1))

# raises error if they are not equal. Else nothing is printed
np.testing.assert_array_almost_equal(np.reshape(prod_linOp,(10,1)), prod_crude)   

# solving the eigenvalue problem
# using numpy.linalg.eig
eigval_1, __ = np.linalg.eig(A)

# using the scipy.sparse.linalg.eigs
eigval_2, __ = scipy.sparse.linalg.eigsh(L_op,k=N-1,which='SM')

print('\n\n')
print('numpy.linalg.eig // using true matrix // all eigenvalues (sorted):\n',np.sort(eigval_1))
print('\n\n')
print('scipy.sparse.linalg.eigsh // using linear operator // N-1 eigenvalues smallest-by-magnitude (sorted):\n', np.sort(eigval_2.real))
