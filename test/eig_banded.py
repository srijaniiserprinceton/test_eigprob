import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import random
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg
import time

N = 3000

# initializing a completely filled matrix with random values
A_sparse = random(N,N, density = 1)
# converting to its dense representation
A_dense = np.array(A_sparse.todense())
# making a symmetric matrix
A_dense = 0.5 * (A_dense + A_dense.T)

# choosing the main and the fifth diagonal (upper and lower)
# when creating the sparse matrix
diag_shift = 50

# defining the diagonals to be used
main_diag_A = np.diag(A_dense)
upper_diag_A = np.diag(A_dense,k=diag_shift)
lower_diag_A = np.diag(A_dense,k=-diag_shift)


# creating the dense version of A as above
A = np.zeros_like(A_dense)
A += np.diag(np.diag(A_dense))
A += np.diag(np.diag(A_dense,k=diag_shift),k=diag_shift)
A += np.diag(np.diag(A_dense,k=-diag_shift),k=-diag_shift)

print('A:\n', A)
print(f'nnz: {np.sum(A != 0)}')

lower_bands = np.zeros((51, N))
lower_bands[0, :] = np.diag(A_dense)
lower_bands[diag_shift, :-diag_shift] = np.diag(A_dense,k=diag_shift)

Niter = 5

# timing the eigenvalue solver for banded matrices
T1 = time.time()
for i in range(Niter):
    eigval_b = scipy.linalg.eig_banded(lower_bands, lower=True, eigvals_only=True)

T2 = time.time()
time_banded = (T2 - T1)

# timing the eigenvalue solver using dense matrix
T1 = time.time()
for i in range(Niter):
    eigval_d = scipy.linalg.eigh(A, eigvals_only=True)

T2 = time.time()
time_dense = (T2 - T1)

print("Banded solve is faster by: ", time_dense/time_banded, "times.")

print(np.sort(eigval_b))

print(np.sort(eigval_d))
