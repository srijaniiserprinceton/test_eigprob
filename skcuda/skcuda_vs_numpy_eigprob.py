from timeit import timeit
import numpy as np
from skcuda.fft import fft
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from skcuda import linalg
linalg.init()

# loading the sparse matrix
supmat = np.load('/scratch/gpfs/sbdas/Helioseismology/qdpy_output/output_files/w135_antia/super_matrix.npy')
supmat_gpu = gpuarray.to_gpu(supmat)

# number of timeit iterations
niter = 10

# timing the numpy eigenvalue solver
time_numpy = timeit(lambda: np.linalg.eigh(supmat), number=niter)

# timing the skcuda eigenvalue solver
time_skcuda = timeit(lambda: linalg.eig(supmat_gpu), number=niter)

time_numpy /= niter
time_skcuda /= niter

print(f'numpy: {time_numpy}')
print(f'skcuda: {time_skcuda}')

# getting eigenvalues and eigenvectors for both cases to compare
eval_np, evec_np = np.linalg.eigh(supmat)
eval_sk_gpu, evec_sk_gpu = linalg.eig(supmat_gpu)
evec_sk, eval_sk = eval_sk_gpu.get(), evec_sk_gpu.get()

# reorderign the skcuda matrix
evec_sk = np.transpose(evec_sk)

# making the max eval positive in both cases
for i in  range(len(supmat)):
    evec_np[i] *= np.sign(evec_np.real[i,np.argmax(np.abs(evec_np[i]))])
    evec_sk[i] *= np.sign(evec_sk.real[i,np.argmax(np.abs(evec_sk[i]))])

print(np.allclose(eval_np, eval_sk, 1e-4))
print(np.allclose(evec_np, evec_sk, 1e-4))
