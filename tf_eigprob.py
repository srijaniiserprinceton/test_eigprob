import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
# import matplotlib.pyplot as plt
import numpy as np
import time
tfd = tfp.distributions

# creating a matrix
def build_matrix(size):
    return tf.random.uniform((size, size))

# getting the eigenvalues
def solve_eigprob(n):
    mat = build_matrix(n)
    t1 = time.time()
    N_iter = 10
    for j in range(N_iter): evals, evecs = tf.linalg.eigh(mat)
    t2 = time.time()
    return ((t2-t1)/N_iter)

# timing computation of different size matrices
N = np.logspace(1,3.5,6,dtype='int')
time_arr = np.zeros_like(N,dtype='float')
for i, n in enumerate(N):
    time_arr[i] = solve_eigprob(n)
    print(f'{n}x{n} matrix: {time_arr[i]} seconds.')
