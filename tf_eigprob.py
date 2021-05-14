import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
# import matplotlib.pyplot as plt
import numpy as np
import time


# creating a matrix
def build_matrix(size):
    return tf.random.uniform((size, size))

# getting the eigenvalues
def solve_eigprob(n, device="CPU"):
    N_iter = 10
    if device == "CPU":
        device_name = "CPU:0"
        mat = build_matrix(n)
    elif device == "GPU":
        device_name = "GPU:0"
        mat = tf.convert_to_tensor(build_matrix(n))
    matlist = []
    for j in range(N_iter):
        matlist.append(tf.convert_to_tensor(build_matrix(n)))
    t1 = time.time()
    N_iter = 5
    with tf.device(device_name):
        for j in range(N_iter):
            # evals, evecs = tf.linalg.eigh(matlist[j])
            if device == "CPU":
                evals, evecs = np.linalg.eigh(mat)
            else:
                evals, evecs = tf.linalg.eigh(mat)
    t2 = time.time()
    return ((t2-t1)/N_iter)

# timing computation of different size matrices
N = np.logspace(1,3.5,6,dtype='int')
time_arr = np.zeros_like(N,dtype='float')
for i, n in enumerate(N):
    time_arr[i] = solve_eigprob(n, device="CPU")
    tc = time_arr[i] * 1.0
    print(f'[CPU] {n:>4}x{n:<4} matrix: {time_arr[i]:7.4f} seconds.')
    time_arr[i] = solve_eigprob(n, device="GPU")
    tg = time_arr[i] * 1.0
    print(f'[GPU] {n:>4}x{n:<4} matrix: {time_arr[i]:7.4f} seconds.')
    print(f'GPU is {tc/tg:5.2f}x faster than CPU')
    print(f"{' ':*<30}")

