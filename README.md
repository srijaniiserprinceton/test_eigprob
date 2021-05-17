## About the package ##

This is a package to compare compuatational time of different packages in Python for solving eigenvalue problems for a sparse matrix. The sparse matrix is generated
using ```scipy.sparse.random``` and converted to its dense form using ```scipy.sparse.csr_matrix.todense``` function. This matrix is used to solve the 
eigenvalue problems using the following packages

* ```numpy.linalg.eigh```
* ```scipy.linalg.eigh```
* ```scipy.sparse.linalg.eigsh```
* ```tensorflow.linalg.eigh```

## Running the code ##

Run the following snippet inside the ```test_eigprob``` directory from your terminal:

```python run_time_eigprob.py```

## Output ##

The timing of matrices of different sizes on 10 CPU cores and 1 GPU on Princeton's Tiger Cluster are tabulated below:

| Function | Time taken (in seconds) |
| --- | --- |
|  |  |
| Matrix size (10 x 10) | |
|  |  |
| numpy.linalg.eigh | 0.0000 |
| scipy.linalg.eigh | 0.0001 |
| scipy.sparse.linalg.eigsh | 0.0004 |
| tensorflow.linalg.eigh | 0.0004 |
|  |  |
| Matrix size (31 x 31) | |
|  |  |
| numpy.linalg.eigh | 0.0000 |
| scipy.linalg.eigh | 0.0001 |
| scipy.sparse.linalg.eigsh | 0.0007 |
| tensorflow.linalg.eigh | 0.0008 |
|  |  |
| Matrix size (100 x 100) | |
|  |  |
| numpy.linalg.eigh | 0.0005 |
| scipy.linalg.eigh | 0.0007 |
| scipy.sparse.linalg.eigsh | 0.0009 |
| tensorflow.linalg.eigh | 0.0042 |
|  |  |
| Matrix size (316 x 316) | |
|  |  |
| numpy.linalg.eigh | 0.0052 |
| scipy.linalg.eigh | 0.0087 |
| scipy.sparse.linalg.eigsh | 0.0134 |
| tensorflow.linalg.eigh | 0.0209 |
|  |  |
| Matrix size (1000 x 1000) | |
|  |  |
| numpy.linalg.eigh | 0.0540 |
| scipy.linalg.eigh | 0.1570 |
| scipy.sparse.linalg.eigsh | 0.0275 |
| tensorflow.linalg.eigh | 0.0859 |
|  |  |
| Matrix size (3162 x 3162) | |
|  |  |
| numpy.linalg.eigh | 1.0206 |
| scipy.linalg.eigh | 2.1212 |
| scipy.sparse.linalg.eigsh | 0.7109 |
| tensorflow.linalg.eigh | 0.4971 |
