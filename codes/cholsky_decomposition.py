import numpy as np
from is_positive_definite import *

def cholesky_decomposition(matrix):
    n = matrix.shape[0]
    L = np.zeros_like(matrix, dtype=float)

    for i in range(n):
        for j in range(i+1):
            if i == j:
                L[i, i] = np.sqrt(matrix[i, i] - np.sum(L[i, :i]**2))
            else:
                L[i, j] = (matrix[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L






if __name__=="__main__":
  matrix = np.array([[16,4,-4]
                     ,[4,10,5]
                     ,[-4,5,9]])
  if is_positive_definite(matrix):
    print(cholesky_decomposition(matrix))  
  else:
    print("input matrix must be positive definite")