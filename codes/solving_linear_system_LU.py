import numpy as np
from LU_decomposition import *
from matrix_rank import *

def forward_substitution(matrix, b):
    """
    Solve a system of equations using backward or forward substitution.

    Args:
        matrix (numpy.ndarray): The coefficient matrix (n x n),lower triangular.
        b (numpy.ndarray): The right-hand side vector (n x 1).

    Returns:
        numpy.ndarray: The solution vector (n x 1).
    """
    n = len(matrix)
    x = np.zeros(n)

    
    # Lower triangular matrix, use forward substitution
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]

    return x

def backward_substitution(matrix,b):
    """
    Solve a system of equations using backward or forward substitution.

    Args:
        matrix (numpy.ndarray): The coefficient matrix (n x n), upper triangular.
        b (numpy.ndarray): The right-hand side vector (n x 1).

    Returns:
        numpy.ndarray: The solution vector (n x 1).
    """
    n = len(matrix)
    x = np.zeros(n)

    
    # Lower triangular matrix, use forward substitution
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
              x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]

    return x




def solve(matrix,b):
    L,U = LU_decomposition(matrix)
    
    Y = forward_substitution(L,b)
    X = backward_substitution(U,Y)
    
    return X
    


if __name__=="__main__":
  matrix = np.array([[4, -1, 0,0],
                       [16,-1, 2,0],
                       [0, 15, 8,5],
                       [0,0,-2,13]])
  b = np.array([[13,65,66,9]]).T
  A_rank = Matrix_rank(matrix)
  additive_matrix_rank = Matrix_rank(np.hstack((matrix,b)))
  
  
  if A_rank==additive_matrix_rank==matrix.shape[0]:
    X = solve(matrix,b)
    print("solution is: ",X)
  elif (A_rank==additive_matrix_rank)  and A_rank<matrix.shape[0]:
      print("there are infinite solutions")
  else:
      print("system has no solution")
      
  


