import numpy as np
from LU_decomposition import *


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
  matrix = np.array([[4, -2, 2],
                       [4, -3, -2],
                       [2, 3, -1]])
  b = np.array([[6,-8,5]]).T
  X = solve(matrix,b)
  
  print(X)


