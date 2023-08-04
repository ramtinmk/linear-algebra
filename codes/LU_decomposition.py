import numpy as np

def LU_decomposition(matrix):
    """
    Perform LU decomposition on a given matrix.

    Parameters:
        matrix (numpy.ndarray): The input matrix to be decomposed.

    Returns:
        numpy.ndarray, numpy.ndarray: The lower triangular matrix (L) and the upper triangular matrix (U).
    """
    U = matrix.copy()  # Make a copy of the input matrix
    n = len(matrix)

    E = np.identity(n)  # Initialize an identity matrix of size n

    for i in range(n):
        for j in range(i+1,n):
          E[j][i] = -U[j][i] / U[i][i]  # Calculate the elements of the elimination matrix E
        U = np.dot(E, U)  # Apply the elimination matrix E to the upper triangular matrix U
        E = np.identity(n)  # Reset the elimination matrix E for the next iteration

    L = np.dot(matrix, np.linalg.inv(U))  # Calculate the lower triangular matrix L

    # Set very small values in L and U to 0 to improve readability
    L[np.abs(L) < 10e-10] = 0
    U[np.abs(U) < 10e-10] = 0

    return L, U


if __name__ == "__main__":
    # Example matrix for LU decomposition
    matrix = np.array([[4, -2, 2],
                       [4, -3, -2],
                       [2, 3, -1]])

    L, U = LU_decomposition(matrix)
    print("Lower Triangular Matrix (L):")
    print(L)
    print("\nUpper Triangular Matrix (U):")
    print(U)
