import numpy as np
from solving_linear_system_LU import backward_substitution

def scaling(matrix):
    max_values = np.max(matrix, axis=1, keepdims=True)
    # Divide each row by its corresponding maximum value using broadcasting
    matrix = matrix / max_values
    return matrix
    
def gaussian_elimination(matrix,pivoting=True):
    m, n = matrix.shape
    

    
    matrix = scaling(matrix)
    for i in range(min(m, n)):
        # Find the maximum row in the current column and swap
        if pivoting:
            max_row = np.argmax(np.abs(matrix[i:, i])) + i
            matrix[[i, max_row]] = matrix[[max_row, i]]

        # Make the diagonal element 1
        diag_element = matrix[i, i]
        matrix[i, :] /= diag_element

        # Eliminate non-zero elements below the diagonal
        for k in range(i + 1, m):
            factor = matrix[k, i]
            matrix[k, :] -= factor * matrix[i, :]
        #Eliminate non-zero elements above the diagonal
        for l in range(i):
          factor = matrix[l,i]
          matrix[l,:] -= factor * matrix[i,:]

    return matrix

# Example usage:
if __name__ == "__main__":
    # Example system of linear equations in matrix form (Ax = B)
    A = np.array([
        [1, -1, 1,1],
        [2, 7, -1,8],
        [15,1, -8,8]
    ], dtype=float)

    # Perform forward elimination
    A = gaussian_elimination(A)
    print("Matrix after forward elimination:\n", A)
    

