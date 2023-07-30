import numpy as np

def gaussian_elimination(matrix):
    m, n = matrix.shape

    for i in range(min(m, n)):
        # Find the maximum row in the current column and swap
        max_row = np.argmax(np.abs(matrix[i:, i])) + i
        matrix[[i, max_row]] = matrix[[max_row, i]]

        # Make the diagonal element 1
        diag_element = matrix[i, i]
        matrix[i, :] /= diag_element

        # Eliminate non-zero elements below the diagonal
        for k in range(i + 1, m):
            factor = matrix[k, i]
            matrix[k, :] -= factor * matrix[i, :]
        # Eliminate non-zero elements above the diagonal
        for l in range(i):
          factor = matrix[l,i]
          matrix[l,:] -= factor * matrix[i,:]

    return matrix

# Example usage:
if __name__ == "__main__":
    # Example system of linear equations in matrix form (Ax = B)
    A = np.array([
        [2, 3, -1],
        [4, 1, -2],
        [1, 2, 1],
        [3, 4, 2]
    ], dtype=float)

    # Perform forward elimination
    A = gaussian_elimination(A)
    print("Matrix after forward elimination:\n", A)
