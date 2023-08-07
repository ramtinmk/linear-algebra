# Import the required module for Gaussian Elimination
from gaussian_elimination import *

# Define a function to calculate the rank of a matrix
def Matrix_rank(matrix):
    # Use Gaussian Elimination to transform the matrix into its reduced row echelon form
    eliminated = gaussian_elimination(matrix)
    

    # Count the number of non-zero elements in each row of the reduced row echelon form
    temp = np.count_nonzero(eliminated, axis=1)

    # Return the number of non-zero rows, which corresponds to the rank of the matrix
    return np.count_nonzero(temp != 0)

# Entry point of the script
if __name__ == "__main__":
    # Define the matrix 'A'
    A = np.array([
        [2, 3, -1],
        [4, 1, -2],
        [1, 2, 1],
        [3, 4, 2]
    ], dtype=float)

    # Calculate the rank of the matrix 'A' using the Matrix_rank function and print the result
    print(f"rank of matrix is: {Matrix_rank(A)}")
