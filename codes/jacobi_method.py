import numpy as np

# Define a function that performs Jacobi iterative method to solve linear equations Ax = b.
def jacobi_method(A, b, num_iteration=100):
    # Initialize the solution vector X with zeros.
    X = np.zeros(len(b))
    
    # Perform one iteration using the diagonal elements of A to initialize X.
    X = b / np.diag(A)  # Dividing b by diagonal elements of A to initialize X.

    
    # Iterate for the specified number of iterations.
    for _ in range(num_iteration):
        # Create a new vector to store the updated solution.
        newX = np.zeros(len(A))
        
        # Iterate through each equation (row) to update the solution vector X.
        for i in range(len(X)):
            # Store the diagonal element of A and set it to zero for computation.
            temp = A[i][i]
            A[i][i] = 0
            
            # Store the current value of X[i] and set it to zero for computation.
            temp2 = X[i]
            X[i] = 0
            
            # Calculate the new value for X[i] using the Jacobi update formula.
            newX[i] = (1 / temp) * (b[i] - (A[i].dot(X)))
            
            # Restore the original diagonal element of A and the original value of X[i].
            A[i][i] = temp
            X[i] = temp2
        
        # Update the solution vector X with the newly calculated values.
        X = newX
      
    return X

# Entry point of the program.
if __name__ == "__main__":
    # Define the coefficient matrix A and the constant vector b for the linear equations.
    A = np.array([[2, -1, 1],
                  [-2, 5, -1],
                  [-1, -2, 4]])
    b = np.array([-1, 1, 3]).T
    
    # Call the jacobi_method function to solve the linear equations.
    X = jacobi_method(A, b)
    
    # Check if the obtained solution X satisfies the original equation Ax = b.
    assert np.allclose(np.matmul(A, X), b.T)  # Check if Ax is close to b.

  
