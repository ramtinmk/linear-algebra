import numpy as np

from solving_linear_system_LU import forward_substitution

# Define a function that performs gauss seidel iterative method to solve linear equations Ax = b.
def gauss_seidel(A, b, num_iteration=100):
    # Initialize the solution vector X with zeros.
    X = np.zeros(len(b))
    
    # Perform one iteration using the diagonal elements of A to initialize X.
    X = b / np.diag(A)  # Dividing b by diagonal elements of A to initialize X.

    
    # Iterate for the specified number of iterations.
    for _ in range(num_iteration):
        # Iterate through each equation (row) to update the solution vector X.
        for i in range(len(X)):
            
            temp = A[i][i]
            A[i][i] = 0
            temp2 = X[i]
            
            # Calculate the new value for X[i] using the Jacobi update formula.
            X[i] =  (1/ temp) * (b[i] - (A[i].dot(X)))
            
            # Restore the original diagonal element of A and the original value of X[i].
            A[i][i] = temp
        if np.allclose(np.matmul(A, X), b.T):
            print(f"converged at step {_}")
            break      
      
    return X

def gauss_seidel_matrix_form(A,b,num_iteration=100):
    
    X = np.zeros(len(A))
    
    for _ in range(num_iteration):
        D = np.diag(np.diag(A))
        L = np.tril(A,k=-1)
        U = np.triu(A,k=1)
        X = forward_substitution((L+D),b-U.dot(X))
        if np.allclose(np.matmul(A, X), b.T):
            print(f"converged at step {_}")
            break  
    return X  
# Entry point of the program.
if __name__ == "__main__":
    # Define the coefficient matrix A and the constant vector b for the linear equations.
    A = np.array([[2, -1, 1],
                  [-2, 5, -1],
                  [-1, -2, 4]])
    b = np.array([-1, 1, 3]).T
    
    # Call the jacobi_method function to solve the linear equations.
    X = gauss_seidel(A, b)
    print(X,np.matmul(A, X),b.T)
    assert np.allclose(np.matmul(A, X), b.T), "the result does not converge"
    
  
