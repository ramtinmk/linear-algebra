import numpy as np

def is_positive_definite(matrix):
    matrix = np.array(matrix)
    n = matrix.shape[0]

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False

    # Check if all leading principal minors are positive
    for i in range(n):
        sub_matrix = matrix[:i+1, :i+1]
        determinant = np.linalg.det(sub_matrix)
        if determinant <= 0:
            return False

    return True
  

if __name__=="__main__":
  matrix = np.array([[16,4,-4]
                     ,[4,10,5]
                     ,[-4,5,9]])
  
  print(is_positive_definite(matrix))