from solving_linear_system_LU import backward_substitution,forward_substitution
from cholsky_decomposition import *
from matrix_rank import *



if __name__=="__main__":
  matrix = np.array([[9,3,6,3],[3,10,5,-5],[6,5,21,16]])
  A = matrix[:,:-1]
  b = matrix[:,-1]
  

  
  A_rank = Matrix_rank(A)
  additive_matrix_rank = Matrix_rank(matrix)
  if A_rank==additive_matrix_rank==matrix.shape[0]:
    if is_positive_definite(A):
      L = cholesky_decomposition(A)
      Y = forward_substitution(L,b)
      X = backward_substitution(L.T,Y)
      print("solution is: ",X)
    else:
      print("matrix is not positive definite")
  elif (A_rank==additive_matrix_rank)  and A_rank<matrix.shape[0]:
      print("there are infinite solutions")
  else:
      print("system has no solution")