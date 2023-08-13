import numpy as np





def jacobi_method(A,b,num_iteration=100):
    X = np.zeros(len(b))
    
    X = b/np.diag(A)
    print(X)
    
    for i in range(num_iteration):
      newX = np.zeros(len(A))
      for i in range(len(X)):
        temp = A[i][i]
        A[i][i] = 0
        temp2 = X[i]
        X[i]=0
        newX[i] = (1/temp) *(b[i]- (A[i].dot(X)))
        A[i][i] = temp
        X[i] = temp2    
      X = newX
      
    return X


if __name__=="__main__":
  A = np.array([[2,-1,1],
                [-2,5,-1],
                [-1,-2,4]])
  b = np.array([-1,1,3]).T
  
  X = jacobi_method(A,b)
  
  assert np.allclose(np.matmul(A,X),b.T)
  
