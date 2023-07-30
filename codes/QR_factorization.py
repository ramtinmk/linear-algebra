import numpy as np
from gram_schmitt import *  # Assuming the 'gram_schmitt' module is available and contains the required functions.
from gaussian_elimination import *



def Qr_factorization(vectors):
    """
    Perform QR factorization using Gram-Schmidt process.

    Parameters:
        vectors (numpy.ndarray): Input matrix containing the vectors to be factorized.

    Returns:
        Q (numpy.ndarray): isometry matrix Q.
        R (numpy.ndarray): Upper triangular matrix R.
    """

    R = np.identity(len(vectors[0]))  # Initialize R as an identity matrix of appropriate size.
    vectors = vectors.T  # Transpose the input matrix to work with column vectors.

    # Perform Gram-Schmidt orthogonalization on the input vectors.
    ortho_lists = np.array(gram_schmidt(vectors))

    D = np.identity(len(vectors))  # Initialize D as an identity matrix of appropriate size.

    # Compute the diagonal elements of D, which are the norms of the orthogonalized vectors.
    for i in range(len(D)):
        D[i][i] = norm_2(ortho_lists[i])

    # Compute the non-diagonal elements of R using inner products between orthogonalized vectors and original vectors.
    for i in range(len(R)):
        for j in range(len(R[0])):
            if j > i:
                R[i][j] = inner_product(ortho_lists[i], vectors[j]) / (norm_2(ortho_lists[i]) ** 2)

    # Update R with the diagonal scaling provided by D.
    R = D.dot(R)

    # Compute Q by multiplying the transposed orthogonalized vectors with the inverse of D.
    Q = ortho_lists.T.dot(np.linalg.inv(D))

    return Q, R

# Example usage:

if __name__=="__main__":
    inp = np.array([[1, 3,3],
                    [-1,1,2],
                    [3,4,5]])

    Q, R = Qr_factorization(inp)
    print(Q,R)
