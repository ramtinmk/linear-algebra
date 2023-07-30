import numpy as np

# Function to calculate the inner product of two vectors.
def inner_product(v, u):
    return sum([u * v for u, v in zip(u, v)])

# Function to calculate the 2-norm (Euclidean norm) of a vector.
def norm_2(v):
    return inner_product(v, v) ** 0.5

# Function to project vector 'v' onto vector 'u'.
def v_project_on_u(v, u):
    return [inner_product(u, v) / (norm_2(u) ** 2) * i for i in u]

# Function to perform the Gram-Schmidt process on a list of vectors.
def gram_schmidt(vectors):
    e = []
    # Normalize the first vector and add it to the orthogonalized list 'e'.
    e1 = vectors[0] #/ np.linalg.norm(vectors[0])
    e.append(list(e1))

    # Loop through the rest of the vectors to orthogonalize them.
    for i in range(1, len(vectors)):
        # Calculate the sum of projections of the current vector onto the previous orthogonal vectors.
        s = np.sum(np.array([v_project_on_u(vectors[i], e[j]) for j in range(i)]), axis=0)

        # Subtract the sum of projections from the current vector to make it orthogonal to the previous vectors.
        ej = vectors[i] - s

        # Normalize the orthogonalized vector.
        ej = ej #/ (np.linalg.norm(ej))
        e.append(list(ej))

    return e

# Example usage with a 3x4 matrix.
if __name__=='__main__':
    print(gram_schmidt(np.array([[1, 2, 0, 3], [4, 0, 5, 8], [8, 1, 5, 6]])))

    