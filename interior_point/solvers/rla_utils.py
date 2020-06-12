import numpy as np

def get_uniform_probs(m):
    return (1/m) * np.ones(m)

def get_weighted_probs(A, B):
    col_norms = np.linalg.norm(A, axis=0)
    row_norms = np.linalg.norm(B, axis=1)

    return (col_norms * row_norms) / (col_norms * row_norms).sum()

def matmul(A, B, c, weighted=True):
    # Inputs:
    #   A: n x m matrix
    #   B: m x p matrix
    #   c: number of rank-one updates to perform
    #   probs: ndarray(m,) of probabilities of a
    #          particular update
    # 
    # Outputs:
    #   C_hat: n x p approximation
    if weighted:
        probs = get_weighted_probs(A, B)
    else:
        probs = get_uniform_probs(A.shape[1])

    locs = np.random.choice(A.shape[1], c, replace=False, p=probs)

    C = A[:, locs] / np.sqrt(c * probs[locs])
    R = (B[locs, :].T / np.sqrt(c * probs[locs])).T

    return C.dot(R)