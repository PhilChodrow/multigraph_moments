import numpy as np


def adjacency_matrix(G):
    A = np.zeros((G.n, G.n))
    for f in G.C:
        A[f[0], f[1]] += 1
        A[f[1], f[0]] += 1
    return(A)

def degree_sort(A, d):
    row_sorted = A[np.argsort(d)]
    col_sorted = row_sorted[:, np.argsort(d)]
    return(col_sorted)

def W_from_b(b):
    y = 0.5*b.sum()
    BB = np.outer(b, b)
    np.fill_diagonal(BB, 0)
    W = BB / (2*y - BB)
    return(W)

def X_from_b(b):
    y = 0.5*b.sum()
    BB = np.outer(b, b)
    np.fill_diagonal(BB, 0)
    return(BB/(2*y))