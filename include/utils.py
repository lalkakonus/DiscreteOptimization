import numpy as np

def inverse_permutation(permutation):
    inverse_permutation = np.empty(permutation.shape, dtype=np.int)
    for idx, pos in enumerate(permutation):
        inverse_permutation[pos] = idx
    return inverse_permutation
