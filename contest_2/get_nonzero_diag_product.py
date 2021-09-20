import numpy as np


def get_nonzero_diag_product(X):
    D = np.diag(X)[np.diag(X) != 0]
    return None if D.size == 0 else np.prod(D)
