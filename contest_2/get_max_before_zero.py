import numpy as np


def get_max_before_zero(x):
    a = x[1:][(x == 0)[:-1]]
    return None if np.size(a) == 0 else np.max(a)
