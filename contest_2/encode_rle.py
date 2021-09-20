import numpy as np


def encode_rle(x):
    if np.shape(x)[0] == 0:
        return ([], [])
    mask = np.concatenate(([True], x[1:] != x[:-1], [True]))
    return (x[mask[:-1]], np.diff(np.where(mask)[0]))
