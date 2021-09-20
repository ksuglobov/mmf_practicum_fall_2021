import numpy as np


def calc_expectations(h, w, X, Q):
    C = np.cumsum(np.cumsum(Q, axis=1), axis=0)
    S1 = np.roll(C, w, axis=1)
    S1[:, 0:w] = 0
    S2 = np.roll(C, h, axis=0)
    S2[0:h, :] = 0
    S3 = np.roll(np.roll(C, w, axis=1), h, axis=0)
    S3[:, 0:w] = 0
    S3[0:h, :] = 0
    R = C - S1 - S2 + S3
    return X * R
