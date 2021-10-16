import numpy as np


def euclidean_distance(X, Y):
    return np.sqrt(((X**2).sum(axis=1, keepdims=True)
                    - 2 * np.dot(X, Y.T)
                    + (Y.T**2).sum(axis=0, keepdims=True)))


def cosine_distance(X, Y):
    return 1 - np.dot(X / np.sqrt((X**2).sum(axis=1, keepdims=True)),
                      Y.T / np.sqrt((Y.T**2).sum(axis=0, keepdims=True)))
