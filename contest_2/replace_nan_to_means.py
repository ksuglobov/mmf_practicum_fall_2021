import numpy as np


def replace_nan_to_means(X):
    nan_mean = np.where(np.all(np.isnan(X), axis=0), 0, np.nanmean(X, axis=0))
    return np.where(np.isnan(X), nan_mean, X)
