import numpy as np
from nearest_neighbors import KNNClassifier


def get_score(score, x, y):
    if not isinstance(score, str):
        raise TypeError('Wrong type of score (should be str)!')
    if score == 'accuracy':
        return (x == y).mean()
    else:
        raise ValueError(f'Unknown score {score}!')


def kfold(n, n_folds):
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1
    indexes = np.random.permutation(n)
    res = [None] * n_folds
    for i in range(n_folds):
        s = fold_sizes[i]
        res[i] = (indexes[s:], indexes[:s])
        indexes = np.roll(indexes, -s)
    return res


def knn_cross_val_score(X, y, k_list=None, score='accuracy',
                        cv=None, **kwargs):
    X = X.copy()
    y = y.copy()
    if k_list is None:
        k_list = list(range(1, 8))
    if cv is None:
        cv = kfold(X.shape[0], 3)
    cv_res = {k: np.empty(len(cv), dtype=float) for k in k_list}
    for i, (train, test) in enumerate(cv):
        model = KNNClassifier(k=k_list[-1], **kwargs)
        model.fit(X[train], y[train])
        if kwargs['weights']:
            dist, ind = model.find_kneighbors(X[test], return_distance=True)
            weights = 1 / (dist + 1e-5)
        else:
            ind = model.find_kneighbors(X[test], return_distance=False)
            weights = np.full(ind.shape, 1, dtype=int)
        fold_pred = np.empty(
            (X[test].shape[0], len(k_list))).astype(y[train].dtype)

        classes = np.unique(y[train])
        y_train_ind = np.empty(y[train].shape, dtype=int)
        for j, c in enumerate(classes):
            y_train_ind[y[train] == c] = j
        classes_ind = y_train_ind[ind]
        for m in range(X[test].shape[0]):
            count = np.zeros(classes.shape)
            for h, (k_pred, k) in enumerate(zip([0] + k_list, k_list)):
                t = np.bincount(classes_ind[m][k_pred:k], weights[m][k_pred:k])
                if (count.shape[0] - t.shape[0]):
                    t = np.pad(t, (0, count.shape[0] - t.shape[0]), 'constant')
                count += t
                fold_pred[m][h] = classes[count.argmax()]
        for j, k in enumerate(k_list):
            cv_res[k][i] = get_score(score, fold_pred[:, j], y[test])
    return cv_res
