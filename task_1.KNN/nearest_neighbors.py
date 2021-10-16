import numpy as np
import distances
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:

    def __init__(self, k=2, strategy='my_own', metric='euclidean',
                 weights=False, test_block_size=1000, test_aug=None, test_aug_f=None):
        if not isinstance(k, int) and k > 0:
            raise TypeError('Wrong k (should be int > 0)!')
        if not isinstance(strategy, str):
            raise TypeError('Wrong type of strategy (should be str)!')
        if not isinstance(metric, str):
            raise TypeError('Wrong type of metric (should be str)!')
        if not isinstance(weights, bool):
            raise TypeError('Wrong type of weights (should be bool)!')
        if not isinstance(test_block_size, int) and test_block_size > 0:
            raise TypeError('Wrong test_block_size (should be int > 0)!')
        self.eps = 1e-5
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.test_aug = test_aug
        self.test_aug_f = test_aug_f
        if strategy == 'my_own':
            if not (metric in ['euclidean', 'cosine']):
                raise ValueError(f'Unknown metric {metric}!')
        elif strategy in ['brute', 'kd_tree', 'ball_tree']:
            self._model = NearestNeighbors(n_neighbors=self.k,
                                           algorithm=self.strategy,
                                           metric=self.metric)
        else:
            raise ValueError(f'Unknown strategy {strategy}!')

    def fit(self, X, y):
        if self.k > X.shape[0]:
            raise ValueError(f'Small data: {X.shape[0]} '
                             f'train objects < k = {self.k}')
        if self.strategy == 'my_own':
            self.X_train = X.copy()
        else:
            self._model.fit(X, y)
        self.y_train = y.copy()
        self.classes = np.unique(y)
        self.y_train_ind = np.empty(y.shape, dtype=int)
        for i, c in enumerate(self.classes):
            self.y_train_ind[y == c] = i

    def find_kneighbors_batch(self, X, return_distance=False):
        if self.strategy == 'my_own':
            dist = np.array([])
            if self.metric == 'euclidean':
                dist = distances.euclidean_distance(X, self.X_train)
            else:
                dist = distances.cosine_distance(X, self.X_train)
            if (self.test_aug != None):
                for comb in self.test_aug:
                    method = comb[0]
                    cases = comb[1]
                    for case in cases:
                        for option in case:
                            X_t = self.test_aug_f(X, method, option)
                            if self.metric == 'euclidean':
                                t = distances.euclidean_distance(
                                    X_t, self.X_train)
                            else:
                                t = distances.cosine_distance(
                                    X_t, self.X_train)
                            dist = np.minimum(dist, t)
            ind = np.argsort(dist, axis=1)[:, :self.k]
            if return_distance:
                return np.sort(dist, axis=1)[:, :self.k], ind
            else:
                return ind
        else:
            return self._model.kneighbors(X, self.k, return_distance)

    def find_kneighbors(self, X, return_distance=False):
        len = X.shape[0]
        step = self.test_block_size
        neighbors_ind = np.empty((len, self.k), dtype=int)
        if return_distance:
            neighbors_dist = np.empty((len, self.k), dtype=float)
            for i in range(0, len, step):
                neighbors_dist[i:i+step], neighbors_ind[i:i+step] =\
                    self.find_kneighbors_batch(
                    X[i:i+step], return_distance=True)
            return neighbors_dist, neighbors_ind
        else:
            for i in range(0, len, step):
                neighbors_ind[i:i+step] = self.find_kneighbors_batch(
                    X[i:i+step], return_distance=False)
            return neighbors_ind

    def predict(self, X):
        prediction = np.empty(X.shape[0]).astype(self.y_train.dtype)
        if self.weights:
            dist, ind = self.find_kneighbors(X, return_distance=True)
            for i, (raw_of_ind, raw_of_dist) in enumerate(zip(ind, dist)):
                prediction[i] = self.classes[np.bincount(
                    self.y_train_ind[raw_of_ind],
                    weights=1 / (raw_of_dist + self.eps)).argmax()]
        else:
            ind = self.find_kneighbors(X, return_distance=False)
            for i, r in enumerate(ind):
                prediction[i] = self.classes[np.bincount(
                    self.y_train_ind[r]).argmax()]
        return prediction
