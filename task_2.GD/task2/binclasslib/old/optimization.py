import numpy as np
from scipy.special import expit as sgm
from oracles import BinaryLogistic
from time import time


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic',
                 step_alpha=1, step_beta=0, tolerance=1e-5,
                 max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function != 'binary_logistic':
            raise ValueError(f'No {loss_function} loss function!')
        self.oracle = BinaryLogistic(**kwargs)
        self.alpha = step_alpha
        self.beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter

    def fit(self, X, y, w_0=None, trace=True, dataset=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        history['acc']: list of floats, содержит значения качества модели на каждой итерации
        """
        self.w = w_0 if w_0 is not None else self.init_w(X.shape[1])
        time_arr, loss_arr, acc_arr = [], [], []
        for self.n_iter in range(0, self.max_iter + 1):
            timer = time()
            if self.n_iter > 0:
                self.step(X, y)
            loss_arr.append(self.get_objective(X, y))
            time_arr.append(time() - timer)
            if dataset is not None:
                acc_arr.append((self.predict(dataset[0]) == dataset[1]).mean())
            if (self.n_iter > 0 and abs(loss_arr[-1] - loss_arr[-2]) < self.tol):
                break
        if trace:
            history = {'time': time_arr, 'func': loss_arr}
            if dataset is not None:
                history['acc'] = acc_arr
            return history

    def predict(self, X):
        return np.where(X.dot(self.w) > 0, 1, -1)

    def predict_proba(self, X):
        p = sgm(X.dot(self.w))
        return np.c_(1 - p, p)

    def get_objective(self, X, y):
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        return self.w

    def get_lrate(self):
        return self.alpha / self.n_iter ** self.beta

    def init_w(self, n, seed=0):
        bound = 1 / (2 * n)
        return np.random.default_rng(seed).uniform(-bound, bound, size=n)

    def step(self, X, y):
        self.w -= self.get_lrate() * self.get_gradient(X, y)


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size=500, step_alpha=1,
                 step_beta=0, tolerance=1e-5, max_iter=1000, random_seed=153,
                 **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function, step_alpha, step_beta, tolerance, max_iter, **kwargs)
        self.batch_size = batch_size
        self.seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, dataset=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        history['acc']: list of floats, содержит значения качества модели на каждой итерации
        """
        np.random.seed(self.seed)

        self.w = w_0 if w_0 is not None else self.init_w(X.shape[1], self.seed)
        time_arr, loss_arr, acc_arr, weights_diff_arr = [], [], [], []
        batch_quant = X.shape[0] / self.batch_size
        # count of batches that was processed, every update period
        batch_count = [0]
        batch_threshold = log_freq * X.shape[0] / self.batch_size
        tol_chk = False  # tolerance check - stopping criteria

        timer = time()
        timer_chk = True  # timer is on indicator
        loss_arr.append(self.get_objective(X, y))
        time_arr.append(time() - timer)
        timer_chk = False
        weights_diff_arr.append(0.0)
        w_prev = self.w.copy()
        if dataset is not None:
            acc_arr.append((self.predict(dataset[0]) == dataset[1]).mean())

        batch_count.append(batch_count[-1])
        timer = time()
        timer_chk = True
        for self.n_iter in range(1, self.max_iter + 1):  # epochs
            X_ind_shuffled = (np.random.default_rng(
                self.seed).permutation(X.shape[0]))
            epoch_ind = X_ind_shuffled[:X.shape[0] -
                                       X.shape[0] % self.batch_size]
            batch_ind = np.split(epoch_ind, batch_quant)
            for i, ind in enumerate(batch_ind):  # batch iterations
                batch_count[-1] += 1
                self.step(X[ind], y[ind])

                # update and tol control
                if batch_count[-1] - batch_count[-2] > batch_threshold:
                    loss_arr.append(self.get_objective(X, y))
                    time_arr.append(time() - timer)
                    timer_chk = False
                    weights_diff_arr.append(
                        np.inner(w_prev - self.w, w_prev - self.w))
                    w_prev = self.w.copy()
                    if dataset is not None:
                        acc_arr.append(
                            (self.predict(dataset[0]) == dataset[1]).mean())
                    if abs(loss_arr[-1] - loss_arr[-2]) < self.tol:      # tol control
                        tol_chk = True
                        break  # (*) end of all epochs
                    if self.n_iter < self.max_iter or i < batch_quant - 1:
                        batch_count.append(batch_count[-1])
                        timer = time()
                        timer_chk = True
            if tol_chk:
                break  # from (*) - end of all epochs
        if timer_chk:
            loss_arr.append(self.get_objective(X, y))
            time_arr.append(time() - timer)
            timer_chk = False
            weights_diff_arr.append(np.inner(w_prev - self.w, w_prev - self.w))
            w_prev = self.w.copy()
            if dataset is not None:
                acc_arr.append((self.predict(dataset[0]) == dataset[1]).mean())
        if trace:
            epoch_arr = list(np.array(batch_count) *
                             self.batch_size / X.shape[0])
            history = {'time': time_arr,
                       'func': loss_arr,
                       'epoch_num': epoch_arr,
                       'weights_diff': weights_diff_arr}
            if dataset is not None:
                history['accuracy'] = acc_arr
            return history

np.random.seed(10)
clf = SGDClassifier(loss_function='binary_logistic', step_alpha=1,
    step_beta=0, tolerance=1e-4, max_iter=5, l2_coef=0.1)
l, d = 1000, 10
X = np.random.random((l, d))
y = np.random.randint(0, 2, l) * 2 - 1
w = np.random.random(d)
history = clf.fit(X, y, w_0=np.zeros(d), trace=True)
print(history)

print(' '.join([str(x) for x in history['func']]))