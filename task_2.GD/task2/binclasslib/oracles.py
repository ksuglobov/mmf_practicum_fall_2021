import numpy as np
from scipy.special import expit as sgm


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    def __init__(self, l2_coef=0):
        """
        Задание параметров оракула.
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычисление значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        bound_eps = 1e-15
        margin = y * (X.dot(w))
        sigmoid = np.clip(sgm(margin), bound_eps, 1 - bound_eps)
        loss = -np.log(sigmoid)
        return loss.mean() + (self.l2_coef / 2) * np.inner(w, w)

    def grad(self, X, y, w):
        """
        Вычисление градиента функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        bound_eps = 1e-15
        margin = y * (X.dot(w))
        sigmoid = np.clip(sgm(-margin), bound_eps, 1 - bound_eps)
        loss_grad = -X.T.dot(sigmoid * y) / X.shape[0]
        return loss_grad + self.l2_coef * w
