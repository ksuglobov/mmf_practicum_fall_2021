class Polynomial:
    def __init__(self, *coeff):
        self.coeff = list(coeff)
        self.deg = len(self.coeff) - 1
        if self.deg < 0:
            self.deg = 0
            self.coeff = [0]

    def __call__(self, x):
        res = 0
        n = self.deg
        a = self.coeff
        for i in range(n, -1, -1):
            res = res * x + a[i]
        return res
