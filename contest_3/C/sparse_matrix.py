class CooSparseMatrix:
    def __init__(self, ijx_list, shape):
        self.data = {}
        self.check_shape(shape)
        self.shape = shape
        if type(ijx_list) != list:
            raise TypeError('Wrong type of ijx_list!')
        for el in ijx_list:
            self.check_ijx_el(el)
            r, c, v = el
            if r in self.data:
                if c in self.data[r]:
                    raise TypeError('Element re-initialization!')
                elif v != 0:
                    self.data[r][c] = v
            elif v != 0:
                self.data[r] = {}
                self.data[r][c] = v

    def __getitem__(self, idx):
        if type(idx) == int:
            r = idx
            if not self.check_row(r):
                raise TypeError('Index out of range!')
            res = CooSparseMatrix([], shape=(1, self.shape[1]))
            res.data = {0: self.data.get(r, {}).copy()}
            return res
        else:
            self.check_idx(idx)
            r, c = idx
            if r in self.data and c in self.data[r]:
                return self.data[r][c]
            return 0

    def __setitem__(self, idx, v):
        self.check_idx(idx)
        r, c = idx
        if type(v) != int and type(v) != float:
            raise TypeError('Wrong type of value!')
        if v != 0:
            if r not in self.data:
                self.data[r] = {}
            self.data[r][c] = v
        else:
            self.del_el(r, c)

    def __add__(self, other):
        if (self.shape != other.shape):
            raise TypeError('Different dimensions!')
        res = CooSparseMatrix([], shape=self.shape)
        for r in self.data:
            res.data[r] = self.data[r].copy()
        for r in other.data:
            if r in res.data:
                for c in other.data[r]:
                    t = res.data[r].get(c, 0) + other.data[r][c]
                    if t == 0:
                        res.del_el(r, c)
                    else:
                        res.data[r][c] = t
            else:
                res.data[r] = other.data[r].copy()
        return res

    def __sub__(self, other):
        if (self.shape != other.shape):
            raise TypeError('Different dimensions!')
        res = CooSparseMatrix([], shape=self.shape)
        for r in self.data:
            res.data[r] = self.data[r].copy()
        for r in other.data:
            if r in res.data:
                for c in other.data[r]:
                    t = res.data[r].get(c, 0) - other.data[r][c]
                    if t == 0:
                        res.del_el(r, c)
                    else:
                        res.data[r][c] = t
            else:
                res.data[r] = other.data[r].copy()
                for c in res.data[r]:
                    res.data[r][c] *= -1
        return res

    def __mul__(self, other):
        if type(other) != int and type(other) != float:
            raise TypeError('Not a scalar!')
        res = CooSparseMatrix([], shape=self.shape)
        res.data = {}
        if (other == 0):
            return res
        for r in self.data:
            res.data[r] = self.data[r].copy()
        for r in res.data:
            for c in res.data[r]:
                res.data[r][c] = res.data[r][c] * other
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def del_el(self, r, c):
        if r in self.data:
            self.data[r].pop(c, 0)
            if (len(self.data[r]) == 0):
                self.data.pop(r, {})

    def check_row(self, r):
        return 0 <= r <= self.shape[0] - 1

    def check_col(self, c):
        return 0 <= c <= self.shape[1] - 1

    def check_idx(self, idx):
        if type(idx) != tuple:
            raise TypeError('Wrong index type!')
        if len(idx) != 2:
            raise TypeError('Wrong indexation dimension')
        if type(idx[0]) != int or type(idx[1]) != int:
            raise TypeError('Wrong indexation type!')
        if not (self.check_row(idx[0]) and self.check_col(idx[1])):
            raise TypeError('Index out of range!')

    def check_ijx_el(self, el):
        if type(el) != tuple:
            raise TypeError('Wrong type in ijx_list!')
        if len(el) != 3:
            raise TypeError('Wrong dimension in ijx_list!')
        if type(el[0]) != int or type(el[1]) != int:
            raise TypeError('Wrong type in ijx_list tuple index!')
        if type(el[2]) != int and type(el[2]) != float:
            raise TypeError('Wrong type in ijx_list tuple value!')
        if not (self.check_row(el[0]) and self.check_col(el[1])):
            raise TypeError('Wrong indices in initialization!')

    def check_shape(self, shape):
        if type(shape) != tuple:
            raise TypeError('Wrong type of shape!')
        if len(shape) != 2:
            raise TypeError('Wrong dimension of shape!')
        for i in shape:
            if type(i) != int:
                raise TypeError('Wrong type in shape!')
