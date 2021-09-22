class CooSparseMatrix:
    def __init__(self, ijx_list, shape):
        self.data = {}
        self.check_shape(shape)
        self._shape = shape
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
        if (self._shape != other._shape):
            raise TypeError('Different dimensions!')
        res = CooSparseMatrix([], shape=self._shape)
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
        if (self._shape != other._shape):
            raise TypeError('Different dimensions!')
        res = CooSparseMatrix([], shape=self._shape)
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
        res = CooSparseMatrix([], shape=self._shape)
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

    def transform(self, m2, n2):
        n1 = self._shape[1]
        self_res = CooSparseMatrix([], shape=(m2, n2))
        self_res.data = {}
        for r1 in self.data:
            for c1 in self.data[r1]:
                r2 = (c1 + r1 * n1) // n2
                c2 = (c1 + r1 * n1) % n2
                self_res[r2, c2] = self[r1, c1]
        return self_res

    def get_shape(self):
        return self._shape

    def set_shape(self, idx):
        self.check_shape(idx)
        (m, n) = idx
        if (self.shape[0] * self.shape[1] != m * n):
            raise TypeError('Wrong reshape dimensions!')
        res = self.transform(m, n)
        for r in self.data:
            self.data[r].clear()
        self.data.clear()
        for r in res.data:
            self.data[r] = res.data[r].copy()
        self._shape = (m, n)

    def get_transposed(self):
        (m, n) = self._shape
        self_res = CooSparseMatrix([], shape=(n, m))
        self_res.data = {}
        for r1 in self.data:
            for c1 in self.data[r1]:
                r2, c2 = c1, r1
                self_res[r2, c2] = self[r1, c1]
        return self_res

    def set_transposed(self, *args):
        raise AttributeError('Can not assign to transposed matrix!')

    shape = property(get_shape, set_shape, None, "Matrix dimensions")
    T = property(get_transposed, set_transposed, None, "Matrix dimensions")

    def check_row(self, r):
        return 0 <= r <= self._shape[0] - 1

    def check_col(self, c):
        return 0 <= c <= self._shape[1] - 1

    def check_idx(self, idx):
        if type(idx) != tuple:
            raise TypeError('Wrong index type!')
        if len(idx) != 2:
            raise TypeError('Wrong indexation dimension!')
        if type(idx[0]) != int or type(idx[1]) != int:
            raise TypeError('Wrong indexation type!')
        if not (self.check_row(idx[0]) and self.check_col(idx[1])):
            raise TypeError('Index out of range!')

    def check_ijx_el(self, el):
        if type(el) != tuple:
            raise TypeError('Wrong type of ijx_list element!')
        if len(el) != 3:
            raise TypeError('Wrong count in ijx_list tuple!')
        if type(el[0]) != int or type(el[1]) != int:
            raise TypeError('Wrong type of index in ijx_list tuple!')
        if type(el[2]) != int and type(el[2]) != float:
            raise TypeError('Wrong type of value in ijx_list tuple!')
        if not (self.check_row(el[0]) and self.check_col(el[1])):
            raise TypeError('Index out of range in initialization!')

    def check_shape(self, shape):
        if type(shape) != tuple:
            raise TypeError('Wrong type of shape!')
        if len(shape) != 2:
            raise TypeError('Wrong count of dimensions in shape!')
        if type(shape[0]) != int or type(shape[1]) != int:
            raise TypeError('Wrong type of dimension in shape!')
        if shape[0] <= 0 or shape[1] <= 0:
            raise TypeError('Non-positive dimension in shape!')
