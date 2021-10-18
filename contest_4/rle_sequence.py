import numpy as np


class RleSequence:
    def __init__(self, input_sequence):
        self.len = np.shape(input_sequence)[0]
        if self.len == 0:
            self.numbers, self.counters = \
                np.array([]), np.array([])
        else:
            mask = np.concatenate(([True],
                                   input_sequence[1:] != input_sequence[:-1], [True]))
            self.numbers, self.counters = \
                input_sequence[mask[:-1]], np.diff(np.where(mask)[0])
        self.runs_quant = self.numbers.shape[0]

    def norm_ind(self, i):
        if (i < 0):
            i += self.len
        if (i < 0):
            i = 0
        if (i > self.len - 1):
            i = self.len
        return i

    def rle_ind(self, ind):
        s = 0
        for j in range(self.runs_quant):
            s += self.counters[j]
            if s > ind:
                s -= self.counters[j]
                return j, ind - s

    def __getitem__(self, ind):
        if (isinstance(ind, slice)):
            begin, end, step = ind.start, ind.stop, ind.step
            if (isinstance(begin, int)):
                begin = self.norm_ind(begin)
            else:
                begin = 0
            if (isinstance(end, int)):
                end = self.norm_ind(end)
            else:
                end = self.len
            if (isinstance(step, int)):
                pass
            else:
                step = 1
            return np.array(list(RleSequence_Iterator(self, begin, end, step)))
        elif (isinstance(ind, int)):
            if -self.len < ind < self.len:
                if ind < 0:
                    ind += self.len
                return self.numbers[self.rle_ind(ind)[0]]
            else:
                raise IndexError
        else:
            raise TypeError

    def __contains__(self, target_elem):
        for elem in self.numbers:
            if target_elem == elem:
                return True
        return False

    def __iter__(self):
        return RleSequence_Iterator(self, 0, self.len, 1)


class RleSequence_Iterator:
    def __init__(self, obj, begin, end, step):
        end -= 1
        self.obj = obj
        self.runs_ind, self.repeat_ind = self.obj.rle_ind(begin)
        self.runs_ind_end, self.repeat_ind_end = self.obj.rle_ind(end)
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        while self.repeat_ind >= self.obj.counters[self.runs_ind]:
            self.repeat_ind -= self.obj.counters[self.runs_ind]
            self.runs_ind += 1
            if (self.runs_ind > self.runs_ind_end):
                break
        if (self.runs_ind > self.runs_ind_end
                or self.runs_ind == self.runs_ind_end
                and self.repeat_ind > self.repeat_ind_end):
            raise StopIteration
        ret = self.obj.numbers[self.runs_ind]
        self.repeat_ind += self.step
        return ret
