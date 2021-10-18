class WordContextGenerator:
    def __init__(self, words, k):
        """
        :param words: Cписок слов
        :param window_size: Размер окна
        """
        self.words = words
        self.k = k

    def __iter__(self):
        return WordContextGenerator_Iterator(self)


class WordContextGenerator_Iterator:
    def __init__(self, obj):
        self.s = obj.words
        self.k = obj.k
        self.i, self.n = 0, len(obj.words)
        self.j = -min(self.i, self.k)

    def __iter__(self):
        return self

    def __next__(self):
        if self.j == 0:
            self.j += 1
        if self.i + self.j >= self.n or self.j > self.k:
            self.i += 1
            if self.i >= self.n:
                raise StopIteration
            self.j = -min(self.i, self.k)
        res = self.s[self.i], self.s[self.i + self.j]
        self.j += 1
        return res
