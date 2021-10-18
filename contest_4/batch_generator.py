import random


class BatchGenerator:
    def __init__(self, list_of_sequences, batch_size, shuffle=False):
        """
        :param list_of_sequences: Список списков или numpy.array одинаковой длины
        :param batch_size: Размер батчей, на которые нужно разбить входные последовательности.
            Батчи последнего элемента генератора могут быть короче чем batch_size
        :param shuffle: Флаг, позволяющий перемешивать порядок элементов в последовательностях
        """
        self.seq_list = list_of_sequences
        self.seq_quant = len(self.seq_list)
        self.seq_len = len(self.seq_list[0]) if self.seq_quant > 0 else 0
        self.batch_size = batch_size
        self.batch_quant = (self.seq_len // self.batch_size
                            + (1 if self.seq_len % self.batch_size > 0 else 0))

        if shuffle:
            for i in range(self.seq_quant):
                random.shuffle(self.seq_list[i])

    def __iter__(self):
        return BatchGenerator_Iterator(self)


class BatchGenerator_Iterator:
    def __init__(self, obj):
        self.list = obj.seq_list
        self.step = obj.batch_size
        self.i1, self.i2 = 0, self.step
        self.i, self.n = 0, obj.batch_quant

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        res = [el[self.i1:self.i2] for el in self.list]
        self.i1 += self.step
        self.i2 += self.step
        self.i += 1
        return res
