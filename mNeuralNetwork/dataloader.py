import numpy as np


class DataLodaer:
    def __init__(self, batchsize, X, y, shuffle=False):
        self.batchsize = batchsize
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.n = len(X)

    def __iter__(self):
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.i = 0
        return self

    def __next__(self):
        if self.i + self.batchsize > self.n:
            raise StopIteration
        idx = self.indices[self.i : self.i + self.batchsize]
        self.i += self.batchsize
        return self.X[idx], self.y[idx]
