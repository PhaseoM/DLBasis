import numpy as np


class DataLoader:
    def __init__(self, X, y, batchsize=1, shuffle=False):
        self.batchsize = batchsize
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.n = len(X)
        self.size = int(np.ceil(self.n / batchsize))

    def __len__(self):
        return self.size

    def __iter__(self):
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        end = min(self.n, self.i + self.batchsize)
        idx = self.indices[self.i : end]
        self.i += self.batchsize
        return self.X[idx], self.y[idx]

    # delete the tail-data
    # def __next__(self):
    #     if self.i + self.batchsize > self.n:
    #         raise StopIteration
    #     idx = self.indices[self.i : self.i + self.batchsize]
    #     self.i += self.batchsize
    #     return self.X[idx], self.y[idx]
