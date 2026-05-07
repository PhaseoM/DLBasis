import numpy as np
from torch import nn
from abc import ABC, abstractmethod
from .config import Config


class Layer(ABC):
    def __init__(self):
        self.cfg = Config()

    @abstractmethod
    def forward(self, *x):
        pass

    @abstractmethod
    def backward(self, *dout):
        pass

    def update_hyparams(self, cfg):
        self.cfg = cfg


class ParamsLayer(Layer):
    @abstractmethod
    def grad_dn(self, *step):
        pass


class Linearlayer(ParamsLayer):
    def __init__(self, size_in, size_out):
        super().__init__()
        rng = np.random.default_rng()
        self.W = rng.normal(0, np.sqrt(2 / size_in), size=(size_out, size_in))
        self.b = np.zeros(size_out)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        x = np.dot(self.W, x) + self.b
        return x

    def backward(self, dout):
        self.dW = np.dot(dout, self.x.T)
        self.db = dout
        dx = np.dot(self.W.T, dout)
        return dx

    def grad_dn(self):
        self.W = (1 + self.cfg.lamb) * self.W - self.cfg.step * self.dW
        self.b = self.b - self.cfg.step * self.db


class ReLUlayer(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        x[self.mask]
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Logisticlayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        x = 1 / (1 + np.exp(-x))
        self.out = x
        return x

    def backward(self, dout):
        dx = self.out * (1 - self.out) * dout
        return dx


class MSELosslayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None
        self.l2_regular = 1e-12

    def forward(self, pred, y):
        loss = 0.5 * np.mean((pred - y) ** 2) + self.cfg.lamb * self.l2_regular
        self.out = loss
        return loss

    def backward(self):
        pass


class CrossEntropyLosslayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None
        self.l2_regular = self.cfg.eps

    def forward(self, pred, y):
        pred = np.clip(pred, 1e-12, 1.0)
        loss = -np.mean(y * np.log(pred)) + self.cfg.lamb * self.l2_regular
        self.out = loss
        return loss

    def backward(self):
        pass
