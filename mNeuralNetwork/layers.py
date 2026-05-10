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


class LossLayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None
        self.l2_regular = 0.0

    def regular_for_W(self, layers):
        self.l2_regular = 0.0
        for layer in layers.values():
            if isinstance(layer, ParamsLayer):
                self.l2_regular += np.sum(layer.W**2)


class Linearlayer(ParamsLayer):
    def __init__(self, size_in, size_out):
        super().__init__()
        rng = np.random.default_rng()
        # self.W = rng.normal(0, np.sqrt(2 / size_in), size=(size_out, size_in))
        # self.W = 0.01 * rng.normal(0, np.sqrt(2 / size_in), size=(size_in, size_out))
        self.W = 0.1 * np.random.randn(size_in, size_out)
        self.b = np.zeros(size_out)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # print(f"x:{x}\nW:{self.W}\nb:{self.b}")
        self.x = x
        out = np.dot(x, self.W) + self.b
        # print(f"Wx:{np.dot(x, self.W)},out:{out}")
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def grad_dn(self):
        if self.cfg.is_regular:
            self.W -= self.cfg.step * (self.dW + self.cfg.lamb * self.W)
        else:
            self.W -= self.cfg.step * self.dW
        self.b -= self.cfg.step * self.db


class ReLUlayer(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

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


class Tanhlayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        # x = 2 / (1 + np.exp(-2 * x)) - 1
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        dx = (1 - self.out**2) * dout
        return dx


class MSELosslayer(LossLayer):
    def __init__(self):
        super().__init__()
        self.delta_y = None

    def forward(self, pred, y):
        # print(pred)
        self.delta_y = pred - y
        if self.cfg.is_regular:
            loss = 0.5 * (
                np.sum((pred - y) ** 2) / pred.shape[0]
                + self.cfg.lamb * self.l2_regular
            )
        else:
            loss = 0.5 * np.sum((pred - y) ** 2) / pred.shape[0]
        self.out = loss
        return loss

    def backward(self):
        dx = self.delta_y / self.delta_y.shape[0]
        return dx


class CrossEntropyLosslayer(LossLayer):
    def __init__(self):
        super().__init__()
        self.frac_y = None

    def forward(self, pred, y):
        pred = np.clip(pred, 1e-12, 1.0)
        self.frac_y = y / pred
        if self.cfg.is_regular:
            loss = -np.mean(y * np.log(pred)) + 0.5 * self.cfg.lamb * self.l2_regular
        else:
            loss = -np.mean(y * np.log(pred))
        self.out = loss
        # print(
        #     f"BCE fw: pred:{pred.shape},loss:{loss.shape},y:{y.shape},frac_y:{self.frac_y.shape}"
        # )
        return loss

    def backward(self):
        dx = -self.frac_y / self.frac_y.shape[0]
        # print(f"BCE bw: dx:{dx.shape},frac_y:{self.frac_y.shape}")
        return dx
