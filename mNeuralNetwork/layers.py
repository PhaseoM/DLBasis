import numpy as np
import torch
from torch import nn
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class ParamsLayer(Layer):
    @abstractmethod
    def grad_dn(self, step):
        pass


class Linearlayer(ParamsLayer):
    def __init__(self, size_in, size_out, reg=1e-4):
        rng = np.random.default_rng()
        self.W = rng.normal(0, np.sqrt(2 / size_in), size=(size_out, size_in))
        self.b = np.zeros(size_out)
        self.x = None
        self.dW = None
        self.db = None
        self.reg = reg

    def forward(self, x):
        self.x = x
        x = np.dot(self.W, x) + self.b
        return x

    def backward(self, dout):
        self.dW = np.dot(dout, self.x.T)
        self.db = dout
        dx = np.dot(self.W.T, dout)
        return dx

    def grad_dn(self, step):
        self.W = (1 + self.reg) * self.W - step * self.dW
        self.b = self.b - step * self.db


class ReLUlayer(Layer):
    def __init__(self):
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
        self.out = None

    def forward(self, x):
        x = 1 / (1 + np.exp(-x))
        self.out = x
        return x

    def backward(self, dout):
        dx = self.out * (1 - self.out) * dout
        return dx
