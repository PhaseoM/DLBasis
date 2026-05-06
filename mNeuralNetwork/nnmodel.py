import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from .layers import *


class nnModel:
    def __init__(self, model, step):
        self.model = model
        self.step = step

    def forward(self, X):
        for layer in self.model.values():
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for layer in reversed(self.model.values()):
            dout = layer.backward(dout)

    def grad_dn(self):
        for layer in self.model.values():
            if isinstance(layer, ParamsLayer):
                layer.grad_dn(self.step)
