import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from .layers import *
from .config import Config


class nnModel:
    def __init__(self, layers, cfg=Config()):
        self.layers = layers
        self.cfg = cfg
        self.hyparam_distb(cfg)

    def forward(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

    def grad_dn(self):
        for layer in self.layers.values():
            if isinstance(layer, ParamsLayer):
                layer.grad_dn()

    def hyparam_distb(self, cfg):
        for layer in self.layers.values():
            layer.update_hyparams(cfg)


class bpNeuralNetwork:
    def __init__(self, model, loss_layer):
        self.model = model
        self.loss_layer = loss_layer
        self.out = None

    def train(self, dataldr):
        for batch, X, y in enumerate(dataldr):
            pred = self.model.forward(X)
            loss = self.loss_layer.forward(pred, y)
            dout = self.loss_layer.backward()
            self.model.backward(dout)

    def test(self, dataldr):
        pass

    def eval(self):
        pass
