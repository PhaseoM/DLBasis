from .layers import *
from .config import Config


class nnModel:
    def __init__(self, layers, cfg=Config()):
        self.layers = layers
        self.cfg = cfg
        self.hyparam_distb(cfg)

    def forward(self, X):
        for name, layer in self.layers.items():
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for name, layer in reversed(self.layers.items()):
            dout = layer.backward(dout)
        return dout

    def grad_dn(self):
        for name, layer in self.layers.items():
            if isinstance(layer, ParamsLayer):
                layer.grad_dn()

    def hyparam_distb(self, cfg):
        for layer in self.layers.values():
            layer.update_hyparams(cfg)
