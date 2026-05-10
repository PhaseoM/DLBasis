from .layers import *
from .config import Config


class nnModel:
    def __init__(self, layers, loss_layer, cfg=Config()):
        self.layers = layers
        self.loss_layer = loss_layer
        self.cfg = cfg
        self.hyparam_distb()

    def forward(self, X):
        for name, layer in self.layers.items():
            X = layer.forward(X)
        return X

    def backward(self, dout):
        dout = self.loss_layer.backward()
        for name, layer in reversed(self.layers.items()):
            dout = layer.backward(dout)
        return dout

    def loss_forward(self, pred, y):
        if self.cfg.is_regular:
            self.loss_layer.regular_for_W(self.layers)
        out = self.loss_layer.forward(pred, y)
        return out

    def grad_dn(self):
        for name, layer in self.layers.items():
            if isinstance(layer, ParamsLayer):
                layer.grad_dn()

    def hyparam_distb(self):
        self.loss_layer.update_hyparams(self.cfg)
        for layer in self.layers.values():
            layer.update_hyparams(self.cfg)
