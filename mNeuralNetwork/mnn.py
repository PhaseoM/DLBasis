from .layers import *
from .config import Config


class nnModel:
    def __init__(self, layers, loss_layer, cfg=None):
        self.layers = layers
        self.loss_layer = loss_layer
        # 避免可变默认参数陷阱:每个模型自带独立的 Config 实例
        self.cfg = cfg if cfg is not None else Config()
        self.hyparam_distb()

    def forward(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def backward(self, dout=None):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)
        return dout

    def loss_forward(self, pred, y):
        if self.cfg.is_regular:
            self.loss_layer.regular_for_W(self.layers)
        out = self.loss_layer.forward(pred, y)
        return out

    def grad_dn(self):
        for layer in self.layers.values():
            if isinstance(layer, ParamsLayer):
                layer.grad_dn()

    def hyparam_distb(self):
        self.loss_layer.update_hyparams(self.cfg)
        for layer in self.layers.values():
            layer.update_hyparams(self.cfg)
