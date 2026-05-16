import numpy as np
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
    def grad_dn(self):
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
    """全连接层。

    初始化策略支持:
        - "xavier" (默认, 适合 tanh/sigmoid): std = sqrt(1 / fan_in)
        - "he"                    (适合 ReLU): std = sqrt(2 / fan_in)
        - "glorot" (Glorot 对称):              std = sqrt(2 / (fan_in + fan_out))
    原来的固定 0.1 缩放对 fan_in=1 的第一层会让 tanh 退化、对大 fan_in 又接近饱和,
    深层网络下效果不稳定。默认使用 Xavier 自适应方差。
    """

    def __init__(self, size_in, size_out, init="xavier"):
        super().__init__()
        # rng = np.random.default_rng()
        if init == "xavier":
            std = np.sqrt(1.0 / size_in)
        elif init == "he":
            std = np.sqrt(2.0 / size_in)
        elif init == "glorot":
            std = np.sqrt(2.0 / (size_in + size_out))
        else:
            raise ValueError(f"unknown init scheme: {init}")
        self.W = np.random.standard_normal((size_in, size_out)) * std
        self.b = np.zeros(size_out)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

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
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Logisticlayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        # 数值稳定版 sigmoid,避免 exp 溢出
        out = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
        self.out = out
        return out

    def backward(self, dout):
        return self.out * (1 - self.out) * dout


class Tanhlayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return (1 - self.out**2) * dout


class MSELosslayer(LossLayer):
    def __init__(self):
        super().__init__()
        self.delta_y = None

    def forward(self, pred, y):
        n = pred.shape[0]
        self.delta_y = pred - y
        data_loss = 0.5 * np.sum(self.delta_y**2) / n
        if self.cfg.is_regular:
            loss = data_loss + 0.5 * self.cfg.lamb * self.l2_regular
        else:
            loss = data_loss
        self.out = loss
        return loss

    def backward(self):
        return self.delta_y / self.delta_y.shape[0]


class CrossEntropyLosslayer(LossLayer):
    """交叉熵。同时支持两种用法:

    - 多分类: pred 为 softmax 后的概率分布 (N, C),标签 y 为 one-hot (N, C),
      损失为 -sum(y * log(pred)) / N。
    - 二分类: pred 为 sigmoid 输出的正类概率 (N, 1) 或 (N,),标签 y 同形状,
      取值 {0, 1},自动走 BCE:-[y*log(p) + (1-y)*log(1-p)] 的均值。

    两种模式下 forward / backward 都按 batch 维度做平均,梯度尺度一致。
    """

    def __init__(self):
        super().__init__()
        self.pred = None
        self.y = None
        self.mode = None  # "bce" or "ce"

    def _is_binary(self, pred):
        return pred.ndim == 1 or pred.shape[-1] == 1

    def forward(self, pred, y):
        n = pred.shape[0]
        eps = self.cfg.eps
        self.mode = "bce" if self._is_binary(pred) else "ce"
        if self.mode == "bce":
            pred = np.clip(pred, eps, 1.0 - eps)
            self.pred = pred
            self.y = y
            data_loss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / n
        else:
            pred = np.clip(pred, eps, 1.0)
            self.pred = pred
            self.y = y
            data_loss = -np.sum(y * np.log(pred)) / n

        if self.cfg.is_regular:
            loss = data_loss + 0.5 * self.cfg.lamb * self.l2_regular
        else:
            loss = data_loss
        self.out = loss
        return loss

    def backward(self):
        n = self.pred.shape[0]
        if self.mode == "bce":
            # d/dp of BCE = (p - y) / (p * (1 - p))
            return (self.pred - self.y) / (self.pred * (1 - self.pred)) / n
        return -(self.y / self.pred) / n


class LeakyReLUlayer(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] *= self.alpha
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] *= self.alpha
        return dx
