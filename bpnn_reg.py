import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.config import Config
from utils.dataplib import DataLoader
from collections import OrderedDict
from tqdm import tqdm

epochs_reg = 900
batch_size_reg = 16
input_layer_size = 1
hide_layer_size = 128
output_layer_size = 1

traindata_reg = "./data/diabetes/regression/sinc_train"
testdata_reg = "./data/diabetes/regression/sinc_test"

regression_model = mnn.nnModel(
    layers=OrderedDict(
        [
            ("linear_1", Linearlayer(input_layer_size, hide_layer_size)),
            # ("relu_1", ReLUlayer()),
            # ("sigmoid_1", Logisticlayer()),
            ("tanh_1", Tanhlayer()),
            ("linear_2", Linearlayer(hide_layer_size, hide_layer_size)),
            # ("relu_2", ReLUlayer()),
            # ("sigmoid_2", Logisticlayer()),
            ("tanh_2", Tanhlayer()),
            # ("linear_3", Linearlayer(hide_layer_size, hide_layer_size)),
            # ("relu_3", ReLUlayer()),
            # ("sigmoid_3", Logisticlayer()),
            # ("tanh_3", Tanhlayer()),
            ("linear_end", Linearlayer(hide_layer_size, output_layer_size)),
            # 回归任务输出层保持线性,不再接 tanh/sigmoid 这类压缩激活
        ]
    ),
    loss_layer=MSELosslayer(),
    cfg=Config(batchsize=batch_size_reg, step=1e-1, is_regular=True),
)

# 训练数据 X 大致在 [-10, 10],归一化后能让 tanh 避开饱和区
X_SCALE = 10.0


class bpNeuralNetwork:
    def __init__(self, model):
        self.model = model
        self.pred_history = None
        self.loss_history = None
        self.out = None

    def train(self, dataldr, epochs=600, log_every=50):
        loss_arr = []
        n_batches = len(dataldr)
        for epoch in range(epochs):
            if epoch == 300:
                self.model.cfg.step = 1e-3
                self.model.hyparam_distb()
            if epoch == 600:
                self.model.cfg.step = 1e-4
                self.model.hyparam_distb()
            epoch_loss = 0.0
            for batch, (X, y) in enumerate(dataldr):
                X = X.reshape(-1, 1) / X_SCALE
                y = y.reshape(-1, 1)
                pred = self.model.forward(X)

                loss = self.model.loss_forward(pred, y)
                self.model.backward(loss)
                self.model.grad_dn()

                epoch_loss += loss
            loss_arr.append(epoch_loss / n_batches)
            if epoch % log_every == 0 or epoch == epochs - 1:
                print(f"loss:{epoch_loss / n_batches:>7f}[{epoch:>5d}/{epochs:>5d}]")

        self.loss_history = np.array(loss_arr)

    def test(self, dataldr, y_true=None):
        pred_arr = []
        for X, y in dataldr:
            X = np.atleast_2d(X).reshape(-1, 1) / X_SCALE
            pred = self.model.forward(X)
            pred_arr.append(pred.ravel())

        self.pred_history = np.concatenate(pred_arr)
        if y_true is not None:
            mse = float(np.mean((self.pred_history - y_true) ** 2))
            print(f"test MSE: {mse:.6f}")
            return mse
        return None

    def loss_eval(self):
        indices = np.array(list(range(0, len(self.loss_history))))
        plt.plot(indices, self.loss_history)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("regression loss_fn")
        # plt.legend()
        plt.show()


def regsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 0], data[:, 1]

    return X, y


def execute_reg():
    y, X = regsdata_process(traindata_reg)
    datatrain = DataLoader(X, y, batch_size_reg, True)
    y1, X1 = regsdata_process(testdata_reg)
    # 测试集整批推理更快,也避免 DataLoader 逐样本迭代的开销
    datatest = DataLoader(X1, y1, batchsize=len(X1))
    bpnn = bpNeuralNetwork(model=regression_model)
    bpnn.train(dataldr=datatrain, epochs=epochs_reg)
    bpnn.test(dataldr=datatest, y_true=y1)

    # 按 x 排序后连线,避免 sinc_test 本身 x 不单调导致的锯齿
    order = np.argsort(X1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, s=6, alpha=0.4, label="train")
    plt.scatter(X1, y1, s=6, alpha=0.4, label="test")
    plt.plot(X1[order], bpnn.pred_history[order], c="r", lw=1.5, label="pred")
    plt.legend()
    plt.show()

    bpnn.loss_eval()


if __name__ == "__main__":
    execute_reg()
