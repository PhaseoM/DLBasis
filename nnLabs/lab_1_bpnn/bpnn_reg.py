import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.config import Config
from utils.dataplib import DataLoader
from collections import OrderedDict
from tqdm import tqdm

traindata_reg = "./data/diabetes/regression/sinc_train"
testdata_reg = "./data/diabetes/regression/sinc_test"

# 训练数据 X 大致在 [-10, 10],归一化后能让 tanh 避开饱和区
X_SCALE = 10.0


class bpNeuralNetwork:
    def __init__(self, model: mnn.nnModel):
        self.model = model
        self.out = None

    def train(self, dataldr):
        avgloss = 0.0
        n_batches = len(dataldr)
        for batch, (X, y) in enumerate(dataldr):
            X = X.reshape(-1, 1) / X_SCALE
            y = y.reshape(-1, 1)
            pred = self.model.forward(X)

            loss = self.model.loss_forward(pred, y)
            self.model.backward(loss)
            self.model.grad_dn()

            avgloss += loss
        return avgloss / n_batches

    def test(self, dataldr):
        pred_arr = []
        for X, y in dataldr:
            X = np.atleast_2d(X).reshape(-1, 1) / X_SCALE
            pred = self.model.forward(X)
            pred_arr.append(pred.ravel())

        ret_pred_arr = np.concatenate(pred_arr)
        # loss = float(np.mean((self.pred_history - dataldr.y) ** 2))
        loss = self.model.loss_forward(ret_pred_arr, dataldr.y)
        return ret_pred_arr, loss

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


def execute_reg(
    epochs=100,
    batch_size=10,
    hide_layer_size=64,
    lamb=1e-4,
    is_regular=True,
    outlength=50,
):
    print(
        f"============ [HyperParams: epochs:{epochs} batch_size:{batch_size} hide_layer_size:{hide_layer_size}] ================"
    )
    # print(f"HyperParams: epochs:{epochs} batch_size:{batch_size} hide_layer_size:{hide_layer_size}")
    modelParams = f"ep={epochs} bs={batch_size} hs={hide_layer_size}"
    regression_model = mnn.nnModel(
        layers=OrderedDict(
            [
                ("linear_1", Linearlayer(1, hide_layer_size)),
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
                ("linear_end", Linearlayer(hide_layer_size, 1)),
            ]
        ),
        loss_layer=MSELosslayer(),
        cfg=Config(batchsize=batch_size, step=1e-1, lamb=lamb, is_regular=is_regular),
    )
    figures = []
    y, X = regsdata_process(traindata_reg)
    datatrain = DataLoader(X, y, batch_size, True)
    y1, X1 = regsdata_process(testdata_reg)
    datatest = DataLoader(X1, y1, batchsize=len(X1))
    bpnn = bpNeuralNetwork(model=regression_model)

    loss_train_arr = []
    loss_test_arr = []
    for epoch in range(epochs):
        if epoch == 300:
            bpnn.model.cfg.step = 1e-2
            bpnn.model.hyparam_distb()
        if epoch == 600:
            bpnn.model.cfg.step = 1e-4
            bpnn.model.hyparam_distb()
        if epoch == 900:
            bpnn.model.cfg.step = 1e-5
            bpnn.model.hyparam_distb()
        if epoch == 1200:
            bpnn.model.cfg.step = 1e-6
            bpnn.model.hyparam_distb()
        train_loss = bpnn.train(dataldr=datatrain)
        pred_arr, test_loss = bpnn.test(dataldr=datatest)
        loss_train_arr.append((epoch, train_loss))
        loss_test_arr.append((epoch, test_loss))
        if epoch % outlength == 0 or epoch == epochs - 1:
            print(f"train_loss:{train_loss:>7f}, test_loss:{test_loss:>7f}[{epoch:>5d}/{epochs:>5d}]")
    pred_arr, test_loss_ = bpnn.test(dataldr=datatest)
    order = np.argsort(X1)
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"reg_pred [{modelParams}]")
    # plt.scatter(X, y, s=6, alpha=0.4, label="train")
    # plt.scatter(X1, y1, s=6, alpha=0.4, label="test")
    plt.scatter(X, y, label="train")
    plt.scatter(X1, y1, label="test")
    plt.plot(X1[order], pred_arr[order], c="r", label="pred")
    # plt.plot(X1, pred_arr, c="r", label="pred")
    plt.legend()
    figures.append(plt.gcf())
    # plt.show()
    plt.figure()
    loss_train_arr = np.array(loss_train_arr)
    loss_test_arr = np.array(loss_test_arr)
    plt.plot(loss_train_arr[:, 0], loss_train_arr[:, 1], c="red", label="train_loss")
    plt.plot(loss_test_arr[:, 0], loss_test_arr[:, 1], c="green", label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"reg_loss [{modelParams}]")
    plt.legend()
    figures.append(plt.gcf())
    # plt.show()
    return test_loss_, figures


if __name__ == "__main__":
    execute_reg(epochs=1500, batch_size=10, hide_layer_size=64)
