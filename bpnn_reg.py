import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.config import Config
from utils.dataplib import DataLoader
from collections import OrderedDict
from tqdm import tqdm

epochs_reg = 1200
batch_size_reg = 32
input_layer_size = 1
hide_layer_size = 80
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
            ("linear_3", Linearlayer(hide_layer_size, hide_layer_size)),
            # ("relu_3", ReLUlayer()),
            # ("sigmoid_3", Logisticlayer()),
            ("tanh_3", Tanhlayer()),
            ("linear_end", Linearlayer(hide_layer_size, output_layer_size)),
            # ("sigmoid_end", Logisticlayer()),
            ("tanh_end", Tanhlayer()),
        ]
    ),
    loss_layer=MSELosslayer(),
    cfg=Config(batchsize=batch_size_reg, step=1e-1, is_regular=True),
)


class bpNeuralNetwork:
    def __init__(self, model):
        self.model = model
        self.pred_history = None
        self.loss_history = None
        self.out = None

    def train(self, dataldr, epochs=600):
        loss_arr = []
        datasize = len(dataldr)
        for epoch in range(epochs):
            if epoch == 300:
                self.model.cfg.step = 1e-3
                self.model.hyparam_distb()
            if epoch == 600:
                self.model.cfg.step = 1e-4
                self.model.hyparam_distb()
            epoch_loss = 0
            for batch, (X, y) in enumerate(dataldr):
                # print(f"Training epoch:{batch}")
                # if batch == 2:
                #     return
                X = X.reshape(-1, 1)
                y = y.reshape(-1, 1)
                pred = self.model.forward(X)

                loss = self.model.loss_forward(pred, y)
                self.model.backward(loss)
                self.model.grad_dn()

                epoch_loss += loss
            loss_arr.append(epoch_loss / datasize)
            # if batch % 5 == 0 or batch == datasize - 1:
            #     current = batch
            #     print(f"loss:{loss:>7f}[{current:>5d}/{datasize:>5d}]")
            print(f"loss:{epoch_loss / datasize:>7f}[{epoch:>5d}/{epochs:>5d}]")

        self.loss_history = np.array(loss_arr)

    def test(self, dataldr):
        pred_arr = []
        for X, y in dataldr:
            pred = self.model.forward(X)
            pred_arr.append(pred)

        self.pred_history = np.array(pred_arr)

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
    datatest = DataLoader(X1, y1)
    bpnn = bpNeuralNetwork(model=regression_model)
    bpnn.train(dataldr=datatrain, epochs=epochs_reg)
    # return
    bpnn.test(dataldr=datatest)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, label="train")
    plt.scatter(X1, y1, label="test")
    plt.plot(X1, bpnn.pred_history, c="r", label="pred")
    plt.legend()
    plt.show()

    bpnn.loss_eval()


if __name__ == "__main__":
    execute_reg()
