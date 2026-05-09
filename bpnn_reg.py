import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.dataloader import DataLoader
from mNeuralNetwork.config import Config
from collections import OrderedDict
from tqdm import tqdm

batch_size_reg = 4  # 50
input_layer_size = 1
hide_layer_size = 4  # 32
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
            # ("relu_2", ReLUlayer()),
            # ("sigmoid_3", Logisticlayer()),
            # ("tanh_3", Tanhlayer()),
            ("linear_end", Linearlayer(hide_layer_size, output_layer_size)),
            # ("sigmoid_end", Logisticlayer()),
            # ("tanh_end", Tanhlayer()),
        ]
    ),
    cfg=Config(batchsize=batch_size_reg, step=1e-4),
)


class bpNeuralNetwork:
    def __init__(self, model, loss_layer):
        self.model = model
        self.loss_layer = loss_layer
        self.pred_history = None
        self.loss_history = None
        self.out = None

    def train(self, dataldr):
        loss_arr = []
        datasize = len(dataldr)
        # for batch, (X, y) in enumerate(
        #     tqdm(dataldr, desc=f"Training", unit="batch", ncols=80)
        # ):
        for batch, (X, y) in enumerate(dataldr):
            if batch == 3:
                break
            X = X.reshape(-1, 1)
            y = y.reshape(-1, 1)
            pred = self.model.forward(X)

            self.loss_layer.regular_for_W(self.model.layers)
            loss = self.loss_layer.forward(pred, y)
            dout = self.loss_layer.backward()
            self.model.backward(dout)
            self.model.grad_dn()

            loss_arr.append(loss)
            if batch % 5 == 0 or batch == datasize - 1:
                current = batch
                print(f"loss:{loss:>7f}[{current:>5d}/{datasize:>5d}]")
        self.loss_history = np.array(loss_arr)

    def test(self, dataldr):
        pred_arr = []
        for X, y in dataldr:
            pred = self.model.forward(X)
            pred_arr.append(pred)
            # print(pred.shape)

        self.pred_history = np.array(pred_arr)

    def loss_eval(self):
        indices = np.array(list(range(0, len(self.loss_history)))) + 1
        plt.plot(indices, self.loss_history)
        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.title("regression loss_fn")
        # plt.legend()
        # plt.show()


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
    bpnn = bpNeuralNetwork(model=regression_model, loss_layer=MSELosslayer())
    bpnn.train(dataldr=datatrain)
    bpnn.test(dataldr=datatest)

    plt.scatter(X, y)
    plt.scatter(X1, y1)
    # print(X1.shape)
    # print(bpnn.pred_history.shape)
    plt.plot(X1, bpnn.pred_history, c="r")
    plt.show()

    bpnn.loss_eval()


if __name__ == "__main__":
    execute_reg()
