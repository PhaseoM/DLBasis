import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.dataloader import DataLoader
from collections import OrderedDict

batch_size_cls = 16  # 64
input_layer_size = 8
hide_layer_size = 256  # 64
output_layer_size = 1

traindata_cls = "./data/diabetes/classification/diabetes_train"
testdata_cls = "./data/diabetes/classification/diabetes_test"

classfication_model = mnn.nnModel(
    layers=OrderedDict(
        [
            ("linear_1", Linearlayer(input_layer_size, hide_layer_size)),
            ("relu_1", ReLUlayer()),
            ("linear_2", Linearlayer(hide_layer_size, hide_layer_size)),
            ("relu_2", ReLUlayer()),
            ("linear_end", Linearlayer(hide_layer_size, output_layer_size)),
            ("sigmoid_end", Logisticlayer()),
        ]
    ),
    cfg=Config(batchsize=batch_size_cls, step=1e-4),
)


class bpNeuralNetwork:
    def __init__(self, model, loss_layer):
        self.model = model
        self.loss_layer = loss_layer
        self.pred_history = []
        self.out = None
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def train(self, dataldr):
        loss_arr = []
        datasize = dataldr.size
        for batch, (X, y) in enumerate(dataldr):
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
        datasize = dataldr.size
        correct = 0
        for X, y in dataldr:
            y = y.reshape(-1, 1)
            y_hat = self.model.forward(X)
            self.pred_history.append((y_hat, y))
            pred = np.where(y_hat >= 0.5, 1, 0)
            # self.TP += (pred == 1) and (y == 1)
            # self.FP += (pred == 1) and (y == 0)
            # self.TN += (pred == 0) and (y == 0)
            # self.FN += (pred == 0) and (y == 1)
            if pred == 1:
                if pred == y:
                    self.TP += 1
                else:
                    self.FP += 1
            else:
                if pred == y:
                    self.TN += 1
                else:
                    self.FN += 1
            correct += np.sum(pred == y)

        correct /= datasize
        print(f"correct:{correct:>7f}")

    def eval(self):
        pass


def clsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 1:], data[:, 0]

    return X, y


def execute_cls():
    X, y = clsdata_process(traindata_cls)
    datatrain = DataLoader(X, y, batch_size_cls, True)
    X1, y1 = clsdata_process(testdata_cls)
    datatest = DataLoader(X1, y1)
    bpnn = bpNeuralNetwork(
        model=classfication_model, loss_layer=CrossEntropyLosslayer()
    )
    bpnn.train(dataldr=datatrain)
    bpnn.test(dataldr=datatest)


if __name__ == "__main__":
    execute_cls()
