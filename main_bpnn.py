import numpy as np
import torch
import matplotlib.pyplot as plt
import mNeuralNetwork.nnmodel as mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.dataloader import DataLodaer
from collections import OrderedDict

batch_size_regs = 100
batch_size_cls = 64

traindata_regs = "./data/diabetes/regression/sinc_train"
testdata_regs = "./data/diabetes/regression/sinc_test"


traindata_cls = "./data/diabetes/classification/diabetes_train"
testdata_cls = "./data/diabetes/classification/diabetes_test"

regression_model = mnn.nnModel(
    model=OrderedDict(
        [
            ("linear_1", Linearlayer(batch_size_regs * 1, 256)),
            ("relu", ReLUlayer()),
            ("linear_2", Linearlayer(256, batch_size_regs)),
            ("sigmoid", Logisticlayer()),
        ]
    ),
    step=0.1,
)

classfication_model = mnn.nnModel(
    model=OrderedDict(
        [
            ("linear_1", Linearlayer(batch_size_cls * 8, 256)),
            ("relu", ReLUlayer()),
            ("linear_2", Linearlayer(256, batch_size_cls)),
            ("sigmoid", Logisticlayer()),
        ]
    ),
    step=0.1,
)


class bpNeuralNetwork:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.out = None

    def train(self, X, y):
        pass

    def test(self, X, y):
        pass

    def eval(self):
        pass


def regsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 0], data[:, 1]

    return X, y


def clsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 1:], data[:, 0]

    return X, y


def regs_main():
    X, y = regsdata_process(traindata_regs)


def cls_main():
    X, y = clsdata_process(traindata_cls)
    print(y)


if __name__ == "__main__":
    regs_main()
    cls_main()
