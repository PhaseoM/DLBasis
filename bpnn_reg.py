import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.dataloader import DataLodaer
from mNeuralNetwork.config import Config
from collections import OrderedDict

batch_size_reg = 100

traindata_reg = "./data/diabetes/regression/sinc_train"
testdata_reg = "./data/diabetes/regression/sinc_test"

regression_model = mnn.nnModel(
    model=OrderedDict(
        [
            ("linear_1", Linearlayer(batch_size_reg * 1, 256)),
            ("relu", ReLUlayer()),
            ("linear_2", Linearlayer(256, batch_size_reg)),
            ("sigmoid", Logisticlayer()),
        ]
    ),
    cfg=Config(batchsize=batch_size_reg),
)


def regsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 0], data[:, 1]

    return X, y


def execute_reg():
    X, y = regsdata_process(traindata_reg)
    datatrain = DataLodaer(batch_size_reg, X, y)
    X, y = regsdata_process(testdata_reg)
    datatest = DataLodaer(batch_size_reg, X, y)


if __name__ == "__main__":
    execute_reg()
