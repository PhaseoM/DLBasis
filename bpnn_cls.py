import numpy as np
import matplotlib.pyplot as plt
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from mNeuralNetwork.dataloader import DataLodaer
from collections import OrderedDict

batch_size_cls = 64

traindata_cls = "./data/diabetes/classification/diabetes_train"
testdata_cls = "./data/diabetes/classification/diabetes_test"

classfication_model = mnn.nnModel(
    model=OrderedDict(
        [
            ("linear_1", Linearlayer(batch_size_cls * 8, 256)),
            ("relu", ReLUlayer()),
            ("linear_2", Linearlayer(256, batch_size_cls)),
            ("sigmoid", Logisticlayer()),
        ]
    ),
    cfg=Config(batchsize=batch_size_cls),
)


def clsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 1:], data[:, 0]

    return X, y


def execute_cls():
    X, y = clsdata_process(traindata_cls)
    datatrain = DataLodaer(batch_size_cls, X, y)
    X, y = clsdata_process(testdata_cls)
    datatest = DataLodaer(batch_size_cls, X, y)


if __name__ == "__main__":
    execute_cls()
