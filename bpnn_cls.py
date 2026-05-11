import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from utils.dataplib import DataLoader
from collections import OrderedDict

epochs_cls = 600
batch_size_cls = 32
input_layer_size = 8
hide_layer_size = 64
output_layer_size = 1

traindata_cls = "./data/diabetes/classification/diabetes_train"
testdata_cls = "./data/diabetes/classification/diabetes_test"

classfication_model = mnn.nnModel(
    layers=OrderedDict(
        [
            ("linear_1", Linearlayer(input_layer_size, hide_layer_size, init="he")),
            ("relu_1", ReLUlayer()),
            # ("relu_1", Tanhlayer()),
            ("linear_2", Linearlayer(hide_layer_size, hide_layer_size, init="he")),
            ("relu_2", ReLUlayer()),
            # ("relu_2", Tanhlayer()),
            # ("linear_3", Linearlayer(hide_layer_size, hide_layer_size, init="he")),
            # ("relu_3", ReLUlayer()),
            ("linear_end", Linearlayer(hide_layer_size, output_layer_size)),
            ("sigmoid_end", Logisticlayer()),
            # ("tanh_end", Tanhlayer()),
        ]
    ),
    loss_layer=CrossEntropyLosslayer(),
    # loss_layer=MSELosslayer(),
    cfg=Config(batchsize=batch_size_cls, step=1e-1, is_regular=True, lamb=1e-4),
)


class bpNeuralNetwork:
    def __init__(self, model):
        self.model = model
        self.pred_history = []
        self.out = None
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def train(self, dataldr, epochs=200, log_every=10):
        loss_arr = []
        n_batches = dataldr.size
        for epoch in range(epochs):
            if epoch == 300:
                self.model.cfg.step = 1e-3
                self.model.hyparam_distb()
            if epoch == 600:
                self.model.cfg.step = 1e-4
                self.model.hyparam_distb()
            epoch_loss = 0.0
            # pbar = tqdm(dataldr, desc=f"Epoch {epoch}", ncols=100)
            for X, y in dataldr:
                y = y.reshape(-1, 1)
                pred = self.model.forward(X)
                loss = self.model.loss_forward(pred, y)
                self.model.backward(None)
                self.model.grad_dn()
                epoch_loss += loss
            loss_arr.append(epoch_loss / n_batches)
            if epoch % log_every == 0 or epoch == epochs - 1:
                print(f"loss:{epoch_loss / n_batches:>7f}[{epoch:>5d}/{epochs:>5d}]")
                # pbar.set_postfix(
                #     loss=f"{epoch_loss / n_batches:>7f}",
                #     Training=f"[{epoch:>5d}/{epochs:>5d}]",
                # )
        self.loss_history = np.array(loss_arr)

    def test(self, dataldr):
        datasize = dataldr.n
        correct = 0
        for X, y in dataldr:
            y = y.reshape(-1, 1)
            y_hat = self.model.forward(X)
            self.pred_history.append((y_hat, y))
            pred = (y_hat >= 0.5).astype(int)
            # pred = (y_hat >= 0).astype(int)
            y_int = y.astype(int)
            self.TP += int(np.sum((pred == 1) & (y_int == 1)))
            self.FP += int(np.sum((pred == 1) & (y_int == 0)))
            self.TN += int(np.sum((pred == 0) & (y_int == 0)))
            self.FN += int(np.sum((pred == 0) & (y_int == 1)))
            correct += int(np.sum(pred == y_int))
        accuracy = correct / datasize
        # print(f"accuracy:{accuracy:.4f}")
        # print(f"TP={self.TP} FP={self.FP} TN={self.TN} FN={self.FN}")
        return accuracy

    def eval(self):
        print(f"TP={self.TP} FP={self.FP} TN={self.TN} FN={self.FN}")
        print(f"accuracy: {accuracyEval(self.TP,self.FP,self.TN,self.FN):.4f}")
        print(f"precision: {precisionEval(self.TP,self.FP,self.TN,self.FN):.4f}")
        print(f"recall: {recallEval(self.TP,self.FP,self.TN,self.FN):.4f}")
        print(f"f1_score: {f1scoreEval(self.TP,self.FP,self.TN,self.FN):.4f}")
        roc_Eval(self.TP, self.FP, self.TN, self.FN, self.pred_history)


def clsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 1:], data[:, 0]

    return X, y


def execute_cls():
    X, y = clsdata_process(traindata_cls)
    datatrain = DataLoader(X, y, batch_size_cls, True)
    X1, y1 = clsdata_process(testdata_cls)

    datatest = DataLoader(X1, y1, 1)
    bpnn = bpNeuralNetwork(model=classfication_model)
    bpnn.train(dataldr=datatrain, epochs=epochs_cls)
    bpnn.test(dataldr=datatest)

    bpnn.eval()


if __name__ == "__main__":
    execute_cls()
