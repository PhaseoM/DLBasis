import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mNeuralNetwork import mnn
from mNeuralNetwork.layers import *
from mNeuralNetwork.evaluators import *
from utils.datalib.dataloader import DataLoader
from collections import OrderedDict

traindata_cls = "./data/diabetes/classification/diabetes_train"
testdata_cls = "./data/diabetes/classification/diabetes_test"


class bpNeuralNetwork:
    def __init__(self, model: mnn.nnModel, threshold):
        self.model = model
        self.threshold = threshold
        self.pred_history = []
        self.out = None
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def train(self, dataldr):
        avgloss = 0.0
        n_batches = len(dataldr)
        for X, y in dataldr:
            y = y.reshape(-1, 1)
            pred = self.model.forward(X)

            loss = self.model.loss_forward(pred, y)
            self.model.backward(None)
            self.model.grad_dn()

            avgloss += loss
        return avgloss / n_batches

    def test(self, dataldr):
        datasize = dataldr.n
        correct = 0
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0
        self.pred_history = []
        pred_arr = []
        for X, y in dataldr:
            y = y.reshape(-1, 1)
            y_hat = self.model.forward(X)
            pred_arr.append(y_hat)
            self.pred_history.append((y_hat, y))
            pred = (y_hat >= self.threshold).astype(int)
            y_int = y.astype(int)
            self.TP += int(np.sum((pred == 1) & (y_int == 1)))
            self.FP += int(np.sum((pred == 1) & (y_int == 0)))
            self.TN += int(np.sum((pred == 0) & (y_int == 0)))
            self.FN += int(np.sum((pred == 0) & (y_int == 1)))
            correct += int(np.sum(pred == y_int))
        accuracy = correct / datasize
        all_preds = np.concatenate(pred_arr)  # shape (N, 1)
        loss = self.model.loss_layer.forward(all_preds, dataldr.y.reshape(-1, 1))
        return accuracy, loss, pred_arr

    def eval(self):
        print(f"TP={self.TP} FP={self.FP} TN={self.TN} FN={self.FN}")
        print(f"accuracy: {accuracyEval(self.TP, self.FP, self.TN, self.FN):.4f}")
        print(f"precision: {precisionEval(self.TP, self.FP, self.TN, self.FN):.4f}")
        print(f"recall: {recallEval(self.TP, self.FP, self.TN, self.FN):.4f}")
        print(f"f1_score: {f1scoreEval(self.TP, self.FP, self.TN, self.FN):.4f}")
        # roc_Eval([("model", self.pred_history)])


def clsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 1:], data[:, 0]

    return X, y


def execute_cls(
    epochs=100,
    batch_size=32,
    hide_layer_size=64,
    lamb=1e-4,
    is_regular=True,
    outlength=50,
    threshold=0.5,
):
    print(
        f"============ [HyperParams: epochs:{epochs} batch_size:{batch_size} hide_layer_size:{hide_layer_size}] ================"
    )
    modelParams = f"ep={epochs} bs={batch_size} hs={hide_layer_size}"
    # modelParams = f"ep={epochs} bs={batch_size} hs={hide_layer_size} lb={lamb}"
    # print(f"[HyperParams: epochs:{epochs} batch_size:{batch_size} hide_layer_size:{hide_layer_size}]")
    classfication_model = mnn.nnModel(
        layers=OrderedDict(
            [
                ("linear_1", Linearlayer(8, hide_layer_size, init="he")),
                ("relu_1", ReLUlayer()),
                # ("relu_1", Tanhlayer()),
                ("linear_2", Linearlayer(hide_layer_size, hide_layer_size, init="he")),
                ("relu_2", ReLUlayer()),
                # ("relu_2", Tanhlayer()),
                # ("linear_3", Linearlayer(hide_layer_size, hide_layer_size, init="he")),
                # ("relu_3", ReLUlayer()),
                ("linear_end", Linearlayer(hide_layer_size, 1)),
                ("sigmoid_end", Logisticlayer()),
                # ("tanh_end", Tanhlayer()),
            ]
        ),
        loss_layer=CrossEntropyLosslayer(),
        # loss_layer=MSELosslayer(),
        cfg=Config(batchsize=batch_size, step=1e-2, lamb=lamb, is_regular=is_regular),
    )
    figures = []

    X, y = clsdata_process(traindata_cls)
    datatrain = DataLoader(X, y, batch_size, True)
    X1, y1 = clsdata_process(testdata_cls)

    datatest = DataLoader(X1, y1, 1)
    bpnn = bpNeuralNetwork(model=classfication_model, threshold=threshold)

    loss_train_arr = []
    loss_test_arr = []
    for epoch in range(epochs):
        if epoch == 300:
            bpnn.model.cfg.step = 1e-3
            bpnn.model.hyparam_distb()
        if epoch == 600:
            bpnn.model.cfg.step = 1e-4
            bpnn.model.hyparam_distb()
        if epoch == 900:
            bpnn.model.cfg.step = 1e-5
            bpnn.model.hyparam_distb()

        train_loss = bpnn.train(dataldr=datatrain)
        acc, test_loss, pred_arr = bpnn.test(dataldr=datatest)
        loss_train_arr.append((epoch, train_loss))
        loss_test_arr.append((epoch, test_loss))
        if epoch % outlength == 0 or epoch == epochs - 1:
            print(
                f"train_loss:{train_loss:>7f}, test_loss:{test_loss:>7f},test_acc:{acc:>7f}[{epoch:>5d}/{epochs:>5d}]"
            )
    acc, test_loss_, pred_arr = bpnn.test(dataldr=datatest)

    bpnn.eval()

    plt.figure()
    loss_train_arr = np.array(loss_train_arr)
    loss_test_arr = np.array(loss_test_arr)
    plt.plot(loss_train_arr[:, 0], loss_train_arr[:, 1], c="red", label="train_loss")
    plt.plot(loss_test_arr[:, 0], loss_test_arr[:, 1], c="green", label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"cls_loss [{modelParams}]")
    plt.legend()
    figures.append(plt.gcf())
    # plt.show()

    return modelParams, bpnn.pred_history, figures


if __name__ == "__main__":
    execute_cls(epochs=900, is_regular=True)
