import sys
import numpy as np
import matplotlib.pyplot as plt
from nnLabs.lab_1_bpnn import bpnn_reg
from nnLabs.lab_1_bpnn import bpnn_cls
from mNeuralNetwork.evaluators import roc_Eval
from utils import unis

datapath_reg = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\regression.txt"
datapath_cls = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\classification.txt"


@unis.tee_output(datapath_reg)
def exp_reg():
    # np.random.seed(42)
    print("============ Regression ============")
    figpath = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\graphs\\reg\\"
    fig_arr = []

    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    test_loss, figs = bpnn_reg.execute_reg(epochs=300, batch_size=50, hide_layer_size=32)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    # plt.show()


@unis.tee_output(datapath_cls)
def exp_cls():
    # np.random.seed(42)
    print("============ Classification ============")
    ot = 30
    histories = []
    fig_arr = []
    figpath = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\graphs\\cls\\"

    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=300, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=300, batch_size=32, hide_layer_size=128, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    fig_arr.extend(roc_Eval(histories))

    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # exp_reg()
    exp_cls()
