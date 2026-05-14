import sys
import numpy as np
import matplotlib.pyplot as plt
from nnLabs.lab_1_bpnn import bpnn_reg
from nnLabs.lab_1_bpnn import bpnn_cls
from mNeuralNetwork.evaluators import roc_Eval
from utils import unis

datapath_reg = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\regression.txt"
datapath_cls = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\classification.txt"


def exp():
    histories = []
    fig_arr = []
    random_seed = 3412
    print("======== Benchmark =========")
    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, threshold=0.22)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)
    fig_arr.extend(roc_Eval(histories, modelp))
    plt.show()


@unis.tee_output(datapath_reg)
def exp_reg():
    random_seed = 0
    print("============ Regression ============")
    print(f"random_seed:{random_seed}")

    figpath = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\graphs\\reg\\"
    fig_arr = []

    # benchmark
    print("============== Benchmark ==============")
    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    # epochs Test
    print("============== Epochs Test ==============")
    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=300, batch_size=10, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=600, batch_size=10, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=900, batch_size=10, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    # for fig in fig_arr:
    #     title = fig.gca().get_title()
    #     fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    # return

    # batch_size Test
    print("============== Batch_size Test ==============")
    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=1, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=5, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=30, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=50, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=100, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=500, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=2500, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=5000, hide_layer_size=64)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    # hidden_size Test
    print("============== Hidden_size Test ==============")
    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=8)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=16)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=32)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=128)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    np.random.seed(random_seed)
    test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=256)
    fig_arr.extend(figs)
    print(f"test_loss:{test_loss}")

    # regular Test
    # print("============== Regular Test ==============")
    # np.random.seed(random_seed)
    # test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64, lamb=0)
    # fig_arr.extend(figs)
    # print(f"test_loss:{test_loss}")

    # np.random.seed(random_seed)
    # test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64, lamb=1e-1)
    # fig_arr.extend(figs)
    # print(f"test_loss:{test_loss}")

    # np.random.seed(random_seed)
    # test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64, lamb=1e-2)
    # fig_arr.extend(figs)
    # print(f"test_loss:{test_loss}")

    # np.random.seed(random_seed)
    # test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64, lamb=1e-3)
    # fig_arr.extend(figs)
    # print(f"test_loss:{test_loss}")

    # np.random.seed(random_seed)
    # test_loss, figs = bpnn_reg.execute_reg(epochs=1200, batch_size=10, hide_layer_size=64, lamb=1e-5)
    # fig_arr.extend(figs)
    # print(f"test_loss:{test_loss}")
    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    # plt.show()


@unis.tee_output(datapath_cls)
def exp_cls():
    # 3412
    random_seed = 3412
    print("============ Classification ============")
    print(f"random_seed:{random_seed}")
    ot = 30
    histories = []
    fig_arr = []
    figpath = "D:\\PyProjects\\data_analysis\\lab_1_bpnn\\graphs\\cls\\"

    # benchmark
    print("======== Benchmark =========")
    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)
    # epochs Test
    print("============== Epochs Test ==============")
    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=300, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=900, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=1200, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    fig_arr.extend(roc_Eval(histories, modelp))
    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    fig_arr = []
    histories = []
    # return

    # benchmark
    print("======== Benchmark =========")
    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)
    # batch_size Test
    print("============== Batch_size Test ==============")
    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=8, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=16, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=64, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=128, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=288, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=576, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    fig_arr.extend(roc_Eval(histories, modelp))
    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    fig_arr = []

    # benchmark
    print("======== Benchmark =========")
    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)
    # hidden_size Test
    print("============== Hidden_size Test ==============")

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=8, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=16, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=128, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    np.random.seed(random_seed)
    modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=256, outlength=ot)
    histories.append((modelp, pred_h))
    fig_arr.extend(figs)

    fig_arr.extend(roc_Eval(histories, modelp))
    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    fig_arr = []
    histories = []

    # benchmark
    # print("======== Benchmark =========")
    # np.random.seed(random_seed)
    # modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, outlength=ot)
    # histories.append((modelp, pred_h))
    # fig_arr.extend(figs)
    # regular Test
    # print("============== Regular Test ==============")
    # np.random.seed(random_seed)
    # modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, lamb=0, outlength=ot)
    # histories.append((modelp, pred_h))
    # fig_arr.extend(figs)

    # np.random.seed(random_seed)
    # modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, lamb=1e-1, outlength=ot)
    # histories.append((modelp, pred_h))
    # fig_arr.extend(figs)

    # np.random.seed(random_seed)
    # modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, lamb=1e-2, outlength=ot)
    # histories.append((modelp, pred_h))
    # fig_arr.extend(figs)

    # np.random.seed(random_seed)
    # modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, lamb=1e-3, outlength=ot)
    # histories.append((modelp, pred_h))
    # fig_arr.extend(figs)

    # np.random.seed(random_seed)
    # modelp, pred_h, figs = bpnn_cls.execute_cls(epochs=600, batch_size=32, hide_layer_size=64, lamb=1e-5, outlength=ot)
    # histories.append((modelp, pred_h))
    # fig_arr.extend(figs)

    # fig_arr.extend(roc_Eval(histories, modelp))

    # for fig in fig_arr:
    #     title = fig.gca().get_title()
    #     fig.savefig(figpath + f"{title}.png", dpi=150, bbox_inches="tight")
    # fig_arr = []
    # histories = []
    # plt.show()


if __name__ == "__main__":
    # exp_reg()
    exp_cls()
