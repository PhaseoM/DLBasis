import pickle
import numpy as np
import matplotlib.pyplot as plt
from . import conf
from utils.datalib import roc_eval, roc_eval_multicls, loss_eval


def load_pkl(name: str, layers):
    obj = []
    with open(conf.pkl_dump_filepath + f"{name}_{layers}.pkl", "rb") as f:
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
        obj.append(pickle.load(f))
    return obj


def info_save():
    fig_arr_ = []
    plain_20 = load_pkl("plain", 20)
    plain_32 = load_pkl("plain", 32)
    plain_56 = load_pkl("plain", 56)
    resnet_20 = load_pkl("resnet", 20)
    resnet_32 = load_pkl("resnet", 32)
    resnet_56 = load_pkl("resnet", 56)
    plain_110 = load_pkl("plain", 110)
    resnet_110 = load_pkl("resnet", 110)

    def __roc_eval__(fig_arr):
        roc_list = []
        roc_list.append(("Plain-20", plain_20[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-Plain-20", roclist=roc_list))
        roc_list = []
        roc_list.append(("Plain-32", plain_32[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-Plain-32", roclist=roc_list))
        roc_list = []
        roc_list.append(("Plain-56", plain_56[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-Plain-56", roclist=roc_list))
        roc_list = []

        roc_list.append(("ResNet-20", resnet_20[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-ResNet-20", roclist=roc_list))
        roc_list = []
        roc_list.append(("ResNet-32", resnet_32[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-ResNet-32", roclist=roc_list))
        roc_list = []
        roc_list.append(("ResNet-56", resnet_56[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-ResNet-56", roclist=roc_list))
        roc_list = []

    def __loss_eval__(fig_arr):
        train_lists, test_lists, labels = [], [], []
        train_lists.append(plain_20[1])
        test_lists.append(plain_20[2])
        labels.append("Plain-20")
        train_lists.append(plain_32[1])
        test_lists.append(plain_32[2])
        labels.append("Plain-32")
        train_lists.append(plain_56[1])
        test_lists.append(plain_56[2])
        labels.append("Plain-56")
        fig_arr.append(
            loss_eval(train_lists, test_lists, "PlainCNN loss", labels=labels)
        )
        train_lists, test_lists, labels = [], [], []

        train_lists.append(resnet_20[1])
        test_lists.append(resnet_20[2])
        labels.append("ResNet-20")
        train_lists.append(resnet_32[1])
        test_lists.append(resnet_32[2])
        labels.append("ResNet-32")
        train_lists.append(resnet_56[1])
        test_lists.append(resnet_56[2])
        labels.append("ResNet-56")
        fig_arr.append(loss_eval(train_lists, test_lists, "ResNet loss", labels=labels))
        train_lists, test_lists, labels = [], [], []

    def __error_eval__(fig_arr):
        train_lists, test_lists, labels = [], [], []
        train_lists.append(plain_20[3])
        test_lists.append(plain_20[4])
        labels.append("Plain-20")
        train_lists.append(plain_32[3])
        test_lists.append(plain_32[4])
        labels.append("Plain-32")
        train_lists.append(plain_56[3])
        test_lists.append(plain_56[4])
        labels.append("Plain-56")
        fig_arr.append(
            loss_eval(
                train_lists, test_lists, "PlainCNN error", labels=labels, ylabel="error"
            )
        )
        train_lists, test_lists, labels = [], [], []

        train_lists.append(resnet_20[3])
        test_lists.append(resnet_20[4])
        labels.append("ResNet-20")
        train_lists.append(resnet_32[3])
        test_lists.append(resnet_32[4])
        labels.append("ResNet-32")
        train_lists.append(resnet_56[3])
        test_lists.append(resnet_56[4])
        labels.append("ResNet-56")
        fig_arr.append(
            loss_eval(
                train_lists, test_lists, "ResNet error", labels=labels, ylabel="error"
            )
        )
        train_lists, test_lists, labels = [], [], []

    def __deep_net__(fig_arr):
        roc_list = []
        train_lists, test_lists, labels = [], [], []

        roc_list.append(("Plain-110", plain_110[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-Plain-110", roclist=roc_list))
        roc_list = []
        roc_list.append(("ResNet-110", resnet_110[0]))
        fig_arr.append(roc_eval_multicls(title="ROC-ResNet-110", roclist=roc_list))
        roc_list = []

        train_lists.append(plain_110[1])
        test_lists.append(plain_110[2])
        labels.append("Plain-110")
        train_lists.append(resnet_110[1])
        test_lists.append(resnet_110[2])
        labels.append("ResNet-110")
        fig_arr.append(
            loss_eval(train_lists, test_lists, "Plain_ResNet-110 loss", labels=labels)
        )
        train_lists, test_lists, labels = [], [], []

        train_lists.append(plain_110[3])
        test_lists.append(plain_110[4])
        labels.append("Plain-110")
        train_lists.append(resnet_110[3])
        test_lists.append(resnet_110[4])
        labels.append("ResNet-110")
        fig_arr.append(
            loss_eval(
                train_lists,
                test_lists,
                "Plain_ResNet-110 error",
                labels=labels,
                ylabel="error",
            )
        )
        train_lists, test_lists, labels = [], [], []

    __roc_eval__(fig_arr_)
    __loss_eval__(fig_arr_)
    __error_eval__(fig_arr_)
    # __deep_net__(fig_arr_)

    for fig in fig_arr_:
        title = fig.gca().get_title()
        fig.savefig(
            conf.fig_output_filepath + f"{title}.png",
            dpi=150,
            bbox_inches="tight",
        )
