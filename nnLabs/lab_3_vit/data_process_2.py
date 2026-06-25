import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import conf
from utils import unis
from utils.datalib import roc_eval, roc_eval_multicls, loss_eval
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _loss_eval(
    train_lists: list[list[float]] | None = None,
    test_lists: list[list[float]] | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure:
    if not train_lists:
        train_lists = None
    if not test_lists:
        test_lists = None
    if train_lists is None and test_lists is None:
        raise Exception("Train&Test Lists are all empty.")
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    ax.set_title(title if title is not None else "loss eval")

    xlabel = kwargs.get("xlabel", "epoch")
    ylabel = kwargs.get("ylabel", "loss")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    colors = [
        "red",
        "C0",  # lightblue
        "orange",
        "skyblue",
        "green",
        "blue",
        "pink",
        "purple",
        "brown",
        "gray",
        "black",
    ]

    if len(train_lists) > len(colors):
        raise Exception("Too many lines in one figure")

    labels = kwargs.get("labels", None)
    if train_lists is None:
        for i, test_list in enumerate(test_lists):
            size_n = len(test_list)
            indices = [_ for _ in range(size_n)]
            if labels is None:
                ax.plot(indices, test_list, c=colors[i])
            else:
                ax.plot(indices, test_list, c=colors[i], label=labels[i])
    elif test_lists is None:
        for i, train_list in enumerate(train_lists):
            size_n = len(train_list)
            indices = [_ for _ in range(size_n)]
            if labels is None:
                ax.plot(indices, train_list, c=colors[i])
            else:
                ax.plot(indices, train_list, c=colors[i], label=labels[i])
    else:
        for i, (train_list, test_list) in enumerate(list(zip(train_lists, test_lists))):
            size_n = len(train_list)
            indices = [_ for _ in range(size_n)]
            ax.plot(indices, train_list, c=colors[i], label="train")
            if labels is None:
                ax.plot(indices, test_list, c=colors[1])
            else:
                ax.plot(indices, test_list, c=colors[1], label="test")
    if labels is not None:
        ax.legend(fontsize=7)
    return fig


def load_pkl(dump_folder: Path):
    obj_list = []
    for dump_file in dump_folder.iterdir():
        obj = []
        with open(dump_file, "rb") as f:
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
            obj.append(pickle.load(f))
        obj_list.append((str(dump_file.stem), obj))
    return obj_list


def save_fig(fig):
    if isinstance(fig, list):
        fig = fig[0]
    title = fig.gca().get_title()
    fig.savefig(
        Path(r"D:\PyProjects\data_analysis\lab_3_vit_m\graph") / f"{title}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_fig_multi(fig_arr):
    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(
            Path(r"D:\PyProjects\data_analysis\lab_3_vit_m\graph") / f"{title}.png",
            dpi=150,
            bbox_inches="tight",
        )


# - - is train
# --- is test
def run(dump_path: Path = None):
    fig_arr = []
    if dump_path is None:
        dump_path = conf.data_dump_path
    dump_list = load_pkl(dump_path)
    for file_name, output in dump_list:
        logger = unis.create_logger(conf.log_path, name=file_name)
        roc_list = [(str(file_name), output[0])]
        train_loss_lists = [output[1]]
        test_loss_lists = [output[2]]
        train_acc_lists = [output[3]]
        test_acc_lists = [output[4]]
        labels = [str(file_name)]
        fig = roc_eval_multicls(title="ROC-" + str(file_name), roclist=roc_list, logger=logger)
        save_fig(fig)

        fig = _loss_eval(train_loss_lists, test_loss_lists, "ViT Loss-" + str(file_name), labels=labels)
        save_fig(fig)

        fig = _loss_eval(train_acc_lists, test_acc_lists, "ViT Accuracy-" + str(file_name), labels=labels, ylabel="acc")
        save_fig(fig)
