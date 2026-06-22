import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import conf
from utils import unis
from utils.datalib import roc_eval, roc_eval_multicls, loss_eval


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


# - - is train
# --- is test
def run():
    fig_arr = []
    dump_list = load_pkl(conf.data_dump_path)
    for file_name, output in dump_list:
        logger = unis.create_logger(conf.log_path, name=file_name)
        roc_list = [(str(file_name), output[0])]
        train_loss_lists = [output[1]]
        test_loss_lists = [output[2]]
        train_acc_lists = [output[3]]
        test_acc_lists = [output[4]]
        labels = [str(file_name)]
        fig_arr.append(roc_eval_multicls(title="ROC-" + str(file_name), roclist=roc_list, logger=logger))
        fig_arr.append(loss_eval(train_loss_lists, test_loss_lists, "ViT Loss-" + str(file_name), labels=labels))
        fig_arr.append(
            loss_eval(train_acc_lists, test_acc_lists, "ViT Accuracy-" + str(file_name), labels=labels, ylabel="acc")
        )

    for fig in fig_arr:
        title = fig.gca().get_title()
        fig.savefig(
            conf.graph_path / f"{title}.png",
            dpi=150,
            bbox_inches="tight",
        )
