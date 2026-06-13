# import torch
import numpy as np
from .config import conf


def main():
    print(f"lr = {conf.lr}")
    print(f"bs = {conf.batch_size}")
    print(f"wd = {conf.weight_decay}")
    print(conf.config_path)
