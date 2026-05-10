import torch
import torch.nn as nn
import numpy as np


class bpNeuralNetwork(nn.Module):
    def __init__(self):
        pass


def regsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        X, y = data[:, 0], data[:, 1]

    return X, y


if __name__ == "__main__":
    print(torch.cuda.is_available())
