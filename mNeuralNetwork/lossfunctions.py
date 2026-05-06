import numpy as np


def crossEntropyLoss(pred, y):
    pred = np.clip(pred, 1e-12, 1.0)
    ret = -np.mean(y * np.log(pred))
    return ret


def MSELoss(pred, y):
    ret = 0.5 * np.mean((pred - y) ** 2)
    return ret
