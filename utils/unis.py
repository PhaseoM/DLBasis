import sys
import numpy as np
from functools import wraps


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


# feature: output > Terminal & @filename
def tee_output(filename, mode="w"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with open(filename, mode=mode) as f:
                sys.stdout = Tee(sys.__stdout__, f)
                res = func(*args, **kwargs)
                sys.stdout = sys.__stdout__
            return res

        return wrapper

    return decorator


def trapz(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return (0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum()
