import sys
import numpy as np
import tomllib
import argparse
from pathlib import Path
from functools import wraps


def seed_everything(seed: int = 42):
    import os
    import torch

    # random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


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


def _args_parse_default_():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--config", dest="config_path", default=argparse.SUPPRESS)
    parse.add_argument("-lr", "--learning-rate", dest="lr", type=float, default=argparse.SUPPRESS)
    parse.add_argument("-bs", "--batch-size", type=int, default=argparse.SUPPRESS)
    parse.add_argument("-wd", "--weight-decay", type=float, default=argparse.SUPPRESS)

    return parse.parse_args()


def _get_config_path(config_path: Path, config_path_prefix: Path):
    if config_path.is_absolute():
        return config_path

    return config_path_prefix / config_path


def load_config(conf_var, args=None):
    """
    args : argparse.Namespace
    conf_var must have the method of to_dict-like
    """
    if args is None:
        args = _args_parse_default_()
    config_path = Path(conf_var.config_path)
    if hasattr(args, "config_path"):
        config_path = _get_config_path(Path(args.config_path), config_path.parent)
        conf_var.config_path = config_path
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    hyper_param = config["HyperParam"]
    conf_dict = vars(conf_var)
    for key, value in hyper_param.items():
        conf_dict[key] = value

    for key, value in vars(args).items():
        if key != "config_path":
            conf_dict[key] = value
