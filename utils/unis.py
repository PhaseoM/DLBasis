import os
import sys
import numpy as np
import tomllib
import argparse
import random
import functools
from pathlib import Path
from functools import wraps


def seed_everything(seed: int = 42, is_deterministic: bool = False):
    import os
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if is_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


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


def store_model_params(model, model_dump_path):
    import torch

    Path(model_dump_path).mkdir(parents=True, exist_ok=True)
    randomtoken = "".join(random.choices("0123456789abcdefABCDEF", k=7))
    modelpath = str(model_dump_path) + f"-{randomtoken}"
    torch.save(model.state_dict(), modelpath)


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name="log"):
    import logging
    from termcolor import colored

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    # create console handlers for master process
    # if dist_rank == 0:
    #     console_handler = logging.StreamHandler(sys.stdout)
    #     console_handler.setLevel(logging.DEBUG)
    #     console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    #     logger.addHandler(console_handler)

    # create file handlers
    # file_handler = logging.FileHandler(os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a")
    file_handler = logging.FileHandler(os.path.join(output_dir, f"{name}_rank_{dist_rank}.txt"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger
