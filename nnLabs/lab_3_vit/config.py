import argparse
import tomllib
from pathlib import Path
from dataclasses import dataclass
from utils.config_load import load_config

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "lab_3_vit" / "default.toml"


@dataclass
class Config:
    config_path: Path = _CONFIG_PATH
    lr: float = 0.00001
    batch_size: int = 10000
    weight_decay: float = 0.1


def _args_parse_():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--config", dest="config_path", default=argparse.SUPPRESS)
    parse.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parse.add_argument("-bs", "--batch-size", type=int, default=argparse.SUPPRESS)
    parse.add_argument("-wd", "--weight-decay", type=float, default=argparse.SUPPRESS)

    return parse.parse_args()


conf = Config()
load_config(_args_parse_(), conf)
