import argparse
import tomllib
from pathlib import Path
from dataclasses import dataclass
from utils.unis import load_config

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "lab_3_vit" / "default.toml"
_LOG_PATH = Path(__file__).parents[2] / "data_analysis" / "lab_3_vit" / "logging_info"
_MODEL_DUMP_PATH = Path(__file__).parent / "models"
_DATA_DUMP_PATH = Path(__file__).parents[2] / "data_analysis" / "lab_3_vit" / "pkldump"
_GRAPH_PATH = Path(__file__).parents[2] / "data_analysis" / "lab_3_vit" / "graph"


@dataclass
class Config:
    config_path: Path = _CONFIG_PATH
    log_path: Path = _LOG_PATH
    data_dump_path: Path = _DATA_DUMP_PATH
    model_dump_path: Path = _MODEL_DUMP_PATH
    graph_path: Path = _GRAPH_PATH

    seed: int = 3407

    epochs: int = 160
    lr: float = 0.01
    batch_size: int = 128
    weight_decay: float = 0.1
    dropout: float = 0
    beta_1: float = 0.9
    beta_2: float = 0.999
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    # total_steps: int = 64000

    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    mlp_hiddens: int = 3072
    img_size: int = 32
    patch_size: int = 4
    num_classes: int = 10

    def __str__(self):
        return "\n" + "\n".join([f"        {key} = {value} " for key, value in self.__dict__.items()])


conf = Config()


def _args_parse_():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--config", dest="config_path", default=argparse.SUPPRESS)
    parse.add_argument("-lr", "--learning-rate", dest="lr", type=float, default=argparse.SUPPRESS)
    parse.add_argument("-bs", "--batch-size", type=int, default=argparse.SUPPRESS)
    parse.add_argument("-wd", "--weight-decay", type=float, default=argparse.SUPPRESS)

    return parse.parse_args()


def load_configuration():
    load_config(conf, _args_parse_())
