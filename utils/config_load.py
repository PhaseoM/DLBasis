import argparse
import tomllib
from pathlib import Path


def _get_config_path(config_path: Path, config_path_prefix: Path):
    if config_path.is_absolute():
        return config_path

    return config_path_prefix / config_path


def load_config(args, conf_var):
    """
    args : argparse.Namespace
    conf_var must have the method of to_dict-like
    """
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
