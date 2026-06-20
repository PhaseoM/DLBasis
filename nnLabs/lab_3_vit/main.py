# import torch
import numpy as np
from . import config
from . import vit


def bootstrap():
    config.load_configuration()
    vit.run()
