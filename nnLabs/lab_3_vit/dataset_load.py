import torch
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import v2

tranform_normal = v2.Compose(
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
)
