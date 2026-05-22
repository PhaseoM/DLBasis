import torch
import conf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as nnF
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


datapath = "./data/cifar10/"

pixelmean = [0.49139968, 0.48215841, 0.44653091]
pixelstd = [0.24703223, 0.24348513, 0.26158784]

train_data = datasets.CIFAR10(
    root=datapath,
    train=True,
    download=True,
    transform=v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=pixelmean, std=[1.0, 1.0, 1.0]),
        ]
    ),
)

test_data = datasets.CIFAR10(
    root=datapath,
    train=False,
    download=True,
    transform=v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=pixelmean, std=[1.0, 1.0, 1.0]),
        ]
    ),
)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def get_pixel_mean():
    train_data = datasets.CIFAR10(root=datapath, train=True, download=True)
    pixel_mean = train_data.data.mean(axis=(0, 1, 2))
    return pixel_mean


class NormalBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        strides = 1 + int(input_channels != output_channels)
        self.F = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                stride=strides,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.F(X)


def res_block(size_n, input_channels, output_channels):
    res = []


if __name__ == "__main__":
    print(get_pixel_mean() / 255)
