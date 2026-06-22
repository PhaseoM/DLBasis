import torch
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

_DEFAULT_DATA_PATH = Path(__file__).parents[2] / "data"


def load_cifar10(batch_size):
    datapath = _DEFAULT_DATA_PATH / "cifar10"
    pixelmean = [0.49139968, 0.48215841, 0.44653091]
    pixelstd = [0.24703223, 0.24348513, 0.26158784]
    tranform_normal = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=pixelstd),
    ])
    transform_data_augment = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Pad(4),
        v2.RandomCrop(32),
        v2.RandomHorizontalFlip(),
        v2.Normalize(mean=pixelmean, std=pixelstd),
    ])
    train_data = datasets.CIFAR10(
        root=datapath,
        train=True,
        download=True,
        transform=transform_data_augment,
    )
    test_data = datasets.CIFAR10(
        root=datapath,
        train=False,
        download=True,
        transform=tranform_normal,
    )
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def load_cifar10_large(batch_size, img_size=224):
    datapath = _DEFAULT_DATA_PATH / "cifar10"
    pixelmean = [0.49139968, 0.48215841, 0.44653091]
    pixelstd = [0.24703223, 0.24348513, 0.26158784]
    tranform_normal = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=pixelstd),
    ])
    transform_data_augment = v2.Compose([
        v2.ToImage(),
        v2.Pad(4),
        v2.RandomCrop(32),
        v2.RandomHorizontalFlip(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=pixelstd),
    ])
    train_data = datasets.CIFAR10(
        root=datapath,
        train=True,
        download=True,
        transform=transform_data_augment,
    )
    test_data = datasets.CIFAR10(
        root=datapath,
        train=False,
        download=True,
        transform=tranform_normal,
    )
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader
