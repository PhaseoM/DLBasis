from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import v2

_DEFAULT_DATA_PATH = Path(__file__).parents[2] / "data"


class MixupCutmixCollate:
    def __init__(self, num_classes):
        mixup = v2.MixUp(alpha=0.8, num_classes=num_classes)
        cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
        self.transform = v2.RandomChoice([mixup, cutmix])

    def __call__(self, batch):
        return self.transform(*default_collate(batch))


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


def load_cifar10_large(
    batch_size,
    img_size,
    rank,
    world_size,
    num_workers=4,
    num_classes=10,
):
    datapath = _DEFAULT_DATA_PATH / "cifar10"
    pixelmean = [0.49139968, 0.48215841, 0.44653091]
    pixelstd = [0.24703223, 0.24348513, 0.26158784]
    transform_normal = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=pixelstd),
    ])
    transform_data_augment = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=pixelstd),
        v2.RandomErasing(),
    ])
    train_data = datasets.CIFAR10(
        root=datapath,
        train=True,
        download=False,
        transform=transform_data_augment,
    )
    test_data = datasets.CIFAR10(
        root=datapath,
        train=False,
        download=False,
        transform=transform_normal,
    )
    train_sampler = DistributedSampler(train_data, rank=rank, num_replicas=world_size, shuffle=True)
    # test_sampler = DistributedSampler(test_data)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=MixupCutmixCollate(num_classes=num_classes),
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        # sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_dataloader, test_dataloader, train_sampler
