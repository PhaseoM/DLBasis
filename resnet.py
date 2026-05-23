import torch
import conf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as nnF
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

generator = torch.Generator.manual_seed(42)

datapath = "./data/cifar10/"
pixelmean = [0.49139968, 0.48215841, 0.44653091]
pixelstd = [0.24703223, 0.24348513, 0.26158784]

transform_normal = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=[1.0, 1.0, 1.0]),
    ]
)
transform_data_augment = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Pad(4),
        v2.RandomCrop(32),
        v2.RandomHorizontalFlip(),
        v2.Normalize(mean=pixelmean, std=[1.0, 1.0, 1.0]),
    ]
)

transform_train_val = (
    transform_data_augment if conf.is_data_augment else transform_normal
)

train_val_data = datasets.CIFAR10(
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

train_data, val_data = torch.utils.data.random_split(
    dataset=train_val_data,
    lengths=[45000, 5000],
    generator=generator,
)

test_data = datasets.CIFAR10(
    root=datapath,
    train=False,
    download=True,
    transform=transform_normal,
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


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.is_projection = input_channels != output_channels
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
        )
        self.projection = (
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, X):
        H = self.F(X)
        X = self.projection(X)
        H = H + X
        return nnF.relu(H)


def res_block(size_n, input_channels, output_channels):
    res = []
    res.append(ResidualBlock(input_channels, output_channels))
    for i in range(1, size_n):
        res.append(ResidualBlock(output_channels, output_channels))
    return res_block


class ResNet(nn.Module):
    def __init__(self):
        super.__init__()
        self.nnModel = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, padding=1
            ),  # map_size: 32
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            *res_block(conf.block_n, 16, 16),  # map_size: 32
            *res_block(conf.block_n, 16, 32),  # map_size: 16
            *res_block(conf.block_n, 32, 64),  # map_size: 8
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, X):
        return self.nnModel(X)


# 352 iter
def train(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn,
):
    model.train()
    size = len(dataloader.dataset)
    n_batch = dataloader.batch_size
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        logits = model.forward(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        if batch % 100:
            loss_, current = loss.item(), dataloader.batch_size * batch + len(X)
            print(f"loss: {loss_:>7f} [{current:>5d}/{size:>5d}]")
    return train_loss / n_batch


def validate(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    val_loss, correct = 0, 0
    size_n = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            val_loss += loss.item() * X.size(0)

            # prob = nnF.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return val_loss / size_n, correct / size_n


def test(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    size_n = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss.item() * X.size(0)

            # prob = nnF.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return test_loss / size_n, correct / size_n


if __name__ == "__main__":
    train_dataloader = DataLoader(train_data, batch_size=conf.batchsize)
    val_dataloader = DataLoader(val_data, batch_size=conf.batchsize)
    test_dataloader = DataLoader(test_data, batch_size=conf.batchsize)
    model = ResNet()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
        momentum=conf.momentum,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[320000, 480000], gamma=0.1
    )
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(conf.epochs):
        print(f"========== Epoch: [{epoch}/{conf.epochs}] ==========")
        train_loss = train(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
        )
        val_loss, val_accuracy = validate(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
        )
        test_loss, test_accuracy = test(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
        )
        print(f"train loss: {train_loss}")
        print(f"validate loss: {val_loss}, accuracy: {val_accuracy}")
        print(f"test loss: {test_loss}, accuracy: {test_accuracy}")
