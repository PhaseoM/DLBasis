import torch
import pickle
import numpy as np
import utils.unis as unis
import torch.nn.functional as nnF
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from . import conf

seed = 2650
torch.manual_seed(seed)

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
    return res


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.nnModel = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),  # map_size: 32
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            *res_block(18, 16, 16),  # map_size: 32
            *res_block(18, 16, 32),  # map_size: 16
            *res_block(18, 32, 64),  # map_size: 8
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, X):
        return self.nnModel(X)


# 391 iter
def train(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn,
):
    model.train()
    size_n = len(dataloader.dataset)
    train_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item() * X.size(0)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        if batch % 100 == 0:
            loss_, current = loss.item(), dataloader.batch_size * batch + len(X)
            print(f"loss: {loss_:>7f} [{current:>5d}/{size_n:>5d}]")
    return train_loss / size_n, correct / size_n


def test(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    size_n = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss.item() * X.size(0)

            # prob = nnF.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return test_loss / size_n, correct / size_n


def test_prob(dataloader: DataLoader, model: nn.Module):
    model.eval()
    history = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            prob = nnF.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            y_true = y.detach().cpu().tolist()
            preds = pred.detach().cpu().tolist()
            probs = prob.detach().cpu().tolist()

            history.extend(list(zip(y_true, preds, probs)))
    return history


def main():
    transform_train = (
        transform_data_augment if conf.is_data_augment else transform_normal
    )

    @unis.tee_output(conf.info_output_filepath + f"resnet_{18 * 6 + 2}")
    def run():
        print(f"resnet_{18 * 6 + 2}")
        train_data = datasets.CIFAR10(
            root=datapath,
            train=True,
            download=True,
            transform=transform_train,
        )

        test_data = datasets.CIFAR10(
            root=datapath,
            train=False,
            download=True,
            transform=transform_normal,
        )
        train_dataloader = DataLoader(
            train_data, batch_size=conf.batchsize, shuffle=True
        )
        test_dataloader = DataLoader(test_data, batch_size=conf.batchsize)
        model = ResNet().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=conf.learning_rate,
            weight_decay=conf.weight_decay,
            momentum=conf.momentum,
        )

        def lr_lambda(step):
            if step < 400:
                return 0.1
            elif step < 32000:
                return 1.0
            elif step < 48000:
                return 0.1
            else:
                return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lr_lambda
        )
        loss_fn = nn.CrossEntropyLoss()
        train_loss_list = []
        test_loss_list = []
        train_error_list = []
        test_error_list = []
        for epoch in range(conf.epochs):
            print(f"========== Epoch: [{epoch}/{conf.epochs}] ==========")
            train_loss, train_accuracy = train(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
            )
            test_loss, test_accuracy = test(
                dataloader=test_dataloader,
                model=model,
                loss_fn=loss_fn,
            )
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_error = 1 - train_accuracy
            test_error = 1 - test_accuracy
            train_error_list.append(train_error)
            test_error_list.append(test_error)

            print(f"train loss: {train_loss:>7f}")
            print(f"test loss: {test_loss:>7f}, accuracy: {test_accuracy:>7f}")

        history = test_prob(dataloader=test_dataloader, model=model)
        with open(conf.pkl_dump_filepath + f"resnet_{18 * 6 + 2}.pkl", "wb") as f:
            pickle.dump(history, f)
            pickle.dump(train_loss_list, f)
            pickle.dump(test_loss_list, f)
            pickle.dump(train_error_list, f)
            pickle.dump(test_error_list, f)

    run()


if __name__ == "__main__":
    main()
