import tqdm
import torch
import pickle
import numpy as np
import torch.nn.functional as nnF
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from utils import unis
from .config import conf
from . import vit
from . import config
from . import lr_scheduler
from . import dataset_load
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights


def train(
    device,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn,
):
    model.train()
    train_loss, correct = 0, 0
    size_n = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item() * X.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        # if batch % 100 == 0:
        #     loss_, current = loss.item(), dataloader.batch_size * batch + len(X)
        #     print(f"loss: {loss_:>7f} [{current:>5d}/{size_n:>5d}]")
    return train_loss / size_n, correct / size_n


def test(
    device,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn,
):
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

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

    return test_loss / size_n, correct / size_n


def test_info(device, dataloader: DataLoader, model: nn.Module):
    model.eval()
    info = []
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

            info.extend(list(zip(y_true, preds, probs)))
    return info


def run():
    config.load_configuration()

    conf_stem = conf.config_path.stem
    file_stem = Path(__file__).stem + f"__{conf_stem}"
    logger = unis.create_logger(conf.log_path, name=file_stem)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    unis.seed_everything(conf.seed)
    train_dataloader, test_dataloader = dataset_load.load_cifar10_fine_tuning(conf.batch_size, conf.img_size)

    # weights = ViT_B_16_Weights.DEFAULT
    # model = vit_b_16(weights=weights)
    weights = ViT_B_32_Weights.DEFAULT
    model = vit_b_32(weights=weights)
    model.heads.head = nn.Linear(model.hidden_dim, conf.num_classes)
    nn.init.zeros_(model.heads.head.weight)
    nn.init.zeros_(model.heads.head.bias)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum)
    total_steps = conf.epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_scheduler.warmup_cosine_decay(conf.warmup_steps, total_steps),
    )
    loss_fn_train = nn.CrossEntropyLoss()
    loss_fn_test = nn.CrossEntropyLoss()

    train_loss_list = []
    test_loss_list = []
    train_correct_list = []
    test_correct_list = []

    pbar = tqdm.tqdm(
        range(conf.epochs),
        dynamic_ncols=True,
    )

    logger.info(conf)

    for epoch in pbar:
        train_loss, train_correct = train(
            device=device,
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn_train,
        )
        test_loss, test_correct = test(
            device=device,
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn_test,
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_correct_list.append(train_correct)
        test_correct_list.append(test_correct)
        pbar.set_postfix(
            epoch=f"{epoch + 1}/{conf.epochs}",
            train_loss=f"{train_loss:.4f}",
            test_loss=f"{test_loss:.4f}",
        )
        logger.info(
            f"train_loss={train_loss:.4f},test_loss={test_loss:.4f},"
            f"train_acc={train_correct:.4f},test_acc={test_correct:.4f}"
        )

    pred_info = test_info(
        device=device,
        dataloader=test_dataloader,
        model=model,
    )

    model_dump_file = conf.model_dump_path / file_stem
    unis.store_model_params(model, model_dump_file)

    data_dump_file = conf.data_dump_path / f"{file_stem}.pkl"
    with open(data_dump_file, "wb") as f:
        pickle.dump(pred_info, f)
        pickle.dump(train_loss_list, f)
        pickle.dump(test_loss_list, f)
        pickle.dump(train_correct_list, f)
        pickle.dump(test_correct_list, f)
