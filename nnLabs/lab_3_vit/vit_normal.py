import os
import tqdm
import pickle
import torch
import numpy as np
from torch import nn
import torch.nn.functional as nnF
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils import unis
from pathlib import Path
from . import config, dataset_load, lr_scheduler, vit
from .config import conf


def train(
    device,
    amp_enabled,
    scaler,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn,
):
    model.train()
    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(X)
            loss = loss_fn(logits, y)

        batch_size = y.size(0)
        pred = logits.argmax(dim=1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.detach() * batch_size
        target_class = y.argmax(dim=1)
        correct += (pred == target_class).sum()
        sample_count += batch_size

        # if batch % 100 == 0:
        #     loss_, current = loss.item(), dataloader.batch_size * batch + len(X)
        #     print(f"loss: {loss_:>7f} [{current:>5d}/{size_n:>5d}]")
    metrics = torch.cat([train_loss, correct, sample_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    global_loss = (metrics[0] / metrics[2]).item()
    global_acc = (metrics[1] / metrics[2]).item()
    return global_loss, global_acc


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


def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("DDP without CUDA")

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 4:
        raise RuntimeError(f"need 4 GPU,but world_size={world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return local_rank, rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def run():
    config.load_configuration()

    local_rank, rank, world_size, device = setup_ddp()
    try:
        is_main_process = rank == 0
        conf_stem = conf.config_path.stem
        file_stem = Path(__file__).stem + f"__{conf_stem}"
        logger = unis.create_logger(conf.log_path, rank, file_stem)
        # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        unis.seed_everything(conf.seed)
        scaler = torch.amp.GradScaler("cuda", enabled=conf.amp)
        train_dataloader, test_dataloader, train_sampler = dataset_load.load_cifar10_large(
            conf.batch_size, conf.img_size, rank, world_size, conf.num_workers
        )
        model = vit.VisionTransformer(
            size_n=conf.num_layers,
            d_model=conf.d_model,
            num_heads=conf.num_heads,
            num_classes=conf.num_classes,
            mlp_hiddens=conf.mlp_hiddens,
            img_size=conf.img_size,
            patch_size=conf.patch_size,
            dropout=conf.dropout,
            drop_path_rate=conf.drop_path_rate,
        ).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        unis.seed_everything(conf.seed + rank)  # random unique proc strategy
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=conf.lr,
            betas=(conf.beta_1, conf.beta_2),
            weight_decay=conf.weight_decay,
        )
        total_steps = conf.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_scheduler.warmup_cosine_decay(conf.warmup_steps, total_steps),
        )
        loss_fn_train = nn.CrossEntropyLoss(label_smoothing=conf.label_smoothing)
        loss_fn_test = nn.CrossEntropyLoss()

        train_loss_list = []
        test_loss_list = []
        train_correct_list = []
        test_correct_list = []

        pbar = tqdm.tqdm(range(conf.epochs), dynamic_ncols=True) if is_main_process else range(conf.epochs)
        if is_main_process:
            logger.info(conf)

        for epoch in pbar:
            train_sampler.set_epoch(epoch)
            train_loss, train_correct = train(
                device=device,
                scaler=scaler,
                amp_enabled=conf.amp,
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn_train,
            )
            dist.barrier()
            if is_main_process:
                test_loss, test_correct = test(
                    device=device,
                    dataloader=test_dataloader,
                    model=model.module,
                    loss_fn=loss_fn_test,
                )

            if is_main_process:
                test_loss_list.append(test_loss)
                test_correct_list.append(test_correct)
                train_loss_list.append(train_loss)
                train_correct_list.append(train_correct)
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{conf.epochs}",
                    train_loss=f"{train_loss:.4f}",
                    test_loss=f"{test_loss:.4f}",
                )
                logger.info(
                    f"train_loss={train_loss:.4f},test_loss={test_loss:.4f},"
                    f"train_acc={train_correct:.4f},test_acc={test_correct:.4f}"
                )
            dist.barrier()

        dist.barrier()
        if is_main_process:
            pred_info = test_info(
                device=device,
                dataloader=test_dataloader,
                model=model.module,
            )
        dist.barrier()
        if is_main_process:
            model_dump_file = conf.model_dump_path / file_stem
            unis.store_model_params(model.module, model_dump_file)

            data_dump_file = conf.data_dump_path / f"{file_stem}.pkl"
            # Path(data_dump_file).mkdir(parents=True, exist_ok=True)
            with open(data_dump_file, "wb") as f:
                pickle.dump(pred_info, f)
                pickle.dump(train_loss_list, f)
                pickle.dump(test_loss_list, f)
                pickle.dump(train_correct_list, f)
                pickle.dump(test_correct_list, f)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    run()
