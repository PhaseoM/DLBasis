import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_PATH = "./data/diabetes/regression/sinc_train"
TEST_PATH = "./data/diabetes/regression/sinc_test"

EPOCHS = 3000
BATCH_SIZE = 5000
HIDDEN = 256
LR = 1e-2
WEIGHT_DECAY = 1e-5
X_SCALE = 10.0  # 训练数据 x 大致在 [-10, 10]


# -------- device & global flags --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    # 安培及以上允许用 TF32,收益挺明显,精度对回归够用
    torch.set_float32_matmul_precision("high")


# -------- model --------
class MLP(nn.Module):
    def __init__(self, in_dim: int = 1, hidden: int = HIDDEN, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------- data --------
def regsdata_process(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
        y, X = data[:, 0], data[:, 1]
    return X, y


def load_to_device(X_np: np.ndarray, y_np: np.ndarray):
    """一次性把数据搬到目标 device,之后训练循环再也不跨设备拷贝。"""
    X = (
        torch.from_numpy(X_np).float().reshape(-1, 1).to(DEVICE, non_blocking=True)
        / X_SCALE
    )
    y = torch.from_numpy(y_np).float().reshape(-1, 1).to(DEVICE, non_blocking=True)
    return X, y


# -------- training --------
def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    log_every: int = 50,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    loss_history = np.empty(epochs, dtype=np.float32)

    model.train()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        # 打乱索引同样放在 GPU 上做,省掉 pinned memory 的拷贝
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = torch.zeros((), device=DEVICE)

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X[idx], y[idx]

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # 不要在循环里 .item(),那是同步点。
            epoch_loss += loss.detach() * xb.shape[0]

        mean_loss = (epoch_loss / n).item()  # 整轮只同步一次
        loss_history[epoch] = mean_loss
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"loss:{mean_loss:>10.6f}  [{epoch:>5d}/{epochs:>5d}]")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print(f"train elapsed: {time.perf_counter() - t0:.2f}s on {DEVICE}")
    return loss_history


@torch.inference_mode()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    model.eval()
    pred = model(X)
    mse = F.mse_loss(pred, y).item()
    return pred.cpu().numpy().ravel(), mse


# -------- plot --------
def plot_fit(X_tr, y_tr, X_te, y_te, pred_te):
    order = np.argsort(X_te)
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X_tr, y_tr, s=6, alpha=0.4, label="train")
    plt.scatter(X_te, y_te, s=6, alpha=0.4, label="test")
    plt.plot(X_te[order], pred_te[order], c="r", lw=1.5, label="pred")
    plt.legend()


def plot_loss(loss_history):
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("regression loss (torch)")


# -------- entry --------
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    X_tr_np, y_tr_np = regsdata_process(TRAIN_PATH)
    X_te_np, y_te_np = regsdata_process(TEST_PATH)

    X_tr, y_tr = load_to_device(X_tr_np, y_tr_np)
    X_te, y_te = load_to_device(X_te_np, y_te_np)

    model = MLP().to(DEVICE)
    print(f"device={DEVICE}  train={X_tr.shape}  test={X_te.shape}")

    loss_history = train(model, X_tr, y_tr)
    pred_te, mse = evaluate(model, X_te, y_te)
    print(f"test MSE: {mse:.6f}")

    plot_fit(X_tr_np, y_tr_np, X_te_np, y_te_np, pred_te)
    plot_loss(loss_history)
    plt.show()


if __name__ == "__main__":
    main()
