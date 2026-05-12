import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- paths & hyperparameters --------
TRAIN_PATH = "./data/diabetes/classification/diabetes_train"
TEST_PATH = "./data/diabetes/classification/diabetes_test"

EPOCHS = 1500
BATCH_SIZE = 64
HIDDEN = 64
LR = 5e-2
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5


# -------- device & global flags --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# -------- model --------
class MLP(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = HIDDEN, out_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -------- data --------
def clsdata_process(filepath):
    """第 1 列为 y, 第 2-9 列为 X。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = np.array([line.strip().split(",") for line in f], dtype=float)
    X, y = data[:, 1:], data[:, 0]
    return X, y


def load_to_device(X_np: np.ndarray, y_np: np.ndarray):
    X = torch.from_numpy(X_np).float().to(DEVICE, non_blocking=True)
    y = torch.from_numpy(y_np).float().reshape(-1, 1).to(DEVICE, non_blocking=True)
    return X, y


# -------- training --------
def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
          lr: float = LR, weight_decay: float = WEIGHT_DECAY,
          log_every: int = 100):
    loss_fn = nn.BCEWithLogitsLoss()
    # SGD 无 momentum,和手写版一致
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = X.shape[0]
    loss_history = np.empty(epochs, dtype=np.float32)

    model.train()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        # 分段学习率: 让模型在大 lr 下充分探索,再慢慢收敛
        if epoch == 800:
            for g in opt.param_groups:
                g["lr"] = 1e-2
        if epoch == 1200:
            for g in opt.param_groups:
                g["lr"] = 5e-3

        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = torch.zeros((), device=DEVICE)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.detach() * xb.shape[0]

        mean_loss = (epoch_loss / n).item()
        loss_history[epoch] = mean_loss
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"loss:{mean_loss:>10.6f}  [{epoch:>5d}/{epochs:>5d}]")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print(f"train elapsed: {time.perf_counter() - t0:.2f}s on {DEVICE}")
    return loss_history


@torch.inference_mode()
def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(X)
    prob = torch.sigmoid(logits).cpu().numpy().ravel()
    return prob


# -------- metrics --------
def confusion(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def basic_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return acc, precision, recall, f1


def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    y_true = y_true.astype(int)
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        raise ValueError("ROC 要求正负样本都存在")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    distinct = np.r_[np.where(np.diff(s_sorted))[0], s_sorted.size - 1]
    tpr = np.concatenate(([0.0], tps[distinct] / P))
    fpr = np.concatenate(([0.0], fps[distinct] / N))
    return fpr, tpr


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    dx = np.diff(fpr)
    mid = (tpr[1:] + tpr[:-1]) * 0.5
    return float((dx * mid).sum())


# -------- plotting --------
def plot_loss(loss_history):
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("classification loss (torch)")


def plot_roc(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, lw=1.8, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="-.", color="gray")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend(loc="lower right")


# -------- entry --------
def main():
    torch.manual_seed(31)
    np.random.seed(31)

    X_tr_np, y_tr_np = clsdata_process(TRAIN_PATH)
    X_te_np, y_te_np = clsdata_process(TEST_PATH)

    X_tr, y_tr = load_to_device(X_tr_np, y_tr_np)
    X_te, y_te = load_to_device(X_te_np, y_te_np)

    model = MLP(in_dim=X_tr.shape[1]).to(DEVICE)
    print(f"device={DEVICE}  train={tuple(X_tr.shape)}  test={tuple(X_te.shape)}")

    loss_history = train(model, X_tr, y_tr)

    prob = predict_proba(model, X_te)
    pred = (prob >= THRESHOLD).astype(int)

    tp, fp, tn, fn = confusion(y_te_np, pred)
    acc, precision, recall, f1 = basic_metrics(tp, fp, tn, fn)
    fpr, tpr = roc_curve(y_te_np, prob)
    auc = auc_score(fpr, tpr)

    print("\n=== Evaluation on test set ===")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"f1_score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    plot_loss(loss_history)
    plot_roc(fpr, tpr, auc)
    plt.show()


if __name__ == "__main__":
    main()
