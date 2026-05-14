import numpy as np
import matplotlib.pyplot as plt
import utils.unis as ut


def accuracyEval(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0.0


def precisionEval(tp, fp, tn, fn):
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0


def recallEval(tp, fp, tn, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def f1scoreEval(tp, fp, tn, fn):
    p = precisionEval(tp, fp, tn, fn)
    r = recallEval(tp, fp, tn, fn)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0.0


def rocEval(tp, fp, tn, fn, pred_history):
    pred_history.sort(reverse=True)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    x_cur = 0.0
    y_cur = 0.0
    fpr = []
    tpr = []
    for index, (y_hat, y) in enumerate(pred_history):
        if y.astype(int) == 1:
            y_cur += 1 / (tp + fn)
        else:
            x_cur += 1 / (tn + fp)
        fpr.append(x_cur)
        tpr.append(y_cur)
    plt.plot(fpr, tpr)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="-.")
    plt.show()


def roc_Eval(histories, title):
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC" + title)

    for label, pred_history in histories:
        scores = np.array([float(np.asarray(p).ravel()[0]) for p, _ in pred_history])
        labels = np.array([int(np.asarray(y).ravel()[0]) for _, y in pred_history])

        P = int((labels == 1).sum())
        N = int((labels == 0).sum())
        if P == 0 or N == 0:
            print(f"ROC [{label}]: can't plot, Positive or Negative samples are 0")
            continue

        order = np.argsort(-scores, kind="mergesort")
        scores_sorted = scores[order]
        labels_sorted = labels[order]

        tps = np.cumsum(labels_sorted == 1)
        fps = np.cumsum(labels_sorted == 0)

        distinct = np.r_[np.where(np.diff(scores_sorted))[0], scores_sorted.size - 1]
        tpr = np.r_[0.0, tps[distinct] / P]
        fpr = np.r_[0.0, fps[distinct] / N]

        auc = ut.trapz(fpr, tpr)
        print(f"[{label}] auc: {auc:.4f}")
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")

    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="-.")
    plt.legend(loc="lower right")
    return [plt.gcf()]
