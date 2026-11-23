"""Clip-level metrics, F1 threshold search and calibration error. p is always P(class 1 = violent)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_GRID = np.arange(0.05, 0.951, 0.01)


def _ratio(a, b) -> float:
    return float(a / b) if b else 0.0


def _confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, tn, fp, fn


def ece(y, p, bins: int = 15) -> float:
    """Expected calibration error of the class-1 probability over equal-width bins on [0, 1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.sum() * abs(y[sel].mean() - p[sel].mean())
    return float(total / len(y))


def clip_metrics(y, p, thr: float) -> dict:
    """Metrics at prediction = p >= thr. roc_auc / pr_auc are nan when y has a single class."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    tp, tn, fp, fn = _confusion(y, (p >= thr).astype(int))
    both = len(np.unique(y)) == 2
    return {
        "accuracy": _ratio(tp + tn, len(y)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "ece": ece(y, p),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y)),
    }


def best_threshold(y, p, grid=DEFAULT_GRID) -> float:
    """Grid threshold maximising F1; ties go to the threshold closest to 0.5 (then the lower one)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    grid = np.round(np.asarray(grid, float), 6)
    f1 = []
    for t in grid:
        tp, _, fp, fn = _confusion(y, (p >= t).astype(int))
        f1.append(_ratio(2 * tp, 2 * tp + fp + fn))
    f1 = np.asarray(f1)
    cand = grid[np.isclose(f1, f1.max(), rtol=0, atol=1e-12)]
    return float(cand[np.argmin(np.abs(cand - 0.5))])
