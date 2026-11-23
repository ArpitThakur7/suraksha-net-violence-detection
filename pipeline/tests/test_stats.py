import math

import numpy as np

from pipeline.stats import best_threshold, clip_metrics, ece


def test_metrics_known():
    # thr 0.5 -> pred [1,1,0,0,1,0,0,1]; y [1,1,1,0,0,0,0,1] => tp=3 fn=1 fp=1 tn=3
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.1, 0.4, 0.7])
    m = clip_metrics(y, p, 0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"], m["n"]) == (3, 3, 1, 1, 8)
    assert math.isclose(m["accuracy"], 6 / 8)
    assert math.isclose(m["precision"], 3 / 4)
    assert math.isclose(m["recall"], 3 / 4)
    assert math.isclose(m["specificity"], 3 / 4)
    assert math.isclose(m["f1"], 3 / 4)
    # ranking: positives 0.9,0.8,0.7,0.3 vs negatives 0.6,0.4,0.2,0.1 -> 14 of 16 pairs correct
    assert math.isclose(m["roc_auc"], 14 / 16)
    assert 0 < m["pr_auc"] <= 1 and 0 <= m["ece"] <= 1
    assert set(m) == {"accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
                      "ece", "tp", "tn", "fp", "fn", "n"}


def test_metrics_single_class():
    m = clip_metrics(np.zeros(4, int), np.array([0.1, 0.6, 0.2, 0.3]), 0.5)
    assert math.isnan(m["roc_auc"]) and math.isnan(m["pr_auc"])
    assert (m["tn"], m["fp"]) == (3, 1) and m["precision"] == 0.0 and m["recall"] == 0.0


def test_best_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t = best_threshold(y, p)
    assert 0.2 < t <= 0.8
    assert clip_metrics(y, p, t)["f1"] == 1.0
    assert math.isclose(t, 0.5)  # every threshold in (0.2, 0.8] ties; closest to 0.5 wins


def test_ece_perfect():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert math.isclose(ece(np.zeros(100, int), np.full(100, 0.9)), 0.9, abs_tol=1e-9)
