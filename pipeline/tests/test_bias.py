import json
import math

import numpy as np
import pandas as pd

import pipeline.bias as B


def _preds(n, acc, seed, groups=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    pred = np.where(rng.uniform(size=n) < acc, y, 1 - y)
    prob = np.where(pred == 1, 0.8, 0.2)
    g = np.arange(n) if groups is None else groups
    return pd.DataFrame({"clip_id": [f"c{i}" for i in range(n)], "dataset": "rwf", "label": y,
                         "prob": prob, "pred": pred, "global_group": g,
                         "shared_source": rng.uniform(size=n) < 0.5})


def test_accuracy_gap_ci_detects_gap():
    a = [_preds(400, 0.95, s) for s in range(3)]
    b = [_preds(400, 0.80, s + 10) for s in range(3)]
    gap, lo, hi = B.accuracy_gap_ci(a, b, n=300)
    assert 0.08 < gap < 0.22 and 0 < lo < gap < hi


def test_accuracy_gap_ci_no_gap_covers_zero():
    a, b = [_preds(400, 0.85, 1)], [_preds(400, 0.85, 2)]
    gap, lo, hi = B.accuracy_gap_ci(a, b, n=300)
    assert lo < 0 < hi


def test_shared_source_gap_is_paired_within_predictions():
    df = _preds(600, 0.9, 3)
    df.loc[df["shared_source"], "pred"] = df.loc[df["shared_source"], "label"]  # perfect on shared
    gap, lo, hi = B.shared_source_gap(df, n=200)
    acc = lambda d: float((d["pred"] == d["label"]).mean())  # noqa: E731
    assert math.isclose(gap, acc(df[df["shared_source"]]) - acc(df[~df["shared_source"]]))
    assert lo > 0


def test_shared_source_gap_without_shared_clips_is_nan():
    df = _preds(50, 0.9, 4)
    df["shared_source"] = False
    assert all(math.isnan(v) for v in B.shared_source_gap(df, n=20))


def test_threshold_leak_rows(tmp_path):
    d = tmp_path / "runs" / "p4_random" / "mnv3_tsm" / "seed0"
    d.mkdir(parents=True)
    # val threshold 0.5 is wrong for this test set; test-tuned threshold 0.65 is perfect
    df = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1], "prob": [0.1, 0.55, 0.6, 0.7, 0.8, 0.9]})
    df.to_csv(d / "test_preds.csv", index=False)
    (d / "test_metrics.json").write_text(json.dumps({"threshold": 0.5}))
    rows = B.threshold_leak_rows(tmp_path / "runs")
    r = rows[0]
    assert r["run"] == "p4_random/mnv3_tsm/seed0"
    assert r["thr_val"] == 0.5 and 0.6 < r["thr_test"] <= 0.7
    assert r["f1_test_thr"] == 1.0 and r["f1_val_thr"] < 1.0 and r["f1_inflation"] > 0


def test_threshold_leak_rows_no_runs(tmp_path):
    assert B.threshold_leak_rows(tmp_path / "nothing") == []
    md = B.render_md({"a": None, "b": None, "c": None, "leak": []}, "mnv3_tsm", (0,))
    assert "no runs" in md.lower()


def test_run_bias_aggregation_end_to_end(tmp_path, monkeypatch):
    from pipeline.config import load_config
    cfg = load_config()
    cfg["paths"] = {"work": str(tmp_path)}
    cfg["eval"]["bootstrap_resamples"] = 50
    monkeypatch.setattr(B, "train_run", lambda *a, **k: {})

    def fake_eval(out, cfg_, views=1):
        out.mkdir(parents=True, exist_ok=True)
        acc = {"p4_random": 0.97, "p4_grouped": 0.85}.get(out.parts[-3], 0.9)
        _preds(200, acc, hash(out.as_posix()) % 1000).to_csv(out / "test_preds.csv", index=False)
        (out / "test_metrics.json").write_text(json.dumps({"threshold": 0.5}))

    monkeypatch.setattr(B, "evaluate_run", fake_eval)
    res = B.run_bias(cfg, seeds=(0, 1))
    assert res["a"]["gap"] > 0.04 and len(res["leak"]) == 6
    rep = tmp_path / "reports"
    csv = pd.read_csv(rep / "bias.csv")
    assert set(csv["part"]) == {"a", "b", "c"}
    md = (rep / "bias.md").read_text()
    for h in ("## (a)", "## (b)", "## (c)", "paired"):
        assert h in md
