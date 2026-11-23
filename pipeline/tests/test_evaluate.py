import json
import math

import numpy as np
import pandas as pd
import pytest

import pipeline.evaluate as E
import pipeline.train as T
from pipeline.data import cache as C
from pipeline.data.dataset import ClipDataset
from pipeline.tests.test_train import _frames, setup  # noqa: F401  (fixture)

PRED_COLS = ["clip_id", "dataset", "label", "prob", "pred", "global_group", "shared_source"]


@pytest.fixture
def trained(setup, monkeypatch):  # noqa: F811
    cfg, tmp = setup
    cfg["train"]["max_epochs"] = 1
    cfg["eval"]["bootstrap_resamples"] = 20
    m = T.load_protocol("x", cfg)
    extra = [(f"te{i}", "rwf" if i % 2 else "sfd", i % 2, "test", i % 3 == 0) for i in range(2, 10)]
    for k, (cid, *_rest) in enumerate(extra):
        C.write_clip(cid, _frames(12, 100 + k), cfg)
    add = pd.DataFrame([dict(clip_id=c, path="x", dataset=d, label=l, global_group=f"g{c}",
                             shared_source=s, split=sp) for c, d, l, sp, s in extra])
    m = pd.concat([m, add], ignore_index=True)
    monkeypatch.setattr(T, "load_protocol", lambda name, cfg, verify=True: m.copy())
    monkeypatch.setattr(E, "load_protocol", lambda name, cfg, verify=True: m.copy())
    out = tmp / "run"
    T.train_run(cfg, "p1_sfd", "cnn_lstm", 0, out)
    return cfg, out


def test_evaluate_run_outputs(trained, monkeypatch):
    cfg, out = trained
    seen = []
    orig = ClipDataset.__init__

    def spy(self, manifest, split, cfg_, norm, train, allow_test=False, **k):
        seen.append((split, allow_test))
        orig(self, manifest, split, cfg_, norm, train, allow_test, **k)

    monkeypatch.setattr(ClipDataset, "__init__", spy)
    res = E.evaluate_run(out, cfg)
    assert [s for s, _ in seen] == ["test"] and all(a for _, a in seen)
    preds = pd.read_csv(out / "test_preds.csv")
    assert list(preds.columns) == PRED_COLS and len(preds) == 10
    assert set(preds["pred"]) <= {0, 1}
    met = json.loads((out / "test_metrics.json").read_text())
    thr = json.loads((out / "val_summary.json").read_text())["threshold"]
    assert met["threshold"] == thr and met["pooled"]["at_val_thr"]["n"] == 10
    assert set(met) >= {"pooled", "by_dataset", "by_shared_source"}
    assert set(met["by_dataset"]) == {"sfd", "rwf"}
    assert set(met["by_shared_source"]) == {"true", "false"}
    assert met["by_shared_source"]["true"]["at_val_thr"]["n"] == int(preds["shared_source"].sum())
    sec = met["pooled"]
    assert set(sec) >= {"at_val_thr", "at_0.5", "ci"} and set(sec["ci"]) == {"accuracy", "f1", "roc_auc"}
    lo, hi = sec["ci"]["accuracy"]
    assert lo <= sec["at_val_thr"]["accuracy"] <= hi
    assert res["pooled"]["at_val_thr"] == sec["at_val_thr"]
    assert ((preds["prob"] >= thr).astype(int) == preds["pred"]).all()


def test_evaluate_run_three_views(trained):
    cfg, out = trained
    E.evaluate_run(out, cfg, views=3)
    assert (out / "test_preds_v3.csv").exists() and (out / "test_metrics_v3.json").exists()
    assert not (out / "test_preds.csv").exists()
    v3 = pd.read_csv(out / "test_preds_v3.csv")
    assert v3["prob"].between(0, 1).all() and len(v3) == 10


def test_single_class_section_does_not_crash(trained):
    cfg, out = trained
    met = E.evaluate_run(out, cfg)
    # by_dataset sections with one clip class must report nan (None in json), not raise
    sec = E.section(np.zeros(4, int), np.array([.1, .2, .7, .3]), np.arange(4), 0.5, cfg)
    assert sec["at_val_thr"]["roc_auc"] is None and sec["ci"]["roc_auc"] == [None, None]
    assert met["pooled"]["at_val_thr"]["accuracy"] is not None
    assert math.isfinite(sec["ci"]["accuracy"][0])
