import json

import numpy as np
import pandas as pd
import pytest
import torch

import pipeline.train as T
from pipeline.config import load_config
from pipeline.data import cache as C
from pipeline.data.dataset import ClipDataset

FILES = ("config.json", "history.csv", "last.pt", "best.pt", "val_preds.csv", "val_summary.json", "DONE")


def _frames(n, seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (1, 72, 88, 3), dtype=np.uint8)
    return (np.repeat(base, n, axis=0) // 2 + (np.arange(n, dtype=np.uint8)[:, None, None, None] * 5))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["paths"] = {"work": str(tmp_path / "work")}
    cfg["clip"]["T"], cfg["clip"]["crop"] = 4, 64
    cfg["train"].update(workers=0, batch=2, effective_batch=4, max_epochs=3, pretrained=False,
                       device="cpu")
    rows = [(f"tr{i}", "sfd", i % 2, "train") for i in range(8)] + \
           [(f"va{i}", "sfd", i % 2, "val") for i in range(4)] + \
           [(f"te{i}", "sfd", i % 2, "test") for i in range(2)]
    m = pd.DataFrame([dict(clip_id=c, path="x", dataset=d, label=l, global_group=c,
                           shared_source=False, split=s) for c, d, l, s in rows])
    for k, r in enumerate(rows):
        C.write_clip(r[0], _frames(12, k), cfg)
    monkeypatch.setattr(T, "load_protocol", lambda name, cfg, verify=True: m.copy())
    return cfg, tmp_path


def _run(cfg, out, **kw):
    return T.train_run(cfg, "p1_sfd", "cnn_lstm", 0, out, **kw)


def _same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-6), k


def test_runs_and_writes(setup):
    cfg, tmp = setup
    out = tmp / "run"
    s = _run(cfg, out)
    for f in FILES:
        assert (out / f).exists(), f
    h = pd.read_csv(out / "history.csv")
    assert 1 <= len(h) <= 3
    assert list(h.columns) == ["epoch", "train_loss", "train_acc", "val_loss", "val_auc",
                               "val_f1@0.5", "lr", "seconds"]
    conf = json.loads((out / "config.json").read_text())
    assert conf["batch"] == 2 and conf["accum"] == 2 and conf["seed"] == 0
    for k in ("git_commit", "pip_hash", "gpu"):
        assert k in conf["meta"]
    assert conf["cfg"]["clip"]["T"] == 4
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert best["model_name"] == "cnn_lstm" and best["T"] == 4 and best["crop"] == 64
    assert best["threshold"] == s["threshold"] and 0.05 <= s["threshold"] <= 0.95
    preds = pd.read_csv(out / "val_preds.csv")
    assert list(preds.columns) == ["clip_id", "dataset", "label", "prob"]
    assert sorted(preds.clip_id) == [f"va{i}" for i in range(4)]
    assert set(s) >= {"best_epoch", "val_auc", "threshold", "metrics"}


def test_resume_equivalence(setup):
    cfg, tmp = setup
    _run(cfg, tmp / "A")
    cfg_b = json.loads(json.dumps(cfg))
    cfg_b["train"]["max_epochs"] = 2
    _run(cfg_b, tmp / "B")
    assert len(pd.read_csv(tmp / "B" / "history.csv")) == 2
    for f in ("DONE", "val_summary.json"):  # "killed": the run never finished
        (tmp / "B" / f).unlink()
    _run(cfg, tmp / "B")
    assert len(pd.read_csv(tmp / "B" / "history.csv")) == 3
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731
    _same_weights(load(tmp / "A" / "best.pt")["model"], load(tmp / "B" / "best.pt")["model"])
    _same_weights(load(tmp / "A" / "last.pt")["model"], load(tmp / "B" / "last.pt")["model"])


class _Disconnect(Exception):
    pass


def test_resume_after_mid_epoch_kill(setup, monkeypatch):
    cfg, tmp = setup
    _run(cfg, tmp / "A")
    real = T.build_model

    def dying_model(*a, **k):  # 4 train batches per epoch: call 10 is mid-epoch 3
        model, n = real(*a, **k), [0]
        fwd = model.forward

        def forward(x):
            if model.training:
                n[0] += 1
                if n[0] == 10:
                    raise _Disconnect
            return fwd(x)
        model.forward = forward
        return model

    monkeypatch.setattr(T, "build_model", dying_model)
    with pytest.raises(_Disconnect):
        _run(cfg, tmp / "B")
    assert not (tmp / "B" / "DONE").exists()
    assert torch.load(tmp / "B" / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 2
    monkeypatch.setattr(T, "build_model", real)
    _run(cfg, tmp / "B")
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731
    _same_weights(load(tmp / "A" / "last.pt")["model"], load(tmp / "B" / "last.pt")["model"])
    _same_weights(load(tmp / "A" / "best.pt")["model"], load(tmp / "B" / "best.pt")["model"])
    pd.testing.assert_frame_equal(pd.read_csv(tmp / "A" / "history.csv").drop(columns="seconds"),
                                  pd.read_csv(tmp / "B" / "history.csv").drop(columns="seconds"))


def test_no_test_rows_loaded(setup, monkeypatch):
    cfg, tmp = setup
    seen = []
    orig = ClipDataset.__init__

    def spy(self, manifest, split, *a, **k):
        seen.append(split)
        orig(self, manifest, split, *a, **k)

    monkeypatch.setattr(ClipDataset, "__init__", spy)
    _run(cfg, tmp / "run")
    assert "train" in seen and "val" in seen and "test" not in seen


def test_done_skips(setup, monkeypatch):
    cfg, tmp = setup
    first = _run(cfg, tmp / "run")

    def boom(*a, **k):
        raise AssertionError("trainer loop entered")

    monkeypatch.setattr(T, "_train_epoch", boom)
    assert _run(cfg, tmp / "run") == first


def test_accum_must_divide(setup):
    cfg, tmp = setup
    cfg["train"]["batch"] = 3
    with pytest.raises(ValueError, match="effective_batch"):
        _run(cfg, tmp / "run")


def test_variant_and_batch_defaults():
    assert T.variant_name("motionlite", {"diff": False}) == "motionlite__diff-False"
    assert T.variant_name("motionlite", {"diff": True, "tsm": True}) == "motionlite"
    assert T.variant_name("motionlite", {"tsm": False, "diff": False}) == "motionlite__diff-False__tsm-False"
    assert T.variant_name("cnn_lstm", None) == "cnn_lstm"
    assert T.default_batch("motionlite", {"backbone": "effb0"}) == 4
    assert T.default_batch("mvit_s", {}) == 2 and T.default_batch("cnn_lstm", {}) == 16
