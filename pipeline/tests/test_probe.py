import numpy as np
import pandas as pd
import torch

from pipeline import probe


def test_weights_balance_cells():
    df = pd.DataFrame({"dataset": ["a"] * 6 + ["b"] * 2, "label": [0, 0, 0, 0, 1, 1, 0, 1]})
    w = probe._weights(df, by_dataset=True)
    cell = pd.Series(w).groupby(df.dataset + "_" + df.label.astype(str)).sum()
    assert np.allclose(cell.to_numpy(), cell.iloc[0]) and np.isclose(w.sum(), len(w))
    wc = probe._weights(df, by_dataset=False)
    assert np.isclose(wc[df.label == 0].sum(), wc[df.label == 1].sum())


def test_fit_probe_separable_and_view_averaging():
    rng = np.random.default_rng(0)
    def make(n):
        y = rng.integers(0, 2, n)
        X = rng.normal(size=(n, 3, 8)).astype(np.float32)
        X[:, :, 0] += 3 * (2 * y[:, None] - 1)
        return X, y
    Xtr, ytr = make(200)
    Xva, yva = make(80)
    pred, C, auc = probe.fit_probe(Xtr, ytr, np.ones(len(ytr)), Xva, yva)
    assert auc > 0.98 and C in probe.C_GRID
    p = pred(Xva)
    assert p.shape == (80,) and ((p >= 0) & (p <= 1)).all()


def test_extract_with_fake_encoder(tmp_path, monkeypatch):
    # ViewDataset reads frames through the cache API; stub it with a synthetic 20-frame clip
    monkeypatch.setattr(probe, "num_frames", lambda cid, cfg: 20)
    monkeypatch.setattr(probe, "read_clip",
                        lambda cid, cfg, idx: np.full((len(idx), 64, 80, 3), 128, np.uint8))
    cfg = {"paths": {"work": str(tmp_path)}}
    monkeypatch.setitem(probe.BACKBONES, "fake", {"hf": None, "T": 4, "size": 32, "norm": probe.IMAGENET_NORM})
    seen = []
    def encode(x):
        seen.append(tuple(x.shape))
        return x.mean(dim=(1, 3, 4))           # [B*V, 3]
    out = probe.extract(cfg, "fake", views=2, batch=2, workers=0, clip_ids=["a", "b", "c"], encode=encode)
    z = np.load(out)
    assert z["feats"].shape == (3, 2, 3) and list(z["clip_ids"]) == ["a", "b", "c"]
    assert seen[0] == (4, 4, 3, 32, 32)        # batch 2 clips x 2 views, T=4, 32 px
