import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.config import load_config
from pipeline.data import cache as C
from pipeline.data.dataset import ClipDataset, make_loader, sample_indices

NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {"work": str(tmp_path / "work")}
    cfg["train"]["workers"] = 0
    return cfg


def _manifest(rows):
    return pd.DataFrame([dict(clip_id=c, path="x", dataset=d, label=l, global_group=c,
                              shared_source=False, split=s) for c, d, l, s in rows])


def _frames(n, h=256, w=300, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (1, h, w, 3), dtype=np.uint8)
    return np.repeat(base, n, axis=0) // 2 + (np.arange(n, dtype=np.uint8)[:, None, None, None] * 3)


def test_sample_short_clip():
    rng = np.random.default_rng(0)
    for train in (False, True):
        idx = sample_indices(12, 16, train=train, rng=rng)
        assert len(idx) == 16 and idx.max() == 11 and idx.min() == 0
        assert (np.diff(idx) >= 0).all()
        assert (idx[12:] == 11).all()


def test_sample_train_within_bounds():
    rng = np.random.default_rng(1)
    T, n = 16, 75
    for _ in range(200):
        idx = sample_indices(n, T, train=True, rng=rng)
        assert len(idx) == T and idx.min() >= 0 and idx.max() <= n - 1
        assert idx.max() - idx.min() >= 0.6 * (n - 1) * (T - 1) / T


def test_eval_views_are_distinct_windows():
    rng = np.random.default_rng(0)
    assert (sample_indices(100, 16, False, rng) == np.rint(np.linspace(0, 99, 16)).astype(int)).all()
    v = [sample_indices(100, 16, False, rng, view=k, views=3) for k in range(3)]
    for a in v:
        assert len(a) == 16 and (np.diff(a) >= 0).all() and a.min() >= 0 and a.max() <= 99
    assert v[0][0] == 0 and v[2][-1] == 99
    assert all((v[i] != v[j]).any() for i in range(3) for j in range(i + 1, 3))
    assert v[0][0] < v[1][0] < v[2][0] and v[0][-1] < v[1][-1] < v[2][-1]
    short = [sample_indices(10, 16, False, rng, view=k, views=3) for k in range(3)]
    assert all((s_ == short[0]).all() for s_ in short)  # clips shorter than T: all views identical
    near = [sample_indices(17, 16, False, rng, view=k, views=3) for k in range(3)]
    assert all(a.min() >= 0 and a.max() <= 16 for a in near)


def test_test_guard(tmp_path):
    cfg = _cfg(tmp_path)
    m = _manifest([("a", "rwf", 0, "test")])
    with pytest.raises(PermissionError):
        ClipDataset(m, "test", cfg, NORM, train=False, allow_test=False)
    with pytest.raises(PermissionError):
        make_loader(m, "test", cfg, NORM, train=False, allow_test=False)


@pytest.mark.parametrize("n", [30, 12])
@pytest.mark.parametrize("train", [True, False])
def test_item_shape(tmp_path, n, train):
    cfg = _cfg(tmp_path)
    C.write_clip("a", _frames(n), cfg)
    m = _manifest([("a", "rwf", 1, "train")])
    x, y, cid = ClipDataset(m, "train", cfg, NORM, train=train)[0]
    assert x.shape == (16, 3, 224, 224) and x.dtype == torch.float32
    assert torch.isfinite(x).all() and y == 1 and cid == "a"


def test_augment_same_params_all_frames(tmp_path):
    cfg = _cfg(tmp_path)
    # identical frames in -> identical frames out iff params are shared across the clip
    f = np.repeat(_frames(1), 20, axis=0)
    C.write_clip("a", f, cfg)
    m = _manifest([("a", "rwf", 0, "train")])
    ds = ClipDataset(m, "train", cfg, NORM, train=True)
    for _ in range(5):
        x = ds[0][0]
        assert torch.allclose(x, x[0:1].expand_as(x), atol=1e-4)


def test_train_reproducible_with_seed(tmp_path):
    cfg = _cfg(tmp_path)
    C.write_clip("a", _frames(40), cfg)
    m = _manifest([("a", "rwf", 0, "train")])
    xs = [ClipDataset(m, "train", cfg, NORM, train=True, seed=5)[0][0] for _ in range(2)]
    assert torch.equal(xs[0], xs[1])
    other = ClipDataset(m, "train", cfg, NORM, train=True, seed=6)[0][0]
    assert not torch.equal(xs[0], other)


def test_balanced_sampler(tmp_path):
    cfg = _cfg(tmp_path)
    rows = [(f"r{i}", "rwf", i % 2, "train") for i in range(90)] + \
           [(f"s{i}", "sfd", i % 2, "train") for i in range(10)]
    m = _manifest(rows)
    for r in rows:
        C.write_clip(r[0], _frames(16, 256, 256), cfg)
    dl = make_loader(m, "train", cfg, NORM, train=True, balance_datasets=True, seed=0)
    idx = list(dl.sampler)  # sampler draws len(dataset) = 100 per epoch
    assert len(idx) == 100
    draws = []
    for _ in range(50):
        draws += list(dl.sampler)
    share = np.mean([rows[i][1] == "sfd" for i in draws])
    assert abs(share - 0.5) < 0.05


def test_loader_batches(tmp_path):
    cfg = _cfg(tmp_path)
    rows = [(f"c{i}", "rwf", i % 2, "val") for i in range(6)]
    for r in rows:
        C.write_clip(r[0], _frames(20), cfg)
    dl = make_loader(_manifest(rows), "val", cfg, NORM, train=False, batch_size=4)
    xb, yb, ids = next(iter(dl))
    assert xb.shape == (4, 16, 3, 224, 224) and yb.shape == (4,) and len(ids) == 4
