import copy

import numpy as np
import pandas as pd

from pipeline.config import load_config
from pipeline.data import cache as C
from pipeline.data.video import decode_frames


def _cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {"work": str(tmp_path / "work")}
    return cfg


def test_roundtrip(tiny_videos, tmp_path):
    cfg = _cfg(tmp_path)
    ref = decode_frames(tiny_videos["normal"], fps=cfg["cache"]["fps"], short_side=cfg["cache"]["short_side"])
    C.write_clip("c1", ref, cfg)
    back = C.read_clip("c1", cfg)
    assert back.shape == ref.shape and back.dtype == np.uint8
    assert np.abs(back.astype(float) - ref.astype(float)).mean() < 6
    assert C.read_clip("c1", cfg, idx=np.array([0, 5])).shape[0] == 2
    assert C.num_frames("c1", cfg) == len(ref)


def test_key_changes(tmp_path):
    a = _cfg(tmp_path)
    b = copy.deepcopy(a)
    b["cache"]["fps"] = 10
    assert C.cache_key(a) != C.cache_key(b)
    assert C.cache_dir(a) != C.cache_dir(b)


def test_atomic_and_build(tiny_videos, tmp_path):
    cfg = _cfg(tmp_path)
    rows = [("a", tiny_videos["normal"], ""), ("a", tiny_videos["normal"], ""),
            ("b", tiny_videos["short"], ""), ("c", tiny_videos["gray"], ""),
            ("x", tiny_videos["gray"], "corrupt")]
    m = pd.DataFrame(rows, columns=["clip_id", "path", "exclusion"])
    C.build_cache(m, cfg, workers=2)
    d = C.cache_dir(cfg)
    assert not list(d.glob("*.tmp")) and not list(d.glob("*.tmp.*"))
    assert sorted(p.stem for p in d.glob("*.npz")) == ["a", "b", "c"]
    idx = pd.read_csv(d / "index.csv")
    assert set(idx.clip_id) == {"a", "b", "c"}
    assert C.read_clip("c", cfg).shape[-1] == 3
    C.build_cache(m, cfg, workers=2)  # second run skips
    import json
    assert json.loads((d / "build.json").read_text())["n_skipped"] == 3


def test_failure_recorded(tiny_videos, tmp_path):
    import pytest
    cfg = _cfg(tmp_path)
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    m = pd.DataFrame([("a", tiny_videos["normal"], ""), ("z", bad, "")],
                     columns=["clip_id", "path", "exclusion"])
    with pytest.raises(RuntimeError):
        C.build_cache(m, cfg, workers=1)
    d = C.cache_dir(cfg)
    assert (d / "a.npz").exists()
    assert "z" in (d / "failures.csv").read_text()
