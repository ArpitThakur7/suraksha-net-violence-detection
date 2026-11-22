import av

from pipeline.config import load_config, run_metadata
from pipeline.paths import ROOT


def test_extends_and_override(tmp_path):
    base_path = ROOT / "pipeline" / "configs" / "base.yaml"
    base = load_config(base_path)
    child = tmp_path / "child.yaml"
    child.write_text(f"extends: {base_path.as_posix()}\nclip:\n  T: 8\n")
    cfg = load_config(child, ["train.lr=0.5", "new.sec.k=3"])
    assert cfg["train"]["lr"] == 0.5
    assert cfg["clip"]["T"] == 8
    assert cfg["clip"]["crop"] == base["clip"]["crop"]
    assert cfg["new"]["sec"]["k"] == 3


def test_relative_extends(tmp_path):
    (tmp_path / "a.yaml").write_text("x: {p: 1, q: 2}\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nx: {q: 5}\n")
    cfg = load_config(tmp_path / "b.yaml")
    assert cfg["x"] == {"p": 1, "q": 5}


def test_base_values():
    cfg = load_config()
    assert cfg["cache"] == {"fps": 15, "short_side": 256, "jpeg_quality": 90}
    assert cfg["clip"]["T"] == 16
    assert cfg["clip"]["crop"] == 224
    assert cfg["split"]["val_frac"] == 0.15
    assert cfg["split"]["seed"] == 1337
    assert cfg["train"]["patience"] == 6
    assert cfg["train"]["max_epochs"] == 40
    assert cfg["train"]["seeds"] == [0, 1, 2]
    for sec in ("paths", "cache", "clip", "split", "train", "eval", "bench"):
        assert sec in cfg


def test_run_metadata():
    m = run_metadata()
    assert set(m) == {"git_commit", "pip_hash", "gpu", "torch", "python", "time"}


def test_tiny_videos_decodable(tiny_videos):
    expected = {"normal": (40, 30, 320, 240), "short": (12, 30, 320, 240), "gray": (30, 25, 180, 120)}
    for k, (n, fps, w, h) in expected.items():
        with av.open(str(tiny_videos[k])) as c:
            frames = list(c.decode(video=0))
            s = c.streams.video[0]
            assert len(frames) == n
            assert (s.width, s.height) == (w, h)
            assert round(float(s.average_rate)) == fps
            if k == "gray":
                assert s.codec_context.pix_fmt == "gray"
            else:
                assert s.codec_context.pix_fmt == "yuv420p"
