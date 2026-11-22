import sys
from pathlib import Path

import av
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _frame_rgb(i, w, h, gray=False):
    xs = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
    ys = np.linspace(0, 255, h, dtype=np.float32)[:, None].repeat(w, 1)
    shift = i * 9
    r = (xs + shift) % 256
    g = (ys + 2 * shift) % 256
    b = (xs * 0.5 + ys * 0.5 + 3 * shift) % 256
    # moving bright square for motion
    x0 = (i * 7) % max(w - 30, 1)
    y0 = (i * 5) % max(h - 30, 1)
    r[y0:y0 + 30, x0:x0 + 30] = 255
    g[y0:y0 + 30, x0:x0 + 30] = 255
    b[y0:y0 + 30, x0:x0 + 30] = 255
    if gray:
        v = (0.299 * r + 0.587 * g + 0.114 * b)
        r = g = b = v
    return np.stack([r, g, b], -1).astype(np.uint8)


def _write(path, n, fps, w, h, gray=False):
    if gray:
        # genuine single-channel stream: ffv1 with pix_fmt gray
        with av.open(str(path), "w") as c:
            s = c.add_stream("ffv1", rate=fps)
            s.width, s.height, s.pix_fmt = w, h, "gray"
            for i in range(n):
                g = _frame_rgb(i, w, h, gray=True)[..., 0]
                for p in s.encode(av.VideoFrame.from_ndarray(g, format="gray")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        return
    for codec in ("mpeg4", "libx264"):
        try:
            with av.open(str(path), "w") as c:
                s = c.add_stream(codec, rate=fps)
                s.width, s.height, s.pix_fmt = w, h, "yuv420p"
                for i in range(n):
                    f = av.VideoFrame.from_ndarray(_frame_rgb(i, w, h), format="rgb24")
                    for p in s.encode(f):
                        c.mux(p)
                for p in s.encode():
                    c.mux(p)
            return
        except Exception:
            if codec == "libx264":
                raise


@pytest.fixture
def tiny_videos(tmp_path):
    out = {
        "normal": tmp_path / "normal.avi",
        "short": tmp_path / "short.avi",
        "gray": tmp_path / "gray.avi",
    }
    _write(out["normal"], 40, 30, 320, 240)
    _write(out["short"], 12, 30, 320, 240)
    _write(out["gray"], 30, 25, 180, 120, gray=True)
    return out
