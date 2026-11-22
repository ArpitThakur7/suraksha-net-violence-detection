"""Video probing and decoding (PyAV). Always yields RGB uint8."""
from __future__ import annotations

import pipeline.paths  # noqa: F401  (sys.path / env setup)

import av
import numpy as np
from PIL import Image

MIN_FRAMES = 16


def _stream_fps(stream) -> float:
    for r in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        try:
            if r is not None and float(r) > 0:
                return float(r)
        except Exception:
            pass
    return 0.0


def probe_video(path) -> dict:
    out = dict(fps=0.0, frames=0, width=0, height=0, duration_s=0.0, error="")
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise ValueError("no video stream")
            s = c.streams.video[0]
            fps = _stream_fps(s)
            n = 0
            for f in c.decode(s):
                if n == 0:
                    out["width"], out["height"] = int(f.width), int(f.height)
                n += 1
            out["frames"] = n
            out["fps"] = round(fps, 4)
            out["duration_s"] = round(n / fps, 3) if fps > 0 else 0.0
    except Exception as e:  # never raise from probing
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def _resize(arr: np.ndarray, short_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if min(h, w) == short_side:
        return arr
    if h <= w:
        nh, nw = short_side, max(1, int(round(w * short_side / h)))
    else:
        nw, nh = short_side, max(1, int(round(h * short_side / w)))
    return np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR))


def _decode(path, picker, short_side):
    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        src = _stream_fps(s)
        for i, f in enumerate(c.decode(s)):
            if picker(i, src):
                a = f.to_ndarray(format="rgb24")
                frames.append(_resize(a, short_side) if short_side else a)
    return frames


def decode_frames(path, fps: float | None = None, short_side: int | None = None) -> np.ndarray:
    """uint8 [N,H,W,3] RGB. With `fps`, keep frames nearest a fps grid; if that yields
    fewer than 16 frames, return all frames."""
    frames = []
    if fps:
        state = {"k": 0}

        def pick(i, src):
            if src <= 0:
                return True
            target = int(round(state["k"] * src / fps))
            if i >= target:
                state["k"] += 1
                return True
            return False

        frames = _decode(path, pick, short_side)
    if not fps or len(frames) < MIN_FRAMES:
        frames = _decode(path, lambda i, s: True, short_side)
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)
