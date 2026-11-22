from pathlib import Path

import numpy as np

from pipeline.data.ingest import normalize_source, rwf_source_id
from pipeline.data.video import decode_frames, probe_video


def test_rwf_source_id():
    assert rwf_source_id("-1l5631l3fg_0.avi") == "-1l5631l3fg"
    assert rwf_source_id("a_b_c_12.avi") == "a_b_c"


def test_normalize_source():
    assert normalize_source("sfd", "sfd:7q6KJoSgxmY") == "yt:7q6KJoSgxmY"
    assert normalize_source("sfd", "sfd:nofight_062-091") == "sfd:nofight_062-091"
    assert normalize_source("rwf", "0H2s9UJcNJ0") == "yt:0H2s9UJcNJ0"
    assert normalize_source("avd", "avd:0:1") == "avd:0:1"
    assert normalize_source("avd", "0:1") == "avd:0:1"


def test_probe_and_decode(tiny_videos):
    p = probe_video(tiny_videos["normal"])
    assert p["frames"] == 40 and p["error"] == "" and p["width"] == 320
    x = decode_frames(tiny_videos["normal"], fps=15, short_side=256)
    assert x.dtype == np.uint8 and abs(x.shape[0] - 20) <= 1
    assert x.shape[1:] == (256, 341, 3)
    g = decode_frames(tiny_videos["gray"], fps=15, short_side=256)
    assert g.ndim == 4 and g.shape[-1] == 3 and min(g.shape[1:3]) == 256
    s = decode_frames(tiny_videos["short"], fps=15, short_side=256)
    assert s.shape[0] == 12


def test_probe_bad_file(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    p = probe_video(bad)
    assert p["error"] and p["frames"] == 0


def test_duplicate_handling():
    import pandas as pd
    from pipeline.data.ingest import mark_duplicates
    df = pd.DataFrame([
        dict(clip_id="a", path="a", sha256="s1", original_split="train", frames=5, error=""),
        dict(clip_id="b", path="b", sha256="s1", original_split="train", frames=5, error=""),
        dict(clip_id="c", path="c", sha256="s1", original_split="val", frames=5, error=""),
        dict(clip_id="d", path="d", sha256="s2", original_split="val", frames=5, error=""),
    ])
    o = mark_duplicates(df)
    assert o.exclusion.tolist() == ["", "duplicate_sha", "", ""]
    assert o.dup_of.tolist() == ["", "a", "a", ""]


def _row(path):
    import pandas as pd
    from pipeline.data.ingest import _audit_one, mark_duplicates
    rel = str(path)  # absolute path: ROOT / abs == abs
    r = _audit_one(dict(path=rel, dataset="rwf", label=0, source_id="x", original_split="train"))
    return mark_duplicates(pd.DataFrame([r])).iloc[0]


def test_corrupt_file_recorded(tmp_path):
    import os
    bad = tmp_path / "x.avi"
    bad.write_bytes(os.urandom(4096))
    # path outside ROOT: _audit_one does ROOT / path, and an absolute path overrides ROOT
    for p in (bad.as_posix(), (tmp_path / "missing.avi").as_posix()):
        r = _row(p)
        assert r.exclusion == "decode_error" and r.error
        assert r.clip_id.startswith("err-") or r.sha256


def test_zero_frame_video(tmp_path):
    import av
    z = tmp_path / "zero.avi"
    with av.open(str(z), "w") as c:
        s = c.add_stream("mpeg4", rate=25)
        s.width, s.height, s.pix_fmt = 64, 64, "yuv420p"
    p = probe_video(z)
    assert p["frames"] == 0
    r = _row(z.as_posix())
    assert r.exclusion in ("empty", "decode_error")


def test_odd_size_decode(tmp_path):
    import av
    path = tmp_path / "odd.avi"
    w, h = 202, 150
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=25)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(20):
            a = np.full((h, w, 3), i * 10, np.uint8)
            for pk in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    x = decode_frames(path, fps=None, short_side=256)
    assert x.dtype == np.uint8 and x.shape[-1] == 3
    assert min(x.shape[1:3]) == 256 and x.shape[1] == 256
    assert abs(x.shape[2] - round(w * 256 / h)) <= 1
