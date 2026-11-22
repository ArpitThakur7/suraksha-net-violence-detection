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
