import io

import av
import numpy as np
import pandas as pd
from PIL import Image

from pipeline.data.groups import (assign_global_groups, near_duplicate_pairs, phash)


def test_phash_identical_and_different():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    base = np.asarray(Image.fromarray(small).resize((320, 240), Image.BICUBIC))
    buf = io.BytesIO()
    Image.fromarray(base).save(buf, "JPEG", quality=75)
    rec = np.asarray(Image.open(buf).convert("RGB"))
    assert bin(phash(base) ^ phash(rec)).count("1") <= 4
    noise = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    grad = np.repeat(np.linspace(0, 255, 320, dtype=np.uint8)[None, :, None], 240, 0).repeat(3, 2)
    assert bin(phash(noise) ^ phash(grad)).count("1") >= 20


def test_union_find():
    m = pd.DataFrame(dict(
        clip_id=["a1", "a2", "b1", "c1"],
        source_id=["yt:A", "yt:A", "yt:B", "avd:3"],
        dataset=["rwf", "sfd", "rwf", "avd"],
        exclusion=["", "", "", ""], dup_of=["", "", "", ""]))
    pairs = pd.DataFrame(dict(clip_a=["b1"], clip_b=["c1"], distance=[2.0], confirmed=[True]))
    g = assign_global_groups(m, pairs)
    assert g.nunique() == 2
    assert g[0] == g[1] == "yt:A"
    assert g[2] == g[3] == "avd:3"
    pairs2 = pairs.assign(confirmed=False)
    assert assign_global_groups(m, pairs2).nunique() == 3
    shared = m.assign(global_group=g).groupby("global_group").dataset.transform("nunique") > 1
    assert shared[[0, 1]].all()


def test_union_find_sha_dup():
    m = pd.DataFrame(dict(
        clip_id=["a", "a"], source_id=["rwf:x", "rwf:y"], dataset=["rwf", "rwf"],
        exclusion=["", ""], dup_of=["", "a"]))
    g = assign_global_groups(m, pd.DataFrame(columns=["clip_a", "clip_b", "distance", "confirmed"]))
    assert g[0] == g[1] == "rwf:x"


def _reencode(src, dst, w, h):
    with av.open(str(src)) as i, av.open(str(dst), "w") as o:
        s = o.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for f in i.decode(video=0):
            nf = av.VideoFrame.from_ndarray(f.reformat(width=w, height=h, format="rgb24").to_ndarray(), format="rgb24")
            for p in s.encode(nf):
                o.mux(p)
        for p in s.encode():
            o.mux(p)


def test_near_dup_detects_reencode(tiny_videos, tmp_path):
    copy = tmp_path / "copy.avi"
    _reencode(tiny_videos["normal"], copy, 160, 120)
    m = pd.DataFrame(dict(
        clip_id=["n", "c", "g"], path=[str(tiny_videos["normal"]), str(copy), str(tiny_videos["gray"])],
        source_id=["rwf:n", "sfd:c", "avd:g"], dataset=["rwf", "sfd", "avd"],
        exclusion=["", "", ""], dup_of=["", "", ""]))
    p = near_duplicate_pairs(m, cache_path=tmp_path / "sig.npz")
    row = p[(p.clip_a == "n") & (p.clip_b == "c") | (p.clip_a == "c") & (p.clip_b == "n")]
    assert len(row) == 1 and row.distance.iloc[0] <= 6
    assert (tmp_path / "sig.npz").exists()
