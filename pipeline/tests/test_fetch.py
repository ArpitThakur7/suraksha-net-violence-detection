import hashlib

import pandas as pd

from pipeline.fetch import fetch


def _h(b):
    return hashlib.sha256(b).hexdigest()


def test_fetch_places_by_hash_and_reports_missing(tmp_path):
    dl = tmp_path / "download" / "weird" / "layout"
    dl.mkdir(parents=True)
    (dl / "x.avi").write_bytes(b"clip-a")
    (dl / "y.mp4").write_bytes(b"clip-b-reencoded")      # not byte-identical -> must not match
    (dl / "z.mp4").write_bytes(b"extra")                  # not in manifest
    m = pd.DataFrame({
        "clip_id": ["a", "b", "c"],
        "path": ["datasets/public/rwf/train/Fight/a.avi", "datasets/public/sfd/b.mp4", "datasets/public/sfd/c.mp4"],
        "dataset": ["rwf", "sfd", "sfd"],
        "sha256": [_h(b"clip-a"), _h(b"clip-b"), _h(b"clip-c")],
        "exclusion": ["", "", "duplicate"],
    })
    root = tmp_path / "repo"
    rep = fetch([tmp_path / "download"], m, root=root, workers=2)
    assert (root / "datasets/public/rwf/train/Fight/a.avi").read_bytes() == b"clip-a"
    assert not (root / "datasets/public/sfd/b.mp4").exists()
    assert rep["missing"] == ["b"]                       # c is excluded, so not required
    assert rep["needed"] == {"rwf": 1, "sfd": 1} and rep["found"] == {"rwf": 1}
    assert rep["unmatched_files"] == 2 and rep["placed"] == 1
    # a second pass with the file already in place still counts it as found
    rep2 = fetch([tmp_path / "empty"], m, root=root)
    assert rep2["found"] == {"rwf": 1}


def test_fetch_name_match_fallback(tmp_path):
    dl = tmp_path / "dl" / "RWF-2000" / "train" / "Fight"
    dl.mkdir(parents=True)
    (dl / "a.avi").write_bytes(b"reencoded")
    m = pd.DataFrame({"clip_id": ["a"], "path": ["datasets/public/rwf2000/RWF-2000/train/Fight/a.avi"],
                      "dataset": ["rwf"], "sha256": [_h(b"original")], "exclusion": [""]})
    root = tmp_path / "repo"
    assert fetch([tmp_path / "dl"], m, root=root)["missing"] == ["a"]
    rep = fetch([tmp_path / "dl"], m, root=root, allow_name_match=True)
    assert rep["missing"] == [] and rep["name_matched"] == {"rwf": 1}
    assert (root / m.path[0]).read_bytes() == b"reencoded"
