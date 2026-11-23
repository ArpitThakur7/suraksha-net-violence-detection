"""Place downloaded dataset copies at the manifest paths, verified by SHA-256.

`python -m pipeline fetch --src DIR [--src DIR ...]` walks every DIR for video files, hashes them
and, for each manifest clip whose sha256 matches, links (or copies) the file to ROOT/<path>. A clip
is identified by content only, so the folder layout of a download (Kaggle, GitHub) does not
matter and a re-encoded or different copy is never mistaken for ours. Exits 1 if any clip that the
frame cache needs (exclusion == "") has no byte-identical file, listing the counts per dataset.
`--allow-name-match`: a clip with no byte-identical file may instead take a file whose last
NAME_PARTS path components (e.g. train/Fight/x.avi) equal the manifest path's; reported separately
(same video, possibly a different encode, so decoded frames can differ slightly).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline.paths import ROOT, WORK

NAME_PARTS = 3
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg", ".webm", ".m4v", ".wmv", ".flv"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _videos(dirs) -> list[Path]:
    out = []
    for d in dirs:
        for dp, _, fs in os.walk(d):
            out += [Path(dp) / f for f in fs if Path(f).suffix.lower() in VIDEO_EXT]
    return sorted(out)


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _tail(p, n=NAME_PARTS) -> str:
    return "/".join(Path(str(p).replace("\\", "/")).parts[-n:]).lower()


def fetch(src_dirs, manifest: pd.DataFrame, root: Path = ROOT, workers: int = 8,
          allow_name_match: bool = False) -> dict:
    """Returns {"placed": n, "needed": {dataset: n}, "found": {dataset: n}, "missing": [clip_id],
    "unmatched_files": n}. Files already at their manifest path with the right hash count as found."""
    root = Path(root)
    m = manifest.copy()
    m["exclusion"] = m["exclusion"].fillna("")
    files = _videos(src_dirs)
    with ThreadPoolExecutor(workers) as ex:
        hashes = list(ex.map(sha256, files))
    by_hash: dict[str, Path] = {}
    for f, h in zip(files, hashes):
        by_hash.setdefault(h, f)
    want = set(m["sha256"])
    by_name: dict[str, Path] = {}
    for f, h in zip(files, hashes):
        if h not in want:
            by_name.setdefault(_tail(f), f)
    placed, found_ids, name_ids = 0, set(), set()
    for r in m.itertuples():
        dst = root / r.path
        src = by_hash.get(r.sha256)
        if src is not None:
            if src.resolve() != dst.resolve():
                _place(src, dst)
                placed += 1
            found_ids.add(r.clip_id)
        elif dst.exists() and sha256(dst) == r.sha256:
            found_ids.add(r.clip_id)
        elif allow_name_match and _tail(r.path) in by_name:
            _place(by_name[_tail(r.path)], dst)
            placed += 1
            found_ids.add(r.clip_id)
            name_ids.add(r.clip_id)
    need = m[m["exclusion"] == ""].drop_duplicates("clip_id")
    missing = [c for c in need["clip_id"] if c not in found_ids]
    return {"placed": placed,
            "needed": need.groupby("dataset").size().to_dict(),
            "found": need[need["clip_id"].isin(found_ids)].groupby("dataset").size().to_dict(),
            "missing": missing,
            "name_matched": need[need["clip_id"].isin(name_ids)].groupby("dataset").size().to_dict(),
            "unmatched_files": sum(1 for h in hashes if h not in want)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline fetch")
    ap.add_argument("--src", action="append", required=True, help="folder with downloaded videos")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-name-match", action="store_true",
                    help="fall back to matching by the last path components when no byte-identical file exists")
    a = ap.parse_args(argv)
    m = pd.read_csv(WORK / "manifest" / "all_clips.csv", keep_default_na=False)
    rep = fetch(a.src, m, workers=a.workers, allow_name_match=a.allow_name_match)
    for ds, n in sorted(rep["needed"].items()):
        nm = rep["name_matched"].get(ds, 0)
        print(f"[fetch] {ds:5s} {rep['found'].get(ds, 0) - nm:5d} / {n} clips verified by sha256"
              + (f", {nm} matched by file name only (bytes differ)" if nm else ""))
    print(f"[fetch] placed {rep['placed']} files under {ROOT}; "
          f"{rep['unmatched_files']} downloaded videos are not in the manifest (ignored)")
    if rep["missing"]:
        print(f"[fetch] MISSING {len(rep['missing'])} clips, e.g. {rep['missing'][:5]}")
        return 1
    print("[fetch] all clips present")
    return 0
