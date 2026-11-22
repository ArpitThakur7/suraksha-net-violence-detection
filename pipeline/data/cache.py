"""Stage 3: JPEG frame cache. One .npz per clip_id: concatenated JPEG bytes + offsets + shape."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.config import load_config, run_metadata
from pipeline.data.video import decode_frames
from pipeline.paths import ROOT, WORK


def cache_key(cfg) -> str:
    return hashlib.sha1(json.dumps(cfg["cache"], sort_keys=True).encode()).hexdigest()[:8]


def cache_dir(cfg) -> Path:
    work = Path(cfg.get("paths", {}).get("work", WORK))
    return work / "cache" / cache_key(cfg)


def _file(clip_id, cfg) -> Path:
    return cache_dir(cfg) / f"{clip_id}.npz"


def write_clip(clip_id, frames: np.ndarray, cfg) -> Path:
    q = int(cfg["cache"]["jpeg_quality"])
    blobs = []
    for f in frames:
        buf = BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=q)
        blobs.append(buf.getvalue())
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in blobs])
    data = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    shape = np.array(frames.shape, dtype=np.int32)
    out = _file(clip_id, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, data=data, offsets=offsets, shape=shape)
    os.replace(tmp, out)
    return out


def read_clip(clip_id, cfg, idx=None) -> np.ndarray:
    """Decode frames `idx` (any order, repeats allowed) or all frames. Only the requested
    JPEGs are decoded, but np.load of an npz member reads the whole `data` blob."""
    with np.load(_file(clip_id, cfg)) as z:
        data, offsets, shape = z["data"], z["offsets"], z["shape"]
    n, h, w, _ = (int(s) for s in shape)
    sel = np.arange(n, dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64).reshape(-1)
    if sel.size and (sel.min() < 0 or sel.max() >= n):
        raise ValueError(f"frame index out of range for {clip_id}: {sel.tolist()} (n={n})")
    out = np.empty((len(sel), h, w, 3), dtype=np.uint8)
    for k, i in enumerate(sel):
        blob = data[offsets[i]:offsets[i + 1]].tobytes()
        with Image.open(BytesIO(blob)) as im:
            out[k] = np.asarray(im.convert("RGB"))
    return out


def num_frames(clip_id, cfg) -> int:
    with np.load(_file(clip_id, cfg)) as z:
        return int(z["shape"][0])


def _work(args):
    clip_id, path, cfg = args
    try:
        fr = decode_frames(path, fps=cfg["cache"]["fps"], short_side=cfg["cache"]["short_side"])
        write_clip(clip_id, fr, cfg)
        return clip_id, None
    except Exception as e:
        return clip_id, f"{type(e).__name__}: {e}"[:300]


def build_cache(manifest: pd.DataFrame, cfg, workers: int = 6, pool: str = "process") -> dict:
    t0 = time.time()
    d = cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*.tmp"):  # leftovers from an interrupted run
        p.unlink()
    m = manifest[manifest["exclusion"].fillna("") == ""].drop_duplicates("clip_id")
    todo, skipped = [], 0
    for r in m.itertuples():
        if _file(r.clip_id, cfg).exists():
            skipped += 1
        else:
            todo.append((r.clip_id, str(ROOT / r.path), cfg))
    failures = []
    if todo:
        Ex = ProcessPoolExecutor if pool == "process" else ThreadPoolExecutor
        with Ex(max_workers=workers) as ex:
            for i, (cid, err) in enumerate(ex.map(_work, todo, chunksize=1 if pool != "process" else 2), 1):
                if err:
                    failures.append((cid, err))
                    print(f"[cache] WARNING failed {cid}: {err}", flush=True)
                if i % 100 == 0 or i == len(todo):
                    print(f"[cache] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    fpath = d / "failures.csv"
    if failures:
        with open(fpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["clip_id", "error"])
            w.writerows(failures)
    elif fpath.exists():
        fpath.unlink()
    rows, total = [], 0
    for r in m.itertuples():
        f = _file(r.clip_id, cfg)
        if not f.exists():
            continue
        total += f.stat().st_size
        with np.load(f) as z:
            n, h, w_, _ = (int(s) for s in z["shape"])
        rows.append((r.clip_id, n, h, w_))
    pd.DataFrame(rows, columns=["clip_id", "n_frames", "h", "w"]).to_csv(d / "index.csv", index=False)
    info = dict(wall_time_s=round(time.time() - t0, 1), n_clips=len(m), n_written=len(todo) - len(failures),
                n_skipped=skipped, n_failed=len(failures), total_bytes=total, cache=cfg["cache"],
                workers=workers, pool=pool, metadata=run_metadata())
    (d / "build.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} clips failed to cache; see {fpath}")
    return info


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pipeline cache")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pool", choices=("process", "thread"), default="process")
    ap.add_argument("--config", default="pipeline/configs/base.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (benchmarking)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    m = pd.read_csv(WORK / "manifest" / "all_clips.csv", keep_default_na=False)
    if a.limit:
        m = m.head(a.limit)
    try:
        info = build_cache(m, cfg, workers=a.workers, pool=a.pool)
    except RuntimeError as e:
        print(f"[cache] {e}")
        return 1
    print(f"[cache] done: {info['n_written']} written, {info['n_skipped']} skipped, "
          f"{info['total_bytes'] / 1e9:.2f} GB, {info['wall_time_s']}s -> {cache_dir(cfg)}")
    return 0
