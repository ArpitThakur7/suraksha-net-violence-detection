"""Stage 8: experiment grid runner.

`python -m pipeline grid [--shard 1,2/7] [--only motionlite] [--dry-run] [--bundle] [--set key=val ...]`

Cells come from configs/grid.yaml (see its header). A cell's run dir is
paths.runs/<protocol>/<variant>/seed<N> (the same dir `train` and `bias` use), so finished cells are
skipped wherever they were trained. Per cell: train_run (resumes from last.pt) then evaluate_run.

Sharding: shard = int(sha1(cell_id), 16) % n, identical on every machine (never Python hash()).
`--shard 1,2/7` runs the union of shards 1 and 2. Cells run in (priority, seed, grid order).
`--dry-run` prints cells, status and estimated hours from work/bench/epoch_seconds.csv.
`--bundle` writes work/colab_bundle/ (repo.zip, cache.zip, splits/, bench/) for colab/train.ipynb.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import subprocess
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from pipeline.bias import PROTOCOLS as BIAS_PROTOCOLS
from pipeline.config import DEFAULT_CONFIG, _git_commit, apply_overrides, load_config
from pipeline.evaluate import evaluate_run
from pipeline.paths import ROOT, runs_root, work_dir
from pipeline.train import run_dir, train_run, variant_name

DEFAULT_GRID = "pipeline/configs/grid.yaml"
# Estimate assumption: early stopping (patience 6, max 40) ends a run after ~15 epochs on average.
EPOCHS_PER_RUN = 15
REF_PROTOCOL, REF_T, REF_CROP = "p2_all", 16, 224  # conditions of the epoch timing pilot
# Recorded config values that must match for a DONE run to count as this cell.
STALE_KEYS = ("train.max_epochs", "train.patience", "train.lr", "train.effective_batch", "clip.T", "clip.crop")
BUNDLE_EXCLUDE = ("datasets/", "work/", "local_data/")


@dataclass
class Cell:
    protocol: str
    model: str
    seed: int
    model_kw: dict = field(default_factory=dict)
    overrides: tuple = ()   # config overrides that name the variant (clip.T=8)
    priority: int = 1
    setup: tuple = ()       # config overrides that do not name the run (per-model lr, micro-batch)

    @property
    def id(self) -> str:
        return f"{self.protocol}/{variant_name(self.model, self.model_kw, self.overrides)}/seed{self.seed}"


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


def load_spec(path=DEFAULT_GRID) -> dict:
    return yaml.safe_load(_resolve(path).read_text(encoding="utf-8"))


def expand_grid(path=DEFAULT_GRID) -> list[Cell]:
    spec = load_spec(path)
    model_setup = spec.get("model_setup") or {}
    cells = []
    for b in spec["blocks"]:
        for v in b.get("variants") or [{}]:
            for m in b["models"]:
                for p in b["protocols"]:
                    for s in b["seeds"]:
                        cells.append(Cell(p, m, int(s), dict(v.get("model_kw") or {}),
                                          tuple(v.get("overrides") or ()), int(b.get("priority", 1)),
                                          tuple(model_setup.get(m, ())) + tuple(v.get("setup") or ())))
    return cells


def shard_of(cell_id: str, n: int) -> int:
    return int(hashlib.sha1(cell_id.encode("utf-8")).hexdigest(), 16) % n


def parse_shard(s: str) -> tuple[tuple[int, ...], int]:
    """"1,2/7" -> ((1, 2), 7)."""
    try:
        ids, n = s.split("/")
        n = int(n)
        ids = tuple(sorted({int(i) for i in ids.split(",")}))
    except ValueError:
        raise ValueError(f"shard must look like 0/7 or 1,2/7, got {s!r}") from None
    if n < 1 or not ids or any(not 0 <= i < n for i in ids):
        raise ValueError(f"shard ids must be in 0..{n - 1}, got {s!r}")
    return ids, n


def select_cells(cells, shard=((0,), 1), only: str | None = None) -> list[Cell]:
    """Cells of the given shard(s) whose id contains `only`, in (priority, seed, grid order)."""
    ids, n = shard
    ids = {ids} if isinstance(ids, int) else set(ids)
    keep = [(i, c) for i, c in enumerate(cells)
            if shard_of(c.id, n) in ids and (not only or only in c.id)]
    return [c for _, c in sorted(keep, key=lambda t: (t[1].priority, t[1].seed, t[0]))]


def cell_cfg(cell: Cell, cfg) -> dict:
    return apply_overrides(copy.deepcopy(cfg), list(cell.setup) + list(cell.overrides))


def cell_dir(cell: Cell, cfg) -> Path:
    return run_dir(cfg, cell.protocol, cell.model, cell.seed, cell.model_kw, cell.overrides)


def _finished(out: Path) -> bool:
    return (out / "DONE").exists() and (out / "test_metrics.json").exists()


def _get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def stale_keys(out: Path, cfg) -> list[str]:
    """Keys whose value recorded in a DONE run's config.json differs from this cell's config (e.g. a
    smoke run with max_epochs=2). Such a run is skipped like any DONE run but reported."""
    f = out / "config.json"
    if not (out / "DONE").exists() or not f.exists():
        return []
    rec = json.loads(f.read_text(encoding="utf-8")).get("cfg", {})
    return [k for k in STALE_KEYS if _get(rec, k) != _get(cfg, k)]


def status(out: Path) -> str:
    if _finished(out):
        return "done"
    if (out / "DONE").exists():
        return "trained"
    return "partial" if (out / "last.pt").exists() else "todo"


def run_grid(cells, cfg, shard=((0,), 1), only: str | None = None) -> list[Cell]:
    """Train + evaluate every selected cell that is not finished. A failing cell is reported and the
    grid moves on. Returns the failed cells."""
    import torch
    sel = select_cells(cells, shard, only)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"grid: {len(sel)} cells, shard {shard}, runs root {runs_root(cfg)}, gpu {gpu}", flush=True)
    failed = []
    for k, cell in enumerate(sel, 1):
        c = cell_cfg(cell, cfg)
        out = cell_dir(cell, c)
        if _finished(out):
            stale = stale_keys(out, c)
            print(f"[{k}/{len(sel)}] skip {cell.id} (done)"
                  + (f" WARNING stale DONE run, differs in {stale}; delete it to retrain" if stale else ""))
            continue
        print(f"[{k}/{len(sel)}] run {cell.id} (priority {cell.priority})", flush=True)
        t0 = time.time()
        try:
            train_run(c, cell.protocol, cell.model, cell.seed, out, cell.model_kw)
            if not (out / "test_metrics.json").exists():
                evaluate_run(out, c)
            print(f"[{k}/{len(sel)}] finished {cell.id} in {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[{k}/{len(sel)}] FAILED {cell.id}", flush=True)
            failed.append(cell)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failed:
        print("failed cells:\n  " + "\n  ".join(c.id for c in failed))
    return failed


# ---------------------------------------------------------------- estimates / dry run

def load_epoch_seconds(cfg) -> dict:
    f = work_dir(cfg) / "bench" / "epoch_seconds.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return {str(r.model): float(r.seconds_per_epoch) for r in df.itertuples()}


_SIZES: dict = {}


def _protocol_size(protocol: str, cfg) -> int | None:
    """Train + val rows of a protocol (an epoch = one train pass + one val pass)."""
    f = work_dir(cfg) / "splits" / f"{protocol}.csv"
    if f not in _SIZES:
        _SIZES[f] = int(pd.read_csv(f, usecols=["split"])["split"].isin(("train", "val")).sum()) \
            if f.exists() else None
    return _SIZES[f]


def estimate_hours(cell: Cell, sec: dict, cfg) -> float:
    """EPOCHS_PER_RUN x pilot seconds/epoch of the model, scaled by the protocol's train+val size
    relative to p2_all and by T/16 x (crop/224)^2 for the input ablations. nan if the model has no
    pilot timing."""
    if cell.model not in sec:
        return float("nan")
    n, ref = _protocol_size(cell.protocol, cfg), _protocol_size(REF_PROTOCOL, cfg)
    size = n / ref if n and ref else 1.0
    clip = cell_cfg(cell, cfg)["clip"]
    inp = int(clip["T"]) / REF_T * (int(clip["crop"]) / REF_CROP) ** 2
    return sec[cell.model] * EPOCHS_PER_RUN * size * inp / 3600


def _h(x: float) -> str:
    return "?" if math.isnan(x) else f"{x:.1f}"


def dry_run(cells, cfg, shard=((0,), 1), only: str | None = None) -> pd.DataFrame:
    sec = load_epoch_seconds(cfg)
    if not sec:
        print(f"no {work_dir(cfg) / 'bench' / 'epoch_seconds.csv'}; run `python -m pipeline bench --epoch`")
    rows = []
    for c in cells:
        cc = cell_cfg(c, cfg)
        out = cell_dir(c, cc)
        rows.append({"id": c.id, "priority": c.priority, "shard": shard_of(c.id, shard[1]),
                     "status": status(out), "stale": ",".join(stale_keys(out, cc)),
                     "hours": estimate_hours(c, sec, cfg)})
    allrows = pd.DataFrame(rows).set_index("id")
    sel = [c.id for c in select_cells(cells, shard, only)]
    df = allrows.loc[sel].reset_index()
    for r in df.itertuples():
        print(f"p{r.priority} shard{r.shard} {r.status:8s} {_h(r.hours):>6s} h  {r.id}"
              + (f"  [stale DONE: {r.stale}]" if r.stale else ""))
    todo = df[df.status != "done"]
    print(f"\n{len(df)} cells selected (shard {shard}, only={only}), {len(todo)} not finished; "
          f"estimate = {EPOCHS_PER_RUN} epochs/run x pilot s/epoch (laptop GPU), scaled by protocol size and T x crop^2")
    cum = 0.0
    for p, g in todo.groupby("priority"):
        h = g.hours.sum(min_count=1)
        cum += 0.0 if math.isnan(h) else h
        print(f"  priority {p}: {len(g):3d} cells  {_h(h):>7s} h  (cumulative {cum:.1f} h)"
              + (f"  [{int(g.hours.isna().sum())} cells without timing]" if g.hours.isna().any() else ""))
    print(f"  total: {cum:.1f} h")
    if shard[1] > 1:
        pend = allrows[allrows.status != "done"]
        if only:
            pend = pend[pend.index.str.contains(only, regex=False)]
        print(f"remaining hours per shard (of {shard[1]}, all priorities):")
        for s in range(shard[1]):
            g = pend[pend.shard == s]
            print(f"  shard {s}/{shard[1]}: {len(g):3d} cells  {g.hours.sum():.1f} h")
    return df


# ---------------------------------------------------------------- Colab bundle

def _bundle_files(files) -> list[str]:
    return [f for f in files if not f.startswith(BUNDLE_EXCLUDE) and not f.lower().endswith(".zip")]


def _git_files(root: Path) -> list[str]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [f for f in r.stdout.decode("utf-8").split("\0") if f]


def _dirty(root: Path) -> bool:
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def make_bundle(cfg, out: Path, root: Path = ROOT, files=None, cache_key: str | None = None,
                commit: str | None = None) -> dict:
    """repo.zip (tracked files minus datasets/, work/ and zips, plus GIT_COMMIT), cache.zip
    (cache/<key>/*, ZIP_STORED: frames are already JPEG), splits/ and bench/epoch_seconds.csv.
    Returns {name: bytes}."""
    from pipeline.data.cache import cache_key as _key
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(root)
    if files is None:
        files = _git_files(root)
        if commit is None and _dirty(root):
            print("WARNING: tracked files have uncommitted changes; GIT_COMMIT is marked -dirty")
            commit = _git_commit() + "-dirty"
    commit = commit or _git_commit()
    work = work_dir(cfg)
    key = cache_key or _key(cfg)
    cache = work / "cache" / key
    if not cache.is_dir():
        raise FileNotFoundError(f"no frame cache at {cache}; run `python -m pipeline cache`")

    tmp = out / "repo.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in _bundle_files(files):
            if (root / f).is_file():
                z.write(root / f, f)
        z.writestr("GIT_COMMIT", commit + "\n")
    tmp.replace(out / "repo.zip")
    tmp = out / "cache.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in sorted(cache.iterdir()):
            if f.is_file() and f.suffix == ".npz":
                z.write(f, f"cache/{key}/{f.name}")
    tmp.replace(out / "cache.zip")

    if (out / "splits").exists():
        shutil.rmtree(out / "splits")
    shutil.copytree(work / "splits", out / "splits")
    es = work / "bench" / "epoch_seconds.csv"
    if es.exists():
        (out / "bench").mkdir(exist_ok=True)
        shutil.copy2(es, out / "bench" / "epoch_seconds.csv")
    else:
        print(f"WARNING: {es} missing; --dry-run on Colab will have no estimates")
    sizes = {p.relative_to(out).as_posix(): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()}
    return sizes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline grid")
    ap.add_argument("--shard", default="0/1", help="i/n or i,j/n (union of shards), e.g. 1,2/7")
    ap.add_argument("--only", default=None, help="run only cells whose id contains this, e.g. motionlite")
    ap.add_argument("--dry-run", action="store_true", help="print cells, status and estimated hours")
    ap.add_argument("--bundle", action="store_true", help="write work/colab_bundle for Colab")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                    help="config override (repeatable), e.g. paths.runs=/content/drive/MyDrive/x/runs")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--grid", default=DEFAULT_GRID)
    a = ap.parse_args(argv)
    spec = load_spec(a.grid)
    cfg = load_config(a.config, list(spec.get("defaults") or []) + a.overrides)
    if a.bundle:
        out = work_dir(cfg) / "colab_bundle"
        sizes = make_bundle(cfg, out)
        for name, b in sizes.items():
            if "/" not in name or name.startswith("bench/"):
                print(f"  {name:28s} {b / 2**20:10.1f} MB")
        print(f"  splits/ ({sum(1 for n in sizes if n.startswith('splits/'))} files)"
              f" {sum(b for n, b in sizes.items() if n.startswith('splits/')) / 2**20:.1f} MB")
        print(f"bundle written to {out}; upload its contents to the shared Drive folder")
        return 0
    cells = expand_grid(a.grid)
    shard = parse_shard(a.shard)
    if a.dry_run:
        dry_run(cells, cfg, shard, a.only)
        return 0
    return 1 if run_grid(cells, cfg, shard, a.only) else 0
