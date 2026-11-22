"""Stage 1: inventory + audit of RWF-2000, SFD and AIRTLab into one manifest."""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline.paths import ROOT, WORK
from pipeline.data.video import probe_video

YT_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
COLUMNS = ["clip_id", "path", "dataset", "label", "source_id", "original_split", "sha256",
           "fps", "frames", "width", "height", "duration_s", "error", "exclusion", "dup_of"]


def rwf_source_id(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"_\d+$", "", stem)


def normalize_source(dataset: str, raw: str) -> str:
    raw = str(raw)
    if dataset == "avd":
        return "avd:" + (raw[4:] if raw.startswith("avd:") else raw)
    if dataset == "sfd" and raw.startswith("sfd:"):
        raw = raw[4:]
    if YT_RE.match(raw):
        return f"yt:{raw}"
    return f"{dataset}:{raw}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _inventory(cfg) -> list[dict]:
    rows = []
    rwf = ROOT / cfg["paths"]["rwf_root"]
    for split in ("train", "val"):
        for cls, label in (("Fight", 1), ("NonFight", 0)):
            for p in sorted((rwf / split / cls).glob("*.avi")):
                rows.append(dict(path=p.relative_to(ROOT).as_posix(), dataset="rwf", label=label,
                                 source_id=normalize_source("rwf", rwf_source_id(p.name)),
                                 original_split=split))
    meta = pd.read_csv(ROOT / cfg["paths"]["handoff"] / "dataset_metadata.csv")
    for r in meta.itertuples():
        rows.append(dict(path=str(r.path).replace("\\", "/"), dataset=r.dataset, label=int(r.label),
                         source_id=normalize_source(r.dataset, r.event_group),
                         original_split=r.split))
    return rows


def _audit_one(row: dict) -> dict:
    out = dict(row)
    out["error"] = ""
    p = ROOT / row["path"]
    try:
        out["sha256"] = _sha256(p)
    except Exception as e:
        out["sha256"] = ""
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        pr = probe_video(p)
    except Exception as e:  # probe_video should not raise, but never let a clip kill the run
        pr = dict(fps=0.0, frames=0, width=0, height=0, duration_s=0.0,
                  error=f"{type(e).__name__}: {e}"[:200])
    if pr["error"] and not out["error"]:
        out["error"] = pr["error"]
    out.update({k: pr[k] for k in ("fps", "frames", "width", "height", "duration_s")})
    out["clip_id"] = (out["sha256"][:16] if out["sha256"]
                      else "err-" + hashlib.sha1(row["path"].encode()).hexdigest()[:12])
    return out


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Set `exclusion` and `dup_of`. df must be sorted by path with a RangeIndex.
    dup_of = clip_id of the first copy (by path) for later copies, else "".
    duplicate_sha is excluded only when the copy shares original_split with the first copy;
    cross-split copies are kept."""
    df = df.copy()
    excl = pd.Series("", index=df.index)
    excl[df["frames"] == 0] = "empty"
    excl[df["error"] != ""] = "decode_error"
    dup_of = pd.Series("", index=df.index)
    has = df["sha256"] != ""
    first = df[has].groupby("sha256").head(1).set_index("sha256")
    for i in df.index[has & df.duplicated("sha256", keep="first")]:
        f = first.loc[df.at[i, "sha256"]]
        dup_of[i] = f["clip_id"]
        if f["original_split"] == df.at[i, "original_split"] and excl[i] == "":
            excl[i] = "duplicate_sha"
    df["exclusion"], df["dup_of"] = excl, dup_of
    return df


def build_manifest(cfg, out_path=None) -> pd.DataFrame:
    rows = _inventory(cfg)
    results, done = [], 0
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(_audit_one, rows):
            results.append(r)
            done += 1
            if done % 200 == 0:
                print(f"audited {done}/{len(rows)}", flush=True)
    df = pd.DataFrame(results)
    df = df.sort_values("path").reset_index(drop=True)
    df = mark_duplicates(df)
    df = df[COLUMNS]
    out = Path(out_path) if out_path else WORK / "manifest" / "all_clips.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_report(df, out.parent / "audit_report.md")
    return df


def write_report(df: pd.DataFrame, path) -> None:
    L = ["# Dataset audit report", "", f"Total clips: {len(df)}", "",
         "## Counts by dataset x original_split x label", "",
         "| dataset | original_split | label | n |", "|---|---|---|---|"]
    for (d, s, l), n in df.groupby(["dataset", "original_split", "label"]).size().items():
        L.append(f"| {d} | {s} | {l} | {n} |")
    L += ["", "## Exclusions by reason", "", "| dataset | reason | n |", "|---|---|---|"]
    ex = df[df.exclusion != ""]
    if ex.empty:
        L.append("| - | none | 0 |")
    for (d, r), n in ex.groupby(["dataset", "exclusion"]).size().items():
        L.append(f"| {d} | {r} | {n} |")
    L += ["", "## Exact duplicates", ""]
    d = df[df.dup_of != ""]
    pairs, within, cross = [], 0, 0
    for r in d.itertuples():
        f = df[df.clip_id == r.dup_of].iloc[0]
        same = f.original_split == r.original_split
        within += same
        cross += not same
        pairs.append(f"| {f.path} | {r.path} | {r.label} | {f.original_split} / {r.original_split} | {'within-split' if same else 'cross-split'} |")
    L += [f"Duplicate copies: {len(d)} (within-split {within}, cross-split {cross})", "",
          "| first copy | duplicate copy | label | original_split (first / dup) | kind |", "|---|---|---|---|---|"] + pairs
    L += ["", "## Per-dataset summaries (non-excluded clips)", ""]
    ok = df[df.exclusion == ""]
    for d, g in ok.groupby("dataset"):
        res = g.groupby(["width", "height"]).size().sort_values(ascending=False).head(8)
        L += [f"### {d} (n={len(g)})", "",
              f"- fps: min {g.fps.min():.2f}, median {g.fps.median():.2f}, max {g.fps.max():.2f}",
              f"- duration_s: min {g.duration_s.min():.2f}, median {g.duration_s.median():.2f}, max {g.duration_s.max():.2f}",
              f"- frames: min {g.frames.min()}, median {g.frames.median():.0f}, max {g.frames.max()}",
              "- resolutions: " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in res.items()),
              ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")
