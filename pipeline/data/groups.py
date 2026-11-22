"""Stage 2: perceptual near-duplicate detection and cross-dataset global groups."""
from __future__ import annotations

import pipeline.paths  # noqa: F401

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

from pipeline.paths import ROOT, WORK

NEAR_DUP = 6      # confirmed threshold
BORDERLINE = 10   # upper bound of the borderline band


def phash(img: np.ndarray) -> int:
    """64-bit pHash: gray -> 32x32 -> 2D DCT -> 8x8 low band without DC, vs median."""
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR, reducing_gap=None), dtype=np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].ravel()
    bits = d[1:]  # drop DC -> 63 values
    med = np.median(bits)
    code = 0
    for i, b in enumerate(bits > med):
        if b:
            code |= 1 << i
    return code


def _indices(total: int, n: int) -> np.ndarray:
    return np.linspace(0, max(total - 1, 0), n).round().astype(int)


def sample_frames(path, n: int = 8) -> list[np.ndarray]:
    """n RGB frames at evenly spaced positions (repeats the last frame if the clip is short)."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        total = int(s.frames or 0)
        if total <= 0:  # unknown length: count in a first pass
            total = sum(1 for _ in c.decode(s))
            c.seek(0)
        want = _indices(total, n)
        need = set(want.tolist())
        got: dict[int, np.ndarray] = {}
        for i, f in enumerate(c.decode(s)):
            if i in need:
                got[i] = f.to_ndarray(format="rgb24")
            if i >= want[-1]:
                break
    if not got:
        raise ValueError(f"no frames decoded from {path}")
    out, prev = [], got[min(got)]
    for k in want.tolist():
        prev = got.get(k, prev)
        out.append(prev)
    return out


def clip_signature(path, n: int = 8) -> np.ndarray:
    return np.array([phash(f) for f in sample_frames(path, n)], dtype=np.uint64)


def signature_distance(sa: np.ndarray, sb: np.ndarray) -> float:
    """Each frame of a matched to its nearest frame in b and vice versa; mean of the two
    directional mean Hamming distances (symmetric best-alignment distance)."""
    h = np.bitwise_count(sa[:, None] ^ sb[None, :]).astype(float)
    return float((h.min(1).mean() + h.min(0).mean()) / 2)


def load_signatures(manifest: pd.DataFrame, cache_path=None, root=ROOT, workers: int = 6) -> dict:
    cache_path = Path(cache_path) if cache_path else WORK / "manifest" / "signatures.npz"
    sigs: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            sigs = {k: z[k] for k in z.files}
    todo = [(r.clip_id, r.path) for r in manifest.itertuples() if r.clip_id not in sigs]

    def one(t):
        try:
            return t[0], clip_signature(Path(root) / t[1])
        except Exception:
            return t[0], None
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for k, (cid, s) in enumerate(ex.map(one, todo)):
                if s is not None:
                    sigs[cid] = s
                if (k + 1) % 200 == 0:
                    print(f"signatures {k + 1}/{len(todo)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **sigs)
    return sigs


def _active(manifest: pd.DataFrame) -> pd.DataFrame:
    if "exclusion" in manifest:
        return manifest[manifest["exclusion"].fillna("") == ""]
    return manifest


def near_duplicate_pairs(manifest: pd.DataFrame, max_mean_hamming: float = 10, sigs=None,
                         cache_path=None, root=ROOT) -> pd.DataFrame:
    """Pairs of non-excluded clips from different source_ids with distance <= max_mean_hamming."""
    # byte-identical copies share a clip_id: compare each content once (they are unioned by clip_id)
    m = _active(manifest).drop_duplicates("clip_id").reset_index(drop=True)
    if sigs is None:
        sigs = load_signatures(m, cache_path, root)
    m = m[m.clip_id.isin(sigs)].reset_index(drop=True)
    ids = m.clip_id.to_numpy()
    src = m.source_id.to_numpy()
    S = np.stack([sigs[c] for c in ids]) if len(ids) else np.zeros((0, 8), np.uint64)
    rows = []
    for i in range(len(ids) - 1):
        # h[j,p,q] = hamming(frame p of i, frame q of j)
        h = np.bitwise_count(S[i][None, :, None] ^ S[i + 1:][:, None, :]).astype(np.float32)
        d = (h.min(2).mean(1) + h.min(1).mean(1)) / 2
        ok = (d <= max_mean_hamming) & (src[i + 1:] != src[i])
        for j in np.flatnonzero(ok):
            rows.append((ids[i], ids[i + 1 + j], float(d[j])))
    p = pd.DataFrame(rows, columns=["clip_a", "clip_b", "distance"])
    p = p.sort_values("distance", kind="stable").reset_index(drop=True)
    p["confirmed"] = p.distance <= NEAR_DUP
    ds = m.set_index("clip_id").dataset if "dataset" in m else None
    avd = (pd.Series(p.clip_a.map(ds) == "avd") & (p.clip_b.map(ds) == "avd")) if ds is not None         else pd.Series(False, index=p.index)
    p["used_for_grouping"] = p.confirmed & ~avd
    # AIRTLab (avd): single-room staged dataset, publisher scene ids are authoritative
    p["reason"] = np.where(avd, "avd_same_room", np.where(p.confirmed, "", "unconfirmed"))
    return p


def assign_global_groups(manifest: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Union-find over source_id, byte-identical copies (dup_of) and confirmed pairs.
    Label = smallest member source_id. Excluded rows still get a group (via source_id)."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # clip_id = sha256 prefix, so byte-identical copies share a clip_id: union all its source_ids
    sids: dict[str, list[str]] = {}
    for c, s in zip(manifest.clip_id, manifest.source_id):
        sids.setdefault(c, []).append(s)
        find(s)
    for lst in sids.values():
        for s in lst[1:]:
            union(lst[0], s)
    if "dup_of" in manifest:
        for c, d in zip(manifest.clip_id, manifest.dup_of.fillna("")):
            if d and d in sids and c in sids:
                union(sids[c][0], sids[d][0])
    if len(pairs):
        use = pairs["used_for_grouping"] if "used_for_grouping" in pairs else pairs["confirmed"]
        for r in pairs[use.astype(bool)].itertuples():
            if r.clip_a in sids and r.clip_b in sids:
                union(sids[r.clip_a][0], sids[r.clip_b][0])
    out = manifest.source_id.map(find)
    out.name = "global_group"
    return out


def contact_sheet(pairs: pd.DataFrame, manifest: pd.DataFrame, out_path, k: int = 40, root=ROOT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    info = manifest.drop_duplicates("clip_id").set_index("clip_id")
    top = pairs.head(k)
    n = max(len(top), 1)
    fig, axes = plt.subplots(n, 2, figsize=(7, 2.2 * n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, p in enumerate(top.itertuples()):
        for c, cid in enumerate((p.clip_a, p.clip_b)):
            try:
                axes[r, c].imshow(sample_frames(Path(root) / info.at[cid, "path"], 3)[1])
            except Exception:
                axes[r, c].text(0.5, 0.5, "unreadable", ha="center")
            i = info.loc[cid]
            axes[r, c].set_title(f"{cid[:8]} {i.dataset}/{i.original_split}/y{i.label}", fontsize=7)
        axes[r, 0].text(0, 1.28, f"pair {r + 1}  d={p.distance:.2f}", transform=axes[r, 0].transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=70)
    plt.close(fig)


def write_groups_report(m: pd.DataFrame, pairs: pd.DataFrame, sigs: dict, path) -> dict:
    act = m[m.exclusion.fillna("") == ""]
    g = m.groupby("global_group")
    ng = g.ngroups
    multi_ds = int((g.dataset.nunique() > 1).sum())
    multi_sp = int((g.original_split.nunique() > 1).sum())
    rows = []
    for sid, grp in act.groupby("source_id"):
        if grp.dataset.nunique() < 2:
            continue
        recs = list(grp.itertuples())
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a.dataset != b.dataset and a.clip_id in sigs and b.clip_id in sigs:
                    rows.append((sid, a, b, signature_distance(sigs[a.clip_id], sigs[b.clip_id])))
    d = np.array([r[3] for r in rows])
    n_dup, n_bd = int((d <= NEAR_DUP).sum()), int(((d > NEAR_DUP) & (d <= BORDERLINE)).sum())
    n_src = int((d > BORDERLINE).sum())
    shared = {r[0] for r in rows}
    n_rwf = int(((act.dataset == "rwf") & act.source_id.isin(shared)).sum())
    L = ["# Global groups report", "", f"Clips: {len(m)} ({len(act)} non-excluded)", "",
         f"- global groups: {ng}", f"- groups spanning >1 dataset: {multi_ds}",
         f"- groups spanning >1 original split: {multi_sp}",
         f"- clips with shared_source: {int(m.shared_source.sum())}", "",
         "## Shared-source clip pairs (same source_id, different datasets)", "",
         f"- shared source_ids: {len(shared)}; RWF clips with a shared source: {n_rwf}",
         f"- clip pairs: {len(rows)}",
         f"- visual near-duplicates (distance <= {NEAR_DUP}): {n_dup}",
         f"- borderline ({NEAR_DUP + 1}-{BORDERLINE}): {n_bd}",
         f"- same-source-only (> {BORDERLINE}): {n_src}", "",
         "Minimum pairwise pHash distance per shared source_id:", "",
         "| source_id | pairs | min distance | datasets (clips) | dataset/split (clips) |", "|---|---|---|---|---|"]
    for sid in sorted(shared):
        rr = [r for r in rows if r[0] == sid]
        grp = act[act.source_id == sid]
        ds = ", ".join(f"{k}:{v}" for k, v in grp.dataset.value_counts().items())
        sp = ", ".join(f"{a}/{b}:{v}" for (a, b), v in grp.groupby(["dataset", "original_split"]).size().items())
        L.append(f"| {sid} | {len(rr)} | {min(r[3] for r in rr):.2f} | {ds} | {sp} |")
    L += ["", "## Near-duplicate pairs across different source_ids", "",
          f"Total pairs <= {BORDERLINE}: {len(pairs)}; confirmed (<= {NEAR_DUP}): {int(pairs.confirmed.sum())}; "
          f"borderline unconfirmed: {int((~pairs.confirmed).sum())}. Borderline pairs need human inspection "
          "of near_duplicates_sheet.png.", "", "Top 40 by distance (smallest first):", "",
          "| clip_a | a (dataset/split/label) | clip_b | b (dataset/split/label) | distance | confirmed |",
          "|---|---|---|---|---|---|"]
    info = m.drop_duplicates("clip_id").set_index("clip_id")
    for p in pairs.head(40).itertuples():
        a, b = info.loc[p.clip_a], info.loc[p.clip_b]
        L.append(f"| {p.clip_a[:8]} | {a.dataset}/{a.original_split}/{a.label} | {p.clip_b[:8]} | "
                 f"{b.dataset}/{b.original_split}/{b.label} | {p.distance:.2f} | {p.confirmed} |")
    info2 = m.drop_duplicates("clip_id").set_index("clip_id")
    da, db = pairs.clip_a.map(info2.dataset), pairs.clip_b.map(info2.dataset)
    key = pd.Series(["-".join(sorted(t)) for t in zip(da, db)], index=pairs.index)
    anyavd = (da == "avd") | (db == "avd")
    L += ["", "## Near-duplicate pairs by dataset pair", "",
          "AIRTLab (avd) pairs where both clips are avd are NOT used for grouping (reason avd_same_room): "
          "AIRTLab is a single-room staged dataset, so different staged events look alike to pHash; "
          "the publisher scene ids (source_id) are authoritative.", "",
          "| dataset pair | confirmed (<=6) | borderline (7-10) |", "|---|---|---|"]
    for name, mask in [("avd-avd", key == "avd-avd"), ("rwf-rwf", key == "rwf-rwf"),
                       ("rwf-sfd", key == "rwf-sfd"), ("sfd-sfd", key == "sfd-sfd"),
                       ("any avd-cross (avd with rwf/sfd)", anyavd & (key != "avd-avd"))]:
        L.append(f"| {name} | {int((mask & pairs.confirmed).sum())} | {int((mask & ~pairs.confirmed).sum())} |")
    rs = pairs[(key == "rwf-sfd") & pairs.confirmed]
    rs = rs[rs.clip_a.map(info2.source_id) != rs.clip_b.map(info2.source_id)]
    L += ["", "## RWF-SFD confirmed pairs with different source_ids (cross-source re-uploads)", "",
          f"Count: {len(rs)}", "",
          "| clip_a | a (dataset/split/label/source) | clip_b | b (dataset/split/label/source) | distance |",
          "|---|---|---|---|---|"]
    for p in rs.itertuples():
        a, b = info2.loc[p.clip_a], info2.loc[p.clip_b]
        L.append(f"| {p.clip_a[:8]} | {a.dataset}/{a.original_split}/{a.label}/{a.source_id} | {p.clip_b[:8]} | "
                 f"{b.dataset}/{b.original_split}/{b.label}/{b.source_id} | {p.distance:.2f} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return dict(groups=ng, multi_ds=multi_ds, multi_split=multi_sp, shared_ids=len(shared),
                rwf_shared_clips=n_rwf, pairs=len(rows), near=n_dup, border=n_bd, source_only=n_src,
                cross_source_pairs=len(pairs), cross_source_confirmed=int(pairs.confirmed.sum()))


def run_groups(manifest_path=None, root=ROOT) -> dict:
    mp = Path(manifest_path) if manifest_path else WORK / "manifest" / "all_clips.csv"
    out = mp.parent
    m = pd.read_csv(mp, keep_default_na=False, dtype={"exclusion": str, "dup_of": str, "error": str})
    m = m.drop(columns=[c for c in ("global_group", "shared_source") if c in m])
    sigs = load_signatures(_active(m), out / "signatures.npz", root)
    pairs = near_duplicate_pairs(m, BORDERLINE, sigs=sigs)
    pairs.to_csv(out / "near_duplicates.csv", index=False)
    contact_sheet(pairs, m, out / "near_duplicates_sheet.png", root=root)
    m["global_group"] = assign_global_groups(m, pairs)
    m["shared_source"] = m.groupby("global_group").dataset.transform("nunique") > 1
    m.to_csv(mp, index=False)
    return write_groups_report(m, pairs, sigs, out / "groups_report.md")
