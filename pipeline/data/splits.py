"""Base splits, protocol manifests, leak check and split lock (Stage 2).

Precedence inside a protocol is test > val > train. A train/val row whose global_group also
appears in a higher-precedence split is dropped (logged in ``m.attrs["dropped"]``). Test rows
are never dropped. clip_id is unique per protocol manifest.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.paths import work_dir

SPLITS = ("train", "val", "test")
COLS = ["clip_id", "path", "dataset", "label", "global_group", "shared_source", "split"]
ALL = ("rwf", "sfd", "avd")

# name -> (train_sets, test_sets); p4_* use RWF original train only (see _p4_manifest)
PROTOCOLS = {
    "p1_rwf": (("rwf",), ("rwf",)),
    "p1_sfd": (("sfd",), ("sfd",)),
    "p1_avd": (("avd",), ("avd",)),
    "p2_all": (ALL, ALL),
    "p3_no_rwf": (("sfd", "avd"), ("rwf",)),
    "p3_no_sfd": (("rwf", "avd"), ("sfd",)),
    "p3_no_avd": (("rwf", "sfd"), ("avd",)),
    "p4_grouped": (("rwf",), ("rwf",)),
    "p4_random": (("rwf",), ("rwf",)),
    "p2_all_leaky": (ALL, ALL),
}
LEAKY_PROTOCOLS = frozenset({"p4_random", "p2_all_leaky"})
RWF_TEST_PROTOCOLS = ("p1_rwf", "p2_all", "p3_no_rwf", "p2_all_leaky")


class LeakError(Exception):
    pass


class LockError(Exception):
    pass


def _work(cfg) -> Path:
    return work_dir(cfg)


def _splits_dir(cfg) -> Path:
    return _work(cfg) / "splits"


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("true", "1"))


def _stratified_group_split(df: pd.DataFrame, fracs, seed: int, ineligible=frozenset()) -> dict:
    """Assign each global_group to a part index (0 = train, 1.. = fracs[1:]).

    Stratified by group majority label. Part sizes come from all groups of that label;
    ineligible groups always stay in part 0.
    """
    g = df.groupby("global_group").label.mean()
    maj = (g >= 0.5).astype(int)
    rng = np.random.RandomState(seed)
    out = {grp: 0 for grp in g.index}
    for lab in sorted(maj.unique()):
        groups = sorted(maj.index[maj == lab])
        n = len(groups)
        counts = [round(f * n) for f in fracs[1:]]
        elig = [x for x in groups if x not in ineligible]
        perm = rng.permutation(len(elig))
        pos = 0
        for part, c in enumerate(counts, start=1):
            for j in perm[pos:pos + c]:
                out[elig[j]] = part
            pos += c
    return out


def base_splits(manifest: pd.DataFrame, cfg) -> pd.DataFrame:
    m = manifest[manifest["exclusion"].astype(str) == ""].copy()
    m["shared_source"] = _to_bool(m["shared_source"])
    m["label"] = m["label"].astype(int)
    split = pd.Series("", index=m.index, dtype=object)
    rwf = m.dataset == "rwf"
    split[rwf & (m.original_split == "val")] = "test"
    split[~rwf] = m.original_split[~rwf].replace({"validation": "val"})
    tr = m[rwf & (m.original_split == "train")]
    test_groups = set(m[rwf & (m.original_split == "val")].global_group)
    vf = cfg["split"]["val_frac"]
    assign = _stratified_group_split(tr, (1 - vf, vf), cfg["split"]["seed"], ineligible=test_groups)
    split[tr.index] = tr.global_group.map(assign).map({0: "train", 1: "val"})
    m["split"] = split
    bad = set(m.split) - set(SPLITS)
    assert not bad, f"unassigned/unknown splits: {bad}"
    return m.reset_index(drop=True)


def _p4_manifest(base: pd.DataFrame, name: str, cfg) -> pd.DataFrame:
    r = base[(base.dataset == "rwf") & (base.original_split == "train")].copy()
    fr = list(cfg["split"]["p4_fracs"])
    seed = cfg["split"]["seed"]
    if name == "p4_random":
        rng = np.random.RandomState(seed)
        r["split"] = ""
        for lab in (0, 1):
            idx = r.index[r.label == lab].to_numpy()
            idx = idx[rng.permutation(len(idx))]
            n = len(idx)
            nt, nv = round(fr[2] * n), round(fr[1] * n)
            r.loc[idx[:nt], "split"] = "test"
            r.loc[idx[nt:nt + nv], "split"] = "val"
            r.loc[idx[nt + nv:], "split"] = "train"
    else:
        a = _stratified_group_split(r, fr, seed)  # 0 train, 1 val, 2 test
        r["split"] = r.global_group.map(a).map(dict(enumerate(SPLITS)))
    return r


def _dedupe_precedence(m: pd.DataFrame, drop_shared: bool, name: str) -> pd.DataFrame:
    """Drop lower-precedence rows that duplicate a higher split's clip_id (always) or global_group
    (when drop_shared). Log counts per dataset and reason."""
    dropped, keep = [], pd.Series(True, index=m.index)
    for s, hi in (("val", ("test",)), ("train", ("val", "test"))):
        rows = m[m.split == s]
        reasons = [("identical_clip", rows.clip_id.isin(set(m[m.split.isin(hi)].clip_id)))]
        if drop_shared:
            for h in hi:
                reasons.append((f"group_in_{h}", rows.global_group.isin(set(m[m.split == h].global_group))))
        done = pd.Series(False, index=rows.index)
        for reason, mask in reasons:
            mask = mask & ~done
            for ds, n in rows[mask].groupby("dataset").size().items():
                dropped.append(dict(protocol=name, dataset=ds, split=s, reason=reason, n=int(n)))
            keep[rows.index[mask]] = False
            done |= mask
    out = m[keep].copy()
    out.attrs["dropped"] = dropped
    return out


def protocol_manifest(base: pd.DataFrame, name: str, drop_shared: bool = True, cfg=None) -> pd.DataFrame:
    if name not in PROTOCOLS:
        raise KeyError(name)
    if cfg is None:
        from pipeline.config import load_config
        cfg = load_config()
    if name.startswith("p4_"):
        m = _p4_manifest(base, name, cfg)
        dropped = []
    else:
        train_sets, test_sets = PROTOCOLS[name]
        tr = base[base.dataset.isin(train_sets) & base.split.isin(("train", "val"))]
        te = base[base.dataset.isin(test_sets) & (base.split == "test")]
        if name in LEAKY_PROTOCOLS:
            drop_shared = False
        m = _dedupe_precedence(pd.concat([tr, te]), drop_shared, name)
        dropped = m.attrs["dropped"]
    out = m[COLS].copy()
    out["_o"] = out.split.map({s: i for i, s in enumerate(SPLITS)})
    out = out.sort_values(["_o", "dataset", "clip_id"]).drop(columns="_o").reset_index(drop=True)
    out.attrs["dropped"] = dropped
    assert out.clip_id.is_unique, f"{name}: clip_id not unique"
    return out


def check_no_leak(m: pd.DataFrame) -> None:
    n = m.groupby("global_group").split.nunique()
    bad = n[n > 1]
    if len(bad):
        raise LeakError(f"{len(bad)} global_group(s) in more than one split, e.g. {list(bad.index[:3])}")


def _hash(path: Path) -> str:
    # normalise CRLF so the lock survives git autocrlf checkouts
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_base(cfg) -> pd.DataFrame:
    p = _work(cfg) / "manifest" / "all_clips.csv"
    m = pd.read_csv(p, keep_default_na=False, dtype={"exclusion": str, "dup_of": str, "error": str})
    return base_splits(m, cfg)


def write_protocols(cfg, base: pd.DataFrame = None, force: bool = False) -> dict:
    d = _splits_dir(cfg)
    lock = d / "LOCK.json"
    if lock.exists() and not force:
        raise LockError(f"{lock} exists; pass --force to overwrite")
    if base is None:
        base = load_base(cfg)
    d.mkdir(parents=True, exist_ok=True)
    drop = bool(cfg["split"].get("drop_shared", True))
    hashes, manifests = {}, {}
    for name in PROTOCOLS:
        m = protocol_manifest(base, name, drop_shared=drop, cfg=cfg)
        if name not in LEAKY_PROTOCOLS:
            check_no_leak(m)
        p = d / f"{name}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            m.to_csv(f, index=False, lineterminator="\n")
        hashes[name] = _hash(p)
        manifests[name] = m
    lock.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(manifests, d / "summary.md")
    return hashes


def load_protocol(name: str, cfg, verify: bool = True) -> pd.DataFrame:
    d = _splits_dir(cfg)
    p = d / f"{name}.csv"
    if verify:
        lock = d / "LOCK.json"
        if not lock.exists():
            raise LockError("no LOCK.json; run `python -m pipeline split`")
        want = json.loads(lock.read_text(encoding="utf-8")).get(name)
        if want is None or not p.exists() or _hash(p) != want:
            raise LockError(f"{name}: split file does not match LOCK.json")
    m = pd.read_csv(p, keep_default_na=False, dtype={"clip_id": str})
    m["shared_source"] = _to_bool(m["shared_source"])
    m["label"] = m["label"].astype(int)
    return m


def write_summary(manifests: dict, path: Path) -> None:
    L = ["# Split summary", "", "## Counts per protocol x split x dataset x label", "",
         "| protocol | split | dataset | label 0 | label 1 | total |", "|---|---|---|---|---|---|"]
    for name, m in manifests.items():
        for s in SPLITS:
            for ds in ALL:
                x = m[(m.split == s) & (m.dataset == ds)]
                if len(x):
                    n0, n1 = int((x.label == 0).sum()), int((x.label == 1).sum())
                    L.append(f"| {name} | {s} | {ds} | {n0} | {n1} | {len(x)} |")
        L.append(f"| {name} | **all** | all | {int((m.label == 0).sum())} | {int((m.label == 1).sum())} | {len(m)} |")
    L += ["", "## Dropped by precedence rule", "",
          "| protocol | dataset | split dropped | reason | n |", "|---|---|---|---|---|"]
    tot = 0
    for m in manifests.values():
        for r in m.attrs.get("dropped", []):
            L.append(f"| {r['protocol']} | {r['dataset']} | {r['split']} | {r['reason']} | {r['n']} |")
            tot += r["n"]
    L += ["", f"Total dropped rows: {tot}", "", "## Test clips with shared_source True", "",
          "| protocol | test clips | shared_source True |", "|---|---|---|"]
    for name, m in manifests.items():
        t = m[m.split == "test"]
        L.append(f"| {name} | {len(t)} | {int(t.shared_source.sum())} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
