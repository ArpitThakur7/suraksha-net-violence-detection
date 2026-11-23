"""Frozen pretrained backbones + linear probes (the "foundation features" baseline).

    python -m pipeline probe extract --backbone videomae|clip|r2plus1d [--views 4]
    python -m pipeline probe train [--backbone ...] [--protocol ...]
    python -m pipeline probe table

extract: every clip of the frame cache is encoded once by a frozen pretrained model, over `views`
evenly spaced temporal windows (the same windows as `evaluate --views`), centre crop, no
augmentation -> work/features/<backbone>_v<views>.npz (clip_ids, feats [N, views, D] float16).

train: per protocol, a logistic-regression probe on standardised features. Train rows contribute
one sample per view; val/test probabilities are averaged over views. P2/P3 weight samples so every
(dataset, label) cell has equal total weight, as the trainer's dataset-balanced sampler does; other
protocols use class-balanced weights. C is chosen by validation ROC-AUC, the decision threshold by
validation F1 (stats.best_threshold, the trainer's rule), then the test split is scored once.
Results go to work/probe/<protocol>/<backbone>/ in the evaluate.py format (pooled, per dataset,
group-bootstrap CIs), so fine-tuned and frozen models share one table.

table: work/results/summary.{csv,md} with test ROC-AUC [95% CI], accuracy and F1 of every
fine-tuned run (work/runs) and probe (work/probe), seeds averaged.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.paths import ROOT, work_dir

os.environ.setdefault("HF_HOME", str(ROOT / "pretrained_weights" / "huggingface"))

from pipeline import stats  # noqa: E402
from pipeline.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pipeline.data.cache import num_frames, read_clip  # noqa: E402
from pipeline.data.dataset import sample_indices  # noqa: E402
from pipeline.data.splits import load_protocol  # noqa: E402
from pipeline.models import IMAGENET_NORM, R2PLUS1D_NORM  # noqa: E402

CLIP_NORM = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
BACKBONES = {
    # Kinetics-400 video transformer (self-supervised VideoMAE pre-training, then K400 fine-tuning)
    "videomae": {"hf": "MCG-NJU/videomae-base-finetuned-kinetics", "T": 16, "size": 224, "norm": IMAGENET_NORM},
    # image-text model, per-frame image embeddings averaged over the window
    "clip": {"hf": "openai/clip-vit-base-patch16", "T": 8, "size": 224, "norm": CLIP_NORM},
    # torchvision R(2+1)D-18, Kinetics-400 supervised, at its native 112 px
    "r2plus1d": {"hf": None, "T": 16, "size": 112, "norm": R2PLUS1D_NORM},
}
PROTOCOLS = ("p1_rwf", "p1_sfd", "p1_avd", "p2_all", "p3_no_rwf", "p3_no_sfd", "p3_no_avd",
             "p4_random", "p4_grouped", "p2_all_leaky")
C_GRID = (0.001, 0.01, 0.1, 1.0)


# ---------------------------------------------------------------- feature extraction

class ViewDataset(torch.utils.data.Dataset):
    """[views, T, 3, size, size] normalised windows of one clip (deterministic)."""

    def __init__(self, clip_ids, cfg, T, size, norm, views):
        self.ids, self.cfg, self.T, self.size, self.views = list(clip_ids), cfg, T, size, views
        self.mean = torch.tensor(norm[0]).view(1, 3, 1, 1)
        self.std = torch.tensor(norm[1]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torchvision.transforms.functional as TF
        cid = self.ids[i]
        n = num_frames(cid, self.cfg)
        idx = np.concatenate([sample_indices(n, self.T, False, None, v, self.views) for v in range(self.views)])
        x = torch.from_numpy(read_clip(cid, self.cfg, idx)).permute(0, 3, 1, 2)  # [V*T,3,H,W] uint8
        short = int(round(self.size * 256 / 224))  # same crop ratio as the 224-from-256 eval crop
        if min(x.shape[-2:]) != short:
            x = TF.resize(x, short, antialias=True)
        x = TF.center_crop(x, [self.size, self.size]).float() / 255.0
        x = (x - self.mean) / self.std
        return x.view(self.views, self.T, 3, self.size, self.size), cid


def _patch_videomae_bias(net, repo: str):
    """The VideoMAE checkpoints store attention biases as q_bias / v_bias (no key bias), which
    recent transformers versions no longer map onto query/key/value.bias, leaving them randomly
    initialised. Copy them over (key bias = 0, as in the original model)."""
    from huggingface_hub import hf_hub_download
    try:
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
    except Exception:
        sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    n = 0
    with torch.no_grad():
        for i, layer in enumerate(net.videomae.encoder.layer):
            att = layer.attention.attention
            q, v = sd.get(f"videomae.encoder.layer.{i}.attention.attention.q_bias"),                 sd.get(f"videomae.encoder.layer.{i}.attention.attention.v_bias")
            if q is None or v is None or getattr(att.query, "bias", None) is None:
                continue
            att.query.bias.copy_(q)
            att.value.bias.copy_(v)
            att.key.bias.zero_()
            n += 1
    if n and n != len(net.videomae.encoder.layer):
        raise RuntimeError(f"patched q/v bias in {n} of {len(net.videomae.encoder.layer)} layers")
    return net


def load_backbone(name: str, dev):
    """Returns encode(x [B,T,3,S,S] float) -> [B,D] float32, all weights frozen."""
    spec = BACKBONES[name]
    if name == "videomae":
        from transformers import VideoMAEForVideoClassification
        net = _patch_videomae_bias(VideoMAEForVideoClassification.from_pretrained(spec["hf"]), spec["hf"])
        net = net.to(dev).eval()

        def encode(x):
            h = net.videomae(pixel_values=x).last_hidden_state
            return net.fc_norm(h.mean(1)) if net.fc_norm is not None else h[:, 0]
    elif name == "clip":
        from transformers import CLIPVisionModelWithProjection
        net = CLIPVisionModelWithProjection.from_pretrained(spec["hf"]).to(dev).eval()

        def encode(x):
            b, t = x.shape[:2]
            e = net(pixel_values=x.flatten(0, 1)).image_embeds
            return e.view(b, t, -1).mean(1)
    elif name == "r2plus1d":
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
        net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        net.fc = torch.nn.Identity()
        net = net.to(dev).eval()

        def encode(x):
            return net(x.permute(0, 2, 1, 3, 4))  # [B,3,T,H,W]
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {', '.join(BACKBONES)}")
    for p in net.parameters():
        p.requires_grad_(False)
    return encode


def feature_path(cfg, backbone: str, views: int) -> Path:
    return work_dir(cfg) / "features" / f"{backbone}_v{views}.npz"


def clip_universe(cfg) -> list[str]:
    m = pd.read_csv(work_dir(cfg) / "manifest" / "all_clips.csv", keep_default_na=False)
    return m[m["exclusion"] == ""].drop_duplicates("clip_id")["clip_id"].tolist()


@torch.inference_mode()
def extract(cfg, backbone: str, views: int = 4, batch: int = 4, workers: int = 4,
            clip_ids=None, encode=None) -> Path:
    spec = BACKBONES[backbone]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = list(clip_ids) if clip_ids is not None else clip_universe(cfg)
    encode = encode or load_backbone(backbone, dev)
    ds = ViewDataset(ids, cfg, spec["T"], spec["size"], spec["norm"], views)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=dev.type == "cuda")
    feats, order, t0 = [], [], time.time()
    for k, (x, cid) in enumerate(dl):
        b = x.shape[0]
        x = x.to(dev, non_blocking=True).flatten(0, 1)  # [B*V,T,3,S,S]
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            f = encode(x)
        feats.append(f.float().view(b, views, -1).cpu().numpy().astype(np.float16))
        order += list(cid)
        if k % 50 == 0:
            print(f"[probe:{backbone}] {len(order)}/{len(ids)} clips ({time.time() - t0:.0f}s)", flush=True)
    out = feature_path(cfg, backbone, views)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, clip_ids=np.array(order), feats=np.concatenate(feats))
    tmp.replace(out)
    print(f"[probe:{backbone}] wrote {out} {np.concatenate(feats).shape} in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------- probes

def _weights(df: pd.DataFrame, by_dataset: bool) -> np.ndarray:
    keys = list(zip(df["dataset"], df["label"])) if by_dataset else list(df["label"])
    s = pd.Series(keys)
    w = 1.0 / s.map(s.value_counts()).to_numpy(float)
    return w * len(w) / w.sum()


def _rows(df: pd.DataFrame, index: dict, feats: np.ndarray) -> np.ndarray:
    missing = [c for c in df["clip_id"] if c not in index]
    if missing:
        raise KeyError(f"{len(missing)} clips have no features, e.g. {missing[:3]}")
    return feats[[index[c] for c in df["clip_id"]]].astype(np.float32)  # [n, V, D]


def fit_probe(Xtr, ytr, wtr, Xva, yva, c_grid=C_GRID):
    """Xtr [n,V,D] (each view a sample), Xva [m,V,D] (views averaged in probability space).
    Returns (predict(X)->p, C, val_auc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    V = Xtr.shape[1]
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    A = sc.transform(Xtr.reshape(-1, Xtr.shape[-1]))
    y, w = np.repeat(ytr, V), np.repeat(wtr, V)

    def make_pred(clf):
        return lambda X: clf.predict_proba(sc.transform(X.reshape(-1, X.shape[-1])))[:, 1] \
            .reshape(X.shape[0], X.shape[1]).mean(1)

    best = None
    for C in c_grid:
        clf = LogisticRegression(C=C, max_iter=3000).fit(A, y, sample_weight=w)
        pred = make_pred(clf)
        auc = stats.clip_metrics(yva, pred(Xva), 0.5).get("roc_auc", float("nan"))
        auc = -1.0 if auc is None or not np.isfinite(auc) else auc
        if best is None or auc > best[2]:
            best = (pred, C, auc)
    return best


def run_probe(cfg, backbone: str, protocol: str, views: int = 4) -> dict:
    from pipeline.evaluate import section, _clean
    z = np.load(feature_path(cfg, backbone, views))
    index = {c: i for i, c in enumerate(z["clip_ids"].tolist())}
    feats = z["feats"]
    m = load_protocol(protocol, cfg)
    tr, va, te = (m[m["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    by_ds = protocol.startswith(("p2", "p3"))
    pred, C, val_auc = fit_probe(_rows(tr, index, feats), tr["label"].to_numpy(int), _weights(tr, by_ds),
                                 _rows(va, index, feats), va["label"].to_numpy(int))
    tc = cfg["train"]["threshold_grid"]
    grid = np.round(np.arange(tc["start"], tc["stop"] + 1e-9, tc["step"]), 6)
    pva = pred(_rows(va, index, feats))
    thr = stats.best_threshold(va["label"], pva, grid)
    p = pred(_rows(te, index, feats))   # the only place test rows are scored
    df = pd.DataFrame({"clip_id": te["clip_id"], "dataset": te["dataset"], "label": te["label"].astype(int),
                       "prob": p, "pred": (p >= thr).astype(int), "global_group": te["global_group"],
                       "shared_source": te["shared_source"].astype(str).str.lower().isin(("true", "1"))})
    sec = lambda d: section(d["label"], d["prob"], d["global_group"], thr, cfg)  # noqa: E731
    res = {"protocol": protocol, "model": f"probe-{backbone}", "seed": 0, "views": views, "threshold": thr,
           "C": C, "val_auc": val_auc, "n_train": int(len(tr)), "pooled": sec(df),
           "by_dataset": {str(k): sec(g) for k, g in df.groupby("dataset")}}
    out = work_dir(cfg) / "probe" / protocol / backbone
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "test_preds.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------- results table

def _fmt(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def collect(cfg) -> pd.DataFrame:
    from pipeline.paths import runs_root
    rows = []
    files = list(runs_root(cfg).glob("*/*/seed*/test_metrics.json")) + \
        list((work_dir(cfg) / "probe").glob("*/*/test_metrics.json"))
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        kind = "frozen probe" if str(r["model"]).startswith("probe-") else "fine-tuned"
        model = f.parent.name if kind == "frozen probe" else f.parent.parent.name
        a, ci = r["pooled"]["at_val_thr"], r["pooled"]["ci"]
        rows.append({"protocol": r["protocol"], "model": model, "kind": kind, "seed": r["seed"],
                     "roc_auc": a.get("roc_auc"), "auc_lo": ci["roc_auc"][0], "auc_hi": ci["roc_auc"][1],
                     "accuracy": a.get("accuracy"), "f1": a.get("f1"), "n_test": r["pooled"]["n"]})
    return pd.DataFrame(rows)


def table(cfg) -> pd.DataFrame:
    d = collect(cfg)
    if d.empty:
        print("no results yet")
        return d
    g = d.groupby(["protocol", "kind", "model"]).agg(
        seeds=("seed", "nunique"), roc_auc=("roc_auc", "mean"), auc_sd=("roc_auc", "std"),
        auc_lo=("auc_lo", "mean"), auc_hi=("auc_hi", "mean"), accuracy=("accuracy", "mean"),
        f1=("f1", "mean"), n_test=("n_test", "first")).reset_index()
    order = {p: i for i, p in enumerate(PROTOCOLS)}
    g = g.sort_values(["protocol", "kind", "roc_auc"], key=lambda s: s.map(order) if s.name == "protocol" else s,
                      ascending=[True, True, False])
    out = work_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "summary.csv", index=False)
    lines = ["# Test results (frozen validation threshold)", "",
             "ROC-AUC with group-bootstrap 95% CI (mean CI bounds over seeds when seeds > 1).", ""]
    for p, gp in g.groupby("protocol", sort=False):
        lines += [f"## {p} (n_test = {int(gp.n_test.iloc[0])})", "",
                  "| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |", "|---|---|---:|---|---:|---:|"]
        for r in gp.itertuples():
            sd = f" ± {r.auc_sd:.3f}" if r.seeds > 1 and np.isfinite(r.auc_sd) else ""
            lines.append(f"| {r.model} | {r.kind} | {r.seeds} | {_fmt(r.roc_auc)}{sd} "
                         f"[{_fmt(r.auc_lo)}, {_fmt(r.auc_hi)}] | {_fmt(r.accuracy)} | {_fmt(r.f1)} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline probe")
    ap.add_argument("action", choices=("extract", "train", "table"))
    ap.add_argument("--backbone", action="append", default=None, choices=tuple(BACKBONES))
    ap.add_argument("--protocol", action="append", default=None, choices=PROTOCOLS)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="extract: first N clips only (smoke test)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    backbones = a.backbone or list(BACKBONES)
    if a.action == "extract":
        for b in backbones:
            ids = clip_universe(cfg)[: a.limit] if a.limit else None
            extract(cfg, b, a.views, a.batch, a.workers, clip_ids=ids)
    elif a.action == "train":
        for b in backbones:
            if not feature_path(cfg, b, a.views).exists():
                print(f"skip {b}: no features (run `probe extract --backbone {b}`)")
                continue
            for p in a.protocol or PROTOCOLS:
                r = run_probe(cfg, b, p, a.views)
                s = r["pooled"]["at_val_thr"]
                print(f"[probe:{b}] {p:12s} C={r['C']:<6} val_auc={r['val_auc']:.3f}  test auc={_fmt(s.get('roc_auc'))} "
                      f"acc={_fmt(s.get('accuracy'))} f1={_fmt(s.get('f1'))}", flush=True)
    else:
        table(cfg)
    return 0
