"""Stage 7: test-set evaluation of a finished run.

`python -m pipeline evaluate --run <run_dir> [--views 3] | --all`

Loads best.pt (weights, frozen validation threshold, norm, T, crop, model kwargs), scores the test
split of the run's protocol (the only place `allow_test=True` is passed) and writes
`test_preds{_v3}.csv` and `test_metrics{_v3}.json` into the run directory. Metrics are reported
pooled, per dataset and per shared_source value, each at the frozen validation threshold and at
0.5, with group-bootstrap 95% CIs (groups = global_group) for accuracy, F1 and ROC-AUC.

views=3 averages P(violent) over 3 evenly spaced deterministic temporal windows of the clip.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pipeline import stats
from pipeline.config import DEFAULT_CONFIG, load_config
from pipeline.data.dataset import make_loader
from pipeline.data.splits import load_protocol
from pipeline.models import build_model
from pipeline.paths import runs_root
from pipeline.train import _atomic, _evaluate

CI_METRICS = ("accuracy", "f1", "roc_auc")


def _clean(o):
    """nan/inf -> None so the JSON is strict; numpy scalars -> python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _acc(thr):
    return lambda y, p: float(np.mean((p >= thr) == (y == 1)))


def _f1(thr):
    def f(y, p):
        pred = p >= thr
        tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
        return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
    return f


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def section(y, p, groups, thr: float, cfg) -> dict:
    """Metrics at the frozen threshold and at 0.5, plus bootstrap CIs (at the frozen threshold)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    ev = cfg["eval"]
    ece_bins = int(ev["ece_bins"])
    n = int(ev["bootstrap_resamples"])
    metric = {"accuracy": _acc(thr), "f1": _f1(thr), "roc_auc": _auc}
    ci = {k: list(stats.group_bootstrap_ci(y, p, groups, metric[k], n=n)) for k in CI_METRICS}
    at = lambda t: {**stats.clip_metrics(y, p, t), "ece": stats.ece(y, p, ece_bins)}  # noqa: E731
    return _clean({"n": int(len(y)), "at_val_thr": at(thr), "at_0.5": at(0.5), "ci": ci})


def _score(net, m, cfg, ck, batch, dev, views):
    """Mean P(violent) over `views` temporal windows. Returns (y, p, clip_ids)."""
    ps = []
    for v in range(views):
        loader = make_loader(m, "test", cfg, ck["norm"], train=False, allow_test=True,
                             batch_size=batch, seed=0, view=v, views=views)
        _, y, p, ids = _evaluate(net, loader, dev, dev.type == "cuda")
        ps.append(p)
        del loader
    return y, np.mean(ps, axis=0), ids


def evaluate_run(run_dir: Path, cfg, views: int = 1) -> dict:
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if ck.get("threshold") is None:
        raise ValueError(f"{run_dir}: best.pt has no frozen threshold (run not finished)")
    thr = float(ck["threshold"])
    rc = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["clip"]["T"], cfg["clip"]["crop"] = int(ck["T"]), int(ck["crop"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = build_model(ck["model_name"], pretrained=False, T=int(ck["T"]), **ck["model_kw"]).to(dev)
    net.load_state_dict(ck["model"])
    m = load_protocol(rc["protocol"], cfg)
    m = m[m["split"] == "test"].reset_index(drop=True)
    y, p, ids = _score(net, m, cfg, ck, int(rc["batch"]), dev, views)

    meta = m.set_index("clip_id").loc[ids]
    df = pd.DataFrame({"clip_id": ids, "dataset": meta["dataset"].tolist(), "label": y.astype(int),
                       "prob": p, "pred": (p >= thr).astype(int),
                       "global_group": meta["global_group"].tolist(),
                       "shared_source": meta["shared_source"].astype(bool).tolist()})
    sec = lambda d: section(d["label"], d["prob"], d["global_group"], thr, cfg)  # noqa: E731
    res = {"protocol": rc["protocol"], "model": ck["model_name"], "seed": rc["seed"], "views": views,
           "threshold": thr, "pooled": sec(df),
           "by_dataset": {str(k): sec(g) for k, g in df.groupby("dataset")},
           "by_shared_source": {("true" if k else "false"): sec(g) for k, g in df.groupby("shared_source")}}
    sfx = "" if views == 1 else f"_v{views}"
    _atomic(run_dir / f"test_preds{sfx}.csv", lambda t: df.to_csv(t, index=False))
    _atomic(run_dir / f"test_metrics{sfx}.json",
            lambda t: t.write_text(json.dumps(res, indent=2), encoding="utf-8"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline evaluate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="run directory")
    g.add_argument("--all", action="store_true", help="every finished run under paths.runs")
    ap.add_argument("--views", type=int, default=1, choices=(1, 3))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    sfx = "" if a.views == 1 else f"_v{a.views}"
    if a.run:
        res = evaluate_run(a.run, cfg, a.views)
        print(json.dumps({"pooled": res["pooled"]["at_val_thr"], "ci": res["pooled"]["ci"]}, indent=2))
        return 0
    for done in sorted(runs_root(cfg).glob("**/DONE")):
        d = done.parent
        if (d / f"test_metrics{sfx}.json").exists():
            print(f"skip {d} (already evaluated)")
            continue
        evaluate_run(d, cfg, a.views)
        print(f"evaluated {d}", flush=True)
    return 0
