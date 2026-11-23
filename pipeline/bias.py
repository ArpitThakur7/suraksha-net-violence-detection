"""Stage 7: data-leakage bias study.

`python -m pipeline bias [--model mnv3_tsm] [--seeds 0 1 2]`

Uses the protocols locked by `split` (never rebuilds them):
  (a) p4_random (clip-level random split, groups ignored) vs p4_grouped (grouped by global_group):
      test accuracy at the frozen validation threshold; gap = random - grouped.
  (b) every existing run's test predictions: metrics at the threshold tuned on test vs the frozen
      validation threshold. This deliberately re-tunes on test to measure the size of that leak.
  (c) p2_all_leaky (cross-dataset drop rule disabled): accuracy on shared_source test clips vs the
      other test clips of the same runs.

Confidence intervals are percentile (95%) group bootstraps over global_group:
  (a) the two protocols have different test sets, so clips cannot be paired; each bootstrap
      replicate resamples groups once per protocol, applies that resample to every seed of the
      protocol (seeds share one test set), averages accuracy over seeds and subtracts.
  (c) both accuracies come from the same predictions, so each replicate resamples groups once
      and computes both accuracies on that same resample (a paired difference).
Outputs: work/reports/bias.csv (long format: part, label, metric, value, ci_lo, ci_hi) and bias.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import stats
from pipeline.config import DEFAULT_CONFIG, load_config
from pipeline.evaluate import _work, evaluate_run
from pipeline.train import run_dir, train_run

PROTOCOLS = ("p4_random", "p4_grouped", "p2_all_leaky")
NAN3 = (float("nan"),) * 3


def _correct(df: pd.DataFrame) -> pd.Series:
    return (df["pred"] == df["label"]).astype(float)


def _group_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per global_group (correct, total), indexed by group."""
    return _correct(df).groupby(df["global_group"]).agg(correct="sum", total="count")


def _stack(runs):
    """[seeds, groups] correct/total matrices; seeds of a protocol share one test set, so the group
    axis is common (a group missing from a run counts 0)."""
    cs = [_group_counts(d) for d in runs]
    groups = sorted(set().union(*(c.index for c in cs)))
    cs = [c.reindex(groups, fill_value=0.0) for c in cs]
    return (np.stack([c["correct"].to_numpy(float) for c in cs]),
            np.stack([c["total"].to_numpy(float) for c in cs]))


def _ci(vals, alpha):
    vals = np.asarray([v for v in vals if not math.isnan(v)])
    if not len(vals):
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def accuracy_gap_ci(a_runs, b_runs, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean accuracy over the runs in a minus in b: (point estimate, lo, hi). Group bootstrap: the
    two protocols are resampled independently (different test sets), but within a protocol one group
    resample per replicate is applied to every seed (all seeds are scored on the same test set)."""
    (ca, ta), (cb, tb) = _stack(a_runs), _stack(b_runs)
    mean_acc = lambda c, t, i: float(np.mean(c[:, i].sum(1) / t[:, i].sum(1)))  # noqa: E731
    all_a, all_b = np.arange(ca.shape[1]), np.arange(cb.shape[1])
    gap = mean_acc(ca, ta, all_a) - mean_acc(cb, tb, all_b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ia, ib = rng.integers(0, len(all_a), len(all_a)), rng.integers(0, len(all_b), len(all_b))
        vals.append(mean_acc(ca, ta, ia) - mean_acc(cb, tb, ib))
    lo, hi = _ci(vals, alpha)
    return gap, lo, hi


def shared_source_gap(df: pd.DataFrame, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Accuracy on shared_source clips minus accuracy on the others, on the same predictions
    (paired group bootstrap). nan when there are no clips of one kind."""
    sh = df["shared_source"].astype(bool)
    if sh.all() or not sh.any():
        return NAN3
    ok = _correct(df)
    groups = df["global_group"]
    per = pd.DataFrame({
        "cs": ok.where(sh, 0.0).groupby(groups).sum(), "ts": sh.astype(float).groupby(groups).sum(),
        "co": ok.where(~sh, 0.0).groupby(groups).sum(), "to": (~sh).astype(float).groupby(groups).sum()})
    cs, ts, co, to = (per[k].to_numpy() for k in ("cs", "ts", "co", "to"))
    gap = cs.sum() / ts.sum() - co.sum() / to.sum()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(per), len(per))
        if ts[i].sum() and to[i].sum():
            vals.append(cs[i].sum() / ts[i].sum() - co[i].sum() / to[i].sum())
    lo, hi = _ci(vals, alpha)
    return float(gap), lo, hi


def threshold_leak_rows(runs_root: Path) -> list[dict]:
    """Part (b): per run with test predictions, metrics at the frozen validation threshold vs at the
    threshold that maximises F1 on the test predictions themselves."""
    rows = []
    for f in sorted(Path(runs_root).glob("**/test_preds.csv")):
        d = f.parent
        if not (d / "test_metrics.json").exists():
            continue
        df = pd.read_csv(f)
        thr_val = float(json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))["threshold"])
        thr_test = stats.best_threshold(df["label"], df["prob"])
        mv = stats.clip_metrics(df["label"], df["prob"], thr_val)
        mt = stats.clip_metrics(df["label"], df["prob"], thr_test)
        rows.append({"run": d.relative_to(runs_root).as_posix(), "thr_val": thr_val, "thr_test": thr_test,
                     "acc_val_thr": mv["accuracy"], "acc_test_thr": mt["accuracy"],
                     "f1_val_thr": mv["f1"], "f1_test_thr": mt["f1"],
                     "acc_inflation": mt["accuracy"] - mv["accuracy"],
                     "f1_inflation": mt["f1"] - mv["f1"]})
    return rows


def _fmt(x, d=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _gap_md(title, g):
    if g is None:
        return [f"### {title}", "", "No runs available yet.", ""]
    return [f"### {title}", "", f"- {g['left_name']}: mean accuracy {_fmt(g['left_acc'])}",
            f"- {g['right_name']}: mean accuracy {_fmt(g['right_acc'])}",
            f"- gap ({g['left_name']} - {g['right_name']}): {_fmt(g['gap'])} "
            f"(95% CI {_fmt(g['lo'])} to {_fmt(g['hi'])}); n = {g['n']}", ""]


def render_md(res: dict, model: str, seeds) -> str:
    L = ["# Data-leakage bias study", "",
         f"Model: `{model}`, seeds: {list(seeds)}. Accuracy is at the frozen validation threshold on the "
         "test split. CIs: 95% percentile group bootstrap over `global_group`.", "",
         "Method: (a) the two protocols have different test sets, so replicates resample groups "
         "independently per protocol, apply the same group resample to every seed of that protocol "
         "(seeds share one test set), average accuracy over seeds and subtract. "
         "(c) both accuracies come from the same predictions, so each replicate resamples groups once "
         "and computes both accuracies on it (paired).", "",
         "## (a) Random vs grouped split", ""]
    L += _gap_md("p4_random - p4_grouped", res["a"])
    L += ["## (b) Threshold tuned on test vs validation threshold", ""]
    if not res["leak"]:
        L += ["No runs with test predictions yet.", ""]
    else:
        L += ["| run | thr_val | thr_test | acc@val | acc@test | f1@val | f1@test | f1 inflation |",
              "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['run']} | {r['thr_val']:.2f} | {r['thr_test']:.2f} | {_fmt(r['acc_val_thr'])} | "
              f"{_fmt(r['acc_test_thr'])} | {_fmt(r['f1_val_thr'])} | {_fmt(r['f1_test_thr'])} | "
              f"{_fmt(r['f1_inflation'])} |" for r in res["leak"]]
        L += ["", f"Mean F1 inflation: {_fmt(float(np.mean([r['f1_inflation'] for r in res['leak']])))}", ""]
    L += ["## (c) Cross-dataset leakage (p2_all_leaky)", ""]
    L += _gap_md("shared_source test clips - other test clips", res["c"])
    return "\n".join(L) + "\n"


def _acc(df) -> float:
    return float(_correct(df).mean()) if len(df) else float("nan")


def run_bias(cfg, model: str = "mnv3_tsm", seeds=(0, 1, 2)) -> dict:
    seeds = tuple(seeds)
    n = int(cfg["eval"]["bootstrap_resamples"])
    preds = {p: [] for p in PROTOCOLS}
    for p in PROTOCOLS:
        for s in seeds:
            out = run_dir(cfg, p, model, s)
            train_run(cfg, p, model, s, out)  # DONE marker makes reruns free
            if not (out / "test_metrics.json").exists():
                evaluate_run(out, cfg)
            preds[p].append(pd.read_csv(out / "test_preds.csv"))

    ra, rg = preds["p4_random"], preds["p4_grouped"]
    gap, lo, hi = accuracy_gap_ci(ra, rg, n=n)
    a = {"left_name": "p4_random", "right_name": "p4_grouped",
         "left_acc": float(np.mean([_acc(d) for d in ra])), "right_acc": float(np.mean([_acc(d) for d in rg])),
         "gap": gap, "lo": lo, "hi": hi, "n": f"{len(ra[0])}/{len(rg[0])} test clips per seed"}
    leaky = pd.concat(preds["p2_all_leaky"], ignore_index=True)
    sh = leaky["shared_source"].astype(bool)
    cgap, clo, chi = shared_source_gap(leaky, n=n)
    c = {"left_name": "shared_source clips", "right_name": "other clips",
         "left_acc": _acc(leaky[sh]), "right_acc": _acc(leaky[~sh]), "gap": cgap, "lo": clo, "hi": chi,
         "n": f"{int(sh.sum())}/{int((~sh).sum())} clips pooled over seeds"}
    res = {"a": a, "c": c, "leak": threshold_leak_rows(_work(cfg) / "runs")}

    rows = []
    for part, g in (("a", a), ("c", c)):
        rows.append({"part": part, "label": g["left_name"], "metric": "accuracy", "value": g["left_acc"]})
        rows.append({"part": part, "label": g["right_name"], "metric": "accuracy", "value": g["right_acc"]})
        rows.append({"part": part, "label": f"{g['left_name']} - {g['right_name']}", "metric": "accuracy_gap",
                     "value": g["gap"], "ci_lo": g["lo"], "ci_hi": g["hi"]})
    for r in res["leak"]:
        rows += [{"part": "b", "label": r["run"], "metric": k, "value": v}
                 for k, v in r.items() if k != "run"]
    rep = _work(cfg) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["part", "label", "metric", "value", "ci_lo", "ci_hi"]).to_csv(
        rep / "bias.csv", index=False)
    (rep / "bias.md").write_text(render_md(res, model, seeds), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline bias")
    ap.add_argument("--model", default="mnv3_tsm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    run_bias(load_config(a.config, a.overrides), a.model, a.seeds)
    print("wrote bias.csv and bias.md")
    return 0
