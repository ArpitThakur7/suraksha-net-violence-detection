"""Paper / presentation figures from work/results and the test predictions.

    python -m pipeline figures   ->  work/results/figs/*.png

fig_auc_protocols.png   test ROC-AUC with group-bootstrap 95% CI per protocol and model
fig_roc_<protocol>.png  ROC curves of every model on one protocol (pooled test set)
fig_confusion_<model>_<protocol>.png  confusion matrix at the frozen validation threshold
Colours: fixed categorical order from the validated reference palette; every model also has its
own marker shape, so identity never depends on colour alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import DEFAULT_CONFIG, load_config
from pipeline.paths import runs_root, work_dir

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# fixed slot per model (never re-assigned when a model is missing)
MODEL_STYLE = {
    "videomae": ("#2a78d6", "o", "VideoMAE-B (frozen)"),
    "r2plus1d": ("#eb6834", "s", "R(2+1)D-18 (frozen / fine-tuned)"),
    "motionlite": ("#1baf7a", "D", "MotionLite (ours, fine-tuned)"),
    "mnv3_tsm": ("#eda100", "^", "MobileNetV3-TSM (fine-tuned)"),
    "clip": ("#e87ba4", "v", "CLIP ViT-B/16 (frozen)"),
    "cnn_lstm": ("#6250d6", "P", "CNN-LSTM (fine-tuned)"),
    "mvit_s": ("#e34948", "X", "MViT-S (fine-tuned)"),
    "motionlite_kd": ("#008300", "*", "MotionLite + VideoMAE distillation (ours)"),
    "videomae_ft": ("#2a78d6", "h", "VideoMAE-B (partially fine-tuned)"),
}


def canon(model: str) -> str:
    """Run-dir variant name -> MODEL_STYLE key."""
    return "motionlite_kd" if "distill_teacher" in model else model
PROTOCOL_LABEL = {
    "p1_rwf": "RWF-2000 only", "p1_sfd": "SFD only", "p1_avd": "AIRTLab only",
    "p2_all": "All three pooled", "p3_no_rwf": "Unseen RWF-2000", "p3_no_sfd": "Unseen SFD",
    "p3_no_avd": "Unseen AIRTLab", "p4_random": "Bias: random split", "p4_grouped": "Bias: grouped split",
    "p2_all_leaky": "Bias: leaky pooled",
}


def _style(ax, plt):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _key(model: str, kind: str) -> str:
    return model if model in MODEL_STYLE else "videomae"


def fig_auc(summary: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    summary = summary.assign(model=summary.model.map(canon))
    protos = [p for p in PROTOCOL_LABEL if p in set(summary.protocol)]
    models = [m for m in MODEL_STYLE if m in set(summary.model)]
    fig, ax = plt.subplots(figsize=(8, 0.48 * len(protos) + 1.6), facecolor=SURFACE)
    _style(ax, plt)
    off = np.linspace(-0.28, 0.28, max(len(models), 1)) if len(models) > 1 else [0.0]
    for j, m in enumerate(models):
        c, mk, lab = MODEL_STYLE[m]
        for kind, alpha in (("frozen probe", 1.0), ("fine-tuned", 1.0)):
            d = summary[(summary.model == m) & (summary.kind == kind)]
            if d.empty:
                continue
            y = np.array([protos.index(p) for p in d.protocol]) + off[j]
            ax.hlines(y, d.auc_lo, d.auc_hi, color=c, linewidth=2, alpha=0.9)
            face = c if kind == "frozen probe" else SURFACE   # hollow marker = fine-tuned
            ax.scatter(d.roc_auc, y, s=64, marker=mk, facecolor=face, edgecolor=c, linewidth=2,
                       zorder=3, label=f"{lab.split(' (')[0]} ({'frozen' if kind == 'frozen probe' else 'fine-tuned'})")
    ax.set_yticks(range(len(protos)), [PROTOCOL_LABEL[p] for p in protos], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Test ROC-AUC (95% group-bootstrap CI)", color=INK2)
    ax.set_xlim(0.5, 1.01)
    ax.set_title("Violence detection across protocols (filled = frozen, hollow = fine-tuned)",
                 color=INK, loc="left", fontsize=11, pad=10)
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.4, -0.09), ncol=3, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _pred_files(cfg, protocol: str) -> dict:
    """model label -> test_preds.csv (probes, and seed 0 of each fine-tuned model)."""
    files = {}
    for f in sorted((work_dir(cfg) / "probe" / protocol).glob("*/test_preds.csv")):
        files[f"{f.parent.name}|frozen probe"] = f
    for f in sorted(runs_root(cfg).glob(f"{protocol}/*/seed0/test_preds.csv")):
        name = f.parent.parent.name
        if "__" not in name or "distill_teacher" in name:
            files[f"{canon(name)}|fine-tuned"] = f
    return files


def fig_roc(cfg, protocol: str, out: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    files = _pred_files(cfg, protocol)
    if not files:
        return False
    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    _style(ax, plt)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    for key, f in files.items():
        model, kind = key.split("|")
        c, mk, lab = MODEL_STYLE.get(model, MODEL_STYLE["videomae"])
        d = pd.read_csv(f)
        fpr, tpr, _ = roc_curve(d.label, d.prob)
        auc = roc_auc_score(d.label, d.prob)
        ax.plot(fpr, tpr, color=c, linewidth=2, linestyle="-" if kind == "frozen probe" else (0, (4, 2)),
                label=f"{lab.split(' (')[0]} ({'frozen' if kind == 'frozen probe' else 'fine-tuned'}) AUC {auc:.3f}")
    ax.set_xlabel("False positive rate", color=INK2)
    ax.set_ylabel("True positive rate", color=INK2)
    ax.set_title(f"ROC: {PROTOCOL_LABEL.get(protocol, protocol)}", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_confusion(f: Path, title: str, out: Path):
    import matplotlib.pyplot as plt
    d = pd.read_csv(f)
    cm = pd.crosstab(d.label, d.pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    fig, ax = plt.subplots(figsize=(3.6, 3.3), facecolor=SURFACE)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.25)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Non-violent", "Violent"], color=INK)
    ax.set_yticks([0, 1], ["Non-violent", "Violent"], color=INK)
    ax.set_xlabel("Predicted", color=INK2)
    ax.set_ylabel("Actual", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_tradeoff(summary: pd.DataFrame, speed: pd.DataFrame, out: Path, protocol: str = "p2_all"):
    """Pooled test ROC-AUC against GPU latency per clip (log x): the accuracy / cost trade-off."""
    import matplotlib.pyplot as plt
    d = summary[summary.protocol == protocol].assign(model=summary.model.map(canon))
    sp = speed.set_index("model")
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    _style(ax, plt)
    for r in d.itertuples():
        arch = {"motionlite_kd": "motionlite", "videomae_ft": "videomae"}.get(r.model, r.model)
        if arch not in sp.index or not np.isfinite(sp.loc[arch, "cuda_ms"]):
            continue
        c, mk, lab = MODEL_STYLE.get(r.model, MODEL_STYLE["videomae"])
        x = float(sp.loc[arch, "cuda_ms"])
        frozen = r.kind == "frozen probe"
        ax.errorbar(x, r.roc_auc, yerr=[[r.roc_auc - r.auc_lo], [r.auc_hi - r.roc_auc]], fmt="none",
                    ecolor=c, elinewidth=1.5, alpha=0.8)
        ax.scatter(x, r.roc_auc, s=90, marker=mk, facecolor=c if frozen else SURFACE, edgecolor=c,
                   linewidth=2, zorder=3)
        ax.annotate(f"{lab.split(' (')[0]}
{sp.loc[arch, 'params_m']:.1f} M params", (x, r.roc_auc),
                    xytext=(7, -4), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("GPU latency per 16-frame clip, ms (log scale, RTX 4050 laptop, fp16)", color=INK2)
    ax.set_ylabel("Test ROC-AUC, all three pooled", color=INK2)
    ax.set_title("Accuracy vs cost (filled = frozen backbone, hollow = fine-tuned)", color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_all(cfg) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    out = work_dir(cfg) / "results" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    summ = work_dir(cfg) / "results" / "summary.csv"
    made = []
    if summ.exists():
        fig_auc(pd.read_csv(summ), out / "fig_auc_protocols.png")
        made.append(out / "fig_auc_protocols.png")
    speed = work_dir(cfg) / "bench" / "speed.csv"
    if summ.exists() and speed.exists():
        fig_tradeoff(pd.read_csv(summ), pd.read_csv(speed), out / "fig_tradeoff_p2_all.png")
        made.append(out / "fig_tradeoff_p2_all.png")
    for p in ("p1_rwf", "p2_all", "p3_no_rwf", "p3_no_sfd", "p3_no_avd"):
        if fig_roc(cfg, p, out / f"fig_roc_{p}.png"):
            made.append(out / f"fig_roc_{p}.png")
    for key, f in _pred_files(cfg, "p2_all").items():
        model, kind = key.split("|")
        dst = out / f"fig_confusion_{model}_{'frozen' if kind == 'frozen probe' else 'finetuned'}_p2_all.png"
        fig_confusion(f, f"{MODEL_STYLE.get(model, ('', '', model))[2].split(' (')[0]}: all three pooled", dst)
        made.append(dst)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline figures")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    for p in make_all(load_config(a.config)):
        print(f"wrote {p}")
    return 0
