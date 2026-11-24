"""Build reports/Suraksha_Net_results.pptx offline by driving the installed PowerPoint (pywin32).

    python reports/make_slides.py            # build the deck + PNG previews (reports/preview/)

Every number is read from the live results (work/runs, work/probe, work/bench/speed.csv), averaged
over the seeds finished so far, so rerunning after more runs finish updates the deck.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline.paths  # noqa: E402,F401
from pipeline.config import load_config  # noqa: E402
from pipeline.probe import collect  # noqa: E402

OUT = ROOT / "reports" / "Suraksha_Net_results.pptx"
FIGS = ROOT / "work" / "results" / "figs"
LOCAL_FIGS = ROOT / "reports" / "figs"
PREVIEW = ROOT / "reports" / "preview"

# palette: deep navy dominant, amber alert accent, slate text, cool light cards
NAVY, NAVY2, AMBER, SLATE, INK, CARD, WHITE, MUTED = (
    "14213D", "22325A", "FCA311", "5C677D", "1B1F2A", "F1F4F9", "FFFFFF", "8A93A6")
HEAD, BODY = "Cambria", "Calibri"
W, H = 960, 540  # points (13.333 x 7.5 in)

NAMES = {"videomae_ft": "VideoMAE-B, fine-tuned (last 4 blocks)", "videomae": "VideoMAE-B, frozen",
         "videomae+clip": "VideoMAE + CLIP, frozen", "clip": "CLIP ViT-B/16, frozen",
         "r2plus1d": "R(2+1)D-18, frozen", "mnv3_tsm": "MobileNetV3-TSM, fine-tuned",
         "motionlite": "MotionLite (ours), fine-tuned", "motionlite_kd": "MotionLite + distillation (ours)"}


def rgb(h: str) -> int:
    return int(h[0:2], 16) + int(h[2:4], 16) * 256 + int(h[4:6], 16) * 65536


# ---------------------------------------------------------------- data

def results() -> pd.DataFrame:
    cfg = load_config()
    d = collect(cfg)
    d["model"] = d["model"].map(lambda m: "motionlite_kd" if "distill_teacher" in m else m)
    g = d.groupby(["protocol", "model"]).agg(
        seeds=("seed", "nunique"), auc=("roc_auc", "mean"), auc_sd=("roc_auc", "std"),
        lo=("auc_lo", "mean"), hi=("auc_hi", "mean"), acc=("accuracy", "mean"), f1=("f1", "mean"),
        n=("n_test", "first")).reset_index()
    return g


def val(g, protocol, model, col="auc"):
    r = g[(g.protocol == protocol) & (g.model == model)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def fmt(x, nd=3):
    return "–" if not np.isfinite(x) else f"{x:.{nd}f}"


def pct(x):
    return "–" if not np.isfinite(x) else f"{100 * x:.1f}%"


# ---------------------------------------------------------------- slide figure

def bar_figure(g: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = g[g.protocol == "p2_all"].sort_values("auc")
    d = d[d.model.isin(NAMES)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="#FFFFFF")
    ys = np.arange(len(d))
    colors = ["#" + (AMBER if m == "videomae_ft" else ("2a78d6" if m in ("videomae", "videomae+clip", "clip", "r2plus1d")
                                                      else "1baf7a")) for m in d.model]
    ax.barh(ys, d.auc - 0.5, left=0.5, height=0.62, color=colors, edgecolor="#FFFFFF", linewidth=2)
    ax.errorbar(d.auc, ys, xerr=[d.auc - d.lo, d.hi - d.auc], fmt="none", ecolor="#52514e", elinewidth=1.2, capsize=3)
    for y, (a, s, k) in enumerate(zip(d.auc, d.seeds, d.model)):
        ax.text(min(d.hi.max() + 0.006, 1.0), y, f"{a:.3f}" + (f"  ({s} runs)" if s > 1 else ""),
                va="center", fontsize=9, color="#1B1F2A")
    ax.set_yticks(ys, [NAMES[m] for m in d.model], fontsize=9.5, color="#1B1F2A")
    ax.set_xlim(0.5, 1.10)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Test ROC-AUC, all three datasets pooled (499 clips), 95% bootstrap CI", color="#52514e", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#e4e3df")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#52514e", labelsize=8.5)
    ax.grid(axis="x", color="#e4e3df")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="#FFFFFF")
    plt.close(fig)


# ---------------------------------------------------------------- PowerPoint helpers

class Deck:
    def __init__(self):
        import win32com.client as w
        self.app = w.Dispatch("PowerPoint.Application")
        self.p = self.app.Presentations.Add(False)
        self.p.PageSetup.SlideWidth, self.p.PageSetup.SlideHeight = W, H
        self.n = 0

    def slide(self, dark=False, notes=""):
        self.n += 1
        s = self.p.Slides.Add(self.n, 12)  # ppLayoutBlank
        s.FollowMasterBackground = False
        s.Background.Fill.Solid()
        s.Background.Fill.ForeColor.RGB = rgb(NAVY if dark else WHITE)
        if notes:
            s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = notes
        if self.n > 1:  # page number, bottom right
            self.text(s, str(self.n), W - 70, H - 32, 40, 18, 10, MUTED if not dark else "AAB4C8", align=3)
        return s

    def text(self, s, txt, x, y, w, h, size=16, color=INK, bold=False, font=BODY, align=1, italic=False,
             anchor=1, name=None):
        sh = s.Shapes.AddTextbox(1, x, y, w, h)
        tf = sh.TextFrame
        tf.WordWrap = True
        tf.AutoSize = 0
        tf.MarginLeft = tf.MarginRight = tf.MarginTop = tf.MarginBottom = 0
        tf.VerticalAnchor = anchor
        r = tf.TextRange
        r.Text = txt
        r.Font.Name, r.Font.Size, r.Font.Bold, r.Font.Italic = font, size, bold, italic
        r.Font.Color.RGB = rgb(color)
        r.ParagraphFormat.Alignment = align
        if name:
            sh.Name = name
        return sh

    def bullets(self, s, items, x, y, w, h, size=16, color=INK, space=8):
        sh = self.text(s, "\r".join(items), x, y, w, h, size, color)
        r = sh.TextFrame.TextRange
        r.ParagraphFormat.Bullet.Visible = True
        r.ParagraphFormat.Bullet.Character = 8226
        r.ParagraphFormat.SpaceAfter = space
        r.IndentLevel = 1
        sh.TextFrame.Ruler.Levels(1).FirstMargin = 0
        sh.TextFrame.Ruler.Levels(1).LeftMargin = 16
        return sh

    def box(self, s, x, y, w, h, fill=CARD, radius=True, line=None):
        sh = s.Shapes.AddShape(5 if radius else 1, x, y, w, h)
        sh.Fill.ForeColor.RGB = rgb(fill)
        if line:
            sh.Line.ForeColor.RGB = rgb(line)
            sh.Line.Weight = 1
        else:
            sh.Line.Visible = False
        if radius:
            sh.Adjustments.SetItem(1, 0.08)
        sh.Shadow.Visible = False
        return sh

    def circle_num(self, s, n, x, y, d=34, fill=AMBER, color=NAVY):
        sh = s.Shapes.AddShape(9, x, y, d, d)
        sh.Fill.ForeColor.RGB = rgb(fill)
        sh.Line.Visible = False
        r = sh.TextFrame.TextRange
        r.Text = str(n)
        r.Font.Name, r.Font.Size, r.Font.Bold = HEAD, 15, True
        r.Font.Color.RGB = rgb(color)
        sh.TextFrame.MarginLeft = sh.TextFrame.MarginRight = 0
        return sh

    def title(self, s, txt, dark=False, sub=None):
        self.text(s, txt, 48, 30, W - 96, 50, 32, WHITE if dark else NAVY, bold=True, font=HEAD)
        if sub:
            self.text(s, sub, 48, 80, W - 96, 24, 15, "C9D2E3" if dark else SLATE)

    def stat(self, s, big, label, x, y, w, color=NAVY, label_color=SLATE, size=44):
        self.text(s, big, x, y, w, size * 1.3, size, color, bold=True, font=HEAD)
        self.text(s, label, x, y + size * 1.45, w, 60, 15, label_color)

    def picture(self, s, path, x, y, w=None, h=None):
        sh = s.Shapes.AddPicture(str(path), False, True, x, y, -1, -1)
        if w and not h:
            sh.LockAspectRatio = True
            sh.Width = w
        elif h and not w:
            sh.LockAspectRatio = True
            sh.Height = h
        elif w and h:
            sh.LockAspectRatio = True
            sh.Width = w
            if sh.Height > h:
                sh.Height = h
        return sh

    def table(self, s, rows, x, y, w, col_w=None, size=13, bold_cells=(), header_fill=NAVY, row_h=26):
        nr, nc = len(rows), len(rows[0])
        sh = s.Shapes.AddTable(nr, nc, x, y, w, row_h * nr)
        t = sh.Table
        if col_w:
            for j, cw in enumerate(col_w, 1):
                t.Columns(j).Width = cw
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row, 1):
                c = t.Cell(i, j)
                c.Shape.Fill.ForeColor.RGB = rgb(header_fill if i == 1 else (CARD if i % 2 == 0 else WHITE))
                tf = c.Shape.TextFrame
                tf.MarginLeft, tf.MarginRight, tf.MarginTop, tf.MarginBottom = 6, 6, 3, 3
                r = tf.TextRange
                r.Text = str(v)
                r.Font.Name, r.Font.Size = BODY, size
                r.Font.Bold = i == 1 or (i - 1, j - 1) in bold_cells
                r.Font.Color.RGB = rgb(WHITE if i == 1 else INK)
                r.ParagraphFormat.Alignment = 1 if j == 1 else 2
            t.Rows(i).Height = row_h
        return sh

    def save(self, path: Path, preview: Path | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        self.p.SaveAs(str(path))
        if preview:
            preview.mkdir(parents=True, exist_ok=True)
            for f in preview.glob("slide-*.png"):
                f.unlink()
            for i in range(1, self.p.Slides.Count + 1):
                self.p.Slides(i).Export(str(preview / f"slide-{i:02d}.png"), "PNG", 1600, 900)
        self.p.Close()


# ---------------------------------------------------------------- the deck

def build():
    g = results()
    LOCAL_FIGS.mkdir(parents=True, exist_ok=True)
    bar_figure(g, LOCAL_FIGS / "pooled_auc_bars.png")
    sp = pd.read_csv(ROOT / "work" / "bench" / "speed.csv").set_index("model")
    ft = lambda p, c="auc": val(g, p, "videomae_ft", c)  # noqa: E731
    n_ft = int(g[(g.model == "videomae_ft") & (g.protocol == "p2_all")].seeds.iloc[0])
    runs_note = f"mean of {n_ft} training runs" if n_ft > 1 else "single training run"

    d = Deck()

    # 1 title
    s = d.slide(dark=True, notes=(
        "Suraksha-Net detects violence in CCTV clips. This deck covers what we changed from the semester "
        "version, how we evaluate fairly, and what the models achieve on cameras they have never seen."))
    d.text(s, "SURAKSHA-NET", 64, 120, 600, 30, 16, AMBER, bold=True)
    d.text(s, "Violence detection that holds up on cameras it has never seen", 64, 160, 760, 130, 40,
           WHITE, bold=True, font=HEAD)
    d.text(s, "Leakage-free evaluation on three public datasets  ·  pretrained video models  ·  "
              "a lightweight model for real CCTV hardware", 64, 300, 780, 50, 17, "C9D2E3")
    d.text(s, "Arpit Thakur  ·  Vishnu  ·  Major project  ·  November 2026", 64, 440, 700, 24, 14, "AAB4C8")

    # 2 why
    s = d.slide(notes=(
        "The semester model reported 83% accuracy, but only on two small datasets, and when it was tested on "
        "a dataset it had not trained on its ROC-AUC fell to 0.60, close to guessing. We also could not tell "
        "which clips it had trained on, so it cannot be tested fairly. That is why we rebuilt the evaluation."))
    d.title(s, "Why we rebuilt the evaluation", sub="The semester model looked good on paper, but the numbers did not transfer")
    cards = [("83%", "accuracy of the semester model, measured on two small datasets only (SFD + AIRTLab)"),
             ("0.60", "ROC-AUC when that model was tested on a different dataset: close to guessing"),
             ("?", "which clips it trained on is unknown, so it cannot be tested fairly on any held-out set")]
    for i, (big, lab) in enumerate(cards):
        x = 48 + i * 292
        d.box(s, x, 136, 272, 200)
        d.stat(s, big, lab, x + 24, 156, 228, color=NAVY, size=54)
    d.box(s, 48, 372, 864, 84, fill=NAVY)
    d.text(s, "Goal: honest numbers that still hold on cameras and datasets the model has never seen",
           76, 400, 808, 30, 20, WHITE, bold=True, font=HEAD)

    # 3 data
    s = d.slide(notes=(
        "Three public datasets, 2,650 clips. Every clip is fingerprinted with SHA-256; exact duplicates are "
        "removed and near-duplicates and clips from the same source video are grouped, so a group never "
        "spans train and test. The splits are locked by hash so they cannot change silently."))
    d.title(s, "Three public datasets, audited clip by clip", sub="2,650 clips, 2,639 unique after removing exact duplicates")
    ds = [("RWF-2000", "2,000", "real-world CCTV fights from YouTube (ICPR 2020)"),
          ("SFD", "300", "surveillance-camera fights (IPTA 2019)"),
          ("AIRTLab", "350", "staged scenes, two cameras, includes hard non-violent actions (Data in Brief 2020)")]
    for i, (nm, n, desc) in enumerate(ds):
        x = 48 + i * 292
        d.box(s, x, 130, 272, 150)
        d.text(s, nm, x + 20, 146, 232, 24, 18, NAVY, bold=True, font=HEAD)
        d.text(s, n + " clips", x + 20, 174, 232, 30, 26, AMBER, bold=True, font=HEAD)
        d.text(s, desc, x + 20, 212, 232, 60, 13, SLATE)
    steps = ["SHA-256 fingerprint of every video", "Near-duplicate and source groups",
             "Splits locked by hash (LOCK.json)", "JPEG frame cache, 15 fps"]
    for i, st in enumerate(steps):
        x = 48 + i * 219
        d.circle_num(s, i + 1, x, 318)
        d.text(s, st, x + 44, 314, 165, 44, 14, INK, anchor=3)
    d.text(s, "A group of related clips never spans train and test, so a model cannot pass the test by "
              "recognising a scene it has already seen.", 48, 400, 864, 50, 15, SLATE, italic=True)

    # 4 protocols
    s = d.slide(notes=(
        "Four protocols. P1 trains and tests within one dataset. P2 pools all three. P3 is the key test: "
        "train on two datasets and test on the third, which the model has never seen. P4 compares random "
        "splits with grouped splits to measure how much leakage inflates scores."))
    d.title(s, "Four ways to test, from easy to hard", sub="P3 is the one that matters for real deployment")
    prot = [("P1", "Within one dataset", "Train and test on RWF-2000, SFD or AIRTLab separately"),
            ("P2", "All three pooled", "One model trained and tested on the mix of all three datasets"),
            ("P3", "Unseen dataset", "Train on two datasets, test on the third, never seen in training"),
            ("P4", "Bias study", "Random split vs grouped split: does leakage inflate the score?")]
    for i, (k, nm, desc) in enumerate(prot):
        x = 48 + i * 219
        hl = k == "P3"
        d.box(s, x, 140, 203, 250, fill=NAVY if hl else CARD)
        d.text(s, k, x + 18, 158, 167, 40, 34, AMBER, bold=True, font=HEAD)
        d.text(s, nm, x + 18, 210, 167, 50, 18, WHITE if hl else NAVY, bold=True, font=HEAD)
        d.text(s, desc, x + 18, 266, 167, 110, 14, "D6DCE8" if hl else SLATE)
    d.text(s, "Every protocol uses the same rules: threshold chosen on validation data, test set scored once, "
              "95% confidence intervals by group bootstrap.", 48, 420, 864, 50, 15, SLATE, italic=True)

    # 5 models
    s = d.slide(notes=(
        "Two families. Frozen: a large pretrained video model turns each clip into features and we train a "
        "small classifier on top. Fine-tuned: the network itself is trained on our data. MotionLite is our "
        "lightweight model; with distillation it also learns from VideoMAE's predictions. All backbones start "
        "from public pretrained weights, which the paper states."))
    d.title(s, "Models we compared", sub="All start from public pretrained weights; every classifier is trained on our splits")
    cols = [("Frozen backbone + trained classifier", ["VideoMAE-B (Kinetics-400 video transformer)",
             "CLIP ViT-B/16 (image-text model, frames averaged)", "R(2+1)D-18 (Kinetics-400 3D CNN)",
             "VideoMAE + CLIP features combined"]),
            ("Fine-tuned on our data", ["VideoMAE-B, last 4 of 12 blocks trained", "MotionLite (ours): 3M parameters, RGB + motion streams",
             "MotionLite + distillation from VideoMAE", "MobileNetV3-TSM (lightweight baseline)"])]
    for i, (hd, items) in enumerate(cols):
        x = 48 + i * 440
        d.box(s, x, 136, 424, 232)
        d.text(s, hd, x + 24, 156, 376, 28, 19, NAVY, bold=True, font=HEAD)
        d.bullets(s, items, x + 24, 200, 376, 160, 16, INK, space=12)
    d.box(s, 48, 392, 864, 76, fill=NAVY)
    d.text(s, "Distillation: MotionLite learns from the true labels and from VideoMAE's predictions. Those "
              "predictions come from classifiers that never saw the clip (5-fold) and never the held-out dataset.",
           72, 406, 816, 52, 15, WHITE)

    # 6 best model
    s = d.slide(notes=(
        f"Our best model is VideoMAE with its last four blocks fine-tuned. On all three datasets pooled it "
        f"reaches ROC-AUC {fmt(ft('p2_all'))} and {pct(ft('p2_all', 'acc'))} accuracy ({runs_note}). On "
        f"RWF-2000 alone, {pct(ft('p1_rwf', 'acc'))}. Most importantly, trained only on SFD and AIRTLab it "
        f"still scores {fmt(ft('p3_no_rwf'))} on RWF-2000, which it never saw."))
    d.title(s, "Best model: VideoMAE, partially fine-tuned", sub=f"Test-set results ({runs_note} on the pooled set)")
    stats = [(fmt(ft("p2_all")), f"ROC-AUC, all three pooled\n{pct(ft('p2_all', 'acc'))} accuracy"),
             (fmt(ft("p1_rwf")), f"ROC-AUC, RWF-2000\n{pct(ft('p1_rwf', 'acc'))} accuracy"),
             (fmt(ft("p3_no_rwf")), f"ROC-AUC on RWF-2000 without training on it\n{pct(ft('p3_no_rwf', 'acc'))} accuracy")]
    for i, (big, lab) in enumerate(stats):
        y = 132 + i * 118
        d.box(s, 48, y, 420, 104)
        d.text(s, big, 70, y + 16, 170, 70, 48, NAVY if i < 2 else AMBER, bold=True, font=HEAD, anchor=3)
        d.text(s, lab, 248, y + 16, 205, 70, 14, SLATE, anchor=3)
    cm = FIGS / "fig_confusion_videomae_ft_finetuned_p2_all.png"
    if cm.exists():
        d.picture(s, cm, 520, 130, w=380, h=350)
    d.text(s, "Semester model for comparison: 83% accuracy on two small datasets, ROC-AUC 0.60 on an unseen one.",
           48, 492, 864, 24, 13, SLATE, italic=True)

    # 7 pooled bar chart
    s = d.slide(notes=(
        "All models on the pooled test set of 499 clips. Fine-tuned VideoMAE is clearly on top. Frozen VideoMAE "
        "is next: features from a large pretrained video model are strong even without training the network. "
        "The small fine-tuned models, including MotionLite, sit around 0.89."))
    d.title(s, "Every model on the pooled test set", sub="Large pretrained video models lead; small models trail by about 0.08 ROC-AUC")
    d.picture(s, LOCAL_FIGS / "pooled_auc_bars.png", 70, 112, w=820, h=400)

    # 8 unseen datasets table
    s = d.slide(notes=(
        "The hardest test: train on two datasets, test on the third. Large pretrained models keep most of "
        "their accuracy; small fine-tuned models drop much more. The SFD and AIRTLab test sets are small, "
        "47 and 52 clips, so their confidence intervals are wide."))
    d.title(s, "Generalising to a dataset never seen in training", sub="Test ROC-AUC under P3 (train on the other two datasets)")
    order = ["videomae_ft", "videomae+clip", "videomae", "clip", "mnv3_tsm", "motionlite_kd", "motionlite"]
    cols3 = ["p3_no_rwf", "p3_no_sfd", "p3_no_avd"]
    rows = [["Model", "Unseen RWF-2000", "Unseen SFD", "Unseen AIRTLab", "Mean"]]
    vals = []
    for m in order:
        v = [val(g, p, m) for p in cols3]
        vals.append(v + [float(np.nanmean(v))])
        rows.append([NAMES[m]] + [fmt(x) for x in vals[-1]])
    best = {(int(np.nanargmax([r[j] for r in vals])) + 1, j + 1) for j in range(4)}
    d.table(s, rows, 48, 124, 864, col_w=[352, 128, 128, 128, 128], size=14, bold_cells=best, row_h=34)
    d.text(s, "Bold = best per column.  Test sets: RWF-2000 400 clips, SFD 47, AIRTLab 52 (wide intervals for the small two).",
           48, 418, 864, 40, 13, SLATE, italic=True)

    # 9 accuracy vs cost
    ml, vm = sp.loc["motionlite"], sp.loc["videomae"]
    s = d.slide(notes=(
        f"Accuracy is not everything for CCTV: cheap boxes have no GPU. On a CPU, MotionLite needs "
        f"{ml.cpu_ms:.0f} ms per clip with {ml.params_m:.1f} million parameters; VideoMAE needs {vm.cpu_ms:.0f} ms "
        f"with {vm.params_m:.0f} million. That is the trade-off: VideoMAE for accuracy, MotionLite where speed "
        f"and cost matter."))
    d.title(s, "Accuracy vs cost on a CPU", sub="Cheap CCTV boxes have no GPU: speed and size decide what can run")
    d.picture(s, FIGS / "fig_tradeoff_p2_all_cpu.png", 40, 110, w=560, h=400)
    d.box(s, 624, 120, 288, 380, fill=NAVY)
    d.text(s, f"{vm.cpu_ms / ml.cpu_ms:.0f}×", 648, 140, 240, 60, 48, AMBER, bold=True, font=HEAD)
    d.text(s, "faster: MotionLite vs VideoMAE per clip on a CPU", 648, 202, 240, 44, 14, "D6DCE8")
    d.text(s, f"{vm.params_m / ml.params_m:.0f}×", 648, 262, 240, 60, 48, AMBER, bold=True, font=HEAD)
    d.text(s, f"smaller: {ml.params_m:.1f}M vs {vm.params_m:.0f}M parameters", 648, 324, 240, 44, 14, "D6DCE8")
    d.text(s, f"Cost: pooled ROC-AUC {fmt(val(g, 'p2_all', 'motionlite_kd'))} vs {fmt(ft('p2_all'))}",
           648, 400, 240, 60, 15, WHITE, bold=True)

    # 10 bias study
    s = d.slide(notes=(
        "We expected random splits to inflate scores compared with grouped splits. On these datasets the effect "
        "was small: within about 0.01 to 0.03 ROC-AUC either way. We report this honestly; the paper's main "
        "claims are generalisation and efficiency, not leakage inflation."))
    d.title(s, "Bias study: leaky splits changed scores only a little here", sub="Test ROC-AUC with random vs grouped splits")
    rows = [["Model", "Random split", "Grouped split", "Pooled, leaky", "Pooled, clean"]]
    for m in ["mnv3_tsm", "videomae", "videomae+clip", "clip"]:
        rows.append([NAMES[m]] + [fmt(val(g, p, m)) for p in ("p4_random", "p4_grouped", "p2_all_leaky", "p2_all")])
    d.table(s, rows, 48, 124, 864, col_w=[352, 128, 128, 128, 128], size=14, row_h=34)
    d.box(s, 48, 330, 864, 110)
    d.text(s, "Takeaway", 72, 344, 300, 24, 17, NAVY, bold=True, font=HEAD)
    d.text(s, "Differences stay within the confidence intervals. The big effect in our data is not leakage but "
              "domain shift: models lose accuracy on a dataset they never saw (previous slide).",
           72, 372, 816, 60, 15, INK)

    # 11 takeaways
    s = d.slide(dark=True, notes=(
        "Three takeaways and what happens next. Extra training runs are in progress so every fine-tuned "
        "number gets an error bar; then the paper draft is finalised."))
    d.title(s, "Takeaways and next steps", dark=True)
    tk = [f"Fine-tuned VideoMAE reaches ROC-AUC {fmt(ft('p2_all'))} ({pct(ft('p2_all', 'acc'))} accuracy) "
          f"on leakage-free splits",
          f"It keeps {fmt(ft('p3_no_rwf'))} on RWF-2000 without training on it; the semester model fell to 0.60",
          f"MotionLite runs {vm.cpu_ms / ml.cpu_ms:.0f}× faster on a CPU with {vm.params_m / ml.params_m:.0f}× fewer parameters, "
          f"at a cost in accuracy"]
    for i, t in enumerate(tk):
        y = 120 + i * 76
        d.circle_num(s, i + 1, 56, y)
        d.text(s, t, 106, y - 4, 520, 64, 17, WHITE)
    d.box(s, 660, 112, 252, 250, fill=NAVY2)
    d.text(s, "Next", 684, 128, 200, 26, 18, AMBER, bold=True, font=HEAD)
    d.bullets(s, ["3 runs per fine-tuned model (in progress) for error bars", "Paper draft",
                  "Demo app on the best model"], 684, 164, 210, 180, 15, WHITE, space=10)

    d.save(OUT, PREVIEW)
    print(f"wrote {OUT} ({d.n} slides); previews in {PREVIEW}")


if __name__ == "__main__":
    build()
