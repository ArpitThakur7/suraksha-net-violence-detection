"""Paper draft, generated offline from the live results.

    python reports/make_paper.py   ->  reports/paper_draft.html, .docx and .pdf (via Word)

Tables are filled from work/runs, work/probe and work/bench/speed.csv (seed means where several
seeds have finished), so rerunning after the seed grid updates every number. Prose that depends on
the numbers is also generated; wording around it is a draft to edit.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))
from make_slides import NAMES, results, val  # noqa: E402

FIGS = ROOT / "work" / "results" / "figs"
OUT_HTML = ROOT / "reports" / "paper_draft.html"
OUT_DOCX = ROOT / "reports" / "paper_draft.docx"

EXTRA_REFS = {  # not in docs/literature.md's BibTeX block
    "tong2022videomae": "Z. Tong, Y. Song, J. Wang and L. Wang. VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS, 2022.",
    "radford2021clip": "A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.",
    "tran2018r2plus1d": "D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun and M. Paluri. A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR, 2018.",
    "hinton2015distilling": "G. Hinton, O. Vinyals and J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.",
    "efron1994bootstrap": "B. Efron and R. J. Tibshirani. An Introduction to the Bootstrap. Chapman and Hall/CRC, 1994.",
}


def bib_refs() -> dict:
    t = (ROOT / "docs" / "literature.md").read_text(encoding="utf-8")
    t = t[t.index("```bibtex"):]
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", t, re.S):
        body = m.group(3)

        def g(k):
            r = re.search(k + r"\s*=\s*\{(.*?)\},?\s*\n", body, re.S)
            return re.sub(r"[{}]", "", r.group(1)).strip() if r else ""
        authors = [a.strip() for a in g("author").split(" and ") if a.strip()]
        names = []
        for a in authors[:3]:
            last, _, first = a.partition(",")
            names.append((first.strip()[:1] + ". " if first.strip() else "") + last.strip())
        au = ", ".join(names) + (" et al." if len(authors) > 3 else "")
        venue = g("booktitle") or g("journal") or (f"arXiv:{g('eprint')}" if g("eprint") else "")
        out[m.group(2)] = f"{au}. {g('title')}. {venue}, {g('year')}."
    out.update(EXTRA_REFS)
    return out


class Cite:
    def __init__(self, refs):
        self.refs, self.order = refs, []

    def __call__(self, *keys):
        nums = []
        for k in keys:
            if k not in self.refs:
                raise KeyError(f"unknown reference {k}")
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"


def f3(x):
    return "–" if not np.isfinite(x) else f"{x:.3f}"


def cell(g, p, m, col="auc"):
    r = g[(g.protocol == p) & (g.model == m)]
    if not len(r):
        return "–"
    r = r.iloc[0]
    v = r[col]
    if col == "auc" and r.seeds > 1 and np.isfinite(r.auc_sd):
        return f"{v:.3f} ± {r.auc_sd:.3f}"
    return f"{v:.3f}" if col == "auc" else f"{100 * v:.1f}"


def table(header, rows, caption, bold=()):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in header)
    body = ""
    for i, r in enumerate(rows):
        tds = "".join(f"<td{' class=b' if (i, j) in bold else ''}>{html.escape(str(c))}</td>"
                      for j, c in enumerate(r))
        body += f"<tr>{tds}</tr>"
    return f"<p class=cap>{caption}</p><table><tr>{h}</tr>{body}</table>"


def figure(path: Path, caption: str, width=15.5):
    if not path.exists():
        return ""
    from PIL import Image
    with Image.open(path) as im:
        ar = im.height / im.width
    wpx = int(width / 2.54 * 96)          # Word honours width/height attributes, not CSS width
    return (f'<p class=fig><img src="{path.as_uri()}" width={wpx} height={int(wpx * ar)}></p>'
            f"<p class=cap>{caption}</p>")


def build():
    g = results()
    sp = pd.read_csv(ROOT / "work" / "bench" / "speed.csv").set_index("model")
    c = Cite(bib_refs())
    ft = lambda p, col="auc": val(g, p, "videomae_ft", col)  # noqa: E731
    nseed = lambda m, p="p2_all": int(g[(g.protocol == p) & (g.model == m)].seeds.max())  # noqa: E731
    ml, vm = sp.loc["motionlite"], sp.loc["videomae"]
    lodo = ["p3_no_rwf", "p3_no_sfd", "p3_no_avd"]
    mean_lodo = {m: float(np.nanmean([val(g, p, m) for p in lodo])) for m in NAMES}
    kd_gain = {p: val(g, p, "motionlite_kd") - val(g, p, "motionlite") for p in ["p2_all", "p1_rwf"] + lodo}

    S = []  # html sections
    S.append(f"""
<h1>Do violence detectors generalise? A leakage-controlled, cross-dataset evaluation of video foundation
models and lightweight networks on RWF-2000, SFD and AIRTLab</h1>
<p class=au>Arpit Thakur, Vishnu<br><i>{pd.Timestamp.now():%d %B %Y}. Code, splits and results:
github.com/ArpitThakur7/suraksha-net-violence-detection</i></p>

<h2>Abstract</h2>
<p>Published violence-detection results on RWF-2000 range from 77% to almost 100% accuracy under
protocols that are often unstated, and few models are tested on data from a different source. We
audit three public datasets (RWF-2000, the Surveillance Camera Fight dataset and AIRTLab; 2,650 clips)
at the level of file hashes, perceptual near-duplicates and source videos, and find 57 RWF-2000 clips
that share YouTube source videos with SFD. On grouped, hash-locked splits we compare frozen video
foundation models with linear classifiers, a partially fine-tuned VideoMAE, and lightweight
fine-tuned networks, choosing every threshold on validation data and reporting group-bootstrap
confidence intervals. The partially fine-tuned VideoMAE reaches a test ROC-AUC of {f3(ft('p2_all'))}
({100 * ft('p2_all', 'acc'):.1f}% accuracy) on the pooled datasets and keeps {f3(ft('p3_no_rwf'))} on
RWF-2000 when trained only on the other two. Lightweight models lose more under this
leave-one-dataset-out shift (mean ROC-AUC {f3(mean_lodo['motionlite'])} for MotionLite against
{f3(mean_lodo['videomae_ft'])} for the fine-tuned VideoMAE), but run {vm.cpu_ms / ml.cpu_ms:.0f}× faster on a
CPU with {vm.params_m / ml.params_m:.0f}× fewer parameters. Distilling VideoMAE into MotionLite raises its
ROC-AUC on unseen AIRTLab from {f3(val(g, 'p3_no_avd', 'motionlite'))} to {f3(val(g, 'p3_no_avd', 'motionlite_kd'))}, above the
fine-tuned VideoMAE ({f3(ft('p3_no_avd'))}). Contrary to our expectation, random versus grouped splits
changed scores by at most a few hundredths of ROC-AUC on these data; domain shift, not split leakage, is
the dominant effect. A live monitoring prototype runs the distilled model on a camera stream. We release
the audit manifests, locked splits and code.</p>
""")

    S.append(f"""
<h2>1. Introduction</h2>
<p>Automatic violence detection in surveillance video is usually benchmarked on RWF-2000 {c('cheng2020rwf')},
with smaller sets such as SFD {c('akti2019vision')} and AIRTLab {c('bianculli2020dataset')} used for
additional results. Reported RWF-2000 accuracies span a wide range, and lightweight models under stated
protocols cluster around 82–90% {c('islam2021sepconvlstm', 'zhang2022mobilenettsm', 'veltmeijer2024subgroup')},
while larger models report 94–96% {c('senadeera2024cuenet', 'senadeera2025fusemamba')}. Several practices make
these numbers hard to compare: random clip-level splits of datasets whose clips come from longer
source videos {c('veltmeijer2024subgroup')}, frame-level folds {c('rendon2024crimenet')}, and selecting models or
thresholds on the test split. Cross-dataset tests exist {c('coskun2026privacy', 'veltmeijer2024subgroup', 'tan2026gvd')}
but none we found covers SFD and AIRTLab with grouped splits.</p>
<p>This paper asks a practical question: which kind of model still works on footage from a dataset it
never saw, and at what computational cost? Our contributions are:</p>
<ol>
<li>a source-level audit of RWF-2000, SFD and AIRTLab (hashes, perceptual near-duplicates, source videos and
AIRTLab camera pairs) that finds cross-dataset overlap between RWF-2000 and SFD not reported before, to our
knowledge, together with grouped, hash-locked splits for ten protocols;</li>
<li>a leave-one-dataset-out (LODO) evaluation with validation-only model and threshold selection and
group-bootstrap confidence intervals;</li>
<li>a comparison of frozen foundation models {c('tong2022videomae', 'radford2021clip')}, partial fine-tuning, and
lightweight networks, including distillation from VideoMAE into a 3M-parameter model
{c('hinton2015distilling')}, with measured CPU and GPU latency;</li>
<li>a bias study of random versus grouped splits, which on these data shows only a small effect.</li>
</ol>
""")

    m = pd.read_csv(ROOT / "work" / "manifest" / "all_clips.csv", keep_default_na=False)
    sizes = {p: pd.read_csv(ROOT / "work" / "splits" / f"{p}.csv").split.value_counts().to_dict()
             for p in ["p1_rwf", "p1_sfd", "p1_avd", "p2_all", "p3_no_rwf", "p3_no_sfd", "p3_no_avd",
                       "p4_random", "p4_grouped", "p2_all_leaky"]}
    prot_rows = [[p, desc, sizes[p].get("train", 0), sizes[p].get("val", 0), sizes[p].get("test", 0)] for p, desc in [
        ("p1_rwf", "RWF-2000 only"), ("p1_sfd", "SFD only"), ("p1_avd", "AIRTLab only"),
        ("p2_all", "all three pooled"), ("p3_no_rwf", "train SFD + AIRTLab, test RWF-2000"),
        ("p3_no_sfd", "train RWF-2000 + AIRTLab, test SFD"), ("p3_no_avd", "train RWF-2000 + SFD, test AIRTLab"),
        ("p4_random", "RWF-2000, random split (leaky)"), ("p4_grouped", "RWF-2000, grouped split"),
        ("p2_all_leaky", "pooled, group rule switched off")]]
    S.append(f"""
<h2>2. Data and audit</h2>
<p><b>Datasets.</b> RWF-2000 {c('cheng2020rwf')} has 2,000 five-second clips of real fights and non-fights
collected from YouTube surveillance footage, with an official 1,600/400 split. SFD {c('akti2019vision')}
has 300 two-second surveillance clips (150 fight, 150 non-fight), also collected from YouTube. AIRTLab
{c('bianculli2020dataset')} has 350 staged clips (230 violent, 120 non-violent) recorded by two cameras, with
non-violent actions chosen to resemble violence (hugs, claps).</p>
<p><b>Audit.</b> Every file is hashed with SHA-256 ({len(m)} clips; {int((m.exclusion != '').sum())} exact duplicates
removed), decoded, and summarised with perceptual hashes. Clips are merged into global groups when they
are near-duplicates, come from the same source video (RWF-2000 source IDs and SFD's list of YouTube
links), or are the two camera views of one AIRTLab scene. This gives 1,122 groups, 14 of which span more
than one dataset. Ten YouTube source videos appear in both RWF-2000 and SFD, covering 57 RWF-2000 clips
(including clips in RWF-2000's official validation split) and SFD clips in its training and test
splits; 57 of the 447 cross-dataset clip pairs from these sources are visual near-duplicates.</p>
<p><b>Splits.</b> RWF-2000 keeps its official split as the test set; SFD and AIRTLab are split 70/15/15
stratified by label and group. Inside a protocol, a train or validation clip whose group also appears
in a higher-precedence split (test, then validation) is dropped, so no group spans two splits. Every
split file is hashed into a lock file that training checks before reading it. Table 1 lists the
protocols.</p>
{table(['Protocol', 'Train / test data', 'Train', 'Val', 'Test'], prot_rows,
       'Table 1. Protocols and split sizes (clips). P4 uses RWF-2000 training data only.')}
""")

    S.append(f"""
<h2>3. Methods</h2>
<p><b>Input.</b> Clips are decoded once at 15 fps into a JPEG frame cache (short side 256 px). A training
sample is 16 frames drawn from a random 60–100% segment, randomly resized and cropped to 224 px, flipped,
colour-jittered and occasionally converted to grey; evaluation uses 16 evenly spaced frames and a centre
crop.</p>
<p><b>Frozen foundation models + linear probes.</b> VideoMAE-B fine-tuned on Kinetics-400
{c('tong2022videomae')}, CLIP ViT-B/16 {c('radford2021clip')} (per-frame image embeddings averaged over 8
frames) and R(2+1)D-18 {c('tran2018r2plus1d')} encode each clip over four evenly spaced temporal windows
with all weights frozen. A logistic-regression classifier on standardised features is trained per
protocol, each window being one training sample; validation and test probabilities are averaged over
windows. Pooled and LODO protocols weight samples so that every (dataset, label) cell carries equal
weight. The regularisation strength is chosen from {{0.001, 0.01, 0.1, 1}} by validation ROC-AUC.
We also concatenate VideoMAE and CLIP features. The probes are deterministic, so they are run once.
Recent versions of the transformers library no longer load VideoMAE's attention biases (stored as
q_bias/v_bias); we copy them into the query and value biases with a zero key bias, as in the original
model.</p>
<p><b>Partial fine-tuning.</b> For VideoMAE-B we freeze the patch embedding and the first 8 of 12
transformer blocks and train the last 4 blocks, the final norm and a new two-way head (28.4M trainable
parameters) with learning rate 5×10<sup>-5</sup>.</p>
<p><b>Lightweight networks.</b> MobileNetV3-TSM inserts temporal shift modules into an ImageNet
MobileNetV3-Large. MotionLite (3.0M parameters) adds a frame-difference stream to the shifted RGB stream,
fuses the two with a learned gate in the middle of the network and pools over time with attention. Its
components are published individually {c('kang2021efficient', 'islam2021sepconvlstm', 'zhang2022mobilenettsm')}; we
treat it as a strong lightweight baseline, not as a new architecture.</p>
<p><b>Distillation.</b> MotionLite is also trained with the loss
(1 − α)·CE(y) + α·CE(p<sub>t</sub>), α = 0.5, where p<sub>t</sub> is the VideoMAE probe's probability for the
clip {c('hinton2015distilling')}. To avoid memorised targets, p<sub>t</sub> is out-of-fold: the protocol's
training clips are split into five group-stratified folds and each clip's target comes from a probe
trained on the other four. Only the protocol's own training and validation clips are used, so a LODO
teacher never sees the held-out dataset.</p>
<p><b>Training and selection.</b> AdamW (weight decay 0.05), learning rate 3×10<sup>-4</sup> unless stated,
cosine schedule with 2 warm-up epochs, label smoothing 0.1, effective batch 32, mixed precision, at most
40 epochs with early stopping after 6 epochs without a better validation ROC-AUC. The checkpoint with the
best validation ROC-AUC is kept, and the decision threshold is the value on a 0.05–0.95 grid that
maximises validation F1. The test split of each protocol is scored once, with the frozen threshold.</p>
<p><b>Statistics.</b> 95% confidence intervals use a percentile bootstrap that resamples whole groups
(1,000 resamples) {c('efron1994bootstrap')}. Where several training seeds have finished we report the mean
and standard deviation across seeds.</p>
""")

    main_models = ["videomae_ft", "videomae", "videomae+clip", "clip", "r2plus1d", "mnv3_tsm", "motionlite_kd", "motionlite"]
    rows, bold = [], set()
    for i, mm in enumerate(main_models):
        rows.append([NAMES[mm], nseed(mm), cell(g, "p2_all", mm), cell(g, "p2_all", mm, "acc"),
                     cell(g, "p1_rwf", mm), cell(g, "p1_rwf", mm, "acc")])
    for j, (p, col) in enumerate([("p2_all", "auc"), ("p2_all", "acc"), ("p1_rwf", "auc"), ("p1_rwf", "acc")], start=2):
        best = int(np.nanargmax([val(g, p, mm, col) for mm in main_models]))
        bold.add((best, j))
    S.append(f"""
<h2>4. Results</h2>
<h3>4.1 Within-domain accuracy</h3>
<p>Table 2 gives test results on the pooled protocol (499 test clips) and on RWF-2000 alone (400 clips).
The partially fine-tuned VideoMAE is best on both, with ROC-AUC {f3(ft('p2_all'))} and {f3(ft('p1_rwf'))}.
Frozen VideoMAE features with a linear classifier come second ({f3(val(g, 'p2_all', 'videomae'))} pooled),
ahead of every fine-tuned lightweight model (0.886–0.896).</p>
{table(['Model', 'Runs', 'Pooled ROC-AUC', 'Pooled acc. (%)', 'RWF-2000 ROC-AUC', 'RWF-2000 acc. (%)'], rows,
       'Table 2. Within-domain test results. ROC-AUC is mean ± s.d. over training runs where more than one has finished. Bold: best per column.', bold)}
{figure(FIGS / 'fig_roc_p2_all.png', 'Figure 1. ROC curves on the pooled test set (one training run per model, so the AUCs in the legend can differ slightly from the run means in Table 2).', 11)}
""")

    rows, bold = [], set()
    for mm in main_models:
        v = [val(g, p, mm) for p in lodo]
        rows.append([NAMES[mm]] + [cell(g, p, mm) for p in lodo] + [f3(np.nanmean(v))])
    for j in range(1, 5):
        colv = [val(g, lodo[j - 1], mm) if j < 4 else mean_lodo[mm] for mm in main_models]
        bold.add((int(np.nanargmax(colv)), j))
    S.append(f"""
<h3>4.2 Generalisation to an unseen dataset</h3>
<p>Under LODO (Table 3) every model loses accuracy relative to the pooled protocol, but large pretrained
models lose far less. Trained on SFD and AIRTLab only (531 clips), the fine-tuned VideoMAE still reaches
{f3(ft('p3_no_rwf'))} on RWF-2000. The concatenated VideoMAE + CLIP probe has the best mean over the three
held-out datasets ({f3(mean_lodo['videomae+clip'])}). The lightweight models fall to a mean of
{f3(mean_lodo['mnv3_tsm'])} (MobileNetV3-TSM) and {f3(mean_lodo['motionlite'])} (MotionLite). The SFD and AIRTLab
test sets have only 47 and 52 clips, so their intervals are wide (Figure 2). Accuracy at the frozen
threshold drops more than ROC-AUC: for instance MotionLite on unseen AIRTLab ranks clips with ROC-AUC
{f3(val(g, 'p3_no_avd', 'motionlite'))} but reaches only {100 * val(g, 'p3_no_avd', 'motionlite', 'acc'):.1f}% accuracy, because the
threshold chosen on the training domains does not transfer.</p>
{table(['Model', 'Unseen RWF-2000', 'Unseen SFD', 'Unseen AIRTLab', 'Mean'], rows,
       'Table 3. Leave-one-dataset-out test ROC-AUC (train on the other two datasets). Bold: best per column.', bold)}
{figure(FIGS / 'fig_auc_protocols.png', 'Figure 2. Test ROC-AUC with 95% group-bootstrap intervals for every protocol. Filled markers: frozen backbones; hollow: fine-tuned.', 15.5)}
""")

    rows = []
    for mm in ["mnv3_tsm", "videomae", "videomae+clip", "clip", "r2plus1d"]:
        rows.append([NAMES[mm]] + [f3(val(g, p, mm)) for p in ("p4_random", "p4_grouped", "p2_all_leaky", "p2_all")])
    S.append(f"""
<h3>4.3 Does split leakage inflate scores?</h3>
<p>We expected random splits of RWF-2000 to score higher than grouped splits, and the pooled protocol
without the group rule to score higher than with it. Table 4 shows that the differences are small and
inconsistent in sign: within about 0.01–0.03 ROC-AUC and inside the confidence intervals. The two RWF-2000
splits also have different test sets (240 and 207 clips), which limits the comparison. On these data the
large effect is the shift between datasets (Section 4.2), not leakage inside a dataset. We report this
negative result because the opposite claim is common.</p>
{table(['Model', 'P4 random', 'P4 grouped', 'Pooled, leaky', 'Pooled, grouped'], rows,
       'Table 4. Bias study: test ROC-AUC with and without the group rule.')}
""")

    rows = []
    for mm, arch in [("motionlite", "motionlite"), ("mnv3_tsm", "mnv3_tsm"), ("r2plus1d", "r2plus1d"),
                     ("clip", "clip"), ("videomae", "videomae"), ("videomae_ft", "videomae")]:
        r = sp.loc[arch]
        rows.append([NAMES[mm], f"{r.params_m:.1f}", f"{r.gflops:.0f}", f"{r.cuda_ms:.1f}", f"{r.cpu_ms:.0f}",
                     cell(g, "p2_all", mm)])
    pname = {"p2_all": "pooled", "p1_rwf": "RWF-2000", "p3_no_rwf": "unseen RWF-2000",
             "p3_no_sfd": "unseen SFD", "p3_no_avd": "unseen AIRTLab"}
    gains = ", ".join(f"{v:+.3f} ({pname[p]})" for p, v in kd_gain.items() if np.isfinite(v))
    S.append(f"""
<h3>4.4 Accuracy versus cost, and distillation</h3>
<p>Table 5 lists model size and latency for one 16-frame clip on the laptop's RTX 4050 GPU (fp16) and on
its CPU (fp32, 4 threads). On the CPU, the setting of inexpensive CCTV recorders without a GPU, MotionLite
needs {ml.cpu_ms:.0f} ms per clip against {vm.cpu_ms:.0f} ms for VideoMAE ({vm.cpu_ms / ml.cpu_ms:.0f}× faster) with
{vm.params_m / ml.params_m:.0f}× fewer parameters, at a cost of about {ft('p2_all') - val(g, 'p2_all', 'motionlite'):.2f}
pooled ROC-AUC (Figure 3). Distillation from VideoMAE changed MotionLite's ROC-AUC by {gains}, each a
mean over two training runs. The gains on unseen AIRTLab and on RWF-2000 are larger than the spread
between runs; on the other protocols the change is within that spread. On unseen AIRTLab the distilled
model ({f3(val(g, 'p3_no_avd', 'motionlite_kd'))}) also scores above the fine-tuned VideoMAE ({f3(ft('p3_no_avd'))}) with
{vm.params_m / ml.params_m:.0f}× fewer parameters.</p>
{table(['Model', 'Params (M)', 'GFLOPs', 'GPU ms', 'CPU ms', 'Pooled ROC-AUC'], rows,
       'Table 5. Cost per 16-frame clip (CLIP: 8 frames). Partial fine-tuning does not change VideoMAE inference cost.')}
{figure(FIGS / 'fig_tradeoff_p2_all_cpu.png', 'Figure 3. Pooled test ROC-AUC against CPU latency per clip (log scale).', 14)}
<h3>4.5 Live monitoring prototype</h3>
<p>To check that the lightweight model works on a stream, the distilled MotionLite trained on the pooled
protocol runs in a local web application that reads a webcam, an RTSP camera or a recording played back
in real time. Frames go through the same test-time preprocessing as in evaluation (15 fps, short side 256,
centre crop 224). Every second the model scores the last 4 seconds (16 frames), and an alert with a
snapshot is raised when 3 of the last 5 scores reach the threshold chosen on validation data, with at most
one alert per camera every 20 seconds. On a 30-second sequence assembled from held-out RWF-2000 test
clips (alternating non-violent and violent scenes), non-violent scenes scored about 0.05–0.08 and violent
scenes 0.92–0.96, and the alert fired about 5 seconds after the violent scene began; one forward pass took
about 13 ms on the laptop GPU. This is a functional check, not an evaluation: false alarms per hour on
long, continuous footage remain to be measured.</p>
""")

    S.append(f"""
<h2>5. Discussion and limitations</h2>
<p><b>Why large pretrained models generalise better.</b> VideoMAE and CLIP were trained on far more and
more varied video and images than our 1,728 pooled training clips; their features change less across
camera, resolution and staging differences. Fine-tuning only the last blocks adapts VideoMAE to the
task without discarding that prior.</p>
<p><b>Pretraining overlap.</b> Kinetics-400 and the violence datasets are both drawn largely from
YouTube. We cannot rule out that some source videos appear in both, which would favour the
Kinetics-pretrained models; a similar effect is documented for zero-shot action recognition
{c('gowda2021truezeroshot')}.</p>
<p><b>Small test sets and runs.</b> The SFD and AIRTLab test sets have only 47 and 52 clips. The fine-tuned
VideoMAE has three training runs and both MotionLite variants have two; MobileNetV3-TSM and the
deterministic frozen probes have one.</p>
<p><b>Thresholds under shift.</b> A threshold tuned on the training domains transfers poorly to a new
dataset even when ranking quality (ROC-AUC) holds; deployments should recalibrate on local footage.</p>

<h2>6. Conclusion</h2>
<p>With leakage-controlled splits, a partially fine-tuned VideoMAE detects violence with ROC-AUC
{f3(ft('p2_all'))} on three pooled datasets and {f3(ft('p3_no_rwf'))} on RWF-2000 without training on it, while
small fine-tuned networks lose much more on unseen datasets. Lightweight models remain an order of
magnitude cheaper on a CPU, which keeps them relevant for edge deployment. On these datasets the
measurable problem is domain shift rather than split leakage, and evaluation should include a dataset
the model has never seen. Next steps are deploying the distilled model on edge hardware next to a
camera and measuring false alarms on long, continuous footage.</p>
""")

    refs = c.refs
    S.append("<h2>References</h2><ol class=refs>" + "".join(
        f"<li>{html.escape(refs[k])}</li>" for k in c.order) + "</ol>")

    css = """
body{font-family:'Times New Roman',serif;font-size:11pt;line-height:1.35;max-width:17cm;margin:auto}
h1{font-size:16pt;text-align:center;margin-bottom:4pt} h2{font-size:13pt;margin-top:14pt}
h3{font-size:11.5pt;font-style:italic} p.au{text-align:center} p{text-align:justify}
table{border-collapse:collapse;margin:4pt auto 10pt auto;font-size:9.5pt}
th,td{border-top:1px solid #888;border-bottom:1px solid #888;padding:2pt 6pt;text-align:center}
th{background:#eef1f6} td:first-child,th:first-child{text-align:left} td.b{font-weight:bold}
p.cap{font-size:9.5pt;text-align:center;margin:2pt 0 6pt 0} p.fig{text-align:center;margin:8pt 0 0 0}
ol.refs li{font-size:9.5pt;margin-bottom:2pt}"""
    doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Paper draft</title><style>{css}</style></head><body>{''.join(S)}</body></html>"
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    to_docx(OUT_HTML, OUT_DOCX)


def to_docx(src: Path, dst: Path):
    import win32com.client as w
    word = w.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(str(src), ReadOnly=True)
    for i in range(doc.InlineShapes.Count, 0, -1):   # embed linked pictures
        sh = doc.InlineShapes(i)
        try:
            if sh.LinkFormat is not None:
                sh.LinkFormat.SavePictureWithDocument = True
                sh.LinkFormat.BreakLink()
        except Exception:
            pass
    for t in doc.Tables:   # keep each table on one page with its caption; first column left
        t.Range.ParagraphFormat.KeepWithNext = True
        t.Rows(t.Rows.Count).Range.ParagraphFormat.KeepWithNext = False
        t.Rows.AllowBreakAcrossPages = False
        for r in range(1, t.Rows.Count + 1):
            t.Cell(r, 1).Range.ParagraphFormat.Alignment = 0
        prev = t.Range.Previous(4, 1)   # wdParagraph: the caption above the table
        if prev is not None:
            prev.ParagraphFormat.KeepWithNext = True
    doc.ActiveWindow.View.Type = 3   # print layout
    if dst.exists():
        dst.unlink()
    doc.SaveAs2(str(dst), FileFormat=16)
    pdf = dst.with_suffix(".pdf")
    if pdf.exists():
        pdf.unlink()
    doc.SaveAs2(str(pdf), FileFormat=17)   # wdFormatPDF
    doc.Close(False)
    word.Quit()
    print(f"wrote {dst}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    build()
