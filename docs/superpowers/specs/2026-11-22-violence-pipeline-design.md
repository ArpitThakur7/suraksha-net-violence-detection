# Suraksha-Net v2 — End-to-end research pipeline (design)

Date: 2026-11-22 · Status: draft for approval

## 1. Intent

Build one reproducible pipeline, from raw video to paper-ready tables, for CCTV violence detection on
RWF-2000, SFD (Surveillance Fight Dataset) and AIRTLab. The goal is a publishable paper. The owner has no
fixed venue or deadline. The target is an IEEE / Scopus-indexed conference or journal, so the
contribution has to survive a reviewer who checks for leakage and for test-set tuning.

The earlier work is not paper-grade. It used frozen 112 px ImageNet features and never trained on RWF.
Vishnu's reported 84% picked its threshold on the test set. This pipeline replaces that work. It keeps
the existing demo app (`suraksha/`) as the deployment target.

## 2. Paper shape (chosen approach)

**Approaches considered**

| | Approach | Trade-off |
|---|---|---|
| A | Benchmark-only: leakage-free protocol and cross-dataset tests of existing models | Low risk, but the contribution is evaluation only, which reviewers often call incremental |
| B | New-model-only: claim state of the art on RWF-2000 | Published RWF results are about 90%+. Hard to beat on a 6 GB laptop GPU, and fails if we don't win |
| **C (chosen)** | **Rigorous unified protocol + one lightweight motion-aware model + deployment analysis** | Several independent contributions; still publishable if the model doesn't top in-domain accuracy |

Chosen by Claude (the owner handed over the how). The paper's planned contributions:

1. **Leakage audit and unified protocol** across RWF-2000, SFD and AIRTLab:
   - group splits by source video, including groups that span datasets
   - fixed validation sets for every tuning decision
   - release the split manifests
2. **Quantified evaluation bias:** how much accuracy is inflated by
   - random clip splits versus grouped splits
   - thresholds tuned on test versus on validation
   - cross-dataset source overlap
3. **A lightweight motion-aware model**, called *MotionLite* in this doc (name to be decided):
   - a 2D efficient backbone with an RGB stream plus a cheap temporal-difference stream
   - temporal shift and attention pooling over time
   - each part tested by removing it
4. **Generalization and deployment analysis:**
   - leave-one-dataset-out tests
   - calibration
   - latency and size on laptop CPU and GPU, plus FP16, INT8 and ONNX versions
   - window-level false-alarm rate on long non-violent footage

**Revision after the literature check (2026-11-23, see `docs/literature.md`):**
- Lead with contribution 2, then 1. Contribution 2 has no prior quantification we could find.
- Contribution 1 claims only:
  - the source-level RWF-2000 ↔ SFD overlap, which we found unreported
  - a grouped protocol spanning all three datasets
  - the released manifests

  It does not claim that "leakage exists": Cheng 2020 and Veltmeijer 2024 already discuss same-source
  leakage.
- Contribution 3 is **not** a new architecture. Its closest prior work is Kang et al. 2021
  (MobileNetV3/EfficientNet-B0 + frame-difference attention + temporal attention). Others cover TSM
  (Zhang 2022) and difference streams (Islam 2021). Describe MotionLite as a "fully ablated lightweight
  model evaluated under a leakage-controlled protocol", and cite Kang 2021 as its basis. Re-evaluating
  that family under the clean protocol is itself useful evidence.
- Contribution 4 claims the coverage that is new:
  - SFD and AIRTLab in leave-one-dataset-out tests
  - overlap-removed splits
  - calibration, false alarms per hour and measured latency

  Cross-dataset testing as such is already published.
- Reconcile with FuseMamba-VD (Senadeera 2025), which found no SFD-test near-duplicates against RWF
  training clips. Shared YouTube IDs may be different moments of the same video. Task 4's frame-hash
  check must report how many shared-source clip pairs are visual near-duplicates versus only the same
  source. The paper reports both numbers and treats same-source as leakage of scene, people and
  camera, not of identical frames.
- Do not quote "about 90%+" as the bar. Lightweight models under stated protocols report about 82–90%
  on RWF.

Anti-goals:
- claiming state of the art without matched protocols
- any number tuned on test data
- results from a single seed
- novelty claims made before checking recent literature (literature check: section 9)

## 3. Facts established during design (2026-11-22)

- **RWF-2000** is the zip at `F:\archive (3).zip`, 12.3 GB, not yet extracted.
  - Contents: 2,000 `.avi` clips in `RWF-2000/{train,val}/{Fight,NonFight}`, 800/800/200/200.
  - File name pattern: `<youtubeID>_<n>.avi`.
  - 765 source IDs in train and 180 in val, with zero overlap between them.
  - **150 source IDs contain both Fight and NonFight clips**, so splits must group by source ID, not by
    source and label.
  - The `RWF 2000` folders on the Desktop (a junction to a missing path), on `F:` and under
    `datasets/public` are empty.
- **Cross-dataset overlap:**
  - 57 RWF clips (40 train, 17 val) come from 10 YouTube IDs that are also SFD source groups.
  - SFD `event_group` values are YouTube IDs, for example `sfd:7q6KJoSgxmY`.
- **SFD:**
  - 300 clips, 1.0–4.7 s (median 2.0 s), mostly 30 fps (some 25 and 20)
  - grouped splits already in `handoff/` (208 train / 45 validation / 47 test)
- **AIRTLab:**
  - 350 clips, 2.5–14.5 s, 30 fps
  - multi-camera views of the same event grouped in `handoff/` (246/52/52)
- **Hardware:** RTX 4050 Laptop (6 GB), CUDA working, PyTorch 2.6.0+cu124, torchvision 0.21 in
  `.ml_deps`, PyAV 16. Disk space: `E:` has 159 GB free, `C:` has 13 GB free.
- **Compute plan:** hybrid. Decode and cache on the PC. Train on the PC or on a Colab GPU (T4, or A100
  if Pro), not a TPU.

## 4. Pipeline stages

The pipeline is a new package `pipeline/` with a single command-line entry point:
`python -m pipeline <stage> --config ...`. The old `semester_ml/` stays as an untouched reference.
Every stage reads and writes files under `work/`, can be re-run safely, and records its config and the
code version.

### Stage 0 — Project hygiene
- `git init` the project and add a `.gitignore` for data, caches and weights.
- Pin package versions in `requirements.txt`.
- Configs are YAML files in `pipeline/configs/` (one base file plus per-experiment overrides).
- Each run directory stores its resolved config, git commit, package versions, seed and hardware info.

### Stage 1 — Ingest and audit
- Extract RWF to `datasets/public/rwf2000/` (on `E:`). Never write data to `C:`.
- Build `work/manifest/all_clips.csv` with these columns:
  - path, dataset, label, source_id, original_split
  - sha256, fps, decoded frame count, width, height, duration
  - decode errors, exclusion reason
- Decode every clip fully. Exclude clips that are corrupt or empty, and record why.
- Duplicate checks:
  - exact duplicates by SHA-256
  - **near-duplicates** by perceptual hashes (pHash) of 8 evenly spaced frames, compared across
    datasets
  - flagged pairs go to a CSV and a contact-sheet image for a human to check
- **Source IDs:**
  - RWF: the file name without its `_<n>` suffix
  - SFD: its existing YouTube-ID group
  - AIRTLab: the existing scene group
  - Then a **union-find over shared YouTube IDs and confirmed near-duplicates** merges groups across
    datasets into `global_group`.
- Output: `work/manifest/audit_report.md` with counts, exclusions, overlaps and duration histograms.

### Stage 2 — Splits (frozen once created)
- **RWF:**
  - Official `val` becomes the RWF **test** set, so results compare with the literature. Report both:
    - "official" (all 400 clips)
    - "clean" (minus clips whose `global_group` also appears in any training set used)
  - Validation is 15% of RWF train, split by `global_group` and stratified by label.
- **SFD and AIRTLab:** keep the `handoff/` group splits. Then apply the global-group rule: any clip whose
  global group touches another dataset's test set moves out of training.
- Each protocol gets a split manifest CSV. A test checks that no `global_group` appears in more than one
  of {train, val, test}.
- Protocols:
  - **P1 in-domain:** train and test within each dataset (3 settings)
  - **P2 combined:** train on all three, test per dataset and pooled
  - **P3 leave-one-dataset-out (LODO):** train on two, test on the third (3 settings)
  - **P4 bias study** (RWF train only): a random clip-level split versus a grouped split with the same
    model and budget. Also measures test-tuned versus validation-tuned thresholds on the same
    predictions.
- After creation, split files are hashed into `work/splits/LOCK.json`. Training refuses to run if the
  hashes change.

### Stage 3 — Frame cache
- Decode each clip once and resample to **15 fps**, keeping every frame for clips shorter than 16
  frames at 15 fps.
- Resize so the short side is **256 px**. Store as JPEG (quality 90) in one `.npz` per clip.
- Expected size is about 3 GB in total, small enough to upload to Google Drive for Colab.
- Parallel decode with a bounded queue. The cache key is sha256 plus cache parameters, so changed
  parameters write a new cache instead of silently mixing.

### Stage 4 — Data loading and augmentation
- Clip sampling:
  - **Training:** a random temporal segment covering 60–100% of the clip, resampled to T frames
    (T = 16 by default)
  - **Evaluation:** uniform over the whole clip, plus an optional 3-view temporal average
    (reported separately)
- Spatial augmentation:
  - **Training:** random resized crop to 224 (scale 0.6–1.0), horizontal flip, mild color jitter, and
    random grayscale at p = 0.1 (some CCTV is monochrome)
  - **Evaluation:** center crop to 224
- Mixed batches in P2 and P3 balance datasets by sampling weights, so RWF doesn't drown out SFD.

### Stage 5 — Models (all trained end-to-end, every model gets the same budget)

| ID | Model | Pretraining | Role |
|---|---|---|---|
| M1 | ResNet18 + LSTM (Vishnu's architecture) | ImageNet | continuity with the team's prior work |
| M2 | MobileNetV3-L + TSM, average pooling | ImageNet | efficient 2D baseline |
| M3 | R(2+1)D-18 (torchvision) | Kinetics-400 | 3D CNN baseline |
| M4 | MViTv2-S (torchvision) | Kinetics-400 | transformer baseline (GPU-heavy; run on Colab if VRAM is short) |
| **M5** | **MotionLite (proposed)** | ImageNet | contribution |
| M6 (optional) | VideoMAE-B fine-tuned | Kinetics-400 | strong upper bound; only on a Colab A100 |

**MotionLite design:**
- A MobileNetV3-L or EfficientNet-B0 backbone (pick on validation) with temporal shift modules in the
  inverted-residual blocks.
- Input has two parts:
  - the RGB frames
  - **temporal differences** |f_t − f_{t−1}|, computed on the GPU from the same frames, so it adds no
    decode cost
- The two streams merge in the middle of the network using a learned gate, so the backbone is shared up
  to that point.
- Temporal attention pooling feeds a 2-class head.
- **Ablations** (P2 plus one LODO setting):
  - without the difference stream
  - without TSM
  - average pooling instead of attention
  - early versus mid fusion
  - T = 8 / 16 / 32
  - input 160 versus 224
- The novelty claim depends on the literature check in section 9. Temporal-difference networks (TDN)
  and TSM exist, so the claim is the **combination tuned for CCTV deployment plus its measured
  generalization**, not the individual blocks.

### Stage 6 — Training
- AdamW with cosine schedule and warmup, mixed precision (AMP), gradient accumulation to an effective
  batch of 32, label smoothing 0.1, and gradient clipping.
- The learning rate is the only hyperparameter tuned per model: 3 values, chosen on validation.
- Model selection: best validation ROC-AUC (ties broken by lower validation loss), with early stopping
  (patience 6, at most 40 epochs).
- Threshold: chosen on validation to maximise F1, then frozen. Also report results at 0.5.
- **Seeds:**
  - 3 seeds for every reported cell
  - 5 seeds for the headline comparison (M3, M4 and M5 on P2 and P3)
  - Report mean ± std.
- Checkpoint every epoch, with optimizer, scheduler, RNG and scaler state. A disconnected Colab session
  resumes exactly where it stopped.
- The test set is evaluated only by `evaluate`, and only after a model is frozen. Training code never
  loads test splits (enforced in the data layer).
- Colab: a thin notebook `colab/train.ipynb` that
  - mounts Drive
  - copies the frame cache to local disk
  - installs the pinned requirements
  - calls the same `python -m pipeline train`

### Stage 7 — Evaluation and statistics
- **Clip-level metrics:**
  - accuracy, precision, recall, F1
  - ROC-AUC, PR-AUC
  - specificity
  - ECE calibration plus reliability diagrams
  - all per dataset and pooled
- **Uncertainty:** 1,000-sample bootstrap 95% confidence intervals, resampled by group (not by clip).
- **Significance:**
  - McNemar's test between M5 and each baseline on the same test clips
  - paired seed-wise comparison for the AUC difference
- **Cross-dataset matrix:** train source × test source, for the P3 heat-map figure.
- **Bias study (P4):** the accuracy gap for a random versus grouped split, and for test-tuned versus
  validation-tuned thresholds, with confidence intervals.
- **Deployment:**
  - parameters, GFLOPs per clip
  - latency (median and P95) on laptop CPU (4 threads) and RTX 4050, in FP32, FP16, ONNX and INT8
  - decisions changed relative to FP32
- **Window-level false alarms:**
  - run the existing sliding-window logic from `suraksha/` over concatenated non-violent test footage
  - report false alarms per hour, with and without the existing temporal smoothing rule
- **Explainability:** Grad-CAM for the 2D models on correct and incorrect examples, plus a categorised
  error analysis (crowding, low light, camera shake, sports or hugging).

### Stage 8 — Reporting and release
- `python -m pipeline report` regenerates every table (CSV and LaTeX) and figure (PDF and PNG) from run
  directories. The paper never contains hand-typed numbers.
- Paper figures:
  - pipeline diagram
  - leakage illustration
  - cross-dataset heat-map
  - accuracy-versus-GFLOPs and accuracy-versus-latency scatter plots
  - reliability diagram
  - Grad-CAM panel
- Release bundle:
  - split manifests and the audit report
  - code
  - configs
  - checkpoints of the final models
  - a model card
- The best deployable model is exported in the existing `release/` format, so `suraksha/app.py` serves
  it via `SURAKSHA_MODEL_RELEASE`.

## 5. Directory layout

```
pipeline/
  __main__.py          # CLI: audit | split | cache | train | evaluate | bias | bench | report
  configs/             # base.yaml, models/*.yaml, protocols/*.yaml
  data/                # ingest.py, audit.py, groups.py, splits.py, cache.py, dataset.py, transforms.py
  models/              # registry.py, cnn_lstm.py, tsm.py, r2plus1d.py, mvit.py, motionlite.py
  train.py  evaluate.py  stats.py  bench.py  report.py
  tests/               # split-leakage, cache round-trip, model forward shapes, resume determinism
colab/train.ipynb
work/                  # manifests, splits, cache, runs (git-ignored except manifests/splits)
paper/                 # LaTeX (IEEE template), generated tables/figures
```

## 6. Compute budget (estimate, to refine after a timing pilot)
- Cache build: about 45–60 min on the PC.
- Each cell is one model × one protocol setting × one seed.
  - 2D models on RWF-sized data: about 1–2 min/epoch on the 4050, so about 0.5–1 h per cell
  - R(2+1)D: about 2× that
  - MViTv2-S: about 4× that, better on Colab
- Core grid:
  - full grid for M2, M3 and M5: 7 settings × 3 seeds
  - M1 and M4 on P2 + P3 only
  - plus ablations and the bias study
- That is about 120 cells, roughly 60–100 GPU-hours. Running the PC and Colab in parallel brings this
  to about 1–2 weeks of wall-clock time. A timing pilot at the start sets the final grid. If the budget
  is tight, cut M4 seeds first, then M1.

## 7. Constraints
- No raw data, cache or checkpoints on `C:` (13 GB free).
- Only the free or Pro Colab GPU is available. The TPU path is out of scope.
- All datasets keep their publishers' licences. The release includes manifests and code, not videos.

## 8. Calls made by Claude (owner can override)
- Paper shape C, with model list M1–M5 (M6 optional).
- RWF official `val` is used as test and reported both "official" and "clean".
- 15 fps and 256 px short-side cache. T = 16 and 224 px by default.
- Validation ROC-AUC for model selection, validation-F1 for the threshold.
- Seed counts (3, with 5 for the headline comparison).
- Window false alarms measured on concatenated test negatives. No new dataset download by default.
- Name "MotionLite" is a placeholder.

## 9. Left to the builder, or to resolve early
- **Literature check (first task):**
  - collect 2023–2026 RWF-2000, SFD and AIRTLab results and their protocols
  - confirm the positioning of MotionLite against TDN, TSM, MoViNet, X3D, and CNN-LSTM variants
  - check whether the RWF/SFD overlap has been reported before
  - adjust the contribution claims accordingly
- Exact TSM insertion points and the shift ratio.
- Whether optical flow (for example RAFT-small, cached) is worth adding as an extra ablation. Off by
  default because of its cost.
- Optional long-footage source for false alarms per hour (for example normal videos from UCF-Crime, or
  campus recordings). This needs the owner's go-ahead because it means a large download or new
  recordings.
