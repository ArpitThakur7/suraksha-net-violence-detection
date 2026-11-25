# Suraksha-Net v2 Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, leakage-audited pipeline from raw RWF-2000 / SFD / AIRTLab video to paper-ready tables and figures, training baselines and the proposed MotionLite model.

**Architecture:** A new Python package `pipeline/` with one CLI (`python -m pipeline <stage>`). Stages communicate only through files under `work/` (manifests → splits → frame cache → run directories → reports), so each stage can be re-run on its own and training can move between the PC and Colab. `semester_ml/` and `suraksha/` stay untouched, except that the final export writes the release format `suraksha` already loads.

**Tech Stack:** Python 3.13, PyTorch 2.6 + CUDA 12.4, torchvision 0.21 (in `.ml_deps`, added to `sys.path` the same way `semester_ml/core.py` does), PyAV 16, NumPy, pandas, scikit-learn, SciPy, matplotlib, PyYAML, pytest, onnx/onnxruntime.

**Spec:** `docs/superpowers/specs/2026-11-22-violence-pipeline-design.md`

## Global Constraints

- Never write datasets, caches, runs or checkpoints to `C:`. Everything goes under `E:\Projects\major\work\` or `datasets\`.
- RWF root: `datasets/public/rwf2000/RWF-2000/{train,val}/{Fight,NonFight}/*.avi`.
- Label convention: 0 = non-violent, 1 = violent. Model outputs logits `[B,2]` in the order [NonViolence, Violence].
- Model input: float tensor `[B, T, 3, H, W]`, ImageNet mean/std normalised (Kinetics models use Kinetics mean `[0.43216,0.394666,0.37645]` and std `[0.22803,0.22145,0.216989]`; the model registry exposes the correct one).
- Cache: 15 fps, short side 256 px, JPEG quality 90. Default T = 16 and crop 224.
- Splits: RWF official `val` is used as the test set. RWF validation is 15% of RWF train by `global_group`, stratified by label, seed 1337. SFD and AIRTLab use `handoff/{train,validation,test}.csv`.
- No `global_group` may appear in more than one of {train, val, test} within a protocol.
- Training code may never load test rows. Only `evaluate` passes `allow_test=True`.
- Model selection uses best validation ROC-AUC (ties go to lower validation loss), patience 6, max 40 epochs. The threshold is the value that maximises validation F1 (searched over a 0.05–0.95 grid in steps of 0.01), then frozen.
- Seeds: 3 per reported cell (0, 1, 2), plus seeds 3 and 4 for the headline cells (M3, M4, M5 on P2 and P3).
- Every run directory records: resolved config, git commit, `pip freeze` hash, seed, GPU name.
- No new heavy dependencies. pHash is implemented with `scipy.fft.dct`, not `imagehash`.

## Review Focus

1. **Clips with fewer than T cached frames** (SFD 1.0 s clips give 15 frames at 15 fps): sampling must repeat the last frame, never crash or return a short tensor.
2. **Corrupt or zero-frame videos**: the audit records them with `exclusion`. Cache and loaders skip excluded rows and never raise mid-epoch.
3. **Colab disconnect mid-epoch**: resuming from `last.pt` must give the same weights as an uninterrupted run (same seed, deterministic mode).
4. **Re-running a stage after changing cache parameters**: the cache key changes, so no silent mixing of old and new frames.
5. **Grayscale or odd-sized frames** (some CCTV is 1-channel, and some RWF clips aren't 320×240): decoding always yields RGB uint8 and the resize keeps the short side at 256.

---

## File map

```
pipeline/__init__.py, __main__.py   CLI dispatch
pipeline/paths.py                   ROOT, WORK, DATASETS constants; .ml_deps sys.path insert
pipeline/config.py                  YAML load with `extends` + dotted overrides
pipeline/configs/base.yaml          all defaults from Global Constraints
pipeline/configs/grid.yaml          experiment grid (cells)
pipeline/data/video.py              decode_frames(), probe_video()
pipeline/data/ingest.py             build_manifest(): inventory + audit of 3 datasets
pipeline/data/groups.py             phash, near-duplicate pairs, global groups
pipeline/data/splits.py             base splits, protocol manifests, leak check, lock
pipeline/data/cache.py              JPEG frame cache write/read
pipeline/data/dataset.py            ClipDataset, sampling, transforms, loaders
pipeline/models/__init__.py         build_model(name, **kw), MODEL_NORM
pipeline/models/{cnn_lstm,tsm,video3d,motionlite}.py
pipeline/train.py                   train_run()
pipeline/evaluate.py                evaluate_run()
pipeline/stats.py                   metrics, ECE, grouped bootstrap, McNemar
pipeline/bias.py                    P4 study
pipeline/bench.py                   params, GFLOPs, latency, exports, window false alarms
pipeline/grid.py                    run grid cells, skip finished ones, sharding
pipeline/report.py                  tables (CSV + LaTeX), figures, release export
pipeline/tests/                     pytest suite (conftest makes tiny synthetic videos)
colab/train.ipynb
docs/literature.md
paper/
```

---

### Task 1: Project scaffold, config and test harness

**Files:**
- Create: `pipeline/__init__.py`, `pipeline/__main__.py`, `pipeline/paths.py`, `pipeline/config.py`, `pipeline/configs/base.yaml`, `pipeline/tests/conftest.py`, `pipeline/tests/test_config.py`, `pytest.ini`
- Modify: `.gitignore` (add `work/cache/`, `work/runs/`, `*.npz`, `*.pt`, `*.onnx`. Keep `*.pth` and `*.zip`, which are already listed.)

**Interfaces:**
- Produces:
  - `paths.ROOT: Path`, `paths.WORK: Path`, `paths.DATASETS: Path`
  - `load_config(path: str | Path = "pipeline/configs/base.yaml", overrides: Sequence[str] = ()) -> dict` (overrides look like `"train.lr=0.0003"`; values parsed with `yaml.safe_load`)
  - `run_metadata() -> dict` with keys `git_commit, pip_hash, gpu, torch, python, time`
- Produces (conftest fixture): `tiny_videos(tmp_path) -> dict[str, Path]` writes 3 synthetic videos with PyAV:
  - `"normal"`: 40 frames at 30 fps, 320×240, RGB
  - `"short"`: 12 frames at 30 fps
  - `"gray"`: 30 frames at 25 fps, 180×120, gray-encoded

- [ ] **Step 1: `git init`, then write `test_config.py`**
  - `test_extends_and_override`: a child yaml with `extends: base.yaml` and override `train.lr=0.5` gives `cfg["train"]["lr"] == 0.5`, and an untouched key equals the base value.
  - `test_base_values`: `cfg["cache"] == {"fps": 15, "short_side": 256, "jpeg_quality": 90}`, `cfg["clip"]["T"] == 16`, `cfg["clip"]["crop"] == 224`, `cfg["split"]["val_frac"] == 0.15`, `cfg["split"]["seed"] == 1337`.
- [ ] **Step 2: Run** `python -m pytest pipeline/tests/test_config.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `paths.py`, `config.py` and `base.yaml`, holding every value from Global Constraints. Sections: `paths, cache, clip, split, train, eval, bench`. `__main__.py` dispatches subcommands `audit, split, cache, train, evaluate, grid, bias, bench, report`, each importing its module lazily.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit**: `chore: scaffold pipeline package, config loader, test harness`

### Task 2: Literature check

**Files:** Create `docs/literature.md`

- [ ] **Step 1:** Search the web for 2022–2026 results on RWF-2000, SFD (Aktı et al.) and AIRTLab. For each entry record: paper, venue, protocol (official split?), model, params/GFLOPs, accuracy.
- [ ] **Step 2:** Position MotionLite against TSM, TDN, MoViNet, X3D, CNN-LSTM, and lightweight violence-detection papers. Write the honest novelty claim.
- [ ] **Step 3:** Search for any report of RWF-2000 ↔ SFD source overlap. Record the result.
- [ ] **Step 4:** If the claim needs to change, update spec section 2 and tell the owner.
- [ ] **Step 5: Commit**: `docs: literature review and positioning`

### Task 3: Video decoding and dataset audit (Stage 1)

**Files:**
- Create: `pipeline/data/video.py`, `pipeline/data/ingest.py`, `pipeline/tests/test_ingest.py`

**Interfaces:**
- Produces:
  - `probe_video(path) -> dict(fps: float, frames: int, width: int, height: int, duration_s: float, error: str)`. This decodes the full stream and counts frames. `error` is `""` when the decode succeeds.
  - `decode_frames(path, fps: float | None, short_side: int | None) -> np.ndarray` uint8 `[N,H,W,3]`, RGB. When `fps` is given, it keeps frames nearest to a `fps` grid; if that grid gives fewer than 16 frames, it returns all frames.
  - `rwf_source_id(filename: str) -> str`, which strips a trailing `_<digits>` and the extension, e.g. `"-1l5631l3fg_0.avi" -> "-1l5631l3fg"`.
  - `normalize_source(dataset: str, raw: str) -> str`:
    - RWF IDs and SFD groups that look like YouTube IDs (`^[A-Za-z0-9_-]{11}$` after removing the `sfd:` prefix) → `"yt:<id>"`
    - other SFD groups → `"sfd:<raw>"`
    - AIRTLab groups → `"avd:<raw>"`
  - `build_manifest(cfg) -> pd.DataFrame`, written to `work/manifest/all_clips.csv`. Columns:
    - `clip_id` (first 16 characters of the sha256), `path` (relative to the project), `dataset` (`rwf|sfd|avd`), `label`
    - `source_id`, `original_split` (RWF `train|val`; SFD/AIRTLab copy the handoff split), `sha256`
    - `fps, frames, width, height, duration_s, error, exclusion`
  - SFD and AIRTLab rows come from `handoff/dataset_metadata.csv` (re-probed). RWF rows come from walking the RWF root.
  - Exclusion reasons: `"decode_error"`, `"empty"`, `"duplicate_sha"` (keep the first by sorted path).
  - Uses a thread pool of 6 and prints progress every 200 clips.

- [ ] **Step 1: Write the tests**
  - `test_rwf_source_id`: `"-1l5631l3fg_0.avi"` → `"-1l5631l3fg"`, and `"a_b_c_12.avi"` → `"a_b_c"`.
  - `test_normalize_source`: `("sfd","sfd:7q6KJoSgxmY")` → `"yt:7q6KJoSgxmY"`, `("sfd","sfd:nofight_062-091")` → `"sfd:nofight_062-091"`, `("rwf","0H2s9UJcNJ0")` → `"yt:0H2s9UJcNJ0"`.
  - `test_probe_and_decode(tiny_videos)`: `normal` probes as 40 frames. `decode_frames(normal, fps=15, short_side=256)` has shape `(20, 256, 341, 3)` (±1 frame), dtype uint8. `gray` decodes with 3 channels. `short` decoded at 15 fps returns all 12 frames.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** with PyAV (`frame.to_ndarray(format="rgb24")`). Resize with `PIL.Image.resize(..., BILINEAR)`, short side to 256, keeping the aspect ratio.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Run on real data**: `python -m pipeline audit`. Expected: 2,650 rows; RWF 800/800/200/200; `exclusion` counts printed.
- [ ] **Step 6: Commit**: `feat: video decode and three-dataset audit manifest` (commit the manifest CSV too)

### Task 4: Near-duplicates and global groups

**Files:**
- Create: `pipeline/data/groups.py`, `pipeline/tests/test_groups.py`

**Interfaces:**
- Produces:
  - `phash(img: np.ndarray) -> int`: 64-bit pHash. Grayscale → 32×32 → 2D DCT → top-left 8×8 without the DC term, compared to the median.
  - `clip_signature(path, n=8) -> np.ndarray[uint64]` of shape `(8,)`: pHashes of 8 evenly spaced frames.
  - `near_duplicate_pairs(manifest, max_mean_hamming=10) -> pd.DataFrame[clip_a, clip_b, distance]`. Compares clips from *different* `source_id`s. The distance is the mean Hamming distance over the best frame alignment (each frame of a matched to its nearest frame in b). Uses vectorised XOR+popcount; bucket on the first frame's top 16 bits is allowed as a prefilter only if recall on the test case still holds.
  - `assign_global_groups(manifest, pairs) -> pd.Series` (name `global_group`). Union-find over `source_id`, plus edges from `pairs` that have `confirmed == True`. The group label is the smallest member `source_id`.
  - Writes `work/manifest/near_duplicates.csv` (with a `confirmed` column defaulting to `distance <= 6`) and `work/manifest/near_duplicates_sheet.png`, a contact sheet of the top 40 pairs.
  - Adds `global_group` and `shared_source` (the group spans more than one dataset) to `all_clips.csv`.

- [ ] **Step 1: Write the tests**
  - `test_phash_identical_and_different`: an image and its JPEG re-encode are within ≤ 4 bits; random noise versus a gradient image differ by ≥ 20 bits.
  - `test_union_find`: rows with source_ids `yt:A` (rwf), `yt:A` (sfd), `yt:B`, plus a confirmed pair (yt:B clip, `avd:3` clip) give 2 groups, and `shared_source` is True for all `yt:A` rows.
  - `test_near_dup_detects_reencode(tiny_videos)`: `normal` and a re-encoded copy at a different resolution come out as a pair with distance ≤ 6.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Run on real data** (inside `python -m pipeline audit --groups`). Expected: at least 10 YouTube IDs shared between RWF and SFD, and at least 57 RWF rows with `shared_source`. Open the contact sheet and set `confirmed` by eye for borderline pairs (distance 7–10).
- [ ] **Step 6: Commit**: `feat: perceptual near-duplicate detection and cross-dataset global groups`

### Task 5: Splits, protocols and lock (Stage 2)

**Files:**
- Create: `pipeline/data/splits.py`, `pipeline/tests/test_splits.py`

**Interfaces:**
- Produces:
  - `base_splits(manifest, cfg) -> pd.DataFrame`. Adds `split` ∈ {train, val, test}:
    - RWF `val` → test
    - RWF train → 85/15 train/val by `global_group`, stratified by majority label, seed 1337
    - SFD and AIRTLab → handoff split (`validation` renamed `val`)
    - Excluded rows are dropped.
  - `PROTOCOLS`: dict mapping each protocol name to `(train_sets, test_sets)`:
    - `p1_rwf, p1_sfd, p1_avd`
    - `p2_all`
    - `p3_no_rwf, p3_no_sfd, p3_no_avd` (train and validate on the other two, test on the held-out one)
    - `p4_grouped, p4_random` (RWF train only; see Task 10)
  - `protocol_manifest(base, name) -> pd.DataFrame` with columns `clip_id, path, dataset, label, global_group, shared_source, split`.
    - Train and val rows come from the train datasets. Test rows come from the test datasets' `test` split.
    - Any train/val row whose `global_group` appears in the test rows is dropped and logged.
  - `check_no_leak(m) -> None` raises `LeakError` if a `global_group` appears in more than one split.
  - `write_protocols(cfg)` writes `work/splits/<name>.csv` and `work/splits/LOCK.json` (`{name: sha256 of file}`).
  - `load_protocol(name, cfg, verify=True) -> pd.DataFrame` raises `LockError` on a hash mismatch.
  - CLI: `python -m pipeline split [--force]`. Without `--force`, it refuses to overwrite an existing LOCK.

- [ ] **Step 1: Write the tests** (synthetic manifest of 60 rows across 3 datasets with shared groups)
  - `test_no_leak_all_protocols`: `check_no_leak` passes for every protocol in `PROTOCOLS`.
  - `test_cross_dataset_drop`: an SFD train row sharing `yt:X` with an RWF test row is absent from `p2_all` and `p3_no_rwf`.
  - `test_rwf_val_is_test`: in `p1_rwf`, every RWF `original_split=="val"` row has `split=="test"`.
  - `test_lock_detects_edit`: editing one byte of a written protocol CSV makes `load_protocol` raise `LockError`.
  - `test_val_fraction`: the RWF val share of RWF train groups is within 0.15 ± 0.03.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: Run** `python -m pipeline split`. Print counts per protocol × split × dataset × label to `work/splits/summary.md`.
- [ ] **Step 6: Commit**: `feat: grouped protocols with leak check and split lock` (commit `work/splits/`)

### Task 6: Frame cache (Stage 3)

**Files:**
- Create: `pipeline/data/cache.py`, `pipeline/tests/test_cache.py`

**Interfaces:**
- Produces:
  - `cache_key(cfg) -> str`: first 8 characters of the sha1 of `json.dumps(cfg["cache"], sort_keys=True)`.
  - `cache_dir(cfg) -> Path` = `WORK/"cache"/cache_key(cfg)`.
  - `write_clip(clip_id, frames: np.ndarray, cfg) -> Path` writes `<clip_id>.npz` holding `data` (uint8, concatenated JPEG bytes), `offsets` (int64, N+1) and `shape` (int32 [N,H,W,3]). It writes to `.tmp`, then renames.
  - `read_clip(clip_id, cfg, idx: np.ndarray | None = None) -> np.ndarray` uint8 `[len(idx) or N, H, W, 3]`. It decodes only the requested JPEGs, via `PIL.Image.open(BytesIO(...))`.
  - `num_frames(clip_id, cfg) -> int` reads only `shape`.
  - `build_cache(manifest, cfg, workers=6)` skips clips that already exist and writes `cache_dir/index.csv` (`clip_id, n_frames, h, w`).
  - CLI: `python -m pipeline cache`

- [ ] **Step 1: Write the tests**
  - `test_roundtrip(tiny_videos, tmp cfg)`: for the `normal` clip, `read_clip` shape equals the `decode_frames` shape, and the mean absolute pixel difference is < 6 (JPEG error). Reading with `idx=[0,5]` returns 2 frames.
  - `test_key_changes`: changing `cache.fps` changes `cache_dir`.
  - `test_atomic`: no `.tmp` files remain after `build_cache`.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: Run** `python -m pipeline cache` in the background. Expected: 2,650 minus excluded files, 2–4 GB total. Record wall time in `work/cache/<key>/build.json`.
- [ ] **Step 6: Commit**: `feat: JPEG frame cache`

### Task 7: Dataset, sampling and augmentation (Stage 4)

**Files:**
- Create: `pipeline/data/dataset.py`, `pipeline/tests/test_dataset.py`

**Interfaces:**
- Consumes: `read_clip`, `num_frames`, `load_protocol`, `MODEL_NORM` (from Task 8, passed in as `(mean, std)`).
- Produces:
  - `sample_indices(n: int, T: int, train: bool, rng: np.random.Generator, view: int = 0, views: int = 1) -> np.ndarray[int]` of length T.
    - Training: a random segment covering 60–100% of the clip at a random start; T indices evenly spaced within it.
    - Evaluation: evenly spaced over the whole clip; for `views > 1`, shifted by `view/views` of one step.
    - When n < T, indices are non-decreasing and padded with the last frame.
  - `ClipDataset(manifest, split, cfg, norm, train, allow_test=False)`. `__getitem__` returns `(x: float32 [T,3,crop,crop], y: int, clip_id: str)`. It raises `PermissionError` if `split=="test"` and not `allow_test`.
    - Training transforms are applied identically to all frames of a clip: RandomResizedCrop (scale 0.6–1.0, ratio 3/4–4/3), hflip p = 0.5, ColorJitter (0.2, 0.2, 0.2, 0.02), grayscale p = 0.1.
    - Evaluation: resize short side to 256 (already cached), then center crop.
  - `make_loader(manifest, split, cfg, norm, train, allow_test=False, balance_datasets=False) -> DataLoader`. With `balance_datasets`, a `WeightedRandomSampler` gives each dataset equal total weight. Workers: `cfg["train"]["workers"]` (default 4), `persistent_workers=True`, `pin_memory=True`.

- [ ] **Step 1: Write the tests**
  - `test_sample_short_clip`: `sample_indices(12, 16, train=False, ...)` has length 16, max 11, non-decreasing.
  - `test_sample_train_within_bounds`: over 200 random draws with n = 75, every index is in [0, 74] and each draw spans ≥ 0.6·74 × (T−1)/T.
  - `test_test_guard`: `ClipDataset(m, "test", ..., allow_test=False)` raises `PermissionError`.
  - `test_item_shape` (cached tiny clip): `x.shape == (16,3,224,224)`, dtype float32, finite.
  - `test_balanced_sampler`: with 90 RWF rows and 10 SFD rows, the empirical SFD share over 5,000 draws is within 0.5 ± 0.05.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit**: `feat: clip dataset, temporal sampling, augmentation, balanced loader`

### Task 8: Model zoo (Stage 5)

**Files:**
- Create: `pipeline/models/__init__.py`, `pipeline/models/cnn_lstm.py`, `pipeline/models/tsm.py`, `pipeline/models/video3d.py`, `pipeline/models/motionlite.py`, `pipeline/tests/test_models.py`

**Interfaces:**
- Produces:
  - `build_model(name: str, pretrained: bool = True, **kw) -> nn.Module`. Input `[B,T,3,H,W]` → logits `[B,2]`.
  - `MODEL_NORM: dict[str, tuple[list, list]]`: ImageNet stats for `cnn_lstm, mnv3_tsm, motionlite`; Kinetics stats for `r2plus1d, mvit_s`.
  - Model names:
    - `cnn_lstm`: ResNet18 → LSTM(512→256, 1 layer) → last hidden state → Dropout 0.3 → Linear(256, 2). Matches the semester baseline architecture.
    - `mnv3_tsm`: MobileNetV3-Large with a temporal shift (fold_div = 8) on the input of each inverted-residual block that has a residual connection → global pool → mean over T → Linear(960, 2).
    - `r2plus1d`: torchvision `r2plus1d_18(weights=KINETICS400_V1)` with fc replaced; permutes to `[B,3,T,H,W]`.
    - `mvit_s`: torchvision `mvit_v2_s(weights=KINETICS400_V1)` with the head replaced. Requires T = 16, 224 px.
    - `motionlite`: kwargs `backbone="mnv3"|"effb0"`, `diff=True`, `tsm=True`, `pool="attn"|"avg"`, `fusion="mid"|"early"`.
  - `tsm.py`: `temporal_shift(x: [B*T,C,H,W], T: int, fold_div: int = 8) -> same shape` and `add_tsm(module, T, fold_div) -> module` (wraps qualifying blocks).
  - `motionlite.py` algorithm (the part a builder can't infer):
    ```
    rgb  = frames                                    # [B,T,3,H,W]
    diff = |frames[:,1:] - frames[:,:-1]|, prepend a zero frame → [B,T,3,H,W]
    early fusion: stem input = concat(rgb, diff) on channels (stem conv widened to 6 in;
                  the extra 3 input channels are initialised to the mean of the RGB weights)
    mid fusion:   the stem and first 1/3 of blocks run separately on rgb and diff (diff uses a lighter
                  copy: channel width ×0.5 via a 1x1 projection); then
                  fused = rgb_feat + sigmoid(g) * proj(diff_feat), where g = conv1x1(concat) is a
                  per-pixel gate; the remaining 2/3 of the blocks are shared
    pool:         attn → w_t = softmax_t(Linear(C,1)(feat_t)), clip = Σ w_t feat_t;  avg → mean_t
    head:         Dropout(0.3) → Linear(C, 2)
    diff=False reduces it to mnv3_tsm with attention pooling (ablation)
    ```
- [ ] **Step 1: Write the tests**
  - `test_forward_shapes` (parametrised over all 5 names, `pretrained=False`, input `[2,16,3,224,224]` on CPU, or 112 px where the model allows): output shape `(2,2)`, finite.
  - `test_temporal_shift`: for `x = arange` reshaped `[2*4, 16, 1, 1]` with T = 4, the first `16/8 = 2` channels are shifted one step back in time, the next 2 one step forward, and the rest are unchanged; zero padding at the edges.
  - `test_motionlite_ablations`: every combination of `diff ∈ {T,F}`, `pool ∈ {attn, avg}`, `fusion ∈ {early, mid}` runs forward. The parameter count with `diff=False` is less than with `diff=True`.
  - `test_param_budget`: `motionlite(backbone="mnv3")` has < 8M parameters.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: GPU memory pilot**: a script `python -m pipeline bench --memory` runs one AMP train step per model at batch sizes {2,4,8,16} with T = 16, 224 px, on the 4050. It writes `work/bench/memory.csv` (the largest batch that fits per model), which sets `train.batch` and the gradient-accumulation steps per model in `grid.yaml`.
- [ ] **Step 6: Commit**: `feat: model zoo with TSM and MotionLite`

### Task 9: Trainer (Stage 6)

**Files:**
- Create: `pipeline/train.py`, `pipeline/tests/test_train.py`

**Interfaces:**
- Consumes: `load_protocol`, `make_loader`, `build_model`, `MODEL_NORM`, `stats.clip_metrics`, `stats.best_threshold`.
- Produces:
  - `train_run(cfg, protocol: str, model: str, seed: int, out_dir: Path, model_kw: dict | None = None) -> dict`
  - Writes to `out_dir`:
    - `config.json` (resolved config + `run_metadata()`)
    - `history.csv` (epoch, train_loss, train_acc, val_loss, val_auc, val_f1@0.5, lr, seconds)
    - `last.pt` (model, optimizer, scheduler, scaler, epoch, best, RNG states for python/numpy/torch/cuda, sampler generator state)
    - `best.pt` (model state_dict + `threshold` + `norm` + `model_name` + `model_kw` + `T` + `crop`)
    - `val_preds.csv` (clip_id, dataset, label, prob)
    - `val_summary.json` (best_epoch, val_auc, threshold, val metrics at the threshold)
    - `DONE` marker
  - Returns the `val_summary` dict. If `DONE` exists, it returns the saved summary without training.
  - Training details:
    - AdamW (weight decay 0.05), linear warmup over 2 epochs then cosine
    - AMP via `torch.autocast("cuda", dtype=torch.float16)` + GradScaler
    - grad clipping at 1.0, label smoothing 0.1
    - effective batch 32 = `batch × accum`
    - `torch.backends.cudnn.benchmark=False` and `torch.use_deterministic_algorithms(True, warn_only=True)` when `cfg.train.deterministic`
    - early stopping per Global Constraints
    - balanced datasets for P2 and P3
  - Calls `stats.best_threshold(y, p)` on the validation predictions of the best epoch.
  - CLI: `python -m pipeline train --protocol p1_rwf --model motionlite --seed 0 [--set key=val ...]`
  - The run directory is `WORK/"runs"/protocol/(model + variant suffix)/f"seed{seed}"`. The variant suffix is a sorted `k-v` join of `model_kw` that differ from the defaults, e.g. `motionlite__diff-False`.

- [ ] **Step 1: Write the tests** (tiny synthetic protocol of 8 train, 4 val clips; `cnn_lstm` with `pretrained=False`, T = 4, 64 px, CPU, 3 epochs)
  - `test_runs_and_writes`: every listed file exists, and `history.csv` has ≤ 3 rows.
  - `test_resume_equivalence`: run A trains 3 epochs. Run B trains 2 epochs, is killed (simulated with `max_epochs` = 2 and no `DONE`), then resumes to 3. Their `best.pt` state dicts match within `atol=1e-6`.
  - `test_no_test_rows_loaded`: patch `ClipDataset.__init__` to record splits; "test" never appears.
  - `test_done_skips`: a second call returns immediately (the patched trainer loop is never entered).
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: Smoke run on real data**: `python -m pipeline train --protocol p1_sfd --model mnv3_tsm --seed 0 --set train.max_epochs=2`. Record seconds per epoch.
- [ ] **Step 6: Commit**: `feat: resumable AMP trainer with validation-only selection`

### Task 10: Evaluation, statistics and bias study (Stage 7, part 1)

**Files:**
- Create: `pipeline/stats.py`, `pipeline/evaluate.py`, `pipeline/bias.py`, `pipeline/tests/test_stats.py`

**Interfaces:**
- Produces in `stats.py`:
  - `clip_metrics(y, p, thr) -> dict` with keys `accuracy, precision, recall, specificity, f1, roc_auc, pr_auc, ece, tp, tn, fp, fn, n`
  - `best_threshold(y, p, grid=np.arange(0.05, 0.951, 0.01)) -> float`: maximises F1; ties go to the threshold closest to 0.5
  - `ece(y, p, bins=15) -> float`
  - `group_bootstrap_ci(y, p, groups, metric: Callable[[y,p], float], n=1000, seed=0, alpha=0.05) -> tuple[float, float]`: resamples groups with replacement
  - `mcnemar(y, pred_a, pred_b) -> tuple[float, float]`: exact binomial when the number of discordant pairs is < 25, else chi-square with continuity correction
- Produces in `evaluate.py`:
  - `evaluate_run(run_dir: Path, cfg, views: int = 1) -> dict`. Loads `best.pt`, builds the test loader with `allow_test=True`, and writes `test_preds{_v3}.csv` (`clip_id, dataset, label, prob, pred, global_group, shared_source`) and `test_metrics{_v3}.json`.
  - The metrics JSON has three sections: `pooled`, `by_dataset`, and `by_shared_source` (`true` and `false`). Each holds metrics at the validation threshold and at 0.5, plus bootstrap 95% CIs for accuracy, F1 and AUC.
  - CLI: `python -m pipeline evaluate --run <dir> | --all`
- Produces in `bias.py`, `run_bias(cfg, model="mnv3_tsm", seeds=(0,1,2))`:
  - (a) Builds `p4_random`: a clip-level random 70/15/15 split of RWF train (stratified by label, ignoring groups), and `p4_grouped`: the same proportions grouped by `global_group`. Trains with the same budget and reports the test accuracy gap with a CI.
  - (b) On every existing run's test predictions, compares metrics at the threshold tuned on test against the validation threshold.
  - (c) Trains `p2_all` with the cross-dataset drop rule disabled (`--set split.drop_shared=false`, which writes the variant protocol `p2_all_leaky`). Reports the accuracy gap on `shared_source` test clips.
  - Writes `work/reports/bias.csv` and `bias.md`.

- [ ] **Step 1: Write the tests**
  - `test_metrics_known`: `y=[0,0,1,1]`, `p=[.1,.6,.4,.9]`, `thr=.5` gives tp = 1, fp = 1, fn = 1, tn = 1, accuracy = 0.5, roc_auc = 0.75.
  - `test_best_threshold`: `y=[0,0,1,1]`, `p=[.1,.2,.8,.9]` gives a threshold in (0.2, 0.8], and F1 at that threshold is 1.0.
  - `test_ece_perfect`: y drawn as Bernoulli(p) with n = 20,000 gives ece < 0.02. For constant p = 0.9 with all y = 0, ece is approximately 0.9.
  - `test_group_bootstrap_wider`: when the clips within each group are duplicated 5×, the group-bootstrap CI width is ≥ the clip-bootstrap width.
  - `test_mcnemar`: a = all correct, b = 10 wrong out of 10 discordant pairs gives p < 0.01; identical predictions give p = 1.0.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit**: `feat: evaluation, grouped bootstrap, McNemar, bias study`

### Task 11: Grid runner, Colab notebook and timing pilot

**Files:**
- Create: `pipeline/grid.py`, `pipeline/configs/grid.yaml`, `colab/train.ipynb`, `pipeline/tests/test_grid.py`

**Interfaces:**
- Produces:
  - `expand_grid(path) -> list[Cell]`, where `Cell = dataclass(protocol, model, seed, model_kw: dict, overrides: list[str], priority: int)`.
  - `run_grid(cells, cfg, shard: tuple[int,int] = (0,1), only: str | None = None)`. Skips cells with `DONE`, runs `train_run` then `evaluate_run`. Shard assignment is `hash(cell id) % n == i`, which is stable.
  - CLI: `python -m pipeline grid [--shard 0/2] [--only motionlite] [--dry-run]`. `--dry-run` prints cells and the estimated hours from `work/bench/epoch_seconds.csv`.
- `grid.yaml` contents (from spec section 6):
  - M2 (`mnv3_tsm`), M3 (`r2plus1d`), M5 (`motionlite`) × all 7 protocol settings × seeds 0–2
  - M1 (`cnn_lstm`) and M4 (`mvit_s`) × {p2_all, p3_*} × seeds 0–2
  - seeds 3–4 for M3, M4, M5 on p2_all and p3_*
  - ablations on p2_all and p3_no_rwf × seeds 0–2:
    - `diff=False`, `tsm=False`, `pool=avg`, `fusion=early`
    - `clip.T` ∈ {8, 32}, `clip.crop=160`
  - learning-rate selection, a pre-step on p2_all with seed 0: lr ∈ {1e-4, 3e-4, 1e-3}, picking the best validation AUC per model. Written to `work/runs/lr_choice.json`, which grid cells read.
- `colab/train.ipynb` steps:
  1. Mount Drive.
  2. `rsync` the repo and `work/cache/<key>` from Drive to `/content`.
  3. `pip install -r requirements.txt`.
  4. `python -m pipeline grid --shard 1/2`.
  5. After every cell, copy the run directory back to Drive. Resuming is automatic via `last.pt`.

- [ ] **Step 1: Write the tests**
  - `test_expand_counts`: on `grid.yaml`, assert these counts, with the derivation commented in the test. No duplicate cell IDs.
    - M2: 7 protocols × 3 seeds = 21
    - M3: 21 + 4 headline protocols × 2 extra seeds = 29
    - M5 (non-ablation): 29
    - M1: 4 protocols × 3 seeds = 12
    - M4: 12 + 8 = 20
    - ablations: 7 variants × 2 protocols × 3 seeds = 42
    - learning-rate selection: 5 models × 3 learning rates = 15
    - total: 168
  - `test_shards_partition`: shards 0/3, 1/3 and 2/3 are disjoint and their union is all cells.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: Timing pilot**: 1 epoch of each model on `p2_all` writes `work/bench/epoch_seconds.csv`. `--dry-run` prints total GPU-hours. **Tell the owner the estimate before launching the full grid.** If it exceeds 120 GPU-hours, cut M4 seeds 3–4, then M1 P3 cells.
- [ ] **Step 6: Commit**: `feat: experiment grid runner, sharding, Colab notebook`

### Task 12: Run the experiments

- [ ] **Step 1:** LR selection: `python -m pipeline grid --only lr`.
- [ ] **Step 2:** Launch the local shard in the background: `python -m pipeline grid --shard 0/2`. The owner runs shard 1/2 on Colab (or the PC runs everything if Colab isn't used).
- [ ] **Step 3:** `python -m pipeline bias`.
- [ ] **Step 4:** Check: every `DONE` exists, no NaN histories, validation AUC > 0.5 for every run. Re-run failed cells.
- [ ] **Step 5:** Commit `work/runs/**/val_summary.json`, `test_metrics*.json` and `history.csv` (small files only): `exp: full grid results`.

### Task 13: Deployment benchmarks (Stage 7, part 2)

**Files:**
- Create: `pipeline/bench.py`, `pipeline/tests/test_bench.py`

**Interfaces:**
- Produces:
  - `count_params(model) -> int`
  - `gflops(model, T, crop) -> float`, via `torch.utils.flop_counter.FlopCounterMode` on one clip, counting multiply-adds × 2 / 1e9
  - `latency(model, device, dtype, T, crop, warmup=10, iters=50) -> dict(median_ms, p95_ms)`. CPU uses `torch.set_num_threads(4)`; GPU uses `cuda.synchronize`.
  - `export_onnx(model, path, T, crop)` with input name `frames`, opset 17
  - `int8_dynamic(model)` quantises Linear and LSTM layers
  - `decisions_changed(ref_probs, probs, thr) -> int`
  - `window_false_alarms(run_dir, cfg, window_s=4, stride_s=2, smoothing=("none","5of7")) -> dict`:
    - Concatenates all test-split non-violent clips per dataset in clip_id order into a frame stream at 15 fps (from the cache).
    - Applies sliding windows and counts positive windows / alarms after smoothing.
    - Reports `hours, windows, fp_windows, alarms, alarms_per_hour`.
  - Writes `work/reports/efficiency.csv` and `false_alarms.csv`.
  - CLI: `python -m pipeline bench [--memory]`
- [ ] **Step 1: Write the tests**
  - `test_flops_linear`: a model that is a single `Linear(10,10)` on input `[1,10]` gives ≈ 2e-7 GFLOPs.
  - `test_onnx_parity`: tiny `cnn_lstm` (`pretrained=False`, 64 px, T = 4); onnxruntime and torch logits agree within 1e-4.
  - `test_window_count`: a 60 s stream with window 4 / stride 2 gives 29 windows.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run on the seed-0 `p2_all` model of every model type.
- [ ] **Step 6: Commit**: `feat: efficiency and window-level false-alarm benchmarks`

### Task 14: Reports, figures and release export (Stage 8)

**Files:**
- Create: `pipeline/report.py`, `pipeline/tests/test_report.py`, `paper/tables/`, `paper/figures/`

**Interfaces:**
- Consumes: run directories, `bias.csv`, `efficiency.csv`, `false_alarms.csv`, `audit_report.md`.
- Produces `python -m pipeline report`:
  - `aggregate(runs_root) -> pd.DataFrame`: one row per (protocol, model+variant, test dataset, metric), with mean, std, n_seeds and CI from seed 0.
  - Tables (CSV + LaTeX `booktabs`, best value bolded):
    - `main_results` (P1 + P2)
    - `cross_dataset` (P3)
    - `ablation`
    - `bias`
    - `efficiency`
    - `dataset_stats`
    - `significance` (McNemar p-values: M5 vs each baseline per test set; Holm-corrected)
  - Figures (PDF + PNG, matplotlib, colour-blind-safe palette):
    - `cross_dataset_heatmap`
    - `acc_vs_gflops`, `acc_vs_latency`
    - `reliability` (M5 vs best baseline)
    - `bias_bars`
    - `gradcam_panel` (8 test clips: TP, TN, FP, FN × 2; Grad-CAM on the last shared backbone block of M5, on the frame with the highest attention weight)
    - `pipeline_diagram` (static SVG, drawn once)
  - `export_release(run_dir, dest="work/release")`: writes `best_model.pth`, `model_config.json`, `preprocessing_config.json`, `recommended_threshold.json`, `class_mapping.json` and `REPORT.md`, using the same key names as `ml_runs/two_sources/release/*.json`. If `suraksha/model.py` can't load the new architecture, add a minimal loader branch there (the one allowed edit outside `pipeline/`) and verify with `python semester_ml/infer.py`-style inference on a sample video.
  - `errors.csv` holds every misclassified test clip of M5 (seed 0, p2_all), with a free-text `category` column for manual tagging (crowding, low light, camera shake, sports/hug).
- [ ] **Step 1: Write the tests**
  - `test_aggregate_mean_std`: 3 fake run directories with accuracy 0.8 / 0.9 / 1.0 give mean 0.9 and std 0.1 (ddof = 1).
  - `test_latex_bold_best`: the best value cell in a column is wrapped in `\textbf{}`.
  - `test_release_keys`: exported JSON key sets equal those in `ml_runs/two_sources/release`.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run the report and open every figure to check it by eye.
- [ ] **Step 6: Commit**: `feat: automated tables, figures and deployable release`

### Task 15: Paper draft

**Files:** Create `paper/main.tex` (IEEE conference template, `IEEEtran`), `paper/refs.bib`

- [ ] **Step 1:** Sections: Abstract, Introduction (contributions = spec section 2), Related Work (from `docs/literature.md`), Datasets & Leakage Audit, Protocol, MotionLite, Experimental Setup, Results (P1/P2, P3, ablations, bias, efficiency, false alarms), Discussion & Limitations, Conclusion, Reproducibility statement. Every number comes from `\input{tables/...}`.
- [ ] **Step 2:** Check that every claim in the abstract maps to a table cell. List any claim the results don't support and rewrite it to match.
- [ ] **Step 3: Commit**: `docs: paper draft`
