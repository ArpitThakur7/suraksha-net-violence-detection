# Suraksha-Net: leakage-free violence detection in surveillance video

Binary violence detection (violent / non-violent clips) evaluated with group-aware, leakage-free
splits on three public datasets, including cross-dataset tests and a study of how much leaky splits
inflate reported accuracy. Design: [docs/superpowers/specs](docs/superpowers/specs/2026-11-22-violence-pipeline-design.md),
implementation plan: [docs/superpowers/plans](docs/superpowers/plans/2026-11-22-violence-pipeline.md).

## Layout

| Path | Contents |
|---|---|
| `pipeline/` | The research pipeline: audit, splits, frame cache, models, trainer, evaluation, statistics, experiment grid (`python -m pipeline <command>`) |
| `pipeline/configs/` | `base.yaml` (all settings), `grid.yaml` (full experiment grid), `grid_day.yaml` (one-day subset) |
| `work/manifest/` | Audit of every clip: SHA-256, metadata, near-duplicates, source groups |
| `work/splits/` | Locked protocol splits (`LOCK.json` holds their hashes) |
| `work/runs/` | Training runs and test metrics (not in git) |
| `work/bench/` | GPU memory and epoch-timing pilots |
| `colab/` | Colab worker notebooks for the experiment grid |
| `demo_app/` | Local web demo: upload a video, see flagged intervals (FastAPI) |
| `datasets/public/` | The three datasets (not in git) |
| `pretrained_weights/` | Cached ImageNet / Kinetics-400 backbone weights (not in git) |

## Protocols

- **P1** within-dataset: `p1_rwf`, `p1_sfd`, `p1_avd`
- **P2** all three datasets pooled: `p2_all`
- **P3** leave-one-dataset-out (train on two, test on the third): `p3_no_rwf`, `p3_no_sfd`, `p3_no_avd`
- **P4** bias study: random vs group-aware splits (`p4_random`, `p4_grouped`, `p2_all_leaky`)

## Running

```powershell
python -m pytest                                   # test suite
python -m pipeline grid --dry-run                  # experiment list and estimated GPU hours
python -m pipeline grid --grid pipeline/configs/grid_day.yaml   # train + evaluate
python -m pipeline evaluate --all                  # (re)evaluate finished runs
```

## Demo app

```powershell
python -m uvicorn demo_app.app:app --host 127.0.0.1 --port 8741
```

Open http://127.0.0.1:8741, upload an MP4 and adjust the threshold. Recordings and review
decisions are stored in `demo_app/data/`. The detector is the baseline ResNet18-LSTM checkpoint in
`demo_app/models/` (4-second windows, 2-second stride, 16 frames at 224 px). Extra packages:
`demo_app/requirements.txt`.

## Datasets

- **RWF-2000**: M. Cheng, K. Cai and M. Li, "RWF-2000: An Open Large Scale Video Database for
  Violence Detection", ICPR 2020. Not redistributable without the authors' approval.
- **Surveillance Camera Fight Dataset (SFD)**: Ş. Aktı, G. A. Tataroğlu and H. K. Ekenel,
  "Vision-based Fight Detection from Surveillance Cameras", IPTA 2019.
  [GitHub](https://github.com/seymanurakti/fight-detection-surv-dataset)
- **AIRTLab dataset (AVD)**: M. Bianculli et al., "A dataset for automatic violence detection in
  videos", Data in Brief 33 (2020).
  [GitHub](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)

## Pretrained weights

All models start from public pretrained weights and are fine-tuned or probed on the datasets
above: ImageNet (MobileNetV3, ResNet, EfficientNet backbones) and Kinetics-400 (R(2+1)D, MViT) from
torchvision. Any additional pretrained models are listed with their sources in the paper.

## Credits

Project team: Arpit Thakur and Vishnu. The baseline ResNet18-LSTM checkpoint used by the demo was
trained by Vishnu in the semester phase of the project. Earlier semester material is archived
outside this repository.
