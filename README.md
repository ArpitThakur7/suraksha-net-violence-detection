# Suraksha-Net laptop demo

The measured semester results, checkpoint provenance, benchmark comparison and demo instructions are in [the semester handoff](submission/SEMESTER_HANDOFF.md). The [submission archive](submission/Suraksha_Semester.zip) contains the laptop code, model weights, results and a recorded demonstration. Use the report's named model and dataset when quoting a metric.

Open http://127.0.0.1:8741 on this laptop. To restart the app, run this command from the project directory:

```powershell
python -m uvicorn suraksha.app:app --host 127.0.0.1 --port 8741
```

The review page accepts an MP4 and an adjustable threshold. It saves the recording, window scores, merged incident intervals and review statuses in `local_data/`. Select a saved recording, click an interval to seek to it, and choose **Reviewed** or **False alarm** to save a decision. H.264 MP4 is suitable for browser playback. Source footage remains local. Upload limits are 250 MiB and approximately 15 minutes.

## Detector

The local CPU runner loads `review/checkpoint/best_model.pth` with PyTorch's restricted weights loader and checks all state-dict keys. It uses the ResNet18–LSTM architecture verified against Vishnu's checkpoint. Each four-second window starts two seconds after the previous one. RGB frames are resized to 224 × 224 with bilinear interpolation; 16 frames are uniformly sampled (short windows repeat their final frame). Normalization comes from the checkpoint. The score is the violence-class softmax output. A final partial interval is included. Touching or overlapping flagged windows are merged and the highest-scoring window supplies the incident snapshot. Boundaries describe the windows, so incident timing has window-level precision.

The demo starts at Vishnu's supplied threshold of 0.40. The decision applies to each four-second window. His handoff reports clip-level metrics, and the earlier INT8 benchmark used 0.50. The three reference recordings in the library exercise playback and workflow. Their overlap with historical training data is unknown.

## Dataset handoff

`handoff/dataset_metadata.csv` contains project-relative paths, source, binary label (0 = nonviolence, 1 = violence), event/source group, SHA-256, frame rate, decoded frame count, resolution, duration, assigned split, and any exclusions. Train, validation and test CSVs are saved alongside it.

These are new prospective splits. Vishnu should load the paths from each CSV directly; rerunning his folder-based splitter would discard their group boundaries. Resolve paths against the project root. For an evaluation on this split, start training from the original pretrained backbone. The previous checkpoint's original split membership is unknown.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| SFD | 208 | 45 | 47 |
| AIRTLab | 246 | 52 | 52 |
| Total | 454 | 97 | 99 |

AIRTLab pairs camera views using class and original scene number. SFD uses each source recording in the authors' `videos.txt`. The two supplementary SFD ranges have ambiguous boundaries and share clip 076, so the handoff keeps clips 062–091 from the no-fight class in a single group. All 650 clips were decoded. All file hashes were unique. The split preserves every known source group. Labels come from the publishers. Semantic duplicates across unrelated source IDs have not been manually audited.

- [SFD source](https://github.com/seymanurakti/fight-detection-surv-dataset/tree/ff83b7c521e5ca4eb67a212cc42dbc54f39d6290), Ş. Aktı, G. A. Tataroğlu and H. K. Ekenel (IPTA 2019). Local copy: `datasets/public/sfd/`.
- [AIRTLab source](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/1f7747e104301ccaa82ef5a2f6804b51ced1c398), M. Bianculli and colleagues (Data in Brief 2020). Local copy: `datasets/public/avd/`.
- RWF-2000 is pending. The [Zenodo release](https://zenodo.org/records/15687512) has a 9.4 GB multipart archive. Allow space for both the archive and unpacked videos before importing it. Keep its original holdout separate.

Run `python dataset_handoff.py` to regenerate these manifests from the two pinned local collections. Run `python verify_demo.py` while the service is running to check uploads, playback, scoring, snapshots, review persistence, and invalid-file handling using the three public reference clips. This check temporarily changes and restores review status on those demo records.

On a new laptop install the packages in `requirements-demo.txt` and provide the checkpoint at the path above. On this laptop the dependencies are already installed.

