# Suraksha-Net | Semester handoff

## Run the laptop demonstration

From this project directory, install the dependencies with `python -m pip install -r semester_ml/requirements.txt`. For NVIDIA acceleration use a CUDA-enabled PyTorch and matching torchvision. The local laptop already has the required packages.

```powershell
$env:SURAKSHA_ENGINE='onnx-cpu'
python -m uvicorn suraksha.app:app --host 127.0.0.1 --port 8741
```

Open http://127.0.0.1:8741 and upload `review/vishnu/sample_video.mp4`. This uses Vishnu's ResNet18–LSTM at threshold 0.40. Recording review uses overlapping four-second windows and stores scores, snapshots and decisions in SQLite. `optimization/camera_replay/laptop_demo.mp4` is a recorded example on two labeled AIRTLab clips.

To use the independently trained release in recording review, set the following before starting the same server. It loads its own image size, normalization, sequence length and threshold.

```powershell
$env:SURAKSHA_MODEL_RELEASE='ml_runs/two_sources/release'
```

## What to present

The independent experiment used SFD and AIRTLab. Validation selected **lstm_layers_2** at **0.30**. On **99** held-out clips it achieved **74.7% accuracy, 84.7% violence recall and 0.800 F1**, with **16 false alarms and 9 misses**. The release report gives dataset-specific results, ROC-AUC and the confusion matrix.

The SFD-only test subset had 47 clips and 57.4% accuracy. Campus deployment needs representative local footage and a measured false-alert rate per camera-hour. Clip-level evaluation cannot supply that rate.

Vishnu delivered a usable 11,998,146-parameter checkpoint, architecture, preprocessing, class labels, a sample, and four Grad-CAM images. Its SHA-256 matches the earlier checkpoint (a2583b924d9a43830ff4c6180664b4363919e59658790bbb2f261e88c084d0a4). His own predictor on the supplied sample decoded 158 frames and returned P(violence) = 0.0388, below 0.40. The sample output printed in his README describes 187 frames and 0.0158.

Vishnu reports 84.11% accuracy on 107 clips. The repository evaluator sweeps thresholds on the test loader and publishes the best F1. The archive provides no original split manifest or independent validation threshold log confirming the calibration claim. Treat 84.11% as a historical team-reported figure. The public-data training suite selects the epoch, threshold and model on validation, then evaluates its held-out test clips. A baseline and the later three-source study can share SFD/AIRTLab test clips. The prespecified plan is fixed across those studies.

## Optimization

| Runtime for Vishnu’s checkpoint | Inference file (MiB) | Median (ms) | P95 (ms) | Changed decisions vs FP32 |
|---|---:|---:|---:|---:|
| eager_cpu | 45.91 | 314.5 | 473.8 | 0 / 97 |
| int8_cpu | 43.57 | 336.7 | 456.9 | 0 / 97 |
| fp32_gpu | 45.91 | 28.9 | 108.1 | 0 / 97 |
| fp16_gpu | 22.94 | 20.9 | 72.9 | 0 / 97 |
| prune_10_cuda | 45.85 | 31.9 | 172.1 | 1 / 97 |
| prune_20_cuda | 45.85 | 43.6 | 175.7 | 1 / 97 |
| prune_30_cuda | 45.85 | 33.6 | 166.0 | 0 / 97 |
| onnx_cpu | 45.76 | 235.2 | 306.3 | 0 / 97 |

The 97-clip comparison uses threshold 0.40 and one uniformly sampled sequence per clip. Original training membership is unknown, so these scores measure conversion agreement. CPU runs used four PyTorch threads. GPU timings include device transfers. Decoder, resize and normalization timings and process-wide resource measurements are in optimization/handoff/execution.json. These are single-laptop runs under the load present during the experiment.

ONNX CPU and CUDA FP16 are available as `SURAKSHA_ENGINE=onnx-cpu` and `cuda-fp16`. FP16 is fastest on this RTX 4050. Dynamic INT8 saves a little space and ran slower on this CPU. The pruning experiments retained dense tensors. Retraining is unnecessary for these exports. The ONNX graph accepts one float32 input named frames with shape [1,16,3,224,224] and returns logits in [Non-Violence, Violence] order.

## Camera and backend demonstration

The laptop camera runner accepts up to three --camera entries, each with an ID, location and file, device index or RTSP URL. These replays exercised the camera pipeline using recorded sources. A physical webcam and remote RTSP feed require separate verification.

```powershell
python -m suraksha.cameras --camera replay Lab review/vishnu/sample_video.mp4 --engine onnx-cpu --window 1 --stride 0.4 --history 3 --votes 2 --endpoint http://127.0.0.1:8741/api/alert
```

The replay settings shorten the temporal rule for short clips. The defaults require five positives within seven predictions and use a 30-second incident cooldown. Choose majority, average or consecutive scoring with --smoothing. Camera workers queue incidents on disk before sending. Re-run with just --endpoint and the same --output to retry after a disconnect. Configure SURAKSHA_API_KEY for authenticated intake and status changes. The laptop service binds to loopback.

The integration test recovered 3 buffered incidents and checked idempotent delivery, conflicting IDs, invalid images, API keys and status changes. Timing and smoothing measurements for one to three replay streams are in optimization/camera_replay. Rates are aggregate decoded file frames per wall-clock second. One-second windows produced false alerts on normal footage, including the hug/high-five replay. Production threshold and smoothing calibration needs continuous recorded sessions.

## Reproduce the experiment

1. Prepare the public source directories described in README.md. Run `python dataset_handoff.py` to generate SFD/AIRTLab grouped manifests. Place the RWF archive and clips on a data drive, or set SURAKSHA_RWF_ROOT to the extracted directory containing the upstream train and val trees.
2. Run `python semester_ml/prepare.py` to audit and cache features. The cache uses ImageNet backbones, RGB frames at 112 × 112 and a 32-frame base sequence. Training ablations support 8, 16 and 32 frames. Vishnu’s original checkpoint uses 224 × 224.
3. Run `python semester_ml/train_suite.py`, then `python semester_ml/report.py two_sources`. The single-experiment entry is semester_ml/train.py. The controlled VGG fine-tuning comparison is semester_ml/vgg_tuning.py.
4. Run `python semester_ml/benchmark_handoff.py` for optimization and `python semester_ml/verify_laptop.py` for local backend and replay checks. Recreating the prior dynamic INT8 and TorchScript artifacts uses `python benchmark_quantization.py`.

## Semester boundary

This delivery covers training and evaluation on the available grouped sources, model exports, laptop replay and the local alert dashboard. Raspberry Pi/Jetson timing, sustained physical-camera trials, Android APK and phone notification delivery belong to the next semester. TensorRT and mobile performance have no measured results in this handoff.

Read ml_runs/two_sources/release/REPORT.md for the selected model. Experiment and transfer tables sit one directory above it. Raw training videos and intermediate feature caches remain in the dataset workspace. The submission includes code, the selected checkpoint and the evaluated baselines. Full reproduction needs the public datasets and ImageNet initialization weights.
