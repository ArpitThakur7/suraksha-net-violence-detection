# Suraksha-Net: Model Deployment Handoff Package
**Prepared by:** Vishnu Bhargava (Advanced Deep Learning & Temporal Modeling Lead)  
**Target Recipient:** Arpit (Edge Optimization, ONNX/TensorRT & Android Deployment Lead)  
**Model Architecture:** CNN + LSTM (Backbone: ResNet-18, Temporal LSTM: hidden=256, dropout=0.3)  
**Checkpoint File:** `best_model.pth` (137.4 MB)  
**Self-Contained Status:** ✅ 100% Standalone (Everything needed to run inference and export is inside this folder)  

---

## 1. Quick Test Command (Run from inside this folder)

You can run inference immediately on the included real CCTV test clip:

```bash
# Move into this handoff folder
cd handoff_to_arpit

# Run standalone video inference (Human-Readable Output)
python predict.py --video sample_video.mp4

# Run with JSON output (Ready for FastAPI backend integration)
python predict.py --video sample_video.mp4 --json
```

**Verified Output:**
```
============================================================
SURAKSHA-NET: REAL-TIME VIDEO VIOLENCE DETECTION
============================================================
Target Video:       sample_video.mp4
Total Video Frames: 187 frames
Sampled Sequence:   16 uniform frames (224x224 RGB)
Inference Device:   cpu
Latency:            4755.7 ms
Class Mapping:      0 = Non-Violence | 1 = Violence
Decision Threshold: 0.40
------------------------------------------------------------
P(Non-Violence):    0.9842
P(Violence):        0.0158  (Confidence Score)
------------------------------------------------------------
PREDICTION:         ✅ NON-VIOLENCE
SECURITY ALERT:     Normal state (No threat detected).
============================================================
```

---

## 2. Model & Pipeline Specifications

- **Input Tensor Dimensions:** `(Batch_Size, 16, 3, 224, 224)` — `[B x T x C x H x W]`
- **Temporal Sequence Length:** Exactly `16` frames sampled uniformly across the video clip.
- **Color Format & Spatial Resolution:** `RGB`, resized to `224 × 224` pixels.
- **Normalization (ImageNet Standards):**
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`
- **Class Mapping:**
  - `0`: `"Non-Violence"` (Normal movement, walking, sitting, peaceful social actions)
  - `1`: `"Violence"` (Physical altercation, fighting, brawling, assault)
- **Calibrated Operational Alert Threshold:** $\tau^* = 0.40$
  - *Deployment Rule:* If `P(Violence) >= 0.40`, trigger `SECURITY_ALERT(Level: HIGH)`.
- **Model Parameters:** `11,998,146` parameters (compact 45.8 MB unquantized FP32).

---

## 3. Final Frozen-Threshold Test Metrics (Untouched Test Videos)

The alert threshold was calibrated and frozen strictly on validation data ($\tau^* = 0.40$), then evaluated on held-out untouched test videos:

| Metric | Score at Frozen Threshold ($\tau^* = 0.40$) | Baseline at Default ($\tau = 0.50$) | Operational Security Impact |
|:---|:---:|:---:|:---|
| **Test Accuracy** | **84.11%** | 83.18% | Reliable classification across outdoor and indoor cameras |
| **Precision** | **86.89%** | 87.93% | High confidence upon alert trigger |
| **Violence Recall (Sensitivity)** | **85.48%** | 82.26% | Detects 53 of 62 violent fights; minimizes missed scuffles |
| **F1-Score** | **86.18%** | 85.00% | Peak balanced harmonic mean |
| **ROC-AUC** | **0.8885** | 0.8885 | Strong separability across all operating points |
| **False Alarms (FP)** | **8** | 7 | Controlled false alarms on hugging, clapping, and rapid gestures |
| **Missed Fights (FN)** | **9** | 11 | Missed scuffles reduced by 18.2% |

---

## 4. Copy-Paste Python Inference Code (Inside this folder)

```python
import torch
import cv2
import json
import numpy as np
import torchvision.transforms as transforms
from models.architectures import build_model

# 1. Load Architecture & Trained Weights
with open("model_config.json") as f:
    config = json.load(f)
model = build_model(config)

ckpt = torch.load("best_model.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

# 2. Preprocess Real Video (Uniform 16 Frames, 224x224 RGB, ImageNet Normalized)
cap = cv2.VideoCapture("sample_video.mp4")
raw_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

indices = np.linspace(0, len(raw_frames) - 1, 16, dtype=int)
sampled = [raw_frames[i] for i in indices]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
video_tensor = torch.stack([transform(f) for f in sampled], dim=0).unsqueeze(0)  # (1, 16, 3, 224, 224)

# 3. Model Forward Pass & Threshold Check
with torch.no_grad():
    logits = model(video_tensor)
    prob_violence = torch.softmax(logits, dim=1)[0, 1].item()

# 4. Decision Rule
THRESHOLD = 0.40  # Frozen recommended threshold
is_violence = prob_violence >= THRESHOLD
print(f"Violence Probability: {prob_violence:.4f}")
print(f"Security Alert: {'TRIGGER HIGH PRIORITY ALERT' if is_violence else 'NORMAL'}")
```

---

## 5. Folder Contents Checklist

- `best_model.pth`: Checkpoint containing model weights and state dictionary.
- `predict.py`: Standalone CLI tool to run inference on any video clip.
- `models/`: Self-contained PyTorch model architecture definitions.
- `sample_video.mp4`: Real verified video clip for end-to-end dry-run testing.
- `model_config.json`: Neural architecture specifications for ONNX/TensorRT export.
- `preprocessing_config.json`: Normalization constants, frame resolution, and sequence length.
- `class_mapping.json`: Explicit label-to-name mapping (`0 = Non-Violence`, `1 = Violence`).
- `recommended_threshold.json`: Calibrated operational threshold with complete security rationale.
- `gradcam/`: Explainability heatmaps demonstrating spatial-temporal attention on interacting limbs.
