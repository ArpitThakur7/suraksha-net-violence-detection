# Suraksha-Net: Model Deployment Handoff Package
**Prepared by:** Vishnu Bhargava (Advanced DL Model Development & Evaluation Lead)  
**Target:** Arpit / Deployment & Optimization Team  
**Model Architecture:** CNN_LSTM (Backbone: resnet18)  

---

## 1. Quick Specifications
- **Input Tensor:** `B × 16 × 3 × 224 × 224` (Batch × Sequence × RGB Channels × Height × Width)
- **Input Preprocessing:** RGB, Resized to 224×224, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Classes:** `0 = Non-Violence`, `1 = Violence`
- **Output:** Softmax probabilities across 2 classes
- **Recommended Detection Threshold:** `0.4` (Trigger alert when P(Violence) >= `0.4`)
- **Model Parameters:** `11,998,146`
- **Model File Size:** `137.44 MB`

---

## 2. Key Performance Metrics (Test Set)
- **Accuracy:** `84.11%`
- **Precision:** `86.89%`
- **Recall (Sensitivity):** `85.48%`
- **F1-Score:** `86.18%`
- **ROC-AUC:** `0.8885`
- **False Positive Rate:** `17.78%`
- **False Negative Rate:** `14.52%`

---

## 3. Package File Directory
- `best_model.pth`: PyTorch trained model weights and full state dictionary
- `model_config.json`: Complete neural architecture specification
- `preprocessing_config.json`: Exact normalization and sampling rules
- `class_mapping.json`: Integer to string label mappings
- `recommended_threshold.json`: Calibrated decision threshold with security justification
- `gradcam/`: Visual explainability maps (True Positive, True Negative, False Positive, False Negative)

---

## 4. Quick Inference Snippet for Arpit

```python
import torch
import torchvision.transforms as transforms
from models.architectures import build_model
import json

# 1. Load configurations
with open("handoff_to_arpit/model_config.json") as f:
    model_cfg = json.load(f)

# 2. Build model and load weights
model = build_model(model_cfg)
ckpt = torch.load("handoff_to_arpit/best_model.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 3. Input tensor: (1, 16, 3, 224, 224)
# (Sample 16 frames uniformly, resize 224x224, normalize ImageNet)
dummy_video = torch.randn(1, 16, 3, 224, 224)

# 4. Predict
with torch.no_grad():
    logits = model(dummy_video)
    prob_violence = torch.softmax(logits, dim=1)[0, 1].item()

# 5. Security alert decision
threshold = 0.40  # From recommended_threshold.json
is_violent = prob_violence >= threshold
print(f"Violence Probability: {prob_violence:.3f} | Alert: {is_violent}")
```

