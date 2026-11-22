# Suraksha-Net: Real-Time Violence Detection System
## Module: Advanced Deep Learning Model Development, Temporal Modeling & Evaluation
**Lead Researcher:** Vishnu Bhargava  
**Target Recipient / Deployment Lead:** Arpit  


---

## 1. System Architecture
```
Video / CCTV Stream (RTSP / Local Camera)
           │
           ▼
[Vishnu's DL Module]
  ├── Frame Sequence Generator (16 frames, 224x224 RGB, ImageNet Normalized)
  ├── Backbone Feature Extractor (ResNet18 / EfficientNet-B0 / VGG16)
  ├── Temporal Modeling (LSTM, hidden_size=256, dropout=0.3)
  └── Classification Head (Softmax: 0=Non-Violence, 1=Violence)
           │
           ▼ (best_model.pth + model_config.json + recommended_threshold.json)
[Arpit's Downstream Pipeline]
  ├── Model Optimization (ONNX / TensorRT / FP16)
  ├── Backend API / Streaming Engine
  ├── Android Mobile App
  └── Instant Security Alerts
```

---

## 2. Directory Structure
```
/home/vishnu/Documents/vishnu project/
├── .venv/                         # Python 3.12 environment with CUDA PyTorch
├── configs/
│   └── default.yaml               # Centralized hyperparameters & data paths
├── data/
│   ├── rwf/                       # RWF-2000 dataset (Surveillance action)
│   ├── avd/                       # AIRTLab Violence Dataset (Hard negative human actions)
│   └── sfd/                       # Surveillance Fight Dataset (CCTV style)
├── datasets/
│   ├── video_dataset.py           # Custom PyTorch Dataset/DataLoader (B x T x C x H x W)
│   ├── audit_datasets.py          # Automated dataset health, FPS, corruption scanner
│   └── generate_mock_cctv.py      # Synthetic CCTV generator for verification dry-runs
├── models/
│   └── architectures.py           # VGG16, EfficientNet-B0, CNN-LSTM (ResNet18/VGG/EffNet)
├── training/
│   └── trainer.py                 # Multi-epoch trainer, AMP, Early Stopping, Checkpointing
├── evaluation/
│   ├── evaluator.py               # Precision, Recall, F1, ROC-AUC, FPR/FNR, Confusion Matrix
│   └── gradcam.py                 # Grad-CAM heatmap visualization & explainability
├── checkpoints/                   # Auto-saved epoch & best-checkpoint weights
├── experiments/                   # Experiment tracking (logs, loss curves, metrics.csv)
├── results/                       # Cross-dataset generalization matrices & comparisons
├── train.py                       # Main CLI training script
├── evaluate.py                    # Standalone checkpoint evaluator
├── cross_dataset_test.py          # Generalization matrix generator across datasets
├── compare_models.py              # Multi-model benchmarking table generator
├── export_handoff.py              # Packages all artifacts for Arpit
└── requirements.txt               # Frozen dependencies
```

---

## 3. Step-by-Step Execution Plan

### Step 1: Environment & GPU Verification (Completed ✅)
- GPU: **NVIDIA GeForce RTX 5050 Laptop (8 GB VRAM)**
- CUDA Driver: **CUDA 13.1 / Driver 590.48**
- PyTorch with CUDA support configured in `.venv`
- Batch size: **8-16** for CNN-LSTM, **16-32** for CNN Baselines

### Step 2: Dataset Acquisition & Verification
- Put datasets in `data/rwf`, `data/avd`, and `data/sfd`.
- Verify dataset health:
  ```bash
  source .venv/bin/activate
  python datasets/audit_datasets.py
  ```

### Step 3: Train VGG16 Baseline
```bash
source .venv/bin/activate
python train.py --dataset rwf --model vgg16 --epochs 15 --batch-size 16 --lr 0.0001
```

### Step 4: Train Lightweight CNN (EfficientNet-B0)
```bash
source .venv/bin/activate
python train.py --dataset rwf --model efficientnet --epochs 20 --batch-size 16 --lr 0.0001
```

### Step 5: Train Main Model (CNN + LSTM)
```bash
source .venv/bin/activate
python train.py --dataset rwf --model cnn_lstm --backbone resnet18 --seq-len 16 --epochs 30 --batch-size 8 --lr 0.0001
```

### Step 6: Cross-Dataset Generalization Testing
```bash
source .venv/bin/activate
python cross_dataset_test.py --checkpoint experiments/rwf_cnn_lstm_16frames/best_model.pth --test-datasets rwf avd sfd
```

### Step 7: Model Comparison & Final Selection
```bash
source .venv/bin/activate
python compare_models.py
```

### Step 8: Package Handoff for Arpit
```bash
source .venv/bin/activate
python export_handoff.py \
  --checkpoint experiments/rwf_cnn_lstm_16frames/best_model.pth \
  --test-results experiments/rwf_cnn_lstm_16frames/test_results.json \
  --output-dir handoff_to_arpit
```


