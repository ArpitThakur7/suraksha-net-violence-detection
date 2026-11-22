"""
Suraksha-Net: Standalone Video Inference CLI
============================================
Official deployment and testing tool for Arpit & Edge Engineering Team.
Loads best_model.pth, applies ImageNet preprocessing, samples 16 uniform frames,
and outputs classification, confidence score, and alert status.

Usage:
    python predict.py --video sample_video.mp4
    python predict.py --video path/to/cctv_clip.mp4 --threshold 0.40 --json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.architectures import build_model


def parse_args():
    default_ckpt = os.path.join(SCRIPT_DIR, "best_model.pth")
    default_cfg = os.path.join(SCRIPT_DIR, "model_config.json")
    
    parser = argparse.ArgumentParser(description="Suraksha-Net Real-Time Video Violence Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=default_cfg, help="Model configuration JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override (default: from config or 0.40)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output results in JSON format")
    return parser.parse_args()


def load_and_preprocess_video(
    video_path: str,
    seq_len: int = 16,
    img_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    OpenCV video reader + uniform frame sampler + ImageNet normalization.
    Returns: Tensor of shape (1, seq_len, 3, img_size, img_size)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
        
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
    cap.release()
    
    n_frames = len(raw_frames)
    if n_frames == 0:
        raise ValueError(f"Video contains 0 decodable frames: {video_path}")
        
    # Uniform temporal sampling
    if n_frames >= seq_len:
        indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)
        sampled_frames = [raw_frames[i] for i in indices]
    else:
        # Pad by repeating last frame
        sampled_frames = list(raw_frames)
        while len(sampled_frames) < seq_len:
            sampled_frames.append(raw_frames[-1])
            
    # Torchvision transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    frame_tensors = [transform(f) for f in sampled_frames]
    video_tensor = torch.stack(frame_tensors, dim=0)  # (seq_len, 3, img_size, img_size)
    video_tensor = video_tensor.unsqueeze(0)          # (1, seq_len, 3, img_size, img_size)
    
    return video_tensor, n_frames


def run_inference(
    video_path: str,
    checkpoint_path: str = None,
    config_path: str = None,
    threshold: float = 0.40,
    device_name: str = None
):
    checkpoint_path = checkpoint_path or os.path.join(SCRIPT_DIR, "best_model.pth")
    config_path = config_path or os.path.join(SCRIPT_DIR, "model_config.json")
    
    # Device setup
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 1. Load Model Config
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            model_cfg = json.load(f)
    else:
        model_cfg = {"architecture": "cnn_lstm", "backbone": "resnet18", "lstm_hidden_size": 256}
        
    # 2. Build Model
    model = build_model(model_cfg)
    
    # 3. Load Checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 4. Preprocess Input Video
    seq_len = model_cfg.get("sequence_length", 16)
    img_size = model_cfg.get("image_size", [224, 224])
    if isinstance(img_size, list):
        img_size = img_size[0]
        
    t_start = time.perf_counter()
    video_tensor, total_frames = load_and_preprocess_video(video_path, seq_len=seq_len, img_size=img_size)
    video_tensor = video_tensor.to(device)
    
    # 5. Forward Pass
    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0
    
    prob_nonviolence = float(probs[0])
    prob_violence = float(probs[1])
    
    # 6. Apply Decision Threshold
    is_alert = prob_violence >= threshold
    predicted_label = "Violence" if is_alert else "Non-Violence"
    alert_level = "ALERT_HIGH" if is_alert else "NORMAL"
    
    result = {
        "video": os.path.abspath(video_path),
        "total_video_frames": total_frames,
        "sampled_frames": seq_len,
        "inference_device": str(device),
        "latency_ms": round(latency_ms, 2),
        "threshold_used": round(threshold, 2),
        "probabilities": {
            "Non-Violence": round(prob_nonviolence, 4),
            "Violence": round(prob_violence, 4)
        },
        "predicted_label": predicted_label,
        "violence_score": round(prob_violence, 4),
        "alert_triggered": bool(is_alert),
        "alert_level": alert_level,
        "class_mapping": {"0": "Non-Violence", "1": "Violence"}
    }
    return result


def main():
    args = parse_args()
    
    # Read recommended threshold if not overridden
    threshold = args.threshold
    if threshold is None:
        rec_thresh_file = os.path.join(SCRIPT_DIR, "recommended_threshold.json")
        if os.path.isfile(rec_thresh_file):
            try:
                with open(rec_thresh_file, "r") as f:
                    tdata = json.load(f)
                    threshold = float(tdata.get("recommended_threshold", 0.40))
            except Exception:
                threshold = 0.40
        else:
            threshold = 0.40
            
    try:
        res = run_inference(
            video_path=args.video,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            threshold=threshold,
            device_name=args.device
        )
    except Exception as e:
        print(f"❌ INFERENCE ERROR: {e}")
        sys.exit(1)
        
    if args.output_json:
        print(json.dumps(res, indent=2))
        return
        
    print("\n" + "="*60)
    print("SURAKSHA-NET: REAL-TIME VIDEO VIOLENCE DETECTION")
    print("="*60)
    print(f"Target Video:       {os.path.basename(res['video'])}")
    print(f"Total Video Frames: {res['total_video_frames']} frames")
    print(f"Sampled Sequence:   {res['sampled_frames']} uniform frames (224x224 RGB)")
    print(f"Inference Device:   {res['inference_device']}")
    print(f"Latency:            {res['latency_ms']:.1f} ms")
    print(f"Class Mapping:      0 = Non-Violence | 1 = Violence")
    print(f"Decision Threshold: {res['threshold_used']:.2f}")
    print("-" * 60)
    print(f"P(Non-Violence):    {res['probabilities']['Non-Violence']:.4f}")
    print(f"P(Violence):        {res['probabilities']['Violence']:.4f}  (Confidence Score)")
    print("-" * 60)
    if res["alert_triggered"]:
        print(f"PREDICTION:         🚨 {res['predicted_label'].upper()}")
        print(f"SECURITY ALERT:     ⚠️  ALERT TRIGGERED! (Level: HIGH)")
        print(f"ACTION REQUIRED:    Notify on-duty security guard via mobile app.")
    else:
        print(f"PREDICTION:         ✅ {res['predicted_label'].upper()}")
        print(f"SECURITY ALERT:     Normal state (No threat detected).")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
