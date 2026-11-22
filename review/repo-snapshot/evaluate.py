"""
Suraksha-Net: Standalone Model Evaluator
=======================================
Usage:
  python evaluate.py --checkpoint experiments/rwf_cnn_lstm_16frames/best_model.pth --dataset rwf
"""

import os
import argparse
import json
import torch

from datasets.video_dataset import build_dataloaders
from models.architectures import build_model
from evaluation.evaluator import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Suraksha-Net Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--dataset", type=str, default=None, choices=["rwf", "avd", "sfd", "combined"])
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    if args.dataset:
        config["dataset"]["name"] = args.dataset
        
    ds_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Building data loader for: {ds_name}...")
    _, _, test_loader, info = build_dataloaders(config)
    
    print(f"Instantiating model: {model_name}...")
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Evaluating on {info['test_count']} test videos...")
    results = evaluate_model(model, test_loader, device, out_dir)
    
    m = results["metrics_at_default_0_5"]
    print(f"\n========================================================")
    print(f"EVALUATION SUMMARY ({ds_name.upper()} - {model_name.upper()}):")
    print(f"  Accuracy:  {m['accuracy']*100:.2f}%")
    print(f"  Precision: {m['precision']*100:.2f}%")
    print(f"  Recall:    {m['recall']*100:.2f}% (Violence Sensitivity)")
    print(f"  F1-Score:  {m['f1']*100:.2f}%")
    print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
    print(f"  FP: {m['false_positives']} | FN: {m['false_negatives']}")
    print(f"Recommended Threshold: {results['recommended_threshold']:.2f}")
    print(f"Artifacts saved to: {out_dir}/")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()


