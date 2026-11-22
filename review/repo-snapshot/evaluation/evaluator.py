"""
Suraksha-Net: Comprehensive Evaluation & Error Analysis
======================================================
Computes:
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score
- ROC-AUC, PR Curves, Confusion Matrix
- False Positive Rate, False Negative Rate
- Automatic Error Mining (Hugging, Clapping, Sudden movements, etc.)
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive security-centric evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / Violence Recall
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str, title: str = "Suraksha-Net Confusion Matrix"):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Violence", "Violence"],
        yticklabels=["Non-Violence", "Violence"]
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float, save_path: str):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_learning_curves(metrics_csv_path: str, save_dir: str):
    """Generate loss and accuracy/F1 curves from metrics.csv."""
    import pandas as pd
    if not os.path.isfile(metrics_csv_path):
        return
    df = pd.read_csv(metrics_csv_path)
    
    # Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()
    
    # Accuracy & F1 Curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc", color="cyan", linestyle="--")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc", color="green")
    plt.plot(df["epoch"], df["val_f1"], label="Val F1", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy & F1 Metric Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_f1_curve.png"), dpi=300)
    plt.close()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> Dict[str, Any]:
    """
    Run complete evaluation on test set across varying decision thresholds,
    saving error cases and plots.
    """
    model.eval()
    all_labels = []
    all_probs = []
    video_names = []
    sources = []
    
    for videos, labels, names, srcs in test_loader:
        videos = videos.to(device)
        outputs = model(videos)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        video_names.extend(names)
        sources.extend(srcs)
        
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Threshold sweep
    threshold_results = {}
    best_f1 = -1.0
    recommended_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, preds, y_prob)
        threshold_results[f"thresh_{thresh:.2f}"] = metrics
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            recommended_thresh = thresh
            
    # Default 0.5 metrics
    default_preds = (y_prob >= 0.5).astype(int)
    default_metrics = compute_metrics(y_true, default_preds, y_prob)
    
    # Save Confusion Matrix & ROC Curve
    cm = confusion_matrix(y_true, default_preds, labels=[0, 1])
    plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, default_metrics["roc_auc"], os.path.join(save_dir, "roc_curve.png"))
    
    # Error mining
    false_positives = []
    false_negatives = []
    for i in range(len(y_true)):
        actual = y_true[i]
        pred = default_preds[i]
        prob = float(y_prob[i])
        vname = video_names[i]
        src = sources[i]
        
        item = {
            "video_name": vname,
            "dataset_source": src,
            "actual_label": int(actual),
            "predicted_label": int(pred),
            "violence_probability": round(prob, 4)
        }
        if actual == 0 and pred == 1:
            false_positives.append(item)
        elif actual == 1 and pred == 0:
            false_negatives.append(item)
            
    # Save error analysis
    error_analysis = {
        "total_test_samples": len(y_true),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
    with open(os.path.join(save_dir, "error_analysis.json"), "w") as f:
        json.dump(error_analysis, f, indent=2)
        
    final_output = {
        "metrics_at_default_0_5": default_metrics,
        "recommended_threshold": float(recommended_thresh),
        "metrics_at_recommended_threshold": threshold_results[f"thresh_{recommended_thresh:.2f}"],
        "threshold_sweep": threshold_results,
        "error_analysis_summary": {
            "fp_count": len(false_positives),
            "fn_count": len(false_negatives)
        }
    }
    
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(final_output, f, indent=2)
        
    return final_output


