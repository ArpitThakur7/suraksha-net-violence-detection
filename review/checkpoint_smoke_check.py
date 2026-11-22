"""Checkpoint compatibility check using a matching torch-only architecture.

This diagnostic avoids installing torchvision. Random-input inference checks
tensor compatibility and finite outputs, not violence-detection accuracy.
"""
from pathlib import Path
import csv
import json
import math
import time
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + identity)


class CheckpointCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64)),
            nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128)),
            nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256)),
            nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True, dropout=0)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(128, 2))

    def forward(self, x):
        b, t, c, h, w = x.shape
        features = self.cnn(x.reshape(b*t, c, h, w)).reshape(b, t, 512)
        seq, _ = self.lstm(features)
        return self.classifier(seq[:, -1, :])


checkpoint = torch.load(ROOT / 'checkpoint/best_model.pth', map_location='cpu', weights_only=True)
model = CheckpointCNNLSTM()
incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

torch.manual_seed(42)
sample = torch.randn(1, 16, 3, 224, 224)
start = time.perf_counter()
with torch.inference_mode():
    output = model(sample)
    probabilities = output.softmax(dim=1)

csv_path = ROOT / 'repo-snapshot/experiments/combined_cnn_lstm_16frames/metrics.csv'
with csv_path.open(encoding='utf-8', newline='') as file:
    rows = list(csv.DictReader(file))
comparison_fields = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'lr']
differences = []
comparisons = 0
for entry in checkpoint['history']:
    row = next(r for r in rows if int(r['epoch']) == entry['epoch'])
    for field in comparison_fields:
        comparisons += 1
        if abs(float(row[field]) - entry[field]) > (5.1e-9 if field == 'lr' else 5.1e-7):
            differences.append({'epoch': entry['epoch'], 'field': field, 'checkpoint': entry[field], 'csv': row[field]})

optimizer_states = checkpoint['optimizer_state_dict']['state']
nonzero_moments = sum(bool(torch.count_nonzero(state['exp_avg'])) for state in optimizer_states.values() if 'exp_avg' in state)
report = {
    'architecture': 'ResNet18 feature extractor + 1-layer LSTM(512,256) + Linear(256,128,2)',
    'implementation': 'Matching torch-only diagnostic; repository code unchanged',
    'strict_state_dict_load': 'passed',
    'missing_keys': incompatible.missing_keys,
    'unexpected_keys': incompatible.unexpected_keys,
    'parameter_count': sum(p.numel() for p in model.parameters()),
    'random_input_shape': list(sample.shape),
    'output_shape': list(output.shape),
    'finite_output': bool(torch.isfinite(output).all()),
    'probabilities_sum_to_one': bool(torch.allclose(probabilities.sum(dim=1), torch.ones(1))),
    'optimizer_states_with_nonzero_first_moment': nonzero_moments,
    'history_csv_values_compared': comparisons,
    'history_csv_mismatches': differences,
    'repo_metrics_epochs': len(rows),
    'checkpoint_history_epochs': len(checkpoint['history']),
    'note': 'No real video was used. No accuracy or latency benchmark was performed. Original dataset membership remains unverified.',
}
(ROOT / 'checkpoint-smoke-check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
