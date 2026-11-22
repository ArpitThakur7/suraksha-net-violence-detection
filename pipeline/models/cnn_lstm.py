"""CNN-LSTM baseline: ResNet18 per frame -> LSTM(512->256) -> last hidden -> Dropout -> Linear."""
from __future__ import annotations

from torch import nn

import pipeline.paths  # noqa: F401
from torchvision.models import ResNet18_Weights, resnet18


class CNNLSTM(nn.Module):
    def __init__(self, pretrained: bool = True, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.lstm = nn.LSTM(512, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.cnn(x.flatten(0, 1)).view(B, T, -1)
        _, (h, _) = self.lstm(f)
        return self.fc(self.drop(h[-1]))
