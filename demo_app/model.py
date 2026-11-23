import torch
from torch import nn


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
