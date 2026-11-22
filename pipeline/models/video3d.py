"""Kinetics-pretrained 3D baselines. Input [B,T,3,H,W] is permuted to [B,3,T,H,W]."""
from __future__ import annotations

from torch import nn

import pipeline.paths  # noqa: F401
from torchvision.models.video import (MViT_V2_S_Weights, R2Plus1D_18_Weights, mvit_v2_s,
                                      r2plus1d_18)

MVIT_T, MVIT_CROP = 16, 224


class R2Plus1D(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x):
        return self.net(x.permute(0, 2, 1, 3, 4))


class MViTS(nn.Module):
    """MViTv2-S. Its positional/relative-position tables fix the input to 16 x 224 x 224."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None)
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, x):
        T, H, W = x.shape[1], x.shape[3], x.shape[4]
        if (T, H, W) != (MVIT_T, MVIT_CROP, MVIT_CROP):
            raise ValueError(f"mvit_s needs T={MVIT_T}, {MVIT_CROP}x{MVIT_CROP}; got T={T}, {H}x{W}")
        return self.net(x.permute(0, 2, 1, 3, 4))
