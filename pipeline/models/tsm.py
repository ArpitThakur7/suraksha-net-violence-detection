"""Temporal Shift Module (Lin et al. 2019, residual shift) and the MobileNetV3-Large TSM baseline.

The shift is inserted on the *residual branch* input of every inverted-residual block that has a
residual connection (torchvision `InvertedResidual.use_res_connect` / `MBConv.use_res_connect`),
by wrapping the block's first layer `block.block[0]`. The identity path stays unshifted.
"""
from __future__ import annotations

import torch
from torch import nn

import pipeline.paths  # noqa: F401  (puts .ml_deps on sys.path for torchvision)
from torchvision.models import (EfficientNet_B0_Weights, MobileNet_V3_Large_Weights,
                                efficientnet_b0, mobilenet_v3_large)
from torchvision.models.efficientnet import MBConv
from torchvision.models.mobilenetv3 import InvertedResidual


def temporal_shift(x: torch.Tensor, T: int, fold_div: int = 8) -> torch.Tensor:
    """x: [B*T, C, H, W]. Channels [0, C/fold_div) move one step back in time (out[t] = in[t+1]),
    channels [C/fold_div, 2C/fold_div) one step forward (out[t] = in[t-1]); the rest unchanged.
    Edges are zero-padded. Not in-place."""
    nt, c, h, w = x.shape
    assert nt % T == 0, f"batch dim {nt} not divisible by T={T}"
    v = x.view(nt // T, T, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(v)
    out[:, :-1, :fold] = v[:, 1:, :fold]
    out[:, 1:, fold:2 * fold] = v[:, :-1, fold:2 * fold]
    out[:, :, 2 * fold:] = v[:, :, 2 * fold:]
    return out.view(nt, c, h, w)


class TemporalShift(nn.Module):
    """Shift the input over time, then run the wrapped layer."""

    def __init__(self, net: nn.Module, T: int, fold_div: int = 8):
        super().__init__()
        self.net, self.T, self.fold_div = net, T, fold_div

    def forward(self, x):
        return self.net(temporal_shift(x, self.T, self.fold_div))

    def extra_repr(self):
        return f"T={self.T}, fold_div={self.fold_div}"


def add_tsm(module: nn.Module, T: int, fold_div: int = 8) -> nn.Module:
    """Wrap block[0] of every residual InvertedResidual/MBConv inside `module`. Idempotent.
    Load pretrained weights *before* calling this (wrapping renames block.0 -> block.0.net)."""
    for m in list(module.modules()):
        if isinstance(m, (InvertedResidual, MBConv)) and m.use_res_connect \
                and not isinstance(m.block[0], TemporalShift):
            m.block[0] = TemporalShift(m.block[0], T, fold_div)
    return module


def backbone_features(name: str, pretrained: bool) -> tuple[nn.Sequential, int]:
    """torchvision `features` Sequential and its output width. ImageNet weights if pretrained."""
    if name == "mnv3":
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        return mobilenet_v3_large(weights=w).features, 960
    if name == "effb0":
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return efficientnet_b0(weights=w).features, 1280
    raise ValueError(f"unknown backbone {name!r}; choose 'mnv3' or 'effb0'")


class MobileNetV3TSM(nn.Module):
    """MobileNetV3-Large + TSM -> global average pool -> mean over T -> Linear(960, 2)."""

    def __init__(self, T: int = 16, pretrained: bool = True, fold_div: int = 8):
        super().__init__()
        self.T = T
        self.features, C = backbone_features("mnv3", pretrained)
        add_tsm(self.features, T, fold_div)
        self.fc = nn.Linear(C, 2)

    def forward(self, x):
        B, T = x.shape[:2]
        assert T == self.T, f"model built for T={self.T}, got T={T}"
        f = self.features(x.flatten(0, 1))
        f = f.mean((2, 3)).view(B, T, -1).mean(1)
        return self.fc(f)
