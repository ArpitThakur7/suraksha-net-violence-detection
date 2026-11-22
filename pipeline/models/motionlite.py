"""MotionLite: lightweight 2D-CNN + TSM + frame-difference stream + attention pooling.

Every component is switchable for the ablation table:
  backbone  "mnv3" (MobileNetV3-Large, C=960) | "effb0" (EfficientNet-B0, C=1280)
  diff      add the |frame_t - frame_{t-1}| stream (computed on the normalised input; a zero
            frame is prepended so the length stays T)
  tsm       temporal shift on the residual branch of every residual block (both streams)
  pool      "attn": w = softmax_T(Linear(C,1)(feat_t)), clip = sum_t w_t feat_t | "avg": mean_T
  fusion    "early": stem conv widened to 6 input channels (extra 3 initialised to the mean of the
            RGB filters), input = concat(rgb, diff)
            "mid": the stem + first blocks (features[:split]) run separately on rgb and on diff
            (the diff copy has its own weights, initialised identically); then
            fused = rgb_feat + sigmoid(g) * proj(diff_feat), proj = 1x1 C->C/2 (BN, ReLU) -> 1x1
            C/2->C bottleneck, g = Conv1x1(concat(rgb_feat, proj(diff_feat))) -> 1 channel
            (per-pixel gate); features[split:] are shared.
Mid-fusion split (FUSION_SPLIT): stem + the first 5 inverted-residual blocks in both backbones,
i.e. mnv3 features[:6] (15 blocks total; output 40 ch, stride 8) and effb0 features[:4]
(stem + stages 1-3 = 5 of 16 MBConv blocks; output 40 ch, stride 8).
diff=False ignores `fusion` and reduces to mnv3_tsm with the chosen pooling.
`last_attn` holds the last forward's temporal weights [B,T] (uniform for pool="avg").
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from pipeline.models.tsm import add_tsm, backbone_features

FUSION_SPLIT = {"mnv3": 6, "effb0": 4}


def _out_channels(seq: nn.Module) -> int:
    convs = [m for m in seq.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1].out_channels


def _widen_stem(stem_conv: nn.Conv2d) -> nn.Conv2d:
    """3 -> 6 input channels; the extra 3 are each the mean of the RGB filters."""
    new = nn.Conv2d(6, stem_conv.out_channels, stem_conv.kernel_size, stem_conv.stride,
                    stem_conv.padding, stem_conv.dilation, stem_conv.groups,
                    bias=stem_conv.bias is not None)
    with torch.no_grad():
        w = stem_conv.weight
        new.weight[:, :3] = w
        new.weight[:, 3:] = w.mean(1, keepdim=True).expand(-1, 3, -1, -1)
        if stem_conv.bias is not None:
            new.bias.copy_(stem_conv.bias)
    return new


class MotionLite(nn.Module):
    def __init__(self, T: int = 16, backbone: str = "mnv3", diff: bool = True, tsm: bool = True,
                 pool: str = "attn", fusion: str = "mid", pretrained: bool = True,
                 fold_div: int = 8, dropout: float = 0.3):
        super().__init__()
        if pool not in ("attn", "avg"):
            raise ValueError(f"pool must be 'attn' or 'avg', got {pool!r}")
        if fusion not in ("mid", "early"):
            raise ValueError(f"fusion must be 'mid' or 'early', got {fusion!r}")
        self.T, self.backbone, self.diff, self.tsm = T, backbone, diff, tsm
        self.pool, self.fusion = pool, fusion
        features, C = backbone_features(backbone, pretrained)
        split = FUSION_SPLIT[backbone]
        self.early = features[:split]
        self.late = features[split:]
        self.mid = diff and fusion == "mid"
        if diff and fusion == "early":
            self.early[0][0] = _widen_stem(self.early[0][0])
        if self.mid:
            c = _out_channels(self.early)
            self.diff_early = copy.deepcopy(self.early)
            self.proj = nn.Sequential(
                nn.Conv2d(c, c // 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True),
                nn.Conv2d(c // 2, c, 1))
            self.gate = nn.Conv2d(2 * c, 1, 1)
        if tsm:
            add_tsm(self, T, fold_div)
        self.attn = nn.Linear(C, 1) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(C, 2)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x):
        B, T = x.shape[:2]
        assert T == self.T, f"model built for T={self.T}, got T={T}"
        if self.diff:
            d = (x[:, 1:] - x[:, :-1]).abs()
            d = torch.cat([torch.zeros_like(x[:, :1]), d], 1)
            if not self.mid:
                x = torch.cat([x, d], 2)
        f = self.early(x.flatten(0, 1))
        if self.mid:
            pd = self.proj(self.diff_early(d.flatten(0, 1)))
            g = self.gate(torch.cat([f, pd], 1))
            f = f + torch.sigmoid(g) * pd
        f = self.late(f).mean((2, 3)).view(B, T, -1)          # [B,T,C]
        if self.attn is not None:
            w = torch.softmax(self.attn(f).squeeze(-1).float(), 1).to(f.dtype)  # [B,T]
        else:
            w = torch.full((B, T), 1.0 / T, device=f.device, dtype=f.dtype)
        self.last_attn = w.detach()
        clip = (w.unsqueeze(-1) * f).sum(1)
        return self.fc(self.drop(clip))
