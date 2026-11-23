"""VideoMAE-B (Kinetics-400) with partial fine-tuning: the patch embedding and the first
`frozen_blocks` transformer blocks stay frozen (run without autograd), the remaining blocks, the
final norm and a new 2-way head are trained. Input [B,16,3,224,224], ImageNet normalisation."""
from __future__ import annotations

import torch
from torch import nn

HF_REPO = "MCG-NJU/videomae-base-finetuned-kinetics"


class VideoMAEFT(nn.Module):
    def __init__(self, pretrained: bool = True, frozen_blocks: int = 8, dropout: float = 0.2):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        if pretrained:
            from pipeline.probe import _patch_videomae_bias
            full = _patch_videomae_bias(VideoMAEForVideoClassification.from_pretrained(HF_REPO), HF_REPO)
        else:  # architecture only (tests, benchmarks): the base config, no download
            full = VideoMAEForVideoClassification(VideoMAEConfig(use_mean_pooling=True, num_labels=400))
        self.embeddings = full.videomae.embeddings
        self.blocks = full.videomae.encoder.layer
        self.norm = full.fc_norm if full.fc_norm is not None else nn.LayerNorm(full.config.hidden_size)
        self.frozen_blocks = int(frozen_blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(full.config.hidden_size, 2))
        for p in self.embeddings.parameters():
            p.requires_grad_(False)
        for blk in self.blocks[: self.frozen_blocks]:
            for p in blk.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _run(blk, h):
        out = blk(h)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, x):
        with torch.no_grad():
            h = self.embeddings(x, None)
            for blk in self.blocks[: self.frozen_blocks]:
                h = self._run(blk, h)
        for blk in self.blocks[self.frozen_blocks:]:
            h = self._run(blk, h)
        return self.head(self.norm(h.mean(1)))

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen part stays in eval mode (no dropout noise in the fixed features)
        self.embeddings.eval()
        for blk in self.blocks[: self.frozen_blocks]:
            blk.eval()
        return self
