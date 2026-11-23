"""Stage 4: clip dataset, temporal sampling, clip-level augmentation, loaders.

RNG design: training randomness (temporal segment + augmentation params) comes from a numpy
Generator owned by the dataset *per process*. It is created lazily on first use, seeded from
[base, worker_id] where base = `seed` if given, else torch.initial_seed() (which DataLoader sets
to a distinct value per worker). The generator then advances across calls and epochs, so
persistent workers do not replay the same augmentation every epoch, and different workers never
share a stream. Given the same `seed` and the same worker count, the stream per worker is fixed.
`make_loader(seed=...)` also seeds the sampler/shuffle generator.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import pipeline.paths  # noqa: F401  (puts .ml_deps on sys.path for torchvision)
from torchvision.transforms.v2 import functional as TF

from pipeline.data.cache import num_frames, read_clip


def sample_indices(n: int, T: int, train: bool, rng: np.random.Generator,
                   view: int = 0, views: int = 1) -> np.ndarray:
    """T frame indices into a clip of n frames. n < T: 0..n-1 then the last frame repeated.
    Eval with views > 1: view v is the v-th of `views` evenly spaced temporal windows."""
    if n <= 0:
        raise ValueError("clip has no frames")
    if n < T:
        return np.minimum(np.arange(T), n - 1).astype(np.int64)
    if train:
        seg = n * rng.uniform(0.6, 1.0)
        start = rng.uniform(0.0, n - seg)
        pos = np.linspace(start, start + seg - 1, T)
    elif views <= 1:
        pos = np.linspace(0, n - 1, T)
    else:  # eval window `view` of `views`: evenly spaced starts, each window max(T, n // views) frames
        w = min(n, max(T, n // views))
        start = (n - w) * view / (views - 1)
        pos = np.linspace(start, start + w - 1, T)
    return np.clip(np.rint(pos), 0, n - 1).astype(np.int64)


def _crop_params(h, w, scale, ratio, rng):
    area = h * w
    log_r = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        a = area * rng.uniform(*scale)
        r = math.exp(rng.uniform(*log_r))
        cw, ch = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if 0 < cw <= w and 0 < ch <= h:
            return int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1)), ch, cw
    s = min(h, w)  # fallback: central square
    return (h - s) // 2, (w - s) // 2, s, s


def augment_clip(frames: np.ndarray, crop: int, aug: dict, rng: np.random.Generator) -> torch.Tensor:
    """frames uint8 [T,H,W,3] -> float [T,3,crop,crop] in [0,1]. One param draw for the whole clip."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)
    top, left, ch, cw = _crop_params(x.shape[2], x.shape[3], aug["scale"], aug["ratio"], rng)
    x = TF.resized_crop(x, top, left, ch, cw, [crop, crop], antialias=True).float() / 255.0
    if rng.random() < aug["hflip"]:
        x = TF.horizontal_flip(x)
    b, c, s, h = aug["color_jitter"]
    ops = {
        "b": lambda t, f=rng.uniform(max(0, 1 - b), 1 + b): TF.adjust_brightness(t, f),
        "c": lambda t, f=rng.uniform(max(0, 1 - c), 1 + c): TF.adjust_contrast(t, f),
        "s": lambda t, f=rng.uniform(max(0, 1 - s), 1 + s): TF.adjust_saturation(t, f),
        "h": lambda t, f=rng.uniform(-h, h): TF.adjust_hue(t, f),
    }
    for k in rng.permutation(list(ops)):
        x = ops[str(k)](x)
    if rng.random() < aug["grayscale"]:
        x = TF.rgb_to_grayscale(x, num_output_channels=3)
    return x.clamp(0, 1)


class ClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split: str, cfg, norm, train: bool,
                 allow_test: bool = False, seed: int | None = None, view: int = 0, views: int = 1):
        if split == "test" and not allow_test:
            raise PermissionError("test rows may only be loaded with allow_test=True (evaluate only)")
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.cfg, self.train, self.seed = cfg, train, seed
        self.view, self.views = view, views  # eval-time temporal window (see sample_indices)
        self.T, self.crop = int(cfg["clip"]["T"]), int(cfg["clip"]["crop"])
        self.aug = cfg["clip"]["aug"]
        self.mean = torch.tensor(norm[0], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(norm[1], dtype=torch.float32).view(1, 3, 1, 1)
        self.clip_ids = self.df["clip_id"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.datasets = self.df["dataset"].tolist()
        self._rng = None

    def __len__(self):
        return len(self.df)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            base = self.seed if self.seed is not None else torch.initial_seed()
            self._rng = np.random.default_rng([int(base) & (2**63 - 1), wi.id if wi else 0])
        return self._rng

    def __getitem__(self, i):
        cid = self.clip_ids[i]
        rng = self._get_rng()
        idx = sample_indices(num_frames(cid, self.cfg), self.T, self.train, rng, self.view, self.views)
        frames = read_clip(cid, self.cfg, idx)
        if self.train:
            x = augment_clip(frames, self.crop, self.aug, rng)
        else:
            x = TF.center_crop(torch.from_numpy(frames).permute(0, 3, 1, 2), [self.crop, self.crop]).float() / 255.0
        x = ((x - self.mean) / self.std).contiguous()
        return x, self.labels[i], cid


def make_loader(manifest, split, cfg, norm, train, allow_test=False, balance_datasets=False,
                batch_size: int | None = None, seed: int | None = None, view: int = 0,
                views: int = 1) -> DataLoader:
    ds = ClipDataset(manifest, split, cfg, norm, train, allow_test, seed=seed, view=view, views=views)
    workers = int(cfg["train"].get("workers", 4))
    bs = int(batch_size or cfg["train"].get("effective_batch", 32))
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    else:
        gen.seed()
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=torch.cuda.is_available(),
              generator=gen, drop_last=False)
    if workers > 0:
        kw["persistent_workers"] = True
    if train and balance_datasets:
        counts = pd.Series(ds.datasets).value_counts()
        w = torch.tensor([1.0 / counts[d] for d in ds.datasets], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=gen)
        return DataLoader(ds, sampler=sampler, **kw)
    return DataLoader(ds, shuffle=train, **kw)
