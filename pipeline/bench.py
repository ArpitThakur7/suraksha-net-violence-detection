"""Benchmarks. `python -m pipeline bench --memory`: GPU memory pilot (one AMP train step per model
and batch size) -> work/bench/memory.csv, used to set train.batch / grad accumulation in grid.yaml."""
from __future__ import annotations

import argparse
import gc
import time

import pandas as pd
import torch

import pipeline.paths  # noqa: F401
from pipeline.config import load_config
from pipeline.paths import ROOT

MEMORY_BATCHES = (2, 4, 8, 16)
# (row label, build_model name, extra kwargs). motionlite[effb0] is the backbone ablation.
MEMORY_MODELS = (
    ("cnn_lstm", "cnn_lstm", {}),
    ("mnv3_tsm", "mnv3_tsm", {}),
    ("r2plus1d", "r2plus1d", {}),
    ("mvit_s", "mvit_s", {}),
    ("motionlite", "motionlite", {}),
    ("motionlite[effb0]", "motionlite", {"backbone": "effb0"}),
)


def _train_step(name, kw, batch, T, crop, dev):
    from pipeline.models import build_model
    model = build_model(name, pretrained=False, T=T, **kw).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(batch, T, 3, crop, crop, device=dev)
    y = torch.randint(0, 2, (batch,), device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def memory_pilot(cfg: dict) -> pd.DataFrame:
    """For each model and batch in {2,4,8,16}: one forward+backward+AdamW step under fp16 AMP on
    random [B,T,3,crop,crop] input. Records peak torch.cuda.max_memory_allocated (weights + grads
    + optimizer state + activations) and ok/OOM. The caching allocator is capped at the VRAM free
    at start, so a run that would spill into Windows shared system memory counts as OOM."""
    if not torch.cuda.is_available():
        raise RuntimeError("memory pilot needs CUDA")
    dev = torch.device("cuda", 0)
    T, crop = int(cfg["clip"]["T"]), int(cfg["clip"]["crop"])
    free, total = torch.cuda.mem_get_info(dev)
    torch.cuda.set_per_process_memory_fraction(min(1.0, free / total), dev)
    gpu = torch.cuda.get_device_name(dev)
    print(f"{gpu}: {total / 2**20:.0f} MB total, {free / 2**20:.0f} MB free (allocator cap); "
          f"T={T}, crop={crop}, fp16 AMP")
    rows = []
    for label, name, kw in MEMORY_MODELS:
        for batch in MEMORY_BATCHES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            ok, secs = True, float("nan")
            try:
                secs = _train_step(name, kw, batch, T, crop, dev)
            except torch.cuda.OutOfMemoryError:
                ok = False
            peak = torch.cuda.max_memory_allocated(dev) / 2**20
            gc.collect()
            torch.cuda.empty_cache()
            rows.append({"model": label, "batch": batch, "ok": ok, "peak_mb": round(peak, 1),
                         "seconds": round(secs, 3), "T": T, "crop": crop, "gpu": gpu,
                         "cap_mb": round(free / 2**20, 1)})
            print(f"  {label:18s} b={batch:2d} {'ok ' if ok else 'OOM'} peak={peak:8.1f} MB "
                  f"t={secs:.2f}s")
    df = pd.DataFrame(rows)
    out = ROOT / cfg["paths"]["work"] / "bench" / "memory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print("largest batch that fits (T=%d, %dpx, fp16 AMP):" % (T, crop))
    for label, g in df.groupby("model", sort=False):
        fits = g.loc[g.ok, "batch"]
        print(f"  {label:18s} {int(fits.max()) if len(fits) else 'none (<2)'}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline bench")
    ap.add_argument("--memory", action="store_true", help="GPU memory pilot -> work/bench/memory.csv")
    ap.add_argument("--config", default="pipeline/configs/base.yaml")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    if a.memory:
        memory_pilot(cfg)
        return 0
    ap.print_help()
    return 2
