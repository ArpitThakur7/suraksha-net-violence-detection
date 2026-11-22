"""Benchmarks. `python -m pipeline bench --memory`: GPU memory pilot (3 AMP train steps per model
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


STEPS = 3          # step 1 warms up (allocates AdamW state); peak is taken over steps 2..STEPS
STEPS_MEASURED = STEPS - 1


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e)
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "ALLOC_FAILED" in msg)


def _train_steps(name, kw, batch, T, crop, dev):
    """STEPS AdamW steps under fp16 AMP. Returns (peak MB over steps 2..STEPS, mean s/step there).
    Measuring after step 1 means exp_avg/exp_avg_sq are resident, i.e. steady-state memory."""
    from pipeline.models import build_model
    model = build_model(name, pretrained=False, T=T, **kw).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10)  # avoid a skipped first step
    x = torch.randn(batch, T, 3, crop, crop, device=dev)
    y = torch.randint(0, 2, (batch,), device=dev)

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    step()
    torch.cuda.synchronize()
    assert len(opt.state) > 0, "optimizer state empty after step 1 (GradScaler skipped the step)"
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for _ in range(STEPS_MEASURED):
        step()
    torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / STEPS_MEASURED
    return torch.cuda.max_memory_allocated(dev) / 2**20, secs


def memory_pilot(cfg: dict) -> pd.DataFrame:
    """For each model and batch in {2,4,8,16}: 3 forward+backward+AdamW steps under fp16 AMP on
    random [B,T,3,crop,crop] input. Records the steady-state peak torch.cuda.max_memory_allocated
    over steps 2-3 (weights + grads + resident AdamW state + activations; `seconds` = mean per
    step) and ok/OOM. The caching allocator is capped at the VRAM free
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
                peak, secs = _train_steps(name, kw, batch, T, crop, dev)
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                ok = False
                peak = torch.cuda.max_memory_allocated(dev) / 2**20  # peak before the failure
            gc.collect()
            torch.cuda.empty_cache()
            rows.append({"model": label, "batch": batch, "ok": ok, "peak_mb": round(peak, 1),
                         "seconds": round(secs, 3), "steps_measured": STEPS_MEASURED,
                         "cap_mb": round(free / 2**20, 1), "T": T, "crop": crop, "gpu": gpu})
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
