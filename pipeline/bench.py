"""Benchmarks. `python -m pipeline bench --memory`: GPU memory pilot (3 AMP train steps per model
and batch size) -> work/bench/memory.csv, used to set train.batch / grad accumulation in grid.yaml.
`python -m pipeline bench --epoch`: timing pilot, one real epoch (train + val pass through the
data loader) per model on p2_all -> work/bench/epoch_seconds.csv, read by `grid --dry-run`."""
from __future__ import annotations

import argparse
import gc
import shutil
import time

import pandas as pd
import torch

import pipeline.paths  # noqa: F401
from pipeline.config import load_config
from pipeline.paths import ROOT, work_dir

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


EPOCH_MODELS = ("cnn_lstm", "mnv3_tsm", "r2plus1d", "mvit_s", "motionlite")


def epoch_pilot(cfg: dict, protocol: str = "p2_all", models=EPOCH_MODELS) -> pd.DataFrame:
    """One epoch (train + val) of each model with the grid's default batch, seed 0, under the
    real loader and trainer, into a scratch run dir that is deleted afterwards. Wall clock of
    train_run, so it includes loader start-up and model build (a slight overestimate per epoch).
    Rows are written as they finish, so a crash keeps the models already timed."""
    from pipeline.train import default_batch, train_run
    import copy
    work = work_dir(cfg)
    out_csv = work / "bench" / "epoch_seconds.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scratch = work / "bench" / "epoch_pilot"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {r["model"]: r for r in prev.to_dict("records")}
    rows = [done[m] for m in models if m in done]
    for m in models:
        if m in done:
            print(f"  {m:12s} {done[m]['seconds_per_epoch']:8.1f} s/epoch (already timed)")
            continue
        c = copy.deepcopy(cfg)
        c["train"]["max_epochs"] = 1
        c["train"]["keep_last"] = False
        shutil.rmtree(scratch, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_run(c, protocol, m, 0, scratch / m)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        rows.append({"model": m, "seconds_per_epoch": round(secs, 1), "protocol": protocol,
                     "batch": int(c["train"].get("batch") or default_batch(m)),
                     "T": int(c["clip"]["T"]), "crop": int(c["clip"]["crop"]), "gpu": gpu})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  {m:12s} {secs:8.1f} s/epoch", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"wrote {out_csv}")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline bench")
    ap.add_argument("--memory", action="store_true", help="GPU memory pilot -> work/bench/memory.csv")
    ap.add_argument("--epoch", action="store_true",
                    help="epoch timing pilot on p2_all -> work/bench/epoch_seconds.csv")
    ap.add_argument("--config", default="pipeline/configs/base.yaml")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    if a.memory:
        memory_pilot(cfg)
        return 0
    if a.epoch:
        epoch_pilot(cfg)
        return 0
    ap.print_help()
    return 2
