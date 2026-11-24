"""Stage 6: resumable trainer with validation-only model selection.

`python -m pipeline train --protocol p1_rwf --model motionlite --seed 0 [--set key=val] [--kw key=val]`

Selection: best validation ROC-AUC (ties -> lower validation loss), early stopping after
`train.patience` epochs without improvement, at most `train.max_epochs`. The decision threshold
maximises validation F1 at the best epoch and is frozen into best.pt / val_summary.json.
Only train and val rows are ever handed to the data loaders.

Exact resume: last.pt is written atomically at the end of every epoch. The train loader is
rebuilt each epoch with a seed derived from (seed, epoch), so shuffling, temporal sampling and
augmentation of an epoch do not depend on how many epochs ran in this process; the global
python/numpy/torch/cuda RNG states (dropout) are restored from last.pt. A run killed mid-epoch
therefore replays that epoch exactly and ends with the same weights as an uninterrupted run.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from pipeline import stats
from pipeline.config import DEFAULT_CONFIG, load_config, run_metadata
from pipeline.data.dataset import make_loader
from pipeline.data.splits import load_protocol
from pipeline.models import MODEL_INPUT, MODEL_NAMES, MODEL_NORM, build_model
from pipeline.paths import runs_root

# Per-model micro-batch from the steady-state memory pilot (work/bench/memory.csv, RTX 4050 6 GB).
DEFAULT_BATCH = {"cnn_lstm": 16, "mnv3_tsm": 8, "motionlite": 8, "motionlite[effb0]": 4,
                 "r2plus1d": 4, "mvit_s": 2, "videomae_ft": 4}
# Model classes, for reading constructor defaults when naming variants.
_MODEL_CLASS = {"cnn_lstm": ("pipeline.models.cnn_lstm", "CNNLSTM"),
                "mnv3_tsm": ("pipeline.models.tsm", "MobileNetV3TSM"),
                "motionlite": ("pipeline.models.motionlite", "MotionLite"),
                "r2plus1d": ("pipeline.models.video3d", "R2Plus1D"),
                "mvit_s": ("pipeline.models.video3d", "MViTS"),
                "videomae_ft": ("pipeline.models.videomae_ft", "VideoMAEFT")}
HISTORY_COLS = ["epoch", "train_loss", "train_acc", "val_loss", "val_auc", "val_f1@0.5", "lr", "seconds"]


def default_batch(model: str, model_kw: dict | None = None) -> int:
    effb0 = model == "motionlite" and (model_kw or {}).get("backbone") == "effb0"
    return DEFAULT_BATCH["motionlite[effb0]" if effb0 else model]


def variant_name(model: str, model_kw: dict | None = None, overrides=()) -> str:
    """model + "__k-v" for every model kwarg that differs from the constructor default (sorted),
    then "__key-val" for every naming config override such as clip.T=8 (sorted)."""
    mod, cls = _MODEL_CLASS[model]
    params = inspect.signature(getattr(importlib.import_module(mod), cls).__init__).parameters
    defaults = {k: p.default for k, p in params.items() if p.default is not inspect.Parameter.empty}
    diff = sorted((k, v) for k, v in (model_kw or {}).items() if k not in defaults or defaults[k] != v)
    ovs = sorted(tuple(o.split("=", 1)) for o in overrides)
    return "__".join([model] + [f"{k}-{v}" for k, v in diff] + [f"{k.strip()}-{v.strip()}" for k, v in ovs])


def run_dir(cfg, protocol: str, model: str, seed: int, model_kw: dict | None = None,
            overrides=()) -> Path:
    """runs_root(cfg)/protocol/variant/seedN; `overrides` are the config overrides that name the
    variant (e.g. the grid's clip.T=8 ablation), not every --set."""
    return runs_root(cfg) / protocol / variant_name(model, model_kw, overrides) / f"seed{seed}"


def epoch_seed(seed: int, epoch: int) -> int:
    return int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _save_pt(obj, path: Path) -> None:
    _atomic(path, lambda t: torch.save(obj, t))


def _write_json(obj, path: Path) -> None:
    _atomic(path, lambda t: t.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8"))


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _atomic(path, lambda t: df.to_csv(t, index=False))


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def _lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup to 1 over warmup_steps optimizer steps, then cosine to 0 at total_steps."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return f


def _device(tc) -> torch.device:
    d = str(tc.get("device", "auto"))
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d)


def make_criterion(cfg, protocol: str):
    """crit(logits, y, clip_ids). Label-smoothed CE; with train.distill_teacher set (e.g. videomae)
    it becomes (1 - a) * CE + a * soft CE against the teacher's out-of-fold P(violent) for each
    train clip (work/probe/<protocol>/<teacher>/teacher_train.csv, `probe teacher`), a =
    train.distill_alpha."""
    tc = cfg["train"]
    ce = nn.CrossEntropyLoss(label_smoothing=float(tc["label_smoothing"]))
    teacher = tc.get("distill_teacher")
    if not teacher:
        return lambda logits, y, cid: ce(logits, y)
    from pipeline.paths import work_dir
    f = work_dir(cfg) / "probe" / protocol / str(teacher) / "teacher_train.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; run `python -m pipeline probe teacher --backbone {teacher} --protocol {protocol}`")
    t = pd.read_csv(f, dtype={"clip_id": str})
    tmap = dict(zip(t["clip_id"], t["prob"].astype(float)))
    alpha = float(tc.get("distill_alpha", 0.5))

    def crit(logits, y, cid):
        pt = torch.tensor([tmap[c] for c in cid], device=logits.device, dtype=logits.dtype)
        soft = torch.stack([1 - pt, pt], 1)
        kd = -(soft * logits.log_softmax(1)).sum(1).mean()
        return (1 - alpha) * ce(logits, y) + alpha * kd
    return crit


def _train_epoch(model, loader, opt, sched, scaler, crit, accum, dev, use_amp):
    """One epoch with gradient accumulation; the last, shorter accumulation group is averaged
    over its own size. Returns (mean train loss, train accuracy)."""
    model.train()
    n = len(loader)
    full = (n // accum) * accum
    loss_sum = correct = seen = 0.0
    opt.zero_grad(set_to_none=True)
    for i, (x, y, cid) in enumerate(loader):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = crit(logits.float(), y, cid)
        scaler.scale(loss / (accum if i < full else n - full)).backward()
        if (i + 1) % accum == 0 or i + 1 == n:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return loss_sum / seen, correct / seen


@torch.no_grad()
def _evaluate(model, loader, dev, use_amp):
    """Returns (mean unsmoothed CE loss, labels, P(violent), clip_ids)."""
    model.eval()
    loss_sum, ys, ps, ids = 0.0, [], [], []
    for x, y, cid in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x).float()
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
        ids += list(cid)
    y, p = np.concatenate(ys), np.concatenate(ps)
    return loss_sum / len(y), y, p, ids


def train_run(cfg, protocol: str, model: str, seed: int, out_dir: Path,
              model_kw: dict | None = None) -> dict:
    """Train (or resume) one run into out_dir and return its val_summary. A finished run (DONE
    marker) returns the saved summary without training."""
    out = Path(out_dir)
    if (out / "DONE").exists():
        _drop_last(out, cfg)
        return json.loads((out / "val_summary.json").read_text(encoding="utf-8"))
    model_kw = dict(model_kw or {})
    tc = cfg["train"]
    T, crop = int(cfg["clip"]["T"]), int(cfg["clip"]["crop"])
    need = MODEL_INPUT.get(model, {})
    if need.get("crop", crop) != crop:
        raise ValueError(f"{model} requires crop={need['crop']}, got crop={crop}")
    batch = int(tc.get("batch") or default_batch(model, model_kw))
    eff = int(tc["effective_batch"])
    if eff % batch:
        raise ValueError(f"effective_batch {eff} is not divisible by batch {batch}; set train.batch")
    accum = eff // batch

    det_prev = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    cudnn_prev = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    if tc.get("deterministic", True):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        return _fit(cfg, protocol, model, seed, out, model_kw, T, crop, batch, accum)
    finally:
        torch.use_deterministic_algorithms(det_prev[0], warn_only=det_prev[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn_prev


def _fit(cfg, protocol, model, seed, out, model_kw, T, crop, batch, accum) -> dict:
    tc = cfg["train"]
    out.mkdir(parents=True, exist_ok=True)
    m = load_protocol(protocol, cfg)
    m = m[m["split"].isin(("train", "val"))].reset_index(drop=True)  # test rows never leave here
    norm = MODEL_NORM[model]
    dev = _device(tc)
    use_amp = dev.type == "cuda"
    balance = protocol.startswith(("p2", "p3"))
    max_epochs, patience = int(tc["max_epochs"]), int(tc["patience"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = build_model(model, pretrained=bool(tc.get("pretrained", True)), T=T, **model_kw).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    steps_per_epoch = math.ceil(math.ceil(int((m["split"] == "train").sum()) / batch) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(int(tc["warmup_epochs"]) * steps_per_epoch, max_epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = make_criterion(cfg, protocol)
    # val workers persist through training while idle; keep them few so RAM goes to train workers
    val_loader = make_loader(m, "val", cfg, norm, train=False, batch_size=batch, seed=seed,
                             workers=int(tc.get("val_workers", tc.get("workers", 4))))
    val_df = m[m["split"] == "val"].set_index("clip_id")

    best = {"auc": -math.inf, "loss": math.inf, "epoch": 0, "since": 0}
    rows, start = [], 0
    if (out / "last.pt").exists():
        # CPU load: RNG states must stay CPU ByteTensors; load_state_dict copies onto the device.
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best = int(ck["epoch"]), ck["best"]
        _set_rng_state(ck["rng"])
        if (out / "history.csv").exists():  # drop rows of an epoch that never reached last.pt
            h = pd.read_csv(out / "history.csv")
            rows = h[h["epoch"] <= start].to_dict("records")

    _write_json({"protocol": protocol, "model": model, "model_kw": model_kw, "seed": seed,
                 "batch": batch, "accum": accum, "device": str(dev), "amp": use_amp,
                 "resumed_from_epoch": start, "cfg": cfg, "meta": run_metadata()}, out / "config.json")

    for epoch in range(start, max_epochs):
        if best["since"] >= patience:
            break
        t0 = time.time()
        train_loader = make_loader(m, "train", cfg, norm, train=True, balance_datasets=balance,
                                   batch_size=batch, seed=epoch_seed(seed, epoch))
        tr_loss, tr_acc = _train_epoch(net, train_loader, opt, sched, scaler, crit, accum, dev, use_amp)
        sampler_state = train_loader.generator.get_state()
        del train_loader  # shuts down persistent workers
        v_loss, y, p, ids = _evaluate(net, val_loader, dev, use_amp)
        vm = stats.clip_metrics(y, p, 0.5)
        auc = -math.inf if math.isnan(vm["roc_auc"]) else vm["roc_auc"]
        if auc > best["auc"] or (auc == best["auc"] and v_loss < best["loss"]):
            best = {"auc": auc, "loss": v_loss, "epoch": epoch + 1, "since": 0}
            _save_pt({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                      "threshold": None, "norm": norm, "model_name": model, "model_kw": model_kw,
                      "T": T, "crop": crop, "epoch": epoch + 1}, out / "best.pt")
            _write_csv(pd.DataFrame({"clip_id": ids, "dataset": val_df.loc[ids, "dataset"].tolist(),
                                     "label": y.astype(int), "prob": p}), out / "val_preds.csv")
        else:
            best["since"] += 1
        rows.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": v_loss,
                     "val_auc": vm["roc_auc"], "val_f1@0.5": vm["f1"], "lr": opt.param_groups[0]["lr"],
                     "seconds": round(time.time() - t0, 2)})
        _write_csv(pd.DataFrame(rows, columns=HISTORY_COLS), out / "history.csv")
        _save_pt({"model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                  "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "rng": _rng_state(),
                  "sampler_generator": sampler_state}, out / "last.pt")
        print(f"[{protocol}/{model}/seed{seed}] epoch {epoch + 1}: train_loss {tr_loss:.4f} "
              f"val_loss {v_loss:.4f} val_auc {vm['roc_auc']:.4f} ({rows[-1]['seconds']}s)", flush=True)

    preds = pd.read_csv(out / "val_preds.csv", dtype={"clip_id": str})
    g = tc["threshold_grid"]
    grid = np.arange(g["start"], g["stop"] + g["step"] / 2, g["step"])
    thr = stats.best_threshold(preds["label"], preds["prob"], grid)
    metrics = stats.clip_metrics(preds["label"], preds["prob"], thr)
    best_ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    best_ck["threshold"] = thr
    _save_pt(best_ck, out / "best.pt")
    summary = {"best_epoch": best["epoch"], "val_auc": metrics["roc_auc"], "threshold": thr,
               "metrics": metrics, "epochs_run": len(rows)}
    _write_json(summary, out / "val_summary.json")
    (out / "DONE").write_text("", encoding="utf-8")
    _drop_last(out, cfg)
    return summary


def _drop_last(out: Path, cfg) -> None:
    """train.keep_last=false: a finished run needs only best.pt, so last.pt (weights + optimizer
    state) is deleted once DONE exists. Saves Drive space on Colab."""
    if not cfg["train"].get("keep_last", True):
        (out / "last.pt").unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline train")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                    help="config override, e.g. train.max_epochs=2")
    ap.add_argument("--kw", action="append", default=[], metavar="KEY=VAL",
                    help="model kwarg, e.g. diff=false (names the run variant)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    kw = {k: yaml.safe_load(v) for k, _, v in (s.partition("=") for s in a.kw)}
    out = run_dir(cfg, a.protocol, a.model, a.seed, kw)
    summary = train_run(cfg, a.protocol, a.model, a.seed, out, kw)
    print(json.dumps(summary, indent=2))
    print(f"run dir: {out}")
    return 0
