"""Compare two deployable CPU models on the laptop demo's validation windows."""
import csv
import hashlib
import json
import platform
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from suraksha.video import ROOT, STRIDE, WINDOW, analyze, detector, window_tensor

OUT = ROOT / "optimization"
MANIFEST = ROOT / "handoff/validation.csv"
NAMES = ("fp32", "dynamic_int8")
THRESHOLD = 0.50


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classification(rows, name):
    counts = Counter((int(row["label"]), int(row[name] >= THRESHOLD)) for row in rows)
    tn, fp, fn, tp = (counts[pair] for pair in [(0,0), (0,1), (1,0), (1,1)])
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, recall=tp/(tp+fn), accuracy=(tn+tp)/len(rows),
                f1=2*tp/(2*tp+fp+fn))


def export(model, name):
    path = OUT / (name + ".torchscript.pt")
    temp = path.with_suffix(".tmp")
    torch.jit.save(torch.jit.script(model), str(temp))
    reloaded = torch.jit.load(str(temp), map_location="cpu").eval()
    temp.replace(path)
    return reloaded, path.stat().st_size


def comparison_table(report):
    metrics = report["results"]
    lines = [
        "# Laptop CPU quantization comparison", "",
        f"{report['clips']} validation clips, {report['windows']} windows, threshold {report['threshold']:.2f}.",
        "A clip is flagged when any window reaches the threshold.", "",
        "| Metric | FP32 | Dynamic INT8 |", "| --- | ---: | ---: |",
    ]
    for label, key, scale, fmt in [
        ("Inference file (MiB)", "bytes", 1/1024**2, ".2f"),
        ("Median forward pass (ms)", "median_ms", 1, ".1f"),
        ("95th percentile forward pass (ms)", "p95_ms", 1, ".1f"),
        ("Clip accuracy (%)", "accuracy", 100, ".2f"),
        ("Violence recall (%)", "recall", 100, ".2f"),
        ("Violence F1 (%)", "f1", 100, ".2f"),
        ("Violent clips missed", "fn", 1, ".0f"),
        ("Nonviolent clips flagged", "fp", 1, ".0f"),
    ]:
        values = [format(metrics[name][key]*scale, fmt) for name in NAMES]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines.extend(["", f"Clip decisions changed: {report['changed_clip_decisions']}. "
                  f"Window decisions changed: {report['changed_window_decisions']}.", "",
                  f"PyTorch {report['torch']}, {report['backend']} backend, {report['threads']} CPU threads. "
                  "Timing covers a forward pass on a preprocessed 16-frame input. "
                  "Five warmups per model preceded paired timings with alternating order. "
                  "File sizes refer to reloadable TorchScript inference exports.",
                  "", "INT8 applies to the LSTM and two linear layers. The ResNet18 backbone uses FP32. "
                  "Membership of these videos in the checkpoint's original training split is unknown.",
                  "", "Reproduce from the project root with `python benchmark_quantization.py`."])
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model, mean, std = detector()
    int8 = quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False)
    eager = dict(zip(NAMES, (model, int8)))
    models, sizes = {}, {}
    for name in NAMES:
        models[name], sizes[name] = export(eager[name], name)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(r["split"] == "validation" and not r["exclusion"] for r in rows)
    clips, windows = [], []
    for row in rows:
        clip_windows = []

        def measure(frames):
            x = window_tensor(frames, mean, std)
            if not windows and not clip_windows:
                for name in NAMES:
                    torch.testing.assert_close(models[name](x), eager[name](x), rtol=1e-4, atol=1e-5)
                for _ in range(5):
                    for m in models.values():
                        m(x)
            order = NAMES if (len(windows)+len(clip_windows))%2==0 else NAMES[::-1]
            sample = {}
            # Reverse the order for the second timed pass to balance first-run effects.
            for name in order + order[::-1]:
                tick = time.perf_counter_ns()
                logits = models[name](x)
                elapsed = (time.perf_counter_ns() - tick) / 1e6
                sample[name+"_ms"] = sample.get(name+"_ms", 0) + elapsed/2
                score = float(logits.softmax(1)[0, 1])
                if not np.isfinite(score):
                    raise ValueError("Nonfinite score: "+row["path"])
                sample[name] = score
            clip_windows.append(sample)
            return sample["fp32"]

        _, timeline, _ = analyze(ROOT / row["path"], THRESHOLD, scorer=measure)
        assert len(timeline) == len(clip_windows)
        for item, w in zip(clip_windows, timeline):
            windows.append(dict(path=row["path"], start=w["start"], end=w["end"], **item))
        clips.append(dict(path=row["path"], dataset=row["dataset"], label=int(row["label"]),
                          **{n:max(w[n] for w in clip_windows) for n in NAMES}))
        if len(clips) % 20 == 0:
            print(f"Scored {len(clips)}/{len(rows)} clips, {len(windows)} windows", flush=True)
    results = {}
    for name in NAMES:
        times = [w[name+"_ms"] for w in windows]
        results[name] = dict(bytes=sizes[name], median_ms=float(np.median(times)),
                             p95_ms=float(np.percentile(times,95)), **classification(clips,name))
    with (ROOT / "review/checkpoint/best_model.pth").open("rb") as checkpoint:
        checkpoint_digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    report = dict(results=results, clips=len(clips), windows=len(windows), threshold=THRESHOLD,
                  timing="CPU forward pass, batch 1; two passes in reversed order, averaged per window; five paired warmups",
                  clip_rule="maximum window score >= threshold", window_s=WINDOW, stride_s=STRIDE,
                  frames_per_window=16, changed_clip_decisions=sum((c["fp32"]>=THRESHOLD)!=(c["dynamic_int8"]>=THRESHOLD) for c in clips),
                  changed_window_decisions=sum((w["fp32"]>=THRESHOLD)!=(w["dynamic_int8"]>=THRESHOLD) for w in windows),
                  validation_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  checkpoint_sha256=checkpoint_digest,
                  torch=torch.__version__, backend=torch.backends.quantized.engine, threads=torch.get_num_threads(),
                  interop_threads=torch.get_num_interop_threads(), machine=platform.machine(), platform=platform.platform())
    write_csv(OUT/"clip_predictions.csv", clips)
    write_csv(OUT/"window_timings.csv", windows)
    (OUT/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT/"comparison.md").write_text(comparison_table(report), encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
