"""Read-only source checks. No training, weight loading, or video decoding.

Dataset membership checks execute the repository's original discovery/splitting
functions with lightweight dataset/loader substitutes and empty path fixtures.
"""
from __future__ import annotations

import ast
import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo-snapshot"
RESULTS = {}


def extract_functions(path, names, scope=None):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {} if scope is None else dict(scope)
    exec(compile(module, str(REPO / path), "exec"), ns)
    return ns


class Weights(list):
    def tolist(self):
        return list(self)


class DatasetStub:
    def __init__(self, paths, labels, sources, **kwargs):
        self.video_paths, self.labels, self.sources = paths, labels, sources

    def __len__(self):
        return len(self.video_paths)

    def get_class_weights(self):
        return Weights([1, 1])


class LoaderStub:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset


scope = extract_functions("datasets/video_dataset.py", {"discover_videos", "split_dataset", "build_dataloaders"}, {
    "os": os, "random": random, "logger": logging.getLogger("audit"),
    "VideoDataset": DatasetStub, "DataLoader": LoaderStub,
})


def config(data_root, name):
    return {"dataset": {"name": name, "data_root": str(data_root), "sequence_length": 16, "image_size": 224},
            "training": {"batch_size": 8}, "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
            "experiment": {"seed": 42}}


def markers(root, split, count):
    for label in ("Violence", "NonViolence"):
        folder = root / "rwf" / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (folder / f"clip_{i:03d}.avi").touch()


with tempfile.TemporaryDirectory(prefix="split-check-", dir=ROOT) as tmp:
    data = Path(tmp)
    markers(data, "train", 50)
    markers(data, "val", 10)
    a, b, c, info = scope["build_dataloaders"](config(data, "rwf"))
    RESULTS["train_val_layout_without_test"] = info
    markers(data, "test", 10)
    a, b, c, info = scope["build_dataloaders"](config(data, "combined"))
    moved = [p for p in a.dataset.video_paths if f"{os.sep}test{os.sep}" in p]
    RESULTS["combined_discards_original_split_membership"] = {
        "counts": info, "original_test_paths_now_in_train": len(moved),
        "example": str(Path(moved[0]).relative_to(data)) if moved else None,
    }

paths, labels, sources = [], [], []
for label, count in [(0, 60), (1, 115)]:
    for camera in (1, 2):
        for scene in range(1, count + 1):
            paths.append(f"avd_label{label}_cam{camera}_{scene}.mp4")
            labels.append(label)
            sources.append("avd")
splits = scope["split_dataset"](paths, labels, sources)
groups = {key: {re.sub(r"cam[12]_", "", path) for path in value[0]} for key, value in splits.items()}
RESULTS["paired_camera_scene_fixture"] = {
    "train_test_shared_scenes": len(groups["train"] & groups["test"]),
    "train_val_shared_scenes": len(groups["train"] & groups["val"]),
    "note": "Demonstrates missing scene grouping; not an observation of original unavailable split manifests.",
}

factory = extract_functions("models/architectures.py", {"build_model"})["build_model"]
try:
    factory(json.loads((REPO / "handoff_to_arpit/model_config.json").read_text()))
except Exception as exc:
    RESULTS["documented_handoff_config_call"] = {"exception": type(exc).__name__, "message": str(exc)}

for path, argv in [("train.py", ["--amp"]), ("evaluate.py", ["--model-dir", "experiments/combined_cnn_lstm_16frames"])]:
    parse = extract_functions(path, {"parse_args"}, {"argparse": argparse})["parse_args"]
    old_args = sys.argv
    stream = io.StringIO()
    try:
        sys.argv = [path] + argv
        with contextlib.redirect_stderr(stream):
            parse()
    except SystemExit as exc:
        RESULTS["readme_cli_" + path] = {"exit_code": exc.code, "last_error": stream.getvalue().strip().splitlines()[-1]}
    finally:
        sys.argv = old_args

checked, mismatches = 0, []
for path in REPO.rglob("test_results.json"):
    results = json.loads(path.read_text())
    records = {"default": results["metrics_at_default_0_5"], **results["threshold_sweep"]}
    for name, m in records.items():
        tp, tn, fp, fn = (m[k] for k in ("true_positives", "true_negatives", "false_positives", "false_negatives"))
        expected = {"accuracy": (tp+tn)/(tp+tn+fp+fn), "precision": tp/(tp+fp) if tp+fp else 0,
                    "recall": tp/(tp+fn) if tp+fn else 0, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
        for metric, value in expected.items():
            checked += 1
            if abs(m[metric] - value) > 1e-10:
                mismatches.append(f"{path.relative_to(REPO)} {name} {metric}")
RESULTS["saved_metric_arithmetic"] = {"values_checked": checked, "mismatches": mismatches,
    "note": "Checks arithmetic only; does not independently reproduce predictions or ROC-AUC."}

sample_errors = []
for path in REPO.rglob("error_analysis.json"):
    data = json.loads(path.read_text())
    for kind in ("false_positives", "false_negatives"):
        for item in data[kind]:
            if item["video_name"].startswith("sample_"):
                sample_errors.append({"artifact": str(path.relative_to(REPO)), **item})
RESULTS["synthetic_named_test_errors"] = sample_errors
RESULTS["python_syntax"] = {"files_checked": len(list(REPO.rglob("*.py"))), "valid": True}
for path in REPO.rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"))

(ROOT / "verification-results.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
print(json.dumps(RESULTS, indent=2))
