"""YAML config loading with `extends` and dotted overrides."""
import copy
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

from pipeline.paths import ROOT

DEFAULT_CONFIG = "pipeline/configs/base.yaml"


def _merge(base: dict, child: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    return p


def _load_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        pp = Path(parent)
        if not pp.is_absolute():
            pp = path.parent / pp
        data = _merge(_load_file(pp), data)
    return data


def apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict:
    for ov in overrides:
        key, sep, raw = ov.partition("=")
        if not sep:
            raise ValueError(f"override must look like a.b=value, got {ov!r}")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path=DEFAULT_CONFIG, overrides: Sequence[str] = ()) -> dict:
    cfg = _load_file(_resolve(path))
    return apply_overrides(cfg, overrides)


_PIP_HASH = None


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _pip_hash() -> str:
    global _PIP_HASH
    if _PIP_HASH is None:
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                               text=True, timeout=120)
            _PIP_HASH = hashlib.sha256(r.stdout.encode()).hexdigest() if r.returncode == 0 else "unknown"
        except Exception:
            _PIP_HASH = "unknown"
    return _PIP_HASH


def run_metadata() -> dict:
    import torch
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "cpu"
    return {
        "git_commit": _git_commit(),
        "pip_hash": _pip_hash(),
        "gpu": gpu,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "time": datetime.now().isoformat(timespec="seconds"),
    }
