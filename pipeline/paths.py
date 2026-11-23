"""Filesystem constants. Importing this module also prepares the environment:
torchvision lives in ROOT/.ml_deps and pretrained weights must cache on E:."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
DATASETS = ROOT / "datasets"

_deps = str(ROOT / ".ml_deps")
if _deps not in sys.path:
    sys.path.insert(0, _deps)
os.environ.setdefault("TORCH_HOME", str(ROOT / "ml_runs" / "torch_cache"))


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def work_dir(cfg) -> Path:
    """paths.work (absolute, or relative to ROOT); WORK when unset."""
    return _resolve((cfg.get("paths") or {}).get("work") or WORK)


def runs_root(cfg) -> Path:
    """paths.runs (absolute, or relative to ROOT); <work>/runs when unset. On Colab it points into
    the shared Drive folder so runs survive disconnects while paths.work (cache) stays local."""
    runs = (cfg.get("paths") or {}).get("runs")
    return _resolve(runs) if runs else work_dir(cfg) / "runs"
