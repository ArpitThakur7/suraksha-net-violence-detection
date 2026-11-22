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
