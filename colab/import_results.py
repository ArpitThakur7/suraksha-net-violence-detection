"""Merge Colab result zips (results_<PART>.zip from suraksha_cloud_runs.ipynb) into work/runs.

    python colab/import_results.py path/to/results_motionlite_kd.zip [more.zip ...]

Only finished runs (with DONE) are imported, and a run that is already finished locally is left as
is. Then rebuild the table / figures / slides:
    python -m pipeline probe table && python -m pipeline figures && python reports/make_slides.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"


def import_zip(path: Path) -> tuple[int, int]:
    added = skipped = 0
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.startswith("runs/") and not n.endswith("/")]
        run_dirs = {PurePosixPath(n).parent for n in names if PurePosixPath(n).name == "DONE"}
        for rd in sorted(run_dirs):
            rel = Path(*rd.parts[1:])          # <protocol>/<variant>/seedN
            dst = RUNS / rel
            if (dst / "DONE").exists():
                skipped += 1
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for n in names:
                p = PurePosixPath(n)
                if p.parent == rd:
                    (dst / p.name).write_bytes(z.read(n))
            added += 1
    return added, skipped


def main(argv=None) -> int:
    paths = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        a, s = import_zip(p)
        print(f"{p.name}: imported {a} finished runs, skipped {s} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
