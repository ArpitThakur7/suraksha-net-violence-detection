"""Build colab/suraksha_cloud_runs.ipynb: one self-contained Colab notebook for the remaining seed runs.

    python colab/make_cloud_notebook.py

The notebook embeds (base64) the pipeline code, the clip manifest (SHA-256 of every video), the
locked splits and the distillation teacher targets. On Colab it downloads the three datasets from
their public sources (RWF-2000: Zenodo record 15687512 by the dataset author; SFD and AIRTLab:
GitHub), places every video by SHA-256 (`pipeline fetch`), rebuilds the frame cache and trains the
chosen PART's seeds 1-2 into the user's Google Drive, then zips the small result files.
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "suraksha_cloud_runs.ipynb"

PARTS = {  # PART -> grid --only substring (grid_seeds.yaml cell ids)
    "motionlite_kd": "distill_teacher",
    "motionlite": "/motionlite/",
    "mnv3_tsm": "mnv3_tsm",
    "videomae_ft": "videomae_ft",
}


def code_zip() -> tuple[bytes, str, int]:
    files = subprocess.run(["git", "ls-files", "pipeline", "requirements.txt"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in files if not f.startswith("pipeline/tests/")]
    files += ["work/manifest/all_clips.csv"]
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "work" / "splits").glob("*") if p.is_file())
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "work" / "probe").glob("*/videomae/teacher_train.csv"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(ROOT / f, f)   # raw bytes: split files must keep the hashes in LOCK.json
        z.writestr("GIT_COMMIT", commit + "\n")
    return buf.getvalue(), commit, len(files)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, form=False):
    meta = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build():
    blob, commit, n = code_zip()
    b64 = base64.b64encode(blob).decode()
    parts = ", ".join(f'"{p}"' for p in PARTS)
    cells = [
        md(f"""
# Suraksha-Net: remaining training runs on Colab

**Before running:** Runtime → Change runtime type → **L4 GPU** (or A100) → Save.
Then set `PART` in the first code cell and choose **Runtime → Run all**.

Each Google account runs a different `PART`:

| Account | PART | Runs |
|---|---|---|
| Colab 1 | `"motionlite_kd"` | MotionLite + VideoMAE distillation, seeds 1–2, 5 protocols (10 runs) |
| Colab 2 | `"motionlite"` | MotionLite, seeds 1–2, 5 protocols (10 runs) |
| Colab 3 | `"mnv3_tsm"` | MobileNetV3-TSM, seeds 1–2, 5 protocols (10 runs) |

What it does: downloads RWF-2000 (Zenodo, by the dataset author, CC-BY-4.0), SFD and AIRTLab (GitHub);
checks every video against the project's SHA-256 manifest; rebuilds the frame cache; trains; saves runs
to your Google Drive (`MyDrive/suraksha_cloud/`) and finally writes `results_<PART>.zip` there.

**If Colab disconnects:** just Runtime → Run all again. Finished runs are skipped and the run in
progress resumes from its last saved epoch on Drive (the download and cache steps repeat, ~20 min).

Embedded code: commit `{commit[:10]}`, {n} files.
"""),
        code(f"""
# ===== SETTINGS: change PART for each Google account =====
PART = "motionlite_kd"      # one of: {parts}
DRIVE_DIR = "/content/drive/MyDrive/suraksha_cloud"
"""),
        code(f"""
#@title 1. Check GPU and connect Google Drive
import os, sys, subprocess, torch
PARTS = {json.dumps(PARTS)}
assert PART in PARTS, f"PART must be one of {{list(PARTS)}}"
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > L4 GPU, then Run all again"
print("GPU:", torch.cuda.get_device_name(0), "| CPUs:", os.cpu_count())
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(f"{{DRIVE_DIR}}/runs", exist_ok=True)
print("Runs will be saved to", DRIVE_DIR)
""", form=True),
        code(f"""
#@title 2. Unpack the embedded project code (do not edit)
import base64, io, zipfile
CODE_B64 = "{b64}"
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall("/content/repo")
del CODE_B64
print("code commit", open("/content/repo/GIT_COMMIT").read().strip())
""", form=True),
        code("""
#@title 3. Install the two missing tools (PyAV video decoder, 7-Zip)
!pip -q install av
!apt-get -qq install -y p7zip-full > /dev/null && echo "7z ready"
""", form=True),
        md("**4. Download the datasets** (~12 GB, about 10–20 minutes)"),
        code("""
%%bash
set -e
mkdir -p /content/dl && cd /content/dl
[ -d sfd ] || git clone -q --depth 1 https://github.com/seymanurakti/fight-detection-surv-dataset sfd
[ -d avd ] || git clone -q --depth 1 https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos avd
if [ ! -d rwf ]; then
  mkdir -p rwf_parts
  seq -f "%03g" 1 13 | xargs -P 4 -I{} wget -q -c -O rwf_parts/RWF-2000.7z.{} "https://zenodo.org/records/15687512/files/RWF-2000.7z.{}?download=1"
  ls -l rwf_parts
  7z x -y -orwf rwf_parts/RWF-2000.7z.001 > /dev/null
  rm -rf rwf_parts
fi
du -sh sfd avd rwf
""", form=False),
        code("""
#@title 5. Check every video against the project's SHA-256 manifest
%cd /content/repo
r = subprocess.run([sys.executable, "-m", "pipeline", "fetch", "--src", "/content/dl", "--allow-name-match"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
assert r.returncode == 0, "Some videos are missing: send this output to Arpit / Claude before continuing"
""", form=True),
        code("""
#@title 6. Build the frame cache (~10-15 minutes)
%cd /content/repo
!python -m pipeline cache --workers {os.cpu_count()}
""", form=True),
        code("""
#@title 7. Train (several hours; results go to Google Drive as each run finishes)
%cd /content/repo
W = max(2, min(8, os.cpu_count() - 2))
ONLY = PARTS[PART]
!python -u -m pipeline grid --grid pipeline/configs/grid_seeds.yaml --only "{ONLY}" --seeds 1,2 --set "paths.runs={DRIVE_DIR}/runs" --set train.workers={W} --set train.val_workers=2 2>&1 | tee -a "{DRIVE_DIR}/log_{PART}.txt"
""", form=True),
        code("""
#@title 8. Package the results (small files only) -> Google Drive
import glob
out = f"{DRIVE_DIR}/results_{PART}.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{DRIVE_DIR}/runs/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f, os.path.relpath(f, DRIVE_DIR)); n += 1
done = len(glob.glob(f"{DRIVE_DIR}/runs/*/*/seed*/DONE"))
print(f"{out}: {n} files, {os.path.getsize(out) // 1024} KB, {done} finished runs")
print("Download this zip from Google Drive (MyDrive/suraksha_cloud) and give it to Claude on the laptop.")
""", form=True),
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {OUT.stat().st_size / 2**20:.1f} MB, code zip {len(blob) / 2**20:.2f} MB ({n} files), commit {commit[:10]}")


if __name__ == "__main__":
    build()
