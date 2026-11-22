"""Run the complete three-source release from the verified extracted RWF collection."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RWF=Path(os.environ.get("SURAKSHA_RWF_ROOT",r"F:\RWF 2000\suraksha_public_release\clips"))

def main():
    a=argparse.ArgumentParser();a.add_argument("--wait-for-extraction",action="store_true")
    options=a.parse_args()
    if options.wait_for_extraction:
        while True:
            paths=list(RWF.rglob("*.avi"))
            if len(paths)==2000 and time.time()-max(p.stat().st_mtime for p in paths)>15:
                # Extraction may preserve original mtimes; require stable byte sizes too.
                sizes=[p.stat().st_size for p in paths]
                time.sleep(10)
                if sizes==[p.stat().st_size for p in paths]:break
            time.sleep(10)
    if len(list(RWF.rglob("*.avi")))!=2000:raise ValueError("Expected 2,000 raw AVI files before the RWF audit")
    stages=[("prepare.py",["--rwf"]),("train_suite.py",["--rwf"]),("report.py",["three_sources"]),
            ("dataset_report.py",["--rwf"]),("evaluate_handoff.py",["three_sources"]),("package.py",["three_sources"])]
    for script,args in stages:
        print("Running",script,*args,flush=True)
        subprocess.run([sys.executable,"-u",str(ROOT/"semester_ml"/script),*args],cwd=ROOT,check=True)
if __name__=="__main__":main()

