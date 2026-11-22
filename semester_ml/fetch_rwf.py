"""Fetch the publisher's multipart release with integrity checks and extract AVI clips."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import argparse
from pathlib import Path
import subprocess
import urllib.request

BASE = "https://zenodo.org/records/15687512/files/"
ROOT = Path(r"F:\RWF 2000\suraksha_public_release")
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prefetch-tail",action="store_true",help="Stage the last four volumes while early volumes download")
    args=parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    record = json.load(urllib.request.urlopen("https://zenodo.org/api/records/15687512", timeout=90))
    files = sorted(record["files"], key=lambda x:x["key"])
    assert [f["key"] for f in files] == [f"RWF-2000.7z.{i:03d}" for i in range(1,14)]

    def fetch(item):
        filename = item["key"]
        dest = ROOT / filename
        partial = dest.with_suffix(dest.suffix + (".prefetch" if args.prefetch_tail else ".partial"))
        if not dest.exists():
            subprocess.run(["curl.exe","--fail","--silent","--show-error","--location","--retry","3",
                            "--connect-timeout","25","--speed-time","90","--speed-limit","1000",
                            "--continue-at","-","--output",str(partial),BASE+filename],check=True)
        source = dest if dest.exists() else partial
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "md5").hexdigest()
        if checksum != item["checksum"].split(":")[-1] or source.stat().st_size != item["size"]:
            raise ValueError(f"Invalid download: {source}")
        if source == partial:
            partial.replace(dest)
        print("Verified", filename, flush=True)
    with ThreadPoolExecutor(4 if args.prefetch_tail else 3) as pool:
        list(pool.map(fetch, files[9:][::-1] if args.prefetch_tail else files))
    if args.prefetch_tail: raise SystemExit(0)
    subprocess.run([r"C:\Program Files\7-Zip\7z.exe","x",str(ROOT / files[0]["key"]),
                    "-o"+str(ROOT / "clips"),"-aos","-r","*.avi","-bso0"], check=True)
    print("Extracted",sum(1 for _ in (ROOT/"clips").rglob("*.avi")),"AVI files")
