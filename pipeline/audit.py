"""CLI: python -m pipeline audit [--groups]"""
import argparse

from pipeline.config import load_config
from pipeline.data.ingest import build_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline audit")
    ap.add_argument("--groups", action="store_true", help="global near-duplicate groups (Task 4)")
    ap.add_argument("--config", default="pipeline/configs/base.yaml")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args(argv)
    df = build_manifest(load_config(a.config, a.override))
    print(df.groupby(["dataset", "original_split", "label"]).size().to_string())
    print("exclusion counts:")
    print(df.exclusion.replace("", "(kept)").value_counts().to_string())
    if a.groups:
        print("groups: not implemented yet")
    return 0
