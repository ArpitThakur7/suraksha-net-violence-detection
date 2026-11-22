"""CLI: python -m pipeline split [--force]"""
import argparse

from pipeline.config import load_config
from pipeline.data import splits as S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline split")
    ap.add_argument("--force", action="store_true", help="overwrite an existing LOCK")
    ap.add_argument("--config", default="pipeline/configs/base.yaml")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    base = S.load_base(cfg)
    n_test = int(((base.dataset == "rwf") & (base.split == "test")).sum())
    assert n_test == 400, f"RWF test must be 400 clips, got {n_test}"
    try:
        S.write_protocols(cfg, base=base, force=a.force)
    except S.LockError as e:
        print(f"refusing: {e}")
        return 1
    for name in S.RWF_TEST_PROTOCOLS:
        m = S.load_protocol(name, cfg)
        n = int(((m.dataset == "rwf") & (m.split == "test")).sum())
        assert n == 400, f"{name}: RWF test = {n}"
    for name in S.PROTOCOLS:
        m = S.load_protocol(name, cfg)
        print(name, m.split.value_counts().reindex(S.SPLITS).fillna(0).astype(int).to_dict())
    return 0
