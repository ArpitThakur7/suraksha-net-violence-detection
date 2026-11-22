import json

import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.data.splits import (LEAKY_PROTOCOLS, PROTOCOLS, LeakError, LockError, base_splits,
                                  check_no_leak, load_protocol, protocol_manifest, write_protocols)


def synth():
    rows = []

    def add(i, ds, split, label, grp, shared=False, clip=None):
        rows.append(dict(clip_id=clip or f"c{i:03d}", path=f"d/{ds}/{i}.mp4", dataset=ds, label=label,
                         source_id=grp, original_split=split, global_group=grp,
                         shared_source=shared, exclusion=""))
    i = 0
    # RWF train: 30 clips, own groups; 2 per group for some
    for k in range(30):
        i += 1
        add(i, "rwf", "train", k % 2, f"rwf:g{k // 2}")
    # RWF val (test): 10 clips; one shares group yt:X with an SFD train row, one is byte-identical
    for k in range(10):
        i += 1
        add(i, "rwf", "val", k % 2, f"rwf:t{k}")
    rows[-1].update(global_group="yt:X", shared_source=True)
    i += 1
    add(i, "rwf", "val", 0, "rwf:dupgrp", clip="c001")  # identical bytes as first RWF train clip
    rows[-1]["global_group"] = rows[0]["global_group"]
    # SFD
    for k, sp in enumerate(["train"] * 6 + ["validation"] * 2 + ["test"] * 3):
        i += 1
        add(i, "sfd", sp, k % 2, f"sfd:{k}")
    i += 1
    add(i, "sfd", "train", 1, "yt:X", shared=True)
    # AVD
    for k, sp in enumerate(["train"] * 6 + ["validation"] * 2 + ["test"] * 3):
        i += 1
        add(i, "avd", sp, k % 2, f"avd:{k}")
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"]["work"] = str(tmp_path)
    return c


@pytest.fixture
def base(cfg):
    return base_splits(synth(), cfg)


def test_no_leak_all_protocols(base):
    for name in PROTOCOLS:
        m = protocol_manifest(base, name, drop_shared=name != "p2_all_leaky")
        assert m.clip_id.is_unique
        if name not in LEAKY_PROTOCOLS:
            check_no_leak(m)


def test_leaky_protocols_leak():
    rows = []
    for k in range(60):
        rows.append(dict(clip_id=f"r{k}", path=f"p{k}", dataset="rwf", label=k % 2,
                         source_id="g", original_split="train", global_group=f"G{k // 6}",
                         shared_source=False, exclusion=""))
    b = base_splits(pd.DataFrame(rows), load_config())
    with pytest.raises(LeakError):
        check_no_leak(protocol_manifest(b, "p4_random"))
    check_no_leak(protocol_manifest(b, "p4_grouped"))
    # p2_all_leaky keeps a train row sharing a group with a test row
    s = base_splits(synth(), load_config())
    leaky = protocol_manifest(s, "p2_all_leaky", drop_shared=False)
    with pytest.raises(LeakError):
        check_no_leak(leaky)


def test_cross_dataset_drop(base):
    sfd_x = base[(base.dataset == "sfd") & (base.global_group == "yt:X")].clip_id.iloc[0]
    for name in ("p2_all", "p3_no_rwf"):
        m = protocol_manifest(base, name)
        assert sfd_x not in set(m.clip_id)
    assert sfd_x in set(protocol_manifest(base, "p2_all_leaky", drop_shared=False).clip_id)
    assert (protocol_manifest(base, "p2_all").attrs["dropped"])


def test_identical_clip_unique_and_test_kept(base):
    m = protocol_manifest(base, "p1_rwf")
    assert m.clip_id.is_unique
    assert m[m.clip_id == "c001"].split.tolist() == ["test"]
    assert (m.split == "test").sum() == 11


def test_rwf_val_is_test(base):
    m = protocol_manifest(base, "p1_rwf")
    ids = set(base[(base.dataset == "rwf") & (base.original_split == "val")].clip_id)
    assert set(m[m.clip_id.isin(ids)].split) == {"test"}
    assert (m[m.split == "test"].dataset == "rwf").all()


def test_val_fraction():
    rows = [dict(clip_id=f"r{k}", path=f"p{k}", dataset="rwf", label=k % 2, source_id="s",
                 original_split="train", global_group=f"G{k // 2}", shared_source=False, exclusion="")
            for k in range(400)]
    b = base_splits(pd.DataFrame(rows), load_config())
    tr = b[b.original_split == "train"]
    share = tr[tr.split == "val"].global_group.nunique() / tr.global_group.nunique()
    assert abs(share - 0.15) <= 0.03


def test_val_never_from_test_group(base):
    rv = base[(base.dataset == "rwf") & (base.original_split == "train") & (base.split == "val")]
    assert not set(rv.global_group) & set(base[(base.dataset == "rwf") & (base.original_split == "val")].global_group)


def test_lock_detects_edit(cfg, base):
    write_protocols(cfg, base=base)
    load_protocol("p1_rwf", cfg)
    with pytest.raises(LockError):
        write_protocols(cfg, base=base)  # refuses overwrite
    p = tmp = __import__("pathlib").Path(cfg["paths"]["work"]) / "splits" / "p1_rwf.csv"
    data = bytearray(p.read_bytes())
    data[-3] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(LockError):
        load_protocol("p1_rwf", cfg)
    assert load_protocol("p1_rwf", cfg, verify=False) is not None
    lock = json.loads((p.parent / "LOCK.json").read_text())
    assert set(lock) == set(PROTOCOLS)
    write_protocols(cfg, base=base, force=True)
    assert load_protocol("p1_rwf", cfg).clip_id.is_unique
