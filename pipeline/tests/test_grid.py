import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

import pipeline.grid as G
from pipeline.config import load_config
from pipeline.paths import ROOT, runs_root, work_dir
from pipeline.train import run_dir


@pytest.fixture(scope="module")
def cells():
    return G.expand_grid(G.DEFAULT_GRID)


def test_expand_counts(cells):
    ids = [c.id for c in cells]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]
    main = Counter(c.model for c in cells if not c.model_kw and not c.overrides
                   and c.protocol not in G.BIAS_PROTOCOLS)
    # M2 mnv3_tsm: 7 protocol settings (p1_rwf, p1_sfd, p1_avd, p2_all, p3_no_rwf, p3_no_sfd,
    #    p3_no_avd) x seeds 0-2 = 21 (the bias cells on p4_*/p2_all_leaky are counted separately)
    assert main["mnv3_tsm"] == 21
    # M3 r2plus1d: 21 + 4 headline protocols (p2_all, p3_*) x 2 extra seeds (3, 4) = 29
    assert main["r2plus1d"] == 29
    # M5 motionlite (non-ablation): same as M3 = 29
    assert main["motionlite"] == 29
    # M1 cnn_lstm: 4 protocols (p2_all, p3_*) x 3 seeds = 12
    assert main["cnn_lstm"] == 12
    # M4 mvit_s: 12 + 4 headline protocols x 2 extra seeds = 20
    assert main["mvit_s"] == 20
    # ablations: 7 MotionLite variants (diff/tsm/pool/fusion, clip.T 8/32, crop 160)
    #    x 2 protocols (p2_all, p3_no_rwf) x 3 seeds = 42
    abl = [c for c in cells if c.model_kw or c.overrides]
    assert len(abl) == 42 and {c.model for c in abl} == {"motionlite"}
    # bias study: mnv3_tsm x {p4_random, p4_grouped, p2_all_leaky} x seeds 0-2 = 9
    bias = [c for c in cells if c.protocol in G.BIAS_PROTOCOLS]
    assert len(bias) == 9 and {c.model for c in bias} == {"mnv3_tsm"}
    # no learning-rate pre-step (controller decision 1): total = 168 - 15 + 9 = 162
    assert len(cells) == 21 + 29 + 29 + 12 + 20 + 42 + 9 == 162


def test_protocols_exist(cells):
    locked = json.loads((ROOT / "work" / "splits" / "LOCK.json").read_text())
    assert {c.protocol for c in cells} <= set(locked)


def test_priorities_and_setup(cells):
    by = {c.id: c for c in cells}
    assert by["p1_rwf/mnv3_tsm/seed0"].priority == 1
    assert by["p3_no_avd/mvit_s/seed2"].priority == 1
    assert by["p4_random/mnv3_tsm/seed1"].priority == 1
    assert by["p2_all/motionlite__diff-False/seed0"].priority == 2
    assert by["p3_no_rwf/motionlite__fusion-early/seed2"].priority == 2
    assert by["p2_all/motionlite__clip.T-8/seed0"].priority == 3
    assert by["p2_all/motionlite__clip.crop-160/seed1"].priority == 3
    assert by["p2_all/r2plus1d/seed3"].priority == 3
    assert by["p1_sfd/motionlite/seed0"].priority == 3
    assert by["p1_avd/r2plus1d/seed2"].priority == 3
    # fixed lr: mvit_s 1e-4 via setup (not part of the run name), every other model uses base.yaml
    assert all("train.lr=0.0001" in c.setup for c in cells if c.model == "mvit_s")
    assert not any(s.startswith("train.lr") for c in cells if c.model != "mvit_s" for s in c.setup)
    assert by["p2_all/motionlite__clip.T-8/seed0"].overrides == ("clip.T=8",)


def test_order_by_priority(cells):
    sel = G.select_cells(cells, ((0, 1, 2, 3, 4, 5, 6), 7))
    pr = [c.priority for c in sel]
    assert pr == sorted(pr) and len(sel) == len(cells)


def test_parse_shard():
    assert G.parse_shard("0/7") == ((0,), 7)
    assert G.parse_shard("1,2/7") == ((1, 2), 7)
    for bad in ("7/7", "1", "a/2", "-1/3"):
        with pytest.raises(ValueError):
            G.parse_shard(bad)


def test_shards_partition(cells):
    parts = [{c.id for c in G.select_cells(cells, ((i,), 7))} for i in range(7)]
    assert sum(len(p) for p in parts) == len(cells)
    assert set().union(*parts) == {c.id for c in cells}
    for i in range(7):
        for j in range(i + 1, 7):
            assert not parts[i] & parts[j]
    multi = G.select_cells(cells, ((1, 2), 7))
    assert {c.id for c in multi} == parts[1] | parts[2]
    assert [c.priority for c in multi] == sorted(c.priority for c in multi)


def test_shard_stable_across_processes(cells):
    ids = [c.id for c in cells]
    mine = [G.shard_of(i, 7) for i in ids]
    code = ("import sys, json; from pipeline.grid import shard_of; "
            "print(json.dumps([shard_of(l, 7) for l in sys.stdin.read().splitlines()]))")
    for hs in ("1", "2"):  # different hash() salts: Python's hash() would disagree, sha1 must not
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", code], input="\n".join(ids), capture_output=True,
                           text=True, cwd=ROOT, env=env, check=True)
        assert json.loads(r.stdout) == mine


def test_runs_root_and_run_dir(tmp_path):
    cfg = load_config()
    assert runs_root(cfg) == ROOT / "work" / "runs"
    cfg["paths"]["runs"] = str(tmp_path / "drive" / "runs")
    assert runs_root(cfg) == tmp_path / "drive" / "runs"
    cfg["paths"] = {"work": str(tmp_path / "w")}  # no paths.runs -> <work>/runs
    assert runs_root(cfg) == tmp_path / "w" / "runs" and work_dir(cfg) == tmp_path / "w"
    d = run_dir(cfg, "p2_all", "motionlite", 1, {"diff": False}, ["clip.T=8"])
    assert d == tmp_path / "w" / "runs" / "p2_all" / "motionlite__diff-False__clip.T-8" / "seed1"


def _fake_cells():
    mk = G.Cell
    return [mk("p2_all", "motionlite", 0, {}, ("clip.T=8",), 3),
            mk("p2_all", "mvit_s", 0, {}, (), 1, ("train.lr=0.0001",)),
            mk("p2_all", "mnv3_tsm", 0, {}, (), 1),
            mk("p3_no_rwf", "cnn_lstm", 1, {}, (), 2)]


def _tmp_cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {"work": str(tmp_path / "work"), "runs": str(tmp_path / "runs")}
    return cfg


def test_run_grid_skips_done_and_isolates_failures(tmp_path, monkeypatch, capsys):
    cfg = _tmp_cfg(tmp_path)
    cells = _fake_cells()
    done = run_dir(cfg, "p2_all", "mnv3_tsm", 0)
    done.mkdir(parents=True)
    (done / "DONE").write_text("")
    (done / "test_metrics.json").write_text("{}")
    trained, evaluated = [], []

    def fake_train(c, protocol, model, seed, out, model_kw=None):
        trained.append((model, Path(out), c["clip"]["T"], c["train"]["lr"]))
        if model == "cnn_lstm":
            raise RuntimeError("boom")
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "DONE").write_text("")
        return {}

    monkeypatch.setattr(G, "train_run", fake_train)
    monkeypatch.setattr(G, "evaluate_run", lambda out, c, views=1: evaluated.append(Path(out)))
    failed = G.run_grid(cells, cfg)
    assert [t[0] for t in trained] == ["mvit_s", "cnn_lstm", "motionlite"]  # priority order, done skipped
    assert trained[0][1] == tmp_path / "runs" / "p2_all" / "mvit_s" / "seed0"
    assert trained[0][3] == pytest.approx(1e-4) and trained[2][3] == pytest.approx(3e-4)
    assert trained[2][2] == 8 and trained[2][1].parent.name == "motionlite__clip.T-8"
    assert [e.parent.name for e in evaluated] == ["mvit_s", "motionlite__clip.T-8"]
    assert [c.id for c in failed] == ["p3_no_rwf/cnn_lstm/seed1"]
    assert cfg["clip"]["T"] == 16  # caller's cfg untouched


def test_run_grid_only_and_shard(tmp_path, monkeypatch):
    cfg = _tmp_cfg(tmp_path)
    seen = []
    monkeypatch.setattr(G, "train_run", lambda c, p, m, s, out, model_kw=None: seen.append(m))
    monkeypatch.setattr(G, "evaluate_run", lambda *a, **k: None)
    monkeypatch.setattr(G, "_finished", lambda out: False)
    G.run_grid(_fake_cells(), cfg, only="motionlite")
    assert seen == ["motionlite"]


def test_estimate_hours(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    bench = tmp_path / "work" / "bench"
    bench.mkdir(parents=True)
    pd.DataFrame({"model": ["motionlite", "mnv3_tsm"], "seconds_per_epoch": [360.0, 120.0],
                  "batch": [8, 8], "gpu": ["x", "x"]}).to_csv(bench / "epoch_seconds.csv", index=False)
    splits = tmp_path / "work" / "splits"
    splits.mkdir()
    pd.DataFrame({"split": ["train"] * 6 + ["val"] * 2 + ["test"] * 5}).to_csv(splits / "p2_all.csv")
    pd.DataFrame({"split": ["train"] * 3 + ["val"] * 1 + ["test"] * 9}).to_csv(splits / "p3_no_rwf.csv")
    sec = G.load_epoch_seconds(cfg)
    c = G.Cell("p2_all", "motionlite", 0, {}, (), 1)
    assert G.estimate_hours(c, sec, cfg) == pytest.approx(360 * G.EPOCHS_PER_RUN / 3600)
    half = G.Cell("p3_no_rwf", "motionlite", 0, {}, ("clip.T=8",), 3)
    assert G.estimate_hours(half, sec, cfg) == pytest.approx(360 * G.EPOCHS_PER_RUN / 3600 * 0.5 * 0.5)
    import math
    assert math.isnan(G.estimate_hours(G.Cell("p2_all", "mvit_s", 0, {}, (), 1), sec, cfg))


def test_cli_dry_run_repeated_set(tmp_path, capsys):
    rc = G.main(["--dry-run", "--shard", "1,2/7", "--set", f"paths.work={tmp_path.as_posix()}/w",
                 "--set", f"paths.runs={tmp_path.as_posix()}/r"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "priority 1" in out and "priority 3" in out and "shard" in out
    assert not (tmp_path / "r").exists()


def test_bundle(tmp_path):
    root = tmp_path / "repo"
    for f in ("pipeline/a.py", "colab/train.ipynb", "datasets/x.csv", "work/splits/p2_all.csv",
              "handoff/data.zip", "README.md"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(f)
    cfg = _tmp_cfg(tmp_path)
    cache = tmp_path / "work" / "cache" / "abcd1234"
    cache.mkdir(parents=True)
    (cache / "c1.npz").write_bytes(b"\xff" * 100)
    (cache / "c2.npz").write_bytes(b"\xfe" * 50)
    splits = tmp_path / "work" / "splits"
    splits.mkdir()
    (splits / "LOCK.json").write_text("{}")
    (splits / "p2_all.csv").write_text("split\ntrain\n")
    (tmp_path / "work" / "bench").mkdir()
    (tmp_path / "work" / "bench" / "epoch_seconds.csv").write_text("model\n")
    files = ["pipeline/a.py", "colab/train.ipynb", "datasets/x.csv", "work/splits/p2_all.csv",
             "handoff/data.zip", "README.md"]
    out = tmp_path / "bundle"
    sizes = G.make_bundle(cfg, out, root=root, files=files, cache_key="abcd1234", commit="deadbeef")
    with zipfile.ZipFile(out / "repo.zip") as z:
        names = set(z.namelist())
        assert names == {"pipeline/a.py", "colab/train.ipynb", "README.md", "GIT_COMMIT"}
        assert z.read("GIT_COMMIT").decode().strip() == "deadbeef"
    with zipfile.ZipFile(out / "cache.zip") as z:
        assert sorted(z.namelist()) == ["cache/abcd1234/c1.npz", "cache/abcd1234/c2.npz"]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    assert (out / "splits" / "LOCK.json").exists() and (out / "splits" / "p2_all.csv").exists()
    assert (out / "bench" / "epoch_seconds.csv").exists()
    assert set(sizes) >= {"repo.zip", "cache.zip"}


def test_notebook_valid():
    nb = json.loads((ROOT / "colab" / "train.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for s in ('SHARDS = "2,5"', "N_SHARDS = 7", "DRIVE_DIR", "MACHINE", "drive.mount",
              "torch.cuda.is_available", "repo.zip", "cache.zip", "requirements.txt",
              "pipeline grid --shard", "paths.work=/content/work", "paths.runs=", "pipeline evaluate --all"):
        assert s in src, s


def test_requirements_minimal():
    # requirement lines only (comments stripped): Colab's preinstalled torch must not be reinstalled
    lines = [l.split("#", 1)[0].strip() for l in (ROOT / "requirements.txt").read_text().lower().splitlines()]
    pkgs = [l for l in lines if l]
    assert any(l.startswith("av") for l in pkgs)
    assert not any(l.startswith("torch") for l in pkgs)
