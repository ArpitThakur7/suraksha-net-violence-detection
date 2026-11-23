import pandas as pd

import pipeline.bench as bench
import pipeline.train as train


def test_epoch_pilot_writes_csv_and_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_train_run(cfg, protocol, model, seed, out):
        assert cfg["train"]["max_epochs"] == 1 and protocol == "p2_all"
        calls.append(model)
        return {}

    monkeypatch.setattr(train, "train_run", fake_train_run)
    cfg = {"paths": {"work": str(tmp_path)}, "train": {"max_epochs": 40}, "clip": {"T": 16, "crop": 224}}
    df = bench.epoch_pilot(cfg, models=("cnn_lstm", "motionlite"))
    assert calls == ["cnn_lstm", "motionlite"]
    csv = pd.read_csv(tmp_path / "bench" / "epoch_seconds.csv")
    assert list(csv.model) == ["cnn_lstm", "motionlite"] and (csv.seconds_per_epoch >= 0).all()
    assert cfg["train"]["max_epochs"] == 40          # caller's config untouched
    assert not (tmp_path / "bench" / "epoch_pilot").exists()
    # already-timed models are skipped on a rerun
    bench.epoch_pilot(cfg, models=("cnn_lstm", "motionlite", "mnv3_tsm"))
    assert calls == ["cnn_lstm", "motionlite", "mnv3_tsm"]
    assert len(df) == 2
