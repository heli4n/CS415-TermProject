"""Tests for the 6-split baseline summary (ECS-20)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import baseline_all_splits  # noqa: E402
from src.baselines import logmel_logreg as bl  # noqa: E402
from test_baseline_logreg import synth  # noqa: E402,F401  (shared fixture)


def fake(name, acc, f1=0.5):
    return {"split": name, "test_acc": acc, "test_macro_f1": f1}


def test_summary_hand_computed():
    res = {"random_0": fake("random_0", .70), "artist_0": fake("artist_0", .50),
           "random_1": fake("random_1", .72), "artist_1": fake("artist_1", .48),
           "random_2": fake("random_2", .68), "artist_2": fake("artist_2", .52)}
    s = bl.summarize_splits(res)
    # deltas .20 .24 .16 -> mean .20, sample std .04
    assert s["delta_acc"]["values"] == pytest.approx([.20, .24, .16])
    assert s["delta_acc"]["mean"] == pytest.approx(.20)
    assert s["delta_acc"]["std"] == pytest.approx(.04)
    assert s["acc_random"]["mean"] == pytest.approx(.70)
    assert s["acc_artist"]["std"] == pytest.approx(.02)
    table = bl.summary_table(s)
    assert "0.200 ± 0.040" in table and "artist-disjoint" in table


def test_end_to_end_six_splits(synth, tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    tracks = pd.read_csv(synth / "clean_tracks.csv")
    rng = np.random.default_rng(1)
    for k in range(3):
        a = (tracks.artist_id + k) % 10
        part_a = np.where(a < 8, "train", np.where(a == 8, "val", "test"))
        part_r = rng.permutation(np.array(["train"] * 384 + ["val"] * 48 + ["test"] * 48))
        pd.DataFrame({"track_id": tracks.track_id, "split": part_a}).to_csv(splits / f"artist_{k}.csv", index=False)
        pd.DataFrame({"track_id": tracks.track_id, "split": part_r}).to_csv(splits / f"random_{k}.csv", index=False)

    log, out = tmp_path / "log.jsonl", tmp_path / "summary.json"
    s = baseline_all_splits.main([
        "--cache-dir", str(synth / "cache"), "--tracks", str(synth / "clean_tracks.csv"),
        "--splits-dir", str(splits), "--log", str(log), "--summary-out", str(out), "--C", "0.01", "1",
    ])
    assert len(s["delta_acc"]["values"]) == 3
    assert s["delta_acc"]["mean"] > 0.1
    kinds = [json.loads(l)["experiment"] for l in log.read_text().splitlines()]
    assert kinds.count("baseline_mean_logmel_logreg") == 6
    assert kinds.count("baseline_mean_logmel_logreg_delta") == 3
    assert kinds[-1] == "baseline_mean_logmel_logreg_summary"
    assert set(json.loads(out.read_text())["runs"]) == {f"{t}_{k}" for t in ("random", "artist") for k in range(3)}
