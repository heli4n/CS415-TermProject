"""Tests for the mean log-mel + logistic regression baseline (ECS-19).

Synthetic data with a built-in artist effect: each artist has its own
"timbre" offset on top of a weak genre signal. A random split puts the same
artists in train and test, so it should score higher than an artist-disjoint
split -> Delta_acc > 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import baseline_logreg  # noqa: E402
from src.baselines import logmel_logreg as bl  # noqa: E402

GENRES = ["Rock", "Pop", "Jazz", "Folk"]
N_MELS, N_FRAMES = 16, 20


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    cache = root / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    genre_centres = rng.normal(0, 1.0, (len(GENRES), N_MELS))
    rows, tid = [], 1
    for artist in range(80):
        g = artist % len(GENRES)
        artist_offset = rng.normal(0, 3.0, N_MELS)  # strong artist "timbre"
        for _ in range(6):
            frames = (genre_centres[g] + artist_offset)[:, None] + rng.normal(0, 1, (N_MELS, N_FRAMES))
            np.save(cache / f"{tid:06d}.npy", frames.astype(np.float16))
            rows.append({"track_id": tid, "genre_top": GENRES[g], "artist_id": artist})
            tid += 1
    tracks = pd.DataFrame(rows)
    tracks.to_csv(root / "clean_tracks.csv", index=False)

    splits = root / "splits"
    splits.mkdir()
    # artist-disjoint: whole artists go to one partition (artist % 10)
    part_a = np.where(tracks.artist_id % 10 < 8, "train",
                      np.where(tracks.artist_id % 10 == 8, "val", "test"))
    # random: track level
    part_r = rng.permutation(np.array(["train"] * 384 + ["val"] * 48 + ["test"] * 48))
    pd.DataFrame({"track_id": tracks.track_id, "split": part_a}).to_csv(splits / "artist_0.csv", index=False)
    pd.DataFrame({"track_id": tracks.track_id, "split": part_r}).to_csv(splits / "random_0.csv", index=False)
    return root


def test_mean_logmel_is_time_average(synth):
    ids, X = bl.mean_logmel(synth / "cache", [1, 2, 99999])
    assert ids.tolist() == [1, 2]                      # missing file dropped
    ref = np.load(synth / "cache" / "000001.npy").astype(np.float32).mean(axis=1)
    np.testing.assert_allclose(X[0], ref)


def test_feature_cache_is_written_and_reused(synth, tmp_path):
    fc = tmp_path / "mean_logmel.npz"
    ids1, X1 = bl.mean_logmel(synth / "cache", range(1, 11), fc)
    assert fc.exists()
    ids2, X2 = bl.mean_logmel(tmp_path / "no_cache_here", range(1, 11), fc)  # all from npz
    np.testing.assert_array_equal(ids1, ids2)
    np.testing.assert_array_equal(X1, X2)


def test_read_split_validates(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("track_id,split\n1,train\n2,dev\n")
    with pytest.raises(ValueError, match="dev"):
        bl.read_split(p)
    p.write_text("track_id,split\n1,train\n1,test\n")
    with pytest.raises(ValueError, match="duplicate"):
        bl.read_split(p)


def test_end_to_end_delta_acc_positive_and_logged(synth, tmp_path):
    log = tmp_path / "log.jsonl"
    out = baseline_logreg.main([
        "--cache-dir", str(synth / "cache"), "--tracks", str(synth / "clean_tracks.csv"),
        "--splits-dir", str(synth / "splits"), "--log", str(log), "--C", "0.01", "1",
    ])
    r, a = out["results"]["random_0"], out["results"]["artist_0"]
    assert (r["n_train"], r["n_val"], r["n_test"]) == (384, 48, 48)
    assert (a["n_train"], a["n_val"], a["n_test"]) == (384, 48, 48)
    assert r["best_C"] in (0.01, 1.0)
    assert out["delta"]["delta_acc"] == pytest.approx(r["test_acc"] - a["test_acc"])
    assert out["delta"]["delta_acc"] > 0.1           # the planted artist effect shows up

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["experiment"] for l in lines] == [
        "baseline_mean_logmel_logreg", "baseline_mean_logmel_logreg",
        "baseline_mean_logmel_logreg_delta"]
    assert all("time_utc" in l for l in lines)


def test_test_set_does_not_leak_into_fit_or_model_selection(synth):
    """Scaler is fit on train, C on val: changing test features changes nothing but test acc."""
    tracks = pd.read_csv(synth / "clean_tracks.csv").set_index("track_id")
    split = bl.read_split(synth / "splits" / "artist_0.csv")
    ids, X = bl.mean_logmel(synth / "cache", tracks.index)
    base = bl.run_split(ids, X, tracks, split, "artist_0", Cs=(0.01, 1.0))

    X2 = X.copy()
    X2[split.reindex(ids).to_numpy() == "test"] += 100.0
    shifted = bl.run_split(ids, X2, tracks, split, "artist_0", Cs=(0.01, 1.0))
    assert shifted["val_acc_by_C"] == base["val_acc_by_C"]
    assert shifted["best_C"] == base["best_C"]


def test_tracks_outside_split_or_without_genre_are_ignored(synth):
    tracks = pd.read_csv(synth / "clean_tracks.csv").set_index("track_id")
    split = bl.read_split(synth / "splits" / "random_0.csv").iloc[10:]   # 10 tracks not in split
    ids, X = bl.mean_logmel(synth / "cache", tracks.index)
    res = bl.run_split(ids, X, tracks, split, "random_0", Cs=(1.0,))
    assert res["n_train"] + res["n_val"] + res["n_test"] == len(tracks) - 10


def test_read_split_accepts_create_splits_labels(tmp_path):
    """ECS-15 split files use training/validation/test."""
    import pandas as pd
    from src.baselines.logmel_logreg import read_split
    p = tmp_path / "artist_0.csv"
    pd.DataFrame({"track_id": [1, 2, 3], "genre_top": ["Rock", "Pop", "Folk"],
                  "artist_id": [1, 2, 3],
                  "split": ["training", "validation", "test"]}).to_csv(p, index=False)
    assert list(read_split(p)) == ["train", "val", "test"]
