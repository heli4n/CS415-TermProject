"""Mean log-mel + logistic regression baseline (ECS-19).

Per track: the cached log-mel (128, n_frames) averaged over time -> 128-d
vector. Standardisation is fit on the train partition only (inside the
pipeline), C is chosen on val accuracy, the model is refit on train and
evaluated once on test.

    Delta_acc = Acc(random split) - Acc(artist-disjoint split)

A positive Delta_acc means the random split over-estimates accuracy because
the same artists appear in train and test.

DRAFT input formats (Deniz's files, ECS-14/15, are not in the repo yet; the
column names are arguments so they are easy to change):

* tracks table (``clean_tracks.csv``): ``track_id, genre_top, artist_id``
* split file (``splits/{random,artist}_{k}.csv``): ``track_id, split`` with
  split in {train, val, test}
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CACHE_FILENAME = "{track_id:06d}.npy"  # same as src/data/audio_config.py (ECS-17)
PARTITIONS = ("train", "val", "test")
DEFAULT_CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# ----------------------------------------------------------------- features
def mean_logmel(cache_dir: Path, track_ids, feature_cache: Path | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (track_ids_found, X) with X[i] = time-mean of the cached log-mel.

    If ``feature_cache`` (npz with ``track_id``, ``X``) exists, vectors are read
    from it and only missing tracks are computed; the file is then updated.
    Reading 8000 .npy files from Drive is slow, the 128-d means are ~4 MB.
    """
    cached: dict[int, np.ndarray] = {}
    if feature_cache is not None and Path(feature_cache).exists():
        with np.load(feature_cache) as f:
            cached = dict(zip(f["track_id"].tolist(), f["X"]))

    ids, rows, n_new, n_missing = [], [], 0, 0
    for tid in map(int, track_ids):
        if tid in cached:
            vec = cached[tid]
        else:
            p = Path(cache_dir) / CACHE_FILENAME.format(track_id=tid)
            if not p.exists():
                n_missing += 1
                continue
            vec = np.load(p).astype(np.float32).mean(axis=1)
            cached[tid] = vec
            n_new += 1
        ids.append(tid)
        rows.append(vec)

    if n_missing:
        print(f"warning: {n_missing} track(s) have no cache file, dropped", file=sys.stderr)
    if feature_cache is not None and n_new:
        all_ids = np.array(sorted(cached))
        np.savez(feature_cache, track_id=all_ids, X=np.stack([cached[t] for t in all_ids]))
    if not rows:
        raise ValueError("no features found: is --cache-dir right?")
    return np.array(ids), np.stack(rows).astype(np.float32)


# ------------------------------------------------------------------- splits
def read_split(path: Path, track_col: str = "track_id", split_col: str = "split") -> pd.Series:
    df = pd.read_csv(path)
    for c in (track_col, split_col):
        if c not in df.columns:
            raise KeyError(f"{path}: missing column '{c}' (have {list(df.columns)})")
    bad = set(df[split_col]) - set(PARTITIONS)
    if bad:
        raise ValueError(f"{path}: unknown partitions {sorted(bad)}")
    if df[track_col].duplicated().any():
        raise ValueError(f"{path}: duplicate track ids")
    return df.set_index(track_col)[split_col]


# ---------------------------------------------------------------- the model
def run_split(track_ids: np.ndarray, X: np.ndarray, tracks: pd.DataFrame,
              split: pd.Series, split_name: str, Cs=DEFAULT_CS,
              genre_col: str = "genre_top", seed: int = 0) -> dict:
    """Fit/select/evaluate on one split. ``tracks`` is indexed by track_id."""
    t0 = time.time()
    part = split.reindex(track_ids).to_numpy()
    y = tracks[genre_col].reindex(track_ids).to_numpy()
    keep = pd.notna(part) & pd.notna(y)
    sets = {p: (X[keep & (part == p)], y[keep & (part == p)]) for p in PARTITIONS}
    for p, (Xp, _) in sets.items():
        if len(Xp) == 0:
            raise ValueError(f"{split_name}: empty {p} partition after joining features")

    def fit(C):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=5000, random_state=seed))
        return model.fit(*sets["train"])

    val_acc = {}
    for C in Cs:
        val_acc[C] = accuracy_score(sets["val"][1], fit(C).predict(sets["val"][0]))
    best_C = max(Cs, key=lambda c: (val_acc[c], -c))  # ties -> stronger regularisation

    model = fit(best_C)
    X_test, y_test = sets["test"]
    pred = model.predict(X_test)
    return {
        "experiment": "baseline_mean_logmel_logreg",
        "split": split_name,
        "n_train": len(sets["train"][1]), "n_val": len(sets["val"][1]), "n_test": len(y_test),
        "best_C": best_C,
        "val_acc_by_C": {str(c): round(a, 4) for c, a in val_acc.items()},
        "val_acc": val_acc[best_C],
        "test_acc": accuracy_score(y_test, pred),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "elapsed_s": round(time.time() - t0, 1),
    }


def delta_acc(random_res: dict, artist_res: dict) -> dict:
    return {
        "experiment": "baseline_mean_logmel_logreg_delta",
        "random_split": random_res["split"], "artist_split": artist_res["split"],
        "acc_random": random_res["test_acc"], "acc_artist": artist_res["test_acc"],
        "delta_acc": random_res["test_acc"] - artist_res["test_acc"],
        "f1_random": random_res["test_macro_f1"], "f1_artist": artist_res["test_macro_f1"],
    }


def summarize_splits(results: dict[str, dict], n_pairs: int = 3) -> dict:
    """ECS-20: pair random_k with artist_k, report mean +- std over k.

    std is the sample std (ddof=1) over the k pairs / k splits.
    """
    def stats(vals):
        vals = np.asarray(vals, dtype=float)
        return {"mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals]}

    ks = range(n_pairs)
    deltas = [delta_acc(results[f"random_{k}"], results[f"artist_{k}"]) for k in ks]
    return {
        "experiment": "baseline_mean_logmel_logreg_summary",
        "n_pairs": n_pairs,
        "acc_random": stats([results[f"random_{k}"]["test_acc"] for k in ks]),
        "acc_artist": stats([results[f"artist_{k}"]["test_acc"] for k in ks]),
        "f1_random": stats([results[f"random_{k}"]["test_macro_f1"] for k in ks]),
        "f1_artist": stats([results[f"artist_{k}"]["test_macro_f1"] for k in ks]),
        "delta_acc": stats([d["delta_acc"] for d in deltas]),
    }


def summary_table(summary: dict) -> str:
    """Markdown table for the report / group channel."""
    def fmt(s):
        return f"{s['mean']:.3f} ± {s['std']:.3f}"
    return "\n".join([
        f"| split type | test acc | macro-F1 |  (n = {summary['n_pairs']} splits each)",
        "|---|---|---|",
        f"| random | {fmt(summary['acc_random'])} | {fmt(summary['f1_random'])} |",
        f"| artist-disjoint | {fmt(summary['acc_artist'])} | {fmt(summary['f1_artist'])} |",
        f"| **Δacc** | **{fmt(summary['delta_acc'])}** | |",
    ])


def append_log(log_path: Path, record: dict) -> None:
    """Experiment log = one JSON object per line (draft format)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
