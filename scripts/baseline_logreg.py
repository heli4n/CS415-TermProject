#!/usr/bin/env python3
"""Run the mean log-mel + logistic regression baseline (ECS-19).

Example (Colab)::

    python scripts/baseline_logreg.py \
        --cache-dir /content/drive/MyDrive/cs415/cache/logmel \
        --tracks    data/clean_tracks.csv \
        --splits-dir splits \
        --feature-cache /content/drive/MyDrive/cs415/cache/mean_logmel.npz \
        --log /content/drive/MyDrive/cs415/results/experiment_log.jsonl

By default runs ``random_0`` and ``artist_0`` and logs Delta_acc for the pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import logmel_logreg as bl  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="clean_tracks.csv")
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--random-split", default="random_0")
    ap.add_argument("--artist-split", default="artist_0")
    ap.add_argument("--feature-cache", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=Path("results/experiment_log.jsonl"))
    ap.add_argument("--C", type=float, nargs="+", default=list(bl.DEFAULT_CS))
    ap.add_argument("--track-col", default="track_id")
    ap.add_argument("--genre-col", default="genre_top")
    ap.add_argument("--split-col", default="split")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    tracks = pd.read_csv(args.tracks).set_index(args.track_col)
    ids, X = bl.mean_logmel(args.cache_dir, tracks.index, args.feature_cache)
    print(f"features: {len(ids)} tracks x {X.shape[1]} dims")

    results = {}
    for name in (args.random_split, args.artist_split):
        split = bl.read_split(args.splits_dir / f"{name}.csv", args.track_col, args.split_col)
        res = bl.run_split(ids, X, tracks, split, name, args.C, args.genre_col)
        bl.append_log(args.log, res)
        results[name] = res
        print(f"{name}: C={res['best_C']}  test acc={res['test_acc']:.4f}  "
              f"macro-F1={res['test_macro_f1']:.4f}")

    delta = bl.delta_acc(results[args.random_split], results[args.artist_split])
    bl.append_log(args.log, delta)
    print(json.dumps(delta, indent=2))
    return {"results": results, "delta": delta}


if __name__ == "__main__":
    main()
