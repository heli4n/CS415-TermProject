#!/usr/bin/env python3
"""Baseline on all 6 splits -> Delta_acc mean +- std (ECS-20).

Runs ``random_{0,1,2}`` and ``artist_{0,1,2}``, logs every run and the
pairwise Delta_acc_k = Acc(random_k) - Acc(artist_k), then the summary
(mean +- sample std over k). Same arguments as ``baseline_logreg.py``.

    python scripts/baseline_all_splits.py \
        --cache-dir /content/drive/MyDrive/cs415/cache/logmel \
        --tracks data/clean_tracks.csv --splits-dir splits \
        --feature-cache /content/drive/MyDrive/cs415/cache/mean_logmel.npz \
        --log /content/drive/MyDrive/cs415/results/experiment_log.jsonl \
        --summary-out /content/drive/MyDrive/cs415/results/baseline_summary.json
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


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--feature-cache", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=Path("results/experiment_log.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--C", type=float, nargs="+", default=list(bl.DEFAULT_CS))
    ap.add_argument("--track-col", default="track_id")
    ap.add_argument("--genre-col", default="genre_top")
    ap.add_argument("--split-col", default="split")
    args = ap.parse_args(argv)

    tracks = pd.read_csv(args.tracks).set_index(args.track_col)
    ids, X = bl.mean_logmel(args.cache_dir, tracks.index, args.feature_cache)

    results = {}
    for k in range(args.n_pairs):
        for name in (f"random_{k}", f"artist_{k}"):
            split = bl.read_split(args.splits_dir / f"{name}.csv", args.track_col, args.split_col)
            res = bl.run_split(ids, X, tracks, split, name, args.C, args.genre_col)
            bl.append_log(args.log, res)
            results[name] = res
            print(f"{name}: C={res['best_C']}  acc={res['test_acc']:.4f}  F1={res['test_macro_f1']:.4f}")
        bl.append_log(args.log, bl.delta_acc(results[f"random_{k}"], results[f"artist_{k}"]))

    summary = bl.summarize_splits(results, args.n_pairs)
    bl.append_log(args.log, summary)
    print(bl.summary_table(summary))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
