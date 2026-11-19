#!/usr/bin/env python3
"""Reference GP@k / ASM on mean log-mel vectors (ECS-22).

For each track: mean over time of the cached log-mel -> 128-d vector,
z-scored per dimension with *train* statistics, then GP@k and ASM from
``src/eval/similarity.py``. Also writes ``embeddings_<partition>_logmel.npz``
in the contract format so the same file can be fed to the CLI later.

Needs the log-mel cache (ECS-18) and a metadata CSV with one row per track:
``track_id, genre, artist_id`` and, optionally, a ``split`` column
(train/val/test). Column names are configurable because the clean metadata /
split files (Deniz) are not final yet.

Example::

    python scripts/similarity_reference.py \
        --cache-dir /content/drive/MyDrive/cs415/cache/logmel \
        --meta /content/drive/MyDrive/cs415/splits/artist_0.csv \
        --partition test --out-dir /content/drive/MyDrive/cs415/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import similarity  # noqa: E402

CACHE_FILENAME = "{track_id:06d}.npy"  # same as src/data/audio_config.py (ECS-17)


def mean_logmel(cache_dir: Path, track_ids) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vectors, found_mask); tracks without a cache file are dropped."""
    vecs, found = [], []
    for tid in track_ids:
        p = cache_dir / CACHE_FILENAME.format(track_id=int(tid))
        if p.exists():
            vecs.append(np.load(p).astype(np.float32).mean(axis=1))
            found.append(True)
        else:
            found.append(False)
    return np.stack(vecs), np.array(found)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--partition", default="test", help="value of --split-col to evaluate; 'all' = every row")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--track-col", default="track_id")
    ap.add_argument("--genre-col", default="genre")
    ap.add_argument("--artist-col", default="artist_id")
    ap.add_argument("--split-col", default="split")
    args = ap.parse_args(argv)

    meta = pd.read_csv(args.meta)
    has_split = args.split_col in meta.columns
    if args.partition != "all" and not has_split:
        ap.error(f"--partition {args.partition} needs a '{args.split_col}' column in {args.meta}")

    vecs, found = mean_logmel(args.cache_dir, meta[args.track_col])
    if (~found).any():
        print(f"warning: {(~found).sum()} track(s) have no cache file, dropped", file=sys.stderr)
    meta = meta[found].reset_index(drop=True)

    train = (meta[args.split_col] == "train").to_numpy() if has_split else np.ones(len(meta), bool)
    mu, sd = vecs[train].mean(axis=0), vecs[train].std(axis=0) + 1e-8
    z = (vecs - mu) / sd

    sel = np.ones(len(meta), bool) if args.partition == "all" else \
        (meta[args.split_col] == args.partition).to_numpy()
    data = {
        "track_id": meta[args.track_col].to_numpy()[sel],
        "embedding": z[sel],
        "genre": meta[args.genre_col].astype(str).to_numpy()[sel],
        "artist_id": meta[args.artist_col].to_numpy()[sel],
    }
    res = {"features": "mean_logmel_zscore_train", "partition": args.partition,
           "meta": str(args.meta), **similarity.evaluate(data, args.k)}
    print(json.dumps(res, indent=2))

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        similarity.save_embeddings(args.out_dir / f"embeddings_{args.partition}_logmel.npz", **data)
        (args.out_dir / f"similarity_{args.partition}_logmel.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main()
