#!/usr/bin/env python3
"""t-SNE / UMAP figure of an embeddings npz, coloured by genre and artist (ECS-23).

    python scripts/plot_embeddings.py results/embeddings_test.npz \
        --out figures/tsne_cnn_artist0_test.pdf --method tsne --title "CNN, artist_0, test"

Use the same ``--seed`` and ``--max-points`` for every model you compare.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import embedding_viz as viz  # noqa: E402


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, required=True, help=".png or .pdf")
    ap.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    ap.add_argument("--top-artists", type=int, default=10)
    ap.add_argument("--max-points", type=int, default=None, help="random subsample for speed")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)

    data = viz.subsample(viz.load_npz(args.npz), args.max_points, args.seed)
    xy = viz.project_2d(data["embedding"], args.method, args.seed, args.perplexity)
    out = viz.plot_projection(xy, data["genre"], data["artist_id"], args.out,
                              args.top_artists, args.title)
    print(f"wrote {out} ({len(xy)} points, {args.method})")
    return out


if __name__ == "__main__":
    main()
