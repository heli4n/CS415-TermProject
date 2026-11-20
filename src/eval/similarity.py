"""Artist-filtered similarity metrics (ECS-22).

Both metrics use cosine similarity between track embeddings.

GP@k (proposal Eq. 2) -- genre precision at k with artist filtering. For each
query track q, all tracks by q's artist (and q itself) are removed from the
candidate set; the remaining tracks are ranked by cosine similarity and

    GP@k(q) = |{j in top-k(q) : genre_j == genre_q}| / min(k, n_candidates(q))
    GP@k    = mean over queries with at least one candidate

With ``filter_artist=False`` only q itself is removed (the usual, artist-leaking
precision@k), so the gap between the two shows how much "same genre"
neighbours are really "same artist" neighbours.

ASM (proposal Eq. 3) -- artist similarity margin:

    ASM = mean_{i<j, a_i == a_j} cos(z_i, z_j)
        - mean_{i<j, g_i == g_j, a_i != a_j} cos(z_i, z_j)

i.e. how much closer two tracks of the same artist are than two tracks of the
same genre by different artists. Means are taken over unordered pairs (so
artists with many tracks weigh more). ASM > 0 means the embedding clusters by
artist beyond what genre explains.

Input file ``embeddings_<partition>.npz`` (DRAFT contract, to be merged into
``docs/data_contract.md``):

    track_id   int   (N,)
    embedding  float (N, D)
    genre      str or int (N,)   top-level genre
    artist_id  int   (N,)

CLI::

    python -m src.eval.similarity results/embeddings_test.npz --k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("track_id", "embedding", "genre", "artist_id")


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be (N, D), got shape {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding: cosine similarity undefined")
    return emb / norms


def _check_lengths(emb, *arrays) -> int:
    n = len(emb)
    for a in arrays:
        if len(a) != n:
            raise ValueError(f"length mismatch: {len(a)} labels for {n} embeddings")
    return n


def gp_at_k_per_query(emb: np.ndarray, genres, artists, k: int = 10,
                      filter_artist: bool = True, chunk_size: int = 1024) -> np.ndarray:
    """GP@k for every query track; NaN where the query has no candidates."""
    if k < 1:
        raise ValueError("k must be >= 1")
    genres, artists = np.asarray(genres), np.asarray(artists)
    n = _check_lengths(emb, genres, artists)
    x = _l2_normalize(emb)
    out = np.full(n, np.nan)
    kk = min(k, n - 1)
    if kk < 1:
        return out

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(stop - start)
        sim = x[start:stop] @ x.T                        # (c, n)
        sim[rows, start + rows] = -np.inf                # never retrieve itself
        if filter_artist:
            sim[artists[start:stop, None] == artists[None, :]] = -np.inf
        n_valid = np.isfinite(sim).sum(axis=1)

        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]   # top-kk, unordered
        valid = np.isfinite(np.take_along_axis(sim, top, axis=1))
        hits = (genres[top] == genres[start:stop, None]) & valid
        denom = np.minimum(n_valid, k)
        has = denom > 0
        out[start:stop][has] = hits.sum(axis=1)[has] / denom[has]
    return out


def genre_precision_at_k(emb: np.ndarray, genres, artists, k: int = 10,
                         filter_artist: bool = True, chunk_size: int = 1024) -> float:
    """Mean GP@k over queries that have at least one candidate (Eq. 2)."""
    per_q = gp_at_k_per_query(emb, genres, artists, k, filter_artist, chunk_size)
    if np.all(np.isnan(per_q)):
        raise ValueError("no query has any candidate track")
    return float(np.nanmean(per_q))


def artist_similarity_margin(emb: np.ndarray, genres, artists,
                             chunk_size: int = 1024) -> dict:
    """ASM (Eq. 3) plus its two terms and pair counts."""
    genres, artists = np.asarray(genres), np.asarray(artists)
    n = _check_lengths(emb, genres, artists)
    x = _l2_normalize(emb)
    s_same = s_cross = 0.0
    n_same = n_cross = 0

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sim = x[start:stop] @ x.T
        upper = np.arange(n)[None, :] > np.arange(start, stop)[:, None]   # i < j
        same_a = artists[start:stop, None] == artists[None, :]
        same_g = genres[start:stop, None] == genres[None, :]
        m_same = upper & same_a
        m_cross = upper & same_g & ~same_a
        s_same += sim[m_same].sum()
        n_same += int(m_same.sum())
        s_cross += sim[m_cross].sum()
        n_cross += int(m_cross.sum())

    if n_same == 0:
        raise ValueError("no same-artist pairs: every artist has a single track")
    if n_cross == 0:
        raise ValueError("no same-genre / different-artist pairs")
    sim_same, sim_cross = s_same / n_same, s_cross / n_cross
    return {
        "asm": float(sim_same - sim_cross),
        "sim_same_artist": float(sim_same),
        "sim_same_genre_diff_artist": float(sim_cross),
        "n_pairs_same_artist": n_same,
        "n_pairs_same_genre_diff_artist": n_cross,
    }


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in REQUIRED_KEYS if k not in f.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}; need {list(REQUIRED_KEYS)}")
        data = {k: f[k] for k in REQUIRED_KEYS}
    _check_lengths(data["embedding"], data["track_id"], data["genre"], data["artist_id"])
    return data


def save_embeddings(path: str | Path, track_id, embedding, genre, artist_id) -> None:
    """Writes the npz in the contract format (genre stored as fixed-width str/int)."""
    genre = np.asarray(genre)
    if genre.dtype == object:
        genre = genre.astype(str)
    np.savez(path, track_id=np.asarray(track_id, dtype=np.int64),
             embedding=np.asarray(embedding, dtype=np.float32),
             genre=genre, artist_id=np.asarray(artist_id, dtype=np.int64))


def evaluate(data: dict[str, np.ndarray], k: int = 10) -> dict:
    emb, g, a = data["embedding"], data["genre"], data["artist_id"]
    res = {
        "n_tracks": int(len(emb)),
        "n_artists": int(len(np.unique(a))),
        f"gp@{k}": genre_precision_at_k(emb, g, a, k, filter_artist=True),
        f"gp@{k}_unfiltered": genre_precision_at_k(emb, g, a, k, filter_artist=False),
    }
    res.update(artist_similarity_margin(emb, g, a))
    return res


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="GP@k and ASM for an embeddings_<partition>.npz")
    ap.add_argument("npz", type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None, help="also write results as JSON")
    args = ap.parse_args(argv)
    res = {"file": str(args.npz), **evaluate(load_embeddings(args.npz), args.k)}
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main()
