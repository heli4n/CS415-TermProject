"""2-D embedding plots coloured by genre and by artist (ECS-23).

Pipeline: L2-normalise (so distances match the cosine metrics of ECS-22)
-> PCA to <= 50 dims -> t-SNE (sklearn) or UMAP (if ``umap-learn`` is
installed) -> two side-by-side scatter plots:

* left: coloured by genre
* right: coloured by artist, only the ``top_artists`` artists with the most
  tracks get a colour; everyone else is light grey. If the right panel shows
  tight single-colour islands inside the genre clusters, the embedding is
  grouping by artist (the qualitative counterpart of ASM).

Input: ``embeddings_<partition>.npz`` with ``track_id, embedding, genre,
artist_id`` (draft contract from ECS-22).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("track_id", "embedding", "genre", "artist_id")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in REQUIRED_KEYS if k not in f.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        return {k: f[k] for k in REQUIRED_KEYS}


def subsample(data: dict, n: int | None, seed: int = 0) -> dict:
    m = len(data["embedding"])
    if n is None or n >= m:
        return data
    idx = np.sort(np.random.default_rng(seed).choice(m, n, replace=False))
    return {k: v[idx] for k, v in data.items()}


def project_2d(emb: np.ndarray, method: str = "tsne", seed: int = 0,
               perplexity: float = 30.0) -> np.ndarray:
    from sklearn.decomposition import PCA

    x = np.asarray(emb, dtype=np.float64)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    n_pca = min(50, x.shape[1], x.shape[0])
    if x.shape[1] > n_pca:
        x = PCA(n_components=n_pca, random_state=seed).fit_transform(x)

    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(perplexity, max(2.0, (len(x) - 1) / 3))
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=seed).fit_transform(x)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("method='umap' needs `pip install umap-learn`") from e
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    raise ValueError(f"unknown method {method!r}")


def top_artists(artists: np.ndarray, n: int) -> list:
    """The n artists with the most tracks (ties broken by artist id)."""
    ids, counts = np.unique(artists, return_counts=True)
    order = sorted(range(len(ids)), key=lambda i: (-counts[i], ids[i]))
    return [ids[i].item() for i in order[:n]]


def plot_projection(xy: np.ndarray, genres, artists, out_path: str | Path,
                    n_top_artists: int = 10, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genres, artists = np.asarray(genres), np.asarray(artists)
    fig, (ax_g, ax_a) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for i, g in enumerate(sorted(np.unique(genres).tolist())):
        m = genres == g
        ax_g.scatter(xy[m, 0], xy[m, 1], s=6, alpha=.7, color=cmap(i % 10), label=str(g), linewidths=0)
    ax_g.set_title("by genre")
    ax_g.legend(markerscale=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    top = top_artists(artists, n_top_artists)
    rest = ~np.isin(artists, top)
    ax_a.scatter(xy[rest, 0], xy[rest, 1], s=5, color="#d0d0d0", linewidths=0, label="other")
    for i, a in enumerate(top):
        m = artists == a
        g = genres[m][0]
        ax_a.scatter(xy[m, 0], xy[m, 1], s=14, color=cmap(i % 10), linewidths=0,
                     label=f"artist {a} ({g}, n={m.sum()})")
    ax_a.set_title(f"by artist (top {len(top)} by track count)")
    ax_a.legend(markerscale=2, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for ax in (ax_g, ax_a):
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
