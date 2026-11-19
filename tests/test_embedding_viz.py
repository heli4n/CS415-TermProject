"""Tests for src/eval/embedding_viz.py (ECS-23)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_embeddings  # noqa: E402
from src.eval import embedding_viz as viz  # noqa: E402


@pytest.fixture()
def npz(tmp_path):
    rng = np.random.default_rng(0)
    n_art, per = 12, 10
    artists = np.repeat(np.arange(n_art), per)
    genres = np.array(["Rock", "Pop", "Jazz"])[artists % 3]
    emb = rng.normal(size=(n_art, 32))[artists] + 0.1 * rng.normal(size=(n_art * per, 32))
    path = tmp_path / "embeddings_test.npz"
    np.savez(path, track_id=np.arange(len(artists)), embedding=emb.astype(np.float32),
             genre=genres, artist_id=artists)
    return path


def test_top_artists_by_count_then_id():
    a = np.array([5, 5, 5, 2, 2, 9, 9, 1])
    assert viz.top_artists(a, 2) == [5, 2]
    assert viz.top_artists(a, 10) == [5, 2, 9, 1]


def test_subsample_keeps_rows_aligned(npz):
    d = viz.subsample(viz.load_npz(npz), 30, seed=1)
    assert all(len(v) == 30 for v in d.values())
    full = viz.load_npz(npz)
    idx = d["track_id"]
    np.testing.assert_array_equal(full["artist_id"][idx], d["artist_id"])


@pytest.mark.parametrize("method", ["tsne", "pca"])
def test_projection_shape_and_determinism(npz, method):
    emb = viz.load_npz(npz)["embedding"]
    xy1 = viz.project_2d(emb, method, seed=0)
    xy2 = viz.project_2d(emb, method, seed=0)
    assert xy1.shape == (len(emb), 2)
    np.testing.assert_allclose(xy1, xy2)


def test_cli_writes_figure(npz, tmp_path):
    out = plot_embeddings.main([str(npz), "--out", str(tmp_path / "fig.png"),
                                "--top-artists", "5", "--title", "toy"])
    assert out.exists() and out.stat().st_size > 10_000


def test_umap_missing_gives_clear_error(npz, monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match="umap-learn"):
        viz.project_2d(viz.load_npz(npz)["embedding"], "umap")
