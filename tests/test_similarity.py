"""Tests for src/eval/similarity.py (ECS-22) on hand-computable toy sets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import similarity as sim  # noqa: E402


def unit(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


# 7 tracks on the unit circle. Cosine similarity is monotone in angular
# distance, so the nearest neighbours can be read off the angles.
#        angle  artist genre
GP_SET = [(0,   1, "Rock"),
          (10,  1, "Rock"),
          (25,  2, "Rock"),
          (45,  3, "Pop"),
          (70,  4, "Pop"),
          (100, 4, "Pop"),
          (140, 5, "Rock")]


@pytest.fixture()
def gp_set():
    emb = np.stack([unit(a) for a, _, _ in GP_SET])
    artists = np.array([a for _, a, _ in GP_SET])
    genres = np.array([g for _, _, g in GP_SET])
    return emb, genres, artists


def test_gp_at_2_artist_filtered(gp_set):
    # same-artist tracks removed; top-2 by angle:
    # 0:{25R,45P} 10:{25R,45P} 25:{10R,45P} 45:{25R,70P}
    # 70:{45P,25R} 100:{140R,45P} 140:{100P,70P}
    per_q = sim.gp_at_k_per_query(*gp_set, k=2)
    np.testing.assert_allclose(per_q, [.5, .5, .5, .5, .5, .5, 0.])
    assert sim.genre_precision_at_k(*gp_set, k=2) == pytest.approx(3 / 7)


def test_gp_at_2_unfiltered_is_higher(gp_set):
    # 0:{10R,25R} 10:{0R,25R} 25:{10R,45P} 45:{25R,70P}
    # 70:{45P,100P} 100:{70P,140R} 140:{100P,70P}
    per_q = sim.gp_at_k_per_query(*gp_set, k=2, filter_artist=False)
    np.testing.assert_allclose(per_q, [1, 1, .5, .5, 1, .5, 0])
    assert sim.genre_precision_at_k(*gp_set, k=2, filter_artist=False) == pytest.approx(9 / 14)


def test_gp_at_1_artist_filtered(gp_set):
    # nearest other-artist track: 0->25R 10->25R 25->10R 45->25R 70->45P 100->140R 140->100P
    np.testing.assert_allclose(sim.gp_at_k_per_query(*gp_set, k=1), [1, 1, 1, 0, 1, 0, 0])


def test_k_larger_than_candidates_divides_by_candidates(gp_set):
    # query 0 has 5 other-artist candidates, 2 of them Rock (25, 140)
    assert sim.gp_at_k_per_query(*gp_set, k=10)[0] == pytest.approx(2 / 5)


def test_query_without_candidates_is_skipped():
    emb = np.stack([unit(0), unit(5), unit(90)])
    per_q = sim.gp_at_k_per_query(emb, ["R", "R", "R"], [1, 1, 1], k=2)
    assert np.isnan(per_q).all()
    with pytest.raises(ValueError):
        sim.genre_precision_at_k(emb, ["R", "R", "R"], [1, 1, 1], k=2)

    per_q = sim.gp_at_k_per_query(emb, ["R", "R", "P"], [1, 1, 2], k=2)
    np.testing.assert_allclose(per_q, [0, 0, 0])


def test_gp_invariant_to_scale_and_chunking(gp_set):
    emb, g, a = gp_set
    scaled = emb * np.array([1, 3, .2, 7, 1, 50, .01])[:, None]
    ref = sim.gp_at_k_per_query(emb, g, a, k=3)
    np.testing.assert_allclose(sim.gp_at_k_per_query(scaled, g, a, k=3, chunk_size=2), ref)


# ASM set: cosines are exact numbers.
#            vector      artist genre
ASM_SET = [((1, 0),      1, "Rock"),
           ((0.6, 0.8),  1, "Rock"),
           ((0, 1),      2, "Rock"),
           ((1, 0),      3, "Pop"),
           ((0, 1),      3, "Pop"),
           ((-1, 0),     4, "Pop")]


@pytest.fixture()
def asm_set():
    emb = np.array([v for v, _, _ in ASM_SET], dtype=float)
    return emb, np.array([g for *_, g in ASM_SET]), np.array([a for _, a, _ in ASM_SET])


def test_asm_hand_computed(asm_set):
    # same artist:              (0,1)=0.6  (3,4)=0             -> mean 0.3
    # same genre, diff artist:  (0,2)=0  (1,2)=0.8  (3,5)=-1  (4,5)=0 -> mean -0.05
    res = sim.artist_similarity_margin(*asm_set)
    assert res["sim_same_artist"] == pytest.approx(0.3)
    assert res["sim_same_genre_diff_artist"] == pytest.approx(-0.05)
    assert res["asm"] == pytest.approx(0.35)
    assert res["n_pairs_same_artist"] == 2
    assert res["n_pairs_same_genre_diff_artist"] == 4


def test_asm_invariant_to_order_scale_and_chunking(asm_set):
    emb, g, a = asm_set
    perm = np.array([5, 2, 0, 4, 1, 3])
    res = sim.artist_similarity_margin(emb[perm] * 4.0, g[perm], a[perm], chunk_size=4)
    assert res["asm"] == pytest.approx(0.35)


def test_asm_errors_without_pairs():
    emb = np.eye(3)
    with pytest.raises(ValueError, match="same-artist"):
        sim.artist_similarity_margin(emb, ["R", "R", "P"], [1, 2, 3])
    with pytest.raises(ValueError, match="different-artist"):
        sim.artist_similarity_margin(emb, ["R", "R", "P"], [1, 1, 2])


def test_zero_vector_rejected():
    with pytest.raises(ValueError, match="zero-norm"):
        sim.artist_similarity_margin(np.zeros((2, 3)), ["R", "R"], [1, 1])


def test_npz_round_trip_and_cli(asm_set, tmp_path):
    emb, g, a = asm_set
    path = tmp_path / "embeddings_test.npz"
    sim.save_embeddings(path, np.arange(6), emb, list(g), a)
    data = sim.load_embeddings(path)
    assert data["genre"].dtype.kind == "U"

    out = tmp_path / "res.json"
    res = sim.main([str(path), "--k", "2", "--out", str(out)])
    assert res["asm"] == pytest.approx(0.35)
    assert res["n_tracks"] == 6 and res["n_artists"] == 4
    assert out.exists()


def test_load_rejects_missing_keys(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, embedding=np.eye(2), genre=np.array(["R", "P"]))
    with pytest.raises(KeyError, match="artist_id"):
        sim.load_embeddings(path)
