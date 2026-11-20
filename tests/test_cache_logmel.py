"""Tests for scripts/cache_logmel.py using small synthetic mp3s (needs ffmpeg)."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cache_logmel  # noqa: E402
from src.data import audio_config as cfg  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_mp3(path: Path, seconds: float, freq: int = 440) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate=44100",
         "-ac", "2", "-b:a", "128k", str(path)],
        check=True,
    )


@pytest.fixture()
def fma_like(tmp_path):
    """FMA-style folder: one good, one short, one corrupt, one non-track file."""
    audio = tmp_path / "fma_small"
    make_mp3(audio / "000" / "000002.mp3", 30.0, 440)
    make_mp3(audio / "000" / "000005.mp3", 30.0, 880)
    make_mp3(audio / "001" / "001039.mp3", 10.0)          # short
    (audio / "001" / "001040.mp3").write_bytes(b"\x00" * 4096)  # corrupt
    (audio / "checksums").write_text("ignored")
    return audio


def run(audio, out, *extra):
    bad = out.parent / "bad_files.csv"
    summary = cache_logmel.main(["--audio-dir", str(audio), "--out-dir", str(out),
                                 "--bad-files", str(bad), "--workers", "2", *extra])
    with open(bad, newline="") as f:
        bad_rows = list(csv.DictReader(f))
    return summary, bad_rows


def test_caches_good_files_with_expected_shape_and_dtype(fma_like, tmp_path):
    out = tmp_path / "cache" / "logmel"
    summary, _ = run(fma_like, out)

    assert sorted(p.name for p in out.glob("*.npy")) == ["000002.npy", "000005.npy"]
    arr = np.load(out / "000002.npy")
    assert arr.dtype == np.float16
    assert arr.shape[0] == cfg.N_MELS
    expected_frames = 1 + int(30.0 * cfg.SAMPLE_RATE) // cfg.HOP_LENGTH
    assert abs(arr.shape[1] - expected_frames) <= 3  # mp3 encoder padding
    assert np.isfinite(arr.astype(np.float32)).all()
    assert summary["newly_cached"] == 2


def test_logmel_matches_direct_computation(fma_like, tmp_path):
    out = tmp_path / "cache" / "logmel"
    run(fma_like, out)
    y = cache_logmel.load_audio(fma_like / "000" / "000002.mp3")
    ref = cache_logmel.compute_logmel(y).astype(np.float16)
    np.testing.assert_array_equal(np.load(out / "000002.npy"), ref)


def test_sine_energy_lands_in_the_right_mel_band(fma_like, tmp_path):
    import librosa

    out = tmp_path / "cache" / "logmel"
    run(fma_like, out)
    mel_freqs = librosa.mel_frequencies(n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX)
    for name, freq in [("000002.npy", 440), ("000005.npy", 880)]:
        peak_band = int(np.load(out / name).astype(np.float32).mean(axis=1).argmax())
        assert abs(mel_freqs[peak_band] - freq) < 60, (name, mel_freqs[peak_band])


def test_short_and_corrupt_files_go_to_bad_files(fma_like, tmp_path):
    out = tmp_path / "cache" / "logmel"
    summary, bad = run(fma_like, out)

    reasons = {int(r["track_id"]): r["reason"] for r in bad}
    assert reasons == {1039: "short", 1040: "unreadable"}
    short_row = next(r for r in bad if r["reason"] == "short")
    assert 9.5 < float(short_row["duration_s"]) < 10.5
    assert summary["bad_short"] == 1 and summary["bad_unreadable"] == 1
    assert not (out / "001039.npy").exists() and not (out / "001040.npy").exists()


def test_rerun_skips_existing_and_resumes(fma_like, tmp_path):
    out = tmp_path / "cache" / "logmel"
    run(fma_like, out)
    (out / "000005.npy").unlink()                     # simulate an interrupted run
    (out / "000005.npy.tmp").write_bytes(b"partial")  # leftover half-written file
    mtime_before = (out / "000002.npy").stat().st_mtime_ns

    summary, bad = run(fma_like, out)

    assert summary["already_cached"] == 1
    assert summary["newly_cached"] == 1
    assert (out / "000002.npy").stat().st_mtime_ns == mtime_before
    assert np.load(out / "000005.npy").shape[0] == cfg.N_MELS
    assert len(bad) == 2  # bad files are re-checked, list stays complete


def test_track_list_and_limit(fma_like, tmp_path):
    out = tmp_path / "cache" / "logmel"
    tl = tmp_path / "tracks.csv"
    tl.write_text("track_id,genre_top\n5,Rock\n")
    summary, bad = run(fma_like, out, "--track-list", str(tl))
    assert summary["tracks_found"] == 1 and bad == []
    assert [p.name for p in out.glob("*.npy")] == ["000005.npy"]
