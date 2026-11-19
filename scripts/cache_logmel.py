#!/usr/bin/env python3
"""Cache log-mel spectrograms for FMA tracks (ECS-17).

Every mp3 -> 22.05 kHz mono -> 128-band mel (n_fft 2048, hop 512) -> dB
-> ``{track_id:06d}.npy`` (float16, shape ``(N_MELS, n_frames)``).
All audio settings come from ``src/data/audio_config.py``.

* Files that already exist in ``--out-dir`` are skipped, so a run that was cut
  off (e.g. Colab disconnect) continues where it stopped. Each ``.npy`` is
  written to a temporary file and renamed, so a half-written file is never
  mistaken for a finished one.
* Unreadable files and files shorter than ``MIN_DURATION_S`` are not cached;
  they are written to ``--bad-files`` (CSV) for the data audit (ECS-14).

Example (Colab, Drive mounted)::

    python scripts/cache_logmel.py \
        --audio-dir /content/drive/MyDrive/cs415/raw/fma_small \
        --out-dir   /content/drive/MyDrive/cs415/cache/logmel \
        --bad-files /content/drive/MyDrive/cs415/cache/bad_files.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import audio_config as cfg  # noqa: E402

BAD_FILES_FIELDS = ["track_id", "path", "reason", "duration_s", "error"]


def compute_logmel(y: np.ndarray) -> np.ndarray:
    """Waveform (already at cfg.SAMPLE_RATE) -> log-mel in dB, float32."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.SAMPLE_RATE,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=cfg.DB_REF, amin=cfg.DB_AMIN, top_db=cfg.DB_TOP)


def load_audio(path: Path) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=cfg.SAMPLE_RATE, mono=cfg.MONO)
    return y


def out_path_for(out_dir: Path, track_id: int) -> Path:
    return out_dir / cfg.CACHE_FILENAME.format(track_id=track_id)


def save_atomic(arr: np.ndarray, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, dest)


def process_one(job: tuple[int, str, str]) -> dict:
    """Returns {"status": "ok"|"bad", ...}. Never raises."""
    track_id, path_str, out_str = job
    path, dest = Path(path_str), Path(out_str)
    try:
        y = load_audio(path)
    except Exception as e:  # corrupt / undecodable mp3
        return {"status": "bad", "track_id": track_id, "path": path_str,
                "reason": "unreadable", "duration_s": "", "error": f"{type(e).__name__}: {e}"}

    duration = len(y) / cfg.SAMPLE_RATE
    if duration < cfg.MIN_DURATION_S:
        return {"status": "bad", "track_id": track_id, "path": path_str,
                "reason": "short", "duration_s": f"{duration:.3f}", "error": ""}

    try:
        logmel = compute_logmel(y).astype(cfg.CACHE_DTYPE)
        if not np.all(np.isfinite(logmel)):
            raise ValueError("non-finite values in log-mel")
        save_atomic(logmel, dest)
    except Exception as e:
        return {"status": "bad", "track_id": track_id, "path": path_str,
                "reason": "unreadable", "duration_s": f"{duration:.3f}",
                "error": f"{type(e).__name__}: {e}"}
    return {"status": "ok", "track_id": track_id, "frames": logmel.shape[1]}


def find_mp3s(audio_dir: Path) -> dict[int, Path]:
    """FMA layout: <audio_dir>/<3-digit folder>/<6-digit track_id>.mp3."""
    tracks: dict[int, Path] = {}
    for p in sorted(audio_dir.rglob("*.mp3")):
        if p.stem.isdigit():
            tracks[int(p.stem)] = p
    return tracks


def read_track_list(csv_path: Path) -> set[int]:
    with open(csv_path, newline="") as f:
        return {int(row["track_id"]) for row in csv.DictReader(f)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audio-dir", type=Path, required=True, help="fma_small root folder")
    ap.add_argument("--out-dir", type=Path, required=True, help="where .npy files go")
    ap.add_argument("--bad-files", type=Path, default=None,
                    help="CSV of unreadable/short files (default: <out-dir>/../bad_files.csv)")
    ap.add_argument("--track-list", type=Path, default=None,
                    help="optional CSV with a track_id column; only these tracks are processed")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None, help="process at most N tracks (smoke test)")
    ap.add_argument("--summary", type=Path, default=None, help="also write the run summary as JSON")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bad_path = args.bad_files or args.out_dir.parent / "bad_files.csv"
    bad_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = find_mp3s(args.audio_dir)
    if args.track_list:
        wanted = read_track_list(args.track_list)
        missing = wanted - tracks.keys()
        if missing:
            print(f"warning: {len(missing)} track(s) in --track-list have no mp3", file=sys.stderr)
        tracks = {t: p for t, p in tracks.items() if t in wanted}
    if args.limit is not None:
        tracks = dict(list(tracks.items())[: args.limit])

    jobs, n_existing = [], 0
    for tid, p in tracks.items():
        dest = out_path_for(args.out_dir, tid)
        if dest.exists():
            n_existing += 1
        else:
            jobs.append((tid, str(p), str(dest)))

    print(f"{len(tracks)} tracks found, {n_existing} already cached, {len(jobs)} to process")

    n_ok, bad_rows = 0, []
    # Bad files are never cached, so every run re-checks them; rewriting the
    # CSV each run therefore always yields the complete list.
    with open(bad_path, "w", newline="") as bf:
        writer = csv.DictWriter(bf, fieldnames=BAD_FILES_FIELDS)
        writer.writeheader()
        # spawn, not fork: forking a multi-threaded process (numpy/numba) can deadlock.
        with mp.get_context("spawn").Pool(processes=max(1, args.workers)) as pool:
            for i, res in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4), 1):
                if res["status"] == "ok":
                    n_ok += 1
                else:
                    row = {k: res[k] for k in BAD_FILES_FIELDS}
                    bad_rows.append(row)
                    writer.writerow(row)
                    bf.flush()
                if i % 200 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  ok={n_ok}  bad={len(bad_rows)}  "
                          f"{time.time() - t0:.0f}s", flush=True)

    cache_files = list(args.out_dir.glob("*.npy"))
    summary = {
        "tracks_found": len(tracks),
        "already_cached": n_existing,
        "newly_cached": n_ok,
        "bad_unreadable": sum(r["reason"] == "unreadable" for r in bad_rows),
        "bad_short": sum(r["reason"] == "short" for r in bad_rows),
        "cache_files_total": len(cache_files),
        "cache_size_mb": round(sum(f.stat().st_size for f in cache_files) / 2**20, 1),
        "elapsed_s": round(time.time() - t0, 1),
        "bad_files_csv": str(bad_path),
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
