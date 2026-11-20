# Data contract (draft)

> Draft started in ECS-17. Covers only the files produced so far; other sections
> (clean_tracks.csv, splits, embeddings, `embed()/head()`) get added in their own PRs.
> Final version to be confirmed with the decisions from ECS-34.

## Log-mel cache — `cache/logmel/{track_id:06d}.npy`

Produced by `scripts/cache_logmel.py`. All settings come from `src/data/audio_config.py`.

| | |
|---|---|
| audio | resampled to 22 050 Hz, mono |
| features | mel power spectrogram, n_fft 2048, hop 512, 128 mel bands, fmin 0, fmax 11 025 Hz |
| scale | `librosa.power_to_db(ref=1.0, amin=1e-10, top_db=80)`: absolute dB, **not** normalised per track |
| array | `float16`, shape `(128, n_frames)`; a ~30 s clip gives ~1292 frames |
| size | ~0.33 MB per track, so ~2.6 GB for all 8000 FMA-small tracks |

Normalisation (mean/std) is done downstream using **train split statistics only**.
Only files that are readable and at least 29 s long are cached.

## Bad files — `cache/bad_files.csv`

Written by the same script, used by the data audit (ECS-14) to exclude tracks.

| column | meaning |
|---|---|
| `track_id` | int |
| `path` | mp3 path the script read |
| `reason` | `unreadable` (decode error / non-finite output) or `short` (< `MIN_DURATION_S`) |
| `duration_s` | decoded duration, empty if unreadable |
| `error` | exception message for `unreadable` |
