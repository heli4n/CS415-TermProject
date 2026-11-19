"""Single source of truth for audio / log-mel settings.

Both the log-mel cache (scripts/cache_logmel.py) and the CNN input pipeline
must import these values from here so the cached features and the model input
never drift apart. Change a value here => regenerate the cache.

NOTE: drafted from the ECS-17 spec; to be confirmed against the decisions in
docs/decisions.md (ECS-34).
"""

# Resampling / channels
SAMPLE_RATE = 22_050  # Hz
MONO = True

# STFT / mel
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0.0
FMAX = SAMPLE_RATE / 2  # Nyquist

# Power -> dB. ref=1.0 keeps absolute levels (no per-track max normalisation);
# feature normalisation is done downstream using train-set statistics.
DB_REF = 1.0
DB_AMIN = 1e-10
DB_TOP = 80.0  # clip to (max - 80 dB) per track, librosa default

# Cache
CACHE_DTYPE = "float16"
CACHE_FILENAME = "{track_id:06d}.npy"  # array shape: (N_MELS, n_frames)

# FMA clips are ~30 s; anything shorter than this is reported as "short".
MIN_DURATION_S = 29.0
