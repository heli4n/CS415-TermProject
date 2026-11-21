
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "Desktop" / "CS415"
INPUT = BASE / "fma_small_subset.csv"
BAD = BASE / "bad_files.csv"
OUTPUT = BASE / "audit_outputs"

OUTPUT.mkdir(exist_ok=True)

# 1. Metadata
df = pd.read_csv(INPUT)
required = ["track_id", "genre_top", "artist_id",
            "album_id", "split"]

assert all(c in df.columns for c in required)
assert len(df) == 8000

# 2. Duplicate IDs
duplicates = int(df["track_id"].duplicated().sum())

# 3. Genre balance
genres = df["genre_top"].value_counts().sort_index()

# 4. Tracks per artist
artist_counts = df.groupby("artist_id").size()

plt.figure(figsize=(9, 5))
plt.hist(artist_counts, bins=30, edgecolor="black")
plt.xlabel("Tracks per artist")
plt.ylabel("Number of artists")
plt.title("FMA-small: Tracks per Artist")
plt.tight_layout()
plt.savefig(OUTPUT / "artist_histogram.png", dpi=150)
plt.close()

# 5. Artists appearing in multiple genres
artist_genres = df.groupby("artist_id")["genre_top"].nunique()
multi_genre = int((artist_genres > 1).sum())

# 6. Exclude bad audio files when available
bad_count = 0
bad_status = "PENDING: Berkay's bad_files.csv not provided"

clean = df.drop_duplicates(subset="track_id").copy()

if BAD.exists():
    bad = pd.read_csv(BAD)
    assert "track_id" in bad.columns

    invalid_ids = set(bad["track_id"].dropna().astype(int))
    bad_count = int(clean["track_id"].isin(invalid_ids).sum())

    clean = clean[~clean["track_id"].isin(invalid_ids)]
    bad_status = "Applied bad_files.csv"

# 7. Save cleaned metadata
clean.to_csv(OUTPUT / "clean_tracks.csv", index=False)

# 8. Generate audit report
report = [
    "# FMA-small Dataset Audit Report",
    "",
    "## Dataset Summary",
    f"- Original tracks: {len(df)}",
    f"- Duplicate track IDs: {duplicates}",
    f"- Unique artists: {df['artist_id'].nunique()}",
    f"- Artists appearing in multiple genres: {multi_genre}",
    f"- Removed invalid audio tracks: {bad_count}",
    f"- Remaining tracks: {len(clean)}",
    f"- Audio validation: {bad_status}",
    "",
    "## Genre Distribution (original 8,000 tracks)",
    "",
    "| Genre | Track count |",
    "|---|---:|",
]

for genre, count in genres.items():
    report.append(f"| {genre} | {count} |")

# Statistics after removing the bad audio files: these describe the data the
# experiments actually use (clean_tracks.csv) and are the numbers to report.
clean_artist_counts = clean.groupby("artist_id").size()
clean_multi = int((clean.groupby("artist_id")["genre_top"].nunique() > 1).sum())
report += [
    "",
    "## After removing bad audio (clean_tracks.csv, used for all experiments)",
    f"- Tracks: {len(clean)}",
    f"- Unique artists: {clean['artist_id'].nunique()}",
    f"- Unique albums: {clean['album_id'].nunique()}",
    f"- Artists appearing in multiple genres: {clean_multi}",
    f"- Tracks per artist: mean {clean_artist_counts.mean():.2f}, "
    f"median {clean_artist_counts.median():.2f}, max {clean_artist_counts.max()}",
    "",
    "| Genre | Track count |",
    "|---|---:|",
]
for genre, count in clean["genre_top"].value_counts().sort_index().items():
    report.append(f"| {genre} | {count} |")

report += [
    "",
    "## Tracks per Artist (original 8,000 tracks)",
    f"- Mean: {artist_counts.mean():.2f}",
    f"- Median: {artist_counts.median():.2f}",
    f"- Maximum: {artist_counts.max()}",
    "",
    "![Artist histogram](artist_histogram.png)",
    "",
    "## Notes",
    "- Official FMA-small split labels were preserved.",
    "- Audio exclusions are applied only when bad_files.csv exists.",
    "- The final clean_tracks.csv schema requires team confirmation.",
]

(OUTPUT / "audit_report.md").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)

print("Audit completed.")
print("Duplicates:", duplicates)
print("Unique artists:", df["artist_id"].nunique())
print("Multi-genre artists:", multi_genre)
print("Audio validation:", bad_status)
print("Clean tracks:", len(clean))
print("Results:", OUTPUT)

