
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "audit_outputs" / "clean_tracks.csv"
OUT = ROOT / "splits"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA).sort_values("track_id").reset_index(drop=True)
assert len(df) == 7994
assert df["track_id"].is_unique

PARTITIONS = ["training", "validation", "test"]


def save_split(name, assignments):
    result = df.copy()
    result["split"] = result["track_id"].map(assignments)

    assert result["split"].notna().all()
    assert set(result["split"]) == set(PARTITIONS)

    result.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n{name}")
    print(result["split"].value_counts().to_string())


def random_split(seed):
    train_val, test = train_test_split(
        df,
        test_size=0.10,
        random_state=seed,
        stratify=df["genre_top"]
    )

    train, val = train_test_split(
        train_val,
        test_size=1 / 9,
        random_state=seed,
        stratify=train_val["genre_top"]
    )

    assignments = {}
    for part, label in [
        (train, "training"),
        (val, "validation"),
        (test, "test")
    ]:
        assignments.update(dict.fromkeys(part["track_id"], label))

    save_split(f"random_{seed}", assignments)


def artist_grouped_split(seed):
    # Choose test fold closest to 10% with good genre balance.
    def choose_fold(data, target_fraction, random_seed):
        splitter = StratifiedGroupKFold(
            n_splits=10,
            shuffle=True,
            random_state=random_seed
        )

        overall = data["genre_top"].value_counts(normalize=True)
        best = None

        for train_idx, held_idx in splitter.split(
            data, data["genre_top"], data["artist_id"]
        ):
            held = data.iloc[held_idx]
            proportions = held["genre_top"].value_counts(normalize=True)
            genre_error = sum(
                abs(proportions.get(g, 0) - p)
                for g, p in overall.items()
            )
            size_error = abs(len(held) / len(data) - target_fraction)
            score = 5 * size_error + genre_error

            if best is None or score < best[0]:
                best = (score, train_idx, held_idx)

        return data.iloc[best[1]], data.iloc[best[2]]

    train_val, test = choose_fold(df, 0.10, seed)
    train, val = choose_fold(train_val, 1 / 9, seed + 100)

    assignments = {}
    for part, label in [
        (train, "training"),
        (val, "validation"),
        (test, "test")
    ]:
        assignments.update(dict.fromkeys(part["track_id"], label))

    save_split(f"artist_{seed}", assignments)


# Three genre-stratified random splits
for seed in range(3):
    random_split(seed)

# Artist 0: preserve official FMA labels
save_split("artist_0", dict(zip(df["track_id"], df["split"])))

# Artist 1 and 2: grouped splits
for seed in [1, 2]:
    artist_grouped_split(seed)

print("\nAll 6 split files generated.")

