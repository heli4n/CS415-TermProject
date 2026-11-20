
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = pd.read_csv(ROOT / "audit_outputs" / "clean_tracks.csv")
SPLITS = ROOT / "splits"

expected_ids = set(CLEAN["track_id"])
assert CLEAN["track_id"].is_unique

for kind in ["random", "artist"]:
    for seed in range(3):
        name = f"{kind}_{seed}"
        df = pd.read_csv(SPLITS / f"{name}.csv")

        assert len(df) == len(CLEAN), f"{name}: wrong size"
        assert df["track_id"].is_unique, f"{name}: duplicate IDs"
        assert set(df["track_id"]) == expected_ids, f"{name}: track mismatch"
        assert df["split"].isin(
            ["training", "validation", "test"]
        ).all(), f"{name}: invalid labels"

        print(f"\n===== {name} =====")
        counts = df["split"].value_counts()
        print("Partition sizes:")
        print(counts.to_string())

        print("\nGenre distribution (%):")
        table = pd.crosstab(
            df["genre_top"],
            df["split"],
            normalize="columns"
        ) * 100
        print(table.round(2).to_string())

        if kind == "artist":
            groups = [
                set(df.loc[df["split"] == part, "artist_id"])
                for part in ["training", "validation", "test"]
            ]
            assert not (groups[0] & groups[1])
            assert not (groups[0] & groups[2])
            assert not (groups[1] & groups[2])
            print("Artist overlap: 0 (PASS)")

        ratios = counts / len(df)
        print("Ratios:", ratios.round(4).to_dict())

        if kind == "random":
            for part, target in [
                ("training", 0.8),
                ("validation", 0.1),
                ("test", 0.1)
            ]:
                assert abs(ratios.get(part, 0) - target) < 0.002

print("\nALL SPLIT CHECKS PASSED")

