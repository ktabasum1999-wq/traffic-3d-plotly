"""Summarise DLR UT trajectory CSVs into one row per 15-minute window.

Source: DLR Urban Traffic dataset v1.3.0, raw_data/trajectories.
Licence: CC BY-NC-SA 4.0. Non-commercial use only.
Run from the project root:  python src\\build_dataset.py
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/DLR-Urban-Traffic-dataset_v1-3-0/raw_data/trajectories")
OUT_PATH = Path("data/processed/traffic_windows.csv")
EXPECTED_WINDOWS = 96  # 24 h x 4 windows per hour

# Assumption, not from the documentation: speed below which an object counts as slow.
# Change it and state the value in the README.
SLOW_MS = 2.0

CLASS_COLS = [
    "classifications_pedestrian",
    "classifications_bicycle",
    "classifications_motorbike",
    "classifications_car",
    "classifications_van",
    "classifications_truck",
]


def summarise_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "id", "velocity_magnitude", *CLASS_COLS])
    df["window_start"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("15min")

    # One row per object per window: mean speed and mean class probabilities.
    agg = {"speed": ("velocity_magnitude", "mean")}
    agg.update({c: (c, "mean") for c in CLASS_COLS})
    per_obj = df.groupby(["window_start", "id"]).agg(**agg).reset_index()

    # Objects with no class probability at all are "unknown", not "pedestrian".
    labelled = per_obj[CLASS_COLS].idxmax(axis=1).str.replace("classifications_", "", regex=False)
    per_obj["dominant"] = labelled.where(per_obj[CLASS_COLS].sum(axis=1) > 0, "unknown")

    g = per_obj.groupby("window_start")
    out = pd.DataFrame({
        "n_objects": g.size(),
        "mean_speed_ms": g["speed"].mean(),
        "n_slow": g["speed"].apply(lambda s: int((s < SLOW_MS).sum())),
    })
    by_class = per_obj.groupby(["window_start", "dominant"]).size().unstack(fill_value=0)
    return out.join(by_class.add_prefix("n_"))


def validate(table: pd.DataFrame) -> None:
    if len(table) != EXPECTED_WINDOWS:
        raise ValueError(f"Expected {EXPECTED_WINDOWS} windows, got {len(table)}")

    class_cols = [c for c in table.columns if c.startswith("n_") and c not in ("n_objects", "n_slow")]
    bad = table[table[class_cols].sum(axis=1) != table["n_objects"]]
    if not bad.empty:
        raise ValueError(f"Class counts do not sum to n_objects in {len(bad)} windows:\n{bad.index[:5]}")

    if (table["n_slow"] > table["n_objects"]).any():
        raise ValueError("n_slow exceeds n_objects in at least one window")


def main() -> None:
    files = sorted(RAW_DIR.glob("trajectories_*.csv"))
    print(f"Found {len(files)} trajectory files")
    if not files:
        raise SystemExit(f"No files in {RAW_DIR.resolve()}. Check the path.")

    table = pd.concat([summarise_file(f) for f in files]).sort_index()

    # A class absent from a window must be 0, not blank.
    count_cols = [c for c in table.columns if c.startswith("n_")]
    table[count_cols] = table[count_cols].fillna(0).astype(int)

    validate(table)  # stops the run before anything is written

    table.insert(0, "source", "real")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH)
    print(f"Validated and wrote {len(table)} windows to {OUT_PATH}")


if __name__ == "__main__":
    main()