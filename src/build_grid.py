"""Bin DLR UT trajectory points into 20 m grid cells per 15-minute window.

Source: DLR Urban Traffic dataset v1.3.0, raw_data/trajectories (UTM zone 32U).
Licence: CC BY-NC-SA 4.0. Non-commercial use only.
Run from the project root:  python src\\build_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/DLR-Urban-Traffic-dataset_v1-3-0/raw_data/trajectories")
OUT_PATH = Path("data/processed/grid_cells.csv")

# Intersection extent from DLR documentation, section 3.3 (UTM 32U, metres).
X0, X1 = 604625.0, 604900.0
Y0, Y1 = 5792650.0, 5792875.0

# Assumption for the first version: 20 m cells. Change it here and state it in the README.
CELL_M = 20.0
N_COLS = int(np.ceil((X1 - X0) / CELL_M))  # 14
N_ROWS = int(np.ceil((Y1 - Y0) / CELL_M))  # 12


def bin_file(path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(
        path,
        usecols=["timestamp", "id", "center_easting", "center_northing", "velocity_magnitude"],
    )
    raw_rows = len(df)

    inside = (
        (df["center_easting"] >= X0) & (df["center_easting"] < X1)
        & (df["center_northing"] >= Y0) & (df["center_northing"] < Y1)
    )
    dropped = int((~inside).sum())
    df = df[inside].copy()

    df["window_start"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("15min")
    df["cell_x"] = ((df["center_easting"] - X0) // CELL_M).astype(int)
    df["cell_y"] = ((df["center_northing"] - Y0) // CELL_M).astype(int)

    out = (
        df.groupby(["window_start", "cell_x", "cell_y"])
        .agg(
            n_observations=("id", "size"),
            n_objects=("id", "nunique"),
            mean_speed_ms=("velocity_magnitude", "mean"),
        )
        .reset_index()
    )

    # Every raw row must be either binned or counted as dropped.
    if int(out["n_observations"].sum()) + dropped != raw_rows:
        raise ValueError(f"Row count mismatch in {path.name}")
    return out, dropped


def main() -> None:
    files = sorted(RAW_DIR.glob("trajectories_*.csv"))
    print(f"Found {len(files)} trajectory files")
    if not files:
        raise SystemExit(f"No files in {RAW_DIR.resolve()}. Check the path.")

    parts = []
    total_dropped = 0
    for f in files:
        out, dropped = bin_file(f)
        parts.append(out)
        total_dropped += dropped

    table = pd.concat(parts, ignore_index=True)

    if not table["cell_x"].between(0, N_COLS - 1).all():
        raise ValueError("cell_x out of range")
    if not table["cell_y"].between(0, N_ROWS - 1).all():
        raise ValueError("cell_y out of range")

    table["cell_easting"] = X0 + (table["cell_x"] + 0.5) * CELL_M
    table["cell_northing"] = Y0 + (table["cell_y"] + 0.5) * CELL_M
    table.insert(0, "source", "real")
    table = table.sort_values(["window_start", "cell_y", "cell_x"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"Points dropped outside the intersection extent: {total_dropped}")
    print(f"Wrote {len(table)} cell-window rows to {OUT_PATH}")


if __name__ == "__main__":
    main()