"""Plot 20 m grid cells as a 3D Plotly view, one 15-minute window per slider step.

Source: DLR Urban Traffic dataset v1.3.0, derived via build_grid.py.
Licence: CC BY-NC-SA 4.0. Non-commercial use only.
Run from the project root:  python src\\viz_3d.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

IN_PATH = Path("data/processed/grid_cells.csv")
OUT_PATH = Path("outputs/traffic_3d.html")

# Fixed colour range in m/s so windows are comparable with each other.
SPEED_MIN, SPEED_MAX = 0.0, 15.0


def build_figure(df: pd.DataFrame) -> go.Figure:
    windows = sorted(df["window_start"].unique())
    fig = go.Figure()

    for i, w in enumerate(windows):
        s = df[df["window_start"] == w]
        fig.add_trace(go.Scatter3d(
            x=s["cell_easting"].to_numpy(),
            y=s["cell_northing"].to_numpy(),
            z=s["n_observations"].to_numpy(),
            mode="markers",
            marker=dict(
                size=np.clip(s["n_objects"].to_numpy() * 1.5, 4, 30),
                color=s["mean_speed_ms"].to_numpy(),
                colorscale="RdYlGn",
                cmin=SPEED_MIN,
                cmax=SPEED_MAX,
                opacity=0.85,
                showscale=(i == 0),
                colorbar=dict(title="Mean speed (m/s)") if i == 0 else None,
            ),
            customdata=np.column_stack([
                s["n_objects"].to_numpy(),
                s["mean_speed_ms"].to_numpy(),
            ]),
            hovertemplate=(
                "Easting %{x:.0f} m<br>"
                "Northing %{y:.0f} m<br>"
                "Observations %{z}<br>"
                "Objects %{customdata[0]}<br>"
                "Mean speed %{customdata[1]:.1f} m/s"
                "<extra></extra>"
            ),
            name=str(w),
            visible=(i == 0),
        ))

    steps = []
    for i, w in enumerate(windows):
        steps.append(dict(
            method="update",
            args=[{"visible": [j == i for j in range(len(windows))]},
                  {"title": f"Intersection traffic, window starting {w}"}],
            label=str(w)[11:16],
        ))

    fig.update_layout(
        title=f"Intersection traffic, window starting {windows[0]}",
        scene=dict(
            xaxis_title="Easting (m, UTM 32U)",
            yaxis_title="Northing (m, UTM 32U)",
            zaxis_title="Observations per cell (20 Hz rows)",
        ),
        sliders=[dict(active=0, currentvalue=dict(prefix="Window: "), steps=steps, pad=dict(t=40))],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run build_grid.py first.")
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} cell-window rows across {df['window_start'].nunique()} windows")

    fig = build_figure(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_PATH, include_plotlyjs=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()