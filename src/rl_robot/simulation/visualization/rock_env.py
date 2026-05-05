"""Rock environment visualization helpers."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any, Mapping

import plotly.graph_objects as go

from rl_robot.simulation.tunnel.rock_wall import (
    K_SLUMP,
    L_TUNNEL,
    N_THETA,
    N_Z,
    NOISE_AMPLITUDE,
    NOISE_OCTAVES,
    NOISE_PERSISTENCE,
    NOISE_SCALE,
    NOISE_SEED,
    R_BASE,
    generate_rock_environment,
)


FIG_WIDTH = 1400
FIG_HEIGHT = 900


def create_rock_surface_mesh(rock_env: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    points_grid = rock_env["points_grid"]
    radius_grid = rock_env["radius_grid"]

    x_grid = points_grid[:, :, 0]
    y_grid = points_grid[:, :, 1]
    z_grid = points_grid[:, :, 2]

    return x_grid, y_grid, z_grid, radius_grid


def create_visualization(rock_env: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()

    x_rock, y_rock, z_rock, color_rock = create_rock_surface_mesh(rock_env)
    fig.add_trace(
        go.Surface(
            x=x_rock,
            y=y_rock,
            z=z_rock,
            surfacecolor=color_rock,
            opacity=0.85,
            colorscale="Earth",
            showscale=True,
            colorbar=dict(title="Radius (m)"),
            name="Rock wall",
            hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra>Wall</extra>",
        )
    )

    fig.update_layout(
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        title=dict(text="Shotcrete Tunnel Rock Environment", x=0.5),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def run_rock_env_visualization(cfg: Mapping[str, Any]) -> Path:
    env_cfg = dict(cfg.get("env", {}))

    print("=" * 68)
    print("Shotcrete Tunnel Rock Environment")
    print("=" * 68)
    print("\n[Environment]")
    print(f"  Base tunnel radius: {R_BASE:.2f} m")
    print(f"  Tunnel length: {L_TUNNEL:.2f} m")
    print(
        "  Noise settings: "
        f"scale={NOISE_SCALE}, octaves={NOISE_OCTAVES}, persistence={NOISE_PERSISTENCE}"
    )
    print(f"  Noise amplitude: {NOISE_AMPLITUDE:.2f} m")
    print(f"  Gravity slump coefficient: {K_SLUMP:.2f}")

    rock_env = generate_rock_environment(
        n_theta=int(env_cfg.get("n_theta", N_THETA)),
        n_z=int(env_cfg.get("n_z", N_Z)),
        seed=int(env_cfg.get("seed", NOISE_SEED)),
    )

    print(f"\n  Surface sample count: {len(rock_env['points'])}")
    print("\n[Visualization]")
    fig = create_visualization(rock_env)

    project_root = Path(__file__).resolve().parents[4]
    output_relative = str(env_cfg.get("rock_env_html", "outputs/rock_environment.html"))
    output_path = project_root / output_relative
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path), include_plotlyjs=True, auto_open=False)
    print(f"  Output: {output_path}")
    webbrowser.open(f"file://{output_path}")
    print("\nEnvironment visualization complete.")
    return output_path


__all__ = ["create_rock_surface_mesh", "create_visualization", "run_rock_env_visualization"]
