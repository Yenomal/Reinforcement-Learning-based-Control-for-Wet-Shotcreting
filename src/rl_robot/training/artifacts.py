from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from rl_robot.algorithms.ppo import PPOAgent
from rl_robot.algorithms.sac import SACAgent


def build_run_dir(config: Dict[str, Any]) -> Path:
    train_cfg = config.get("train", {})
    algorithm_cfg = config.get("algorithm", {})
    env_cfg = config.get("env", {})
    rl_cfg = config.get("rl", {})

    runs_root = Path(train_cfg.get("runs_root", "outputs/runs"))
    algorithm_name = str(algorithm_cfg.get("name", "rl")).lower()
    env_name = str(env_cfg.get("name") or rl_cfg.get("env_name", "math_env"))
    env_backend = str(train_cfg.get("env_backend", "classic")).lower()
    run_name = f"{algorithm_name}_{env_name}_{env_backend}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    run_dir: Path,
    filename: str,
    agent: PPOAgent | SACAgent,
    config: Dict[str, Any],
    progress: int,
    metrics: Dict[str, Any],
    extra_state: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "progress": progress,
        "config": config,
        "metrics": metrics,
        "state_dict": agent.state_dict(),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(payload, run_dir / filename)


def load_metrics_csv(run_dir: Path) -> List[Dict[str, Any]]:
    """Load existing metrics history from a run directory if it exists."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []

    history: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed_row: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed_row[key] = value
                    continue
                try:
                    parsed_row[key] = float(value)
                except ValueError:
                    parsed_row[key] = value
            history.append(parsed_row)
    return history


def resolve_resume_checkpoint(config: Dict[str, Any]) -> Path | None:
    """Return the configured checkpoint path when one is provided."""
    train_cfg = config.get("train", {})
    checkpoint = str(train_cfg.get("checkpoint", "")).strip()

    if not checkpoint:
        return None

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def write_metrics_csv(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Persist scalar training metrics to CSV."""
    if not history:
        return

    preferred_fields = [
        "progress",
        "batch_reward_mean",
        "episodes_in_window",
        "success_episodes",
        "success_rate",
        "policy_loss",
        "value_loss",
        "lr",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "env_steps_per_sec",
    ]
    fieldnames = [field for field in preferred_fields if any(field in row for row in history)]

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_training_curves(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Generate an HTML dashboard for batch reward, episode stats, loss, and lr."""
    if not history:
        return

    progress = [row["progress"] for row in history]
    batch_rewards = [row.get("batch_reward_mean", float("nan")) for row in history]
    episodes = [row.get("episodes_in_window", 0.0) for row in history]
    success_episodes = [row.get("success_episodes", 0.0) for row in history]
    success_rates = [row.get("success_rate", float("nan")) for row in history]
    policy_losses = [row["policy_loss"] for row in history]
    value_losses = [row["value_loss"] for row in history]
    lrs = [row.get("lr", float("nan")) for row in history]
    actor_lrs = [row.get("actor_lr", float("nan")) for row in history]
    critic_lrs = [row.get("critic_lr", float("nan")) for row in history]
    alpha_lrs = [row.get("alpha_lr", float("nan")) for row in history]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Batch Reward",
            "Episodes",
            "Success Rate",
            "Loss",
            "Learning Rate",
        ),
    )

    fig.add_trace(
        go.Scatter(x=progress, y=batch_rewards, mode="lines", name="Batch Reward"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=episodes, mode="lines", name="Episodes"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=success_episodes, mode="lines", name="Success Episodes"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=success_rates, mode="lines", name="Success Rate"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=policy_losses, mode="lines", name="Policy Loss"),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=value_losses, mode="lines", name="Value Loss"),
        row=4,
        col=1,
    )
    if any(not np.isnan(value) for value in lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=lrs, mode="lines", name="LR"),
            row=5,
            col=1,
        )
    if any(not np.isnan(value) for value in actor_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=actor_lrs, mode="lines", name="Actor LR"),
            row=5,
            col=1,
        )
    if any(not np.isnan(value) for value in critic_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=critic_lrs, mode="lines", name="Critic LR"),
            row=5,
            col=1,
        )
    if any(not np.isnan(value) for value in alpha_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=alpha_lrs, mode="lines", name="Alpha LR"),
            row=5,
            col=1,
        )

    fig.update_xaxes(title_text="Progress", row=5, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Episodes", row=2, col=1)
    fig.update_yaxes(title_text="Success", row=3, col=1)
    fig.update_yaxes(title_text="Loss", row=4, col=1)
    fig.update_yaxes(title_text="LR", row=5, col=1)
    fig.update_layout(
        title="Training Curves",
        height=1350,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    fig.write_html(str(run_dir / "training_curves.html"), include_plotlyjs=True)


__all__ = [
    "build_run_dir",
    "save_checkpoint",
    "load_metrics_csv",
    "resolve_resume_checkpoint",
    "write_metrics_csv",
    "save_training_curves",
]
