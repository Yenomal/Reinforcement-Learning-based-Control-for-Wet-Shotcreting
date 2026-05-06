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
        "det_success_rate",
        "det_mean_min_goal_distance",
        "approx_kl",
        "explained_variance",
        "ppo_success_ema",
        "ppo_std_streak",
        "ppo_next_log_std",
        "ppo_std_cooldown_remaining",
        "ppo_std_phase",
        "policy_loss",
        "value_loss",
        "ppo_std_mean",
        "ppo_log_std_mean",
        "lr",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "env_steps_per_sec",
    ]
    fieldnames = [field for field in preferred_fields if any(field in row for row in history)]
    extra_fields = [
        field
        for row in history
        for field in row
        if field not in fieldnames and field not in preferred_fields
    ]
    fieldnames.extend(dict.fromkeys(extra_fields))

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_training_curves(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Generate an HTML dashboard with the full parallel-style training panels."""
    if not history:
        return

    progress = [row["progress"] for row in history]
    batch_rewards = [row.get("batch_reward_mean", float("nan")) for row in history]
    episodes = [row.get("episodes_in_window", 0.0) for row in history]
    success_episodes = [row.get("success_episodes", 0.0) for row in history]
    success_rates = [row.get("success_rate", float("nan")) for row in history]
    det_success_rates = [row.get("det_success_rate", float("nan")) for row in history]
    det_goal_distances = [
        row.get("det_mean_min_goal_distance", float("nan")) for row in history
    ]
    policy_losses = [row["policy_loss"] for row in history]
    value_losses = [row["value_loss"] for row in history]
    approx_kls = [row.get("approx_kl", float("nan")) for row in history]
    explained_variances = [row.get("explained_variance", float("nan")) for row in history]
    ppo_success_ema = [row.get("ppo_success_ema", float("nan")) for row in history]
    ppo_std_streak = [row.get("ppo_std_streak", float("nan")) for row in history]
    ppo_next_log_std = [row.get("ppo_next_log_std", float("nan")) for row in history]
    ppo_std_cooldown_remaining = [
        row.get("ppo_std_cooldown_remaining", float("nan")) for row in history
    ]
    ppo_std_mean = [row.get("ppo_std_mean", float("nan")) for row in history]
    ppo_log_std_mean = [row.get("ppo_log_std_mean", float("nan")) for row in history]
    lrs = [row.get("lr", float("nan")) for row in history]
    actor_lrs = [row.get("actor_lr", float("nan")) for row in history]
    critic_lrs = [row.get("critic_lr", float("nan")) for row in history]
    alpha_lrs = [row.get("alpha_lr", float("nan")) for row in history]

    def has_finite_values(values: List[float]) -> bool:
        return any(np.isfinite(value) for value in values)

    fig = make_subplots(
        rows=9,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Batch Reward",
            "Episodes",
            "Success Rate",
            "Deterministic Distance",
            "Loss",
            "Approx KL",
            "Explained Variance",
            "PPO Std",
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
    if has_finite_values(det_goal_distances):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=det_goal_distances,
                mode="lines",
                name="Det Mean Min Goal Distance",
            ),
            row=4,
            col=1,
        )
    if has_finite_values(det_success_rates):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=det_success_rates,
                mode="lines",
                name="Det Success Rate",
            ),
            row=4,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=progress, y=policy_losses, mode="lines", name="Policy Loss"),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=value_losses, mode="lines", name="Value Loss"),
        row=5,
        col=1,
    )
    if has_finite_values(approx_kls):
        fig.add_trace(
            go.Scatter(x=progress, y=approx_kls, mode="lines", name="Approx KL"),
            row=6,
            col=1,
        )
    if has_finite_values(explained_variances):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=explained_variances,
                mode="lines",
                name="Explained Variance",
            ),
            row=7,
            col=1,
        )
    if has_finite_values(ppo_std_mean):
        fig.add_trace(
            go.Scatter(x=progress, y=ppo_std_mean, mode="lines", name="PPO Std Mean"),
            row=8,
            col=1,
        )
    if has_finite_values(ppo_log_std_mean):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=ppo_log_std_mean,
                mode="lines",
                name="PPO Log Std Mean",
            ),
            row=8,
            col=1,
        )
    if has_finite_values(ppo_next_log_std):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=ppo_next_log_std,
                mode="lines",
                name="PPO Next Log Std",
            ),
            row=8,
            col=1,
        )
    if has_finite_values(ppo_success_ema):
        fig.add_trace(
            go.Scatter(x=progress, y=ppo_success_ema, mode="lines", name="PPO Success EMA"),
            row=8,
            col=1,
        )
    if has_finite_values(ppo_std_streak):
        fig.add_trace(
            go.Scatter(x=progress, y=ppo_std_streak, mode="lines", name="PPO Std Streak"),
            row=8,
            col=1,
        )
    if has_finite_values(ppo_std_cooldown_remaining):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=ppo_std_cooldown_remaining,
                mode="lines",
                name="PPO Std Cooldown Remaining",
            ),
            row=8,
            col=1,
        )
    if has_finite_values(lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=lrs, mode="lines", name="LR"),
            row=9,
            col=1,
        )
    if has_finite_values(actor_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=actor_lrs, mode="lines", name="Actor LR"),
            row=9,
            col=1,
        )
    if has_finite_values(critic_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=critic_lrs, mode="lines", name="Critic LR"),
            row=9,
            col=1,
        )
    if has_finite_values(alpha_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=alpha_lrs, mode="lines", name="Alpha LR"),
            row=9,
            col=1,
        )

    fig.update_xaxes(title_text="Progress", row=9, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Episodes", row=2, col=1)
    fig.update_yaxes(title_text="Success", row=3, col=1)
    fig.update_yaxes(title_text="Det Eval", row=4, col=1)
    fig.update_yaxes(title_text="Loss", row=5, col=1)
    fig.update_yaxes(title_text="KL", row=6, col=1)
    fig.update_yaxes(title_text="EV", row=7, col=1)
    fig.update_yaxes(title_text="Std", row=8, col=1)
    fig.update_yaxes(title_text="LR", row=9, col=1)
    fig.update_layout(
        title="Training Curves",
        height=2200,
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
