from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def summarize_episodes(
    episode_summaries: List[Dict[str, Any]],
) -> tuple[float, float, float, float, int]:
    """Compute episode-level summary statistics inside one logging window."""
    if not episode_summaries:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    rewards = [float(ep["return"]) for ep in episode_summaries]
    lengths = [int(ep["length"]) for ep in episode_summaries]
    success = [1.0 if ep["success"] else 0.0 for ep in episode_summaries]
    min_distances = [float(ep["min_goal_distance"]) for ep in episode_summaries]
    return (
        float(np.mean(rewards)),
        float(np.mean(lengths)),
        float(np.mean(success)),
        float(np.mean(min_distances)),
        int(np.sum(success)),
    )


def build_metrics(
    progress: int,
    episode_summaries: List[Dict[str, Any]],
    update_metrics: Dict[str, Any],
    rollout_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble one metrics row for logging and plotting."""
    (
        mean_reward,
        mean_length,
        success_rate,
        mean_min_goal_distance,
        success_episodes,
    ) = summarize_episodes(episode_summaries)
    metrics = {
        "progress": progress,
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "mean_min_goal_distance": mean_min_goal_distance,
        "episodes_in_window": len(episode_summaries),
        "success_episodes": success_episodes,
        "policy_loss": float(update_metrics.get("policy_loss", float("nan"))),
        "value_loss": float(update_metrics.get("value_loss", float("nan"))),
    }
    if rollout_metrics is not None:
        for key, value in rollout_metrics.items():
            metrics[key] = value
    for key, value in update_metrics.items():
        if key not in metrics:
            metrics[key] = value
    return metrics


def _format_metric(value: Any, precision: int = 3) -> str:
    """Format scalars for logs while making missing episode stats explicit."""
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "NA"
    return f"{numeric:.{precision}f}"


def log_metrics(
    prefix: str,
    metrics: Dict[str, Any],
    terminal_metrics: Dict[str, Any] | None = None,
) -> None:
    """Print a compact metrics line suited to batched environments."""
    episode_count = int(metrics.get("episodes_in_window", 0))
    success_episodes = int(metrics.get("success_episodes", 0))
    message = (
        f"[{prefix} {metrics['progress']:05d}] "
        f"batch_r={_format_metric(metrics.get('batch_reward_mean'), 3)} "
        f"episodes={episode_count} "
        f"success_episodes={success_episodes} "
        f"success_rate={_format_metric(metrics.get('success_rate'), 3)} "
        f"policy_loss={_format_metric(metrics['policy_loss'], 4)} "
        f"value_loss={_format_metric(metrics['value_loss'], 4)}"
    )
    if "lr" in metrics:
        message += f" lr={_format_metric(metrics['lr'], 6)}"
    if "actor_lr" in metrics:
        message += f" actor_lr={_format_metric(metrics['actor_lr'], 6)}"
    if "critic_lr" in metrics:
        message += f" critic_lr={_format_metric(metrics['critic_lr'], 6)}"
    if "alpha_lr" in metrics:
        message += f" alpha_lr={_format_metric(metrics['alpha_lr'], 6)}"
    if terminal_metrics:
        for key, value in terminal_metrics.items():
            message += f" {key}={_format_metric(value, 6)}"
    print(message)
    print()


__all__ = [
    "summarize_episodes",
    "build_metrics",
    "_format_metric",
    "log_metrics",
]
