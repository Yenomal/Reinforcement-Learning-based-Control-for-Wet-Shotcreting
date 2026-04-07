#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training entry for PPO on the mathematical planning environment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from plotly.subplots import make_subplots

from .algorithm.ppo import PPOAgent
from .component.rollout_buffer import RolloutBatch, RolloutBuffer
from .config import load_config
from .rl_env.math_env import MathEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the math environment.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config file.",
    )
    return parser.parse_args()

def collect_rollout(
    env: MathEnv,
    agent: PPOAgent,
    rollout_steps: int,
    device: torch.device,
    observation: np.ndarray,
    reset_info: Dict[str, Any],
) -> tuple[RolloutBatch, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    rollout_buffer = RolloutBuffer(capacity=rollout_steps)
    episode_summaries: List[Dict[str, Any]] = []

    current_observation = observation
    current_info = reset_info
    last_transition_done = False

    for _ in range(rollout_steps):
        obs_tensor = torch.as_tensor(
            current_observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        action_tensor, log_prob_tensor, value_tensor = agent.act(obs_tensor)

        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rollout_buffer.add(
            observation=current_observation,
            action=action,
            log_prob=float(log_prob_tensor.squeeze(0).cpu().item()),
            reward=float(reward),
            done=done,
            value=float(value_tensor.squeeze(0).cpu().item()),
        )
        last_transition_done = done

        if done:
            if "episode" in info:
                episode_summaries.append(info["episode"])
            current_observation, current_info = env.reset()
        else:
            current_observation = next_observation
            current_info = info

    rollout_buffer.finalize(
        next_observation=current_observation,
        next_done=last_transition_done,
    )
    batch = rollout_buffer.to_batch(device=device)
    return batch, current_observation, current_info, episode_summaries


def save_checkpoint(
    run_dir: Path,
    filename: str,
    agent: PPOAgent,
    config: Dict[str, Any],
    update: int,
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "update": update,
        "config": config,
        "metrics": metrics,
        "state_dict": agent.state_dict(),
    }
    torch.save(payload, run_dir / filename)


def write_metrics_csv(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Persist scalar training metrics to CSV."""
    if not history:
        return

    fieldnames = list(history[0].keys())
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_training_curves(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Generate an HTML dashboard for reward, success, and loss curves."""
    if not history:
        return

    updates = [row["update"] for row in history]
    mean_rewards = [row["mean_reward"] for row in history]
    success_rates = [row["success_rate"] for row in history]
    policy_losses = [row["policy_loss"] for row in history]
    value_losses = [row["value_loss"] for row in history]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Reward", "Success Rate", "Loss"),
    )

    fig.add_trace(
        go.Scatter(x=updates, y=mean_rewards, mode="lines", name="Mean Reward"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=updates, y=success_rates, mode="lines", name="Success Rate"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=updates, y=policy_losses, mode="lines", name="Policy Loss"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=updates, y=value_losses, mode="lines", name="Value Loss"),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text="Update", row=3, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Success", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=3, col=1)
    fig.update_layout(
        title="Training Curves",
        height=900,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    fig.write_html(str(run_dir / "training_curves.html"), include_plotlyjs=True)


def build_run_dir(config: Dict[str, Any]) -> Path:
    train_cfg = config.get("train", {})

    runs_root = Path(train_cfg.get("runs_root", "outputs/runs"))
    run_name = str(train_cfg.get("run_name", "ppo_math_env"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config.get("train", {})
    env_cfg = config.get("env", {})
    planner_cfg = config.get("planner", {})
    rl_cfg = config.get("rl", {})
    algorithm_cfg = config.get("algorithm", {})
    model_cfg = config.get("model", {})
    algorithm_name = str(algorithm_cfg.get("name", "ppo")).lower()

    if algorithm_name != "ppo":
        raise ValueError("This training entry currently only supports PPO.")

    ppo_cfg = dict(config.get("ppo", {}))
    ppo_cfg["gamma"] = float(algorithm_cfg.get("gamma", 0.99))

    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    requested_device = str(train_cfg.get("device", "cpu")).lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env = MathEnv(env_cfg=env_cfg, planner_cfg=planner_cfg, rl_cfg=rl_cfg)
    agent = PPOAgent(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        model_cfg=model_cfg,
        algorithm_cfg=ppo_cfg,
        device=device,
    )

    run_dir = build_run_dir(config)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    total_updates = int(train_cfg.get("total_updates", 200))
    if total_updates <= 0:
        raise ValueError("train.total_updates must be a positive integer.")

    rollout_steps = int(ppo_cfg.get("rollout_steps", 1024))
    log_interval = int(train_cfg.get("log_interval", 1))
    save_interval = int(train_cfg.get("save_interval", 10))

    observation, reset_info = env.reset(seed=seed)
    best_mean_return = float("-inf")
    metrics_history: List[Dict[str, Any]] = []

    print("=" * 72)
    print("PPO Training On MathEnv")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Total updates: {total_updates}")
    print(f"Rollout steps: {rollout_steps}")

    for update in range(1, total_updates + 1):
        batch, observation, reset_info, episode_summaries = collect_rollout(
            env=env,
            agent=agent,
            rollout_steps=rollout_steps,
            device=device,
            observation=observation,
            reset_info=reset_info,
        )

        update_metrics = agent.update(batch)
        episode_returns = [float(ep["return"]) for ep in episode_summaries]
        episode_lengths = [int(ep["length"]) for ep in episode_summaries]
        episode_success = [1.0 if ep["success"] else 0.0 for ep in episode_summaries]

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
        success_rate = float(np.mean(episode_success)) if episode_success else float("nan")

        metrics = {
            "update": update,
            "mean_reward": mean_return,
            "mean_length": mean_length,
            "success_rate": success_rate,
            "episodes_in_rollout": len(episode_summaries),
            **update_metrics,
        }
        metrics_history.append(metrics)

        if log_interval > 0 and update % log_interval == 0:
            print(
                f"[Update {update:04d}] "
                f"reward={mean_return:.3f} "
                f"length={mean_length:.2f} "
                f"success={success_rate:.3f} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f}"
            )

        if not np.isnan(mean_return) and mean_return > best_mean_return:
            best_mean_return = mean_return
            save_checkpoint(run_dir, "best.pt", agent, config, update, metrics)

        if save_interval > 0 and update % save_interval == 0:
            save_checkpoint(run_dir, "latest.pt", agent, config, update, metrics)

    write_metrics_csv(run_dir, metrics_history)
    save_training_curves(run_dir, metrics_history)
    save_checkpoint(run_dir, "final.pt", agent, config, total_updates, metrics)
    print(f"\nTraining complete. Final artifacts saved to: {run_dir}")
    print(f"Metrics CSV: {run_dir / 'metrics.csv'}")
    print(f"Curves HTML: {run_dir / 'training_curves.html'}")


if __name__ == "__main__":
    main()
