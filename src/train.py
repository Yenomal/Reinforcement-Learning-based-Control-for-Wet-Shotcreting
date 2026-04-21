#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training entry for reinforcement learning on the mathematical planning environment."""

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

from .algorithm.ppo import PPOAgent, build_ppo_config
from .algorithm.sac import SACAgent, build_sac_config
from .component.buffer import OnPolicyBatch, OnPolicyBuffer, ReplayBatch, ReplayBuffer
from .config import load_config
from .rl_env.math_env import MathEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on the math environment.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config file.",
    )
    return parser.parse_args()


def build_device(requested_device: str) -> torch.device:
    """Resolve the requested compute device."""
    normalized = requested_device.lower()
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_run_dir(config: Dict[str, Any]) -> Path:
    train_cfg = config.get("train", {})
    algorithm_cfg = config.get("algorithm", {})
    rl_cfg = config.get("rl", {})

    runs_root = Path(train_cfg.get("runs_root", "outputs/runs"))
    algorithm_name = str(algorithm_cfg.get("name", "rl")).lower()
    env_name = str(rl_cfg.get("env_name", "math_env"))
    run_name = f"{algorithm_name}_{env_name}"
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

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_training_curves(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Generate an HTML dashboard for reward, success, min distance, and loss curves."""
    if not history:
        return

    progress = [row["progress"] for row in history]
    mean_rewards = [row["mean_reward"] for row in history]
    success_rates = [row["success_rate"] for row in history]
    mean_min_distances = [row["mean_min_goal_distance"] for row in history]
    policy_losses = [row["policy_loss"] for row in history]
    value_losses = [row["value_loss"] for row in history]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Reward", "Success Rate", "Min Goal Distance", "Loss"),
    )

    fig.add_trace(
        go.Scatter(x=progress, y=mean_rewards, mode="lines", name="Mean Reward"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=progress, y=success_rates, mode="lines", name="Success Rate"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=progress,
            y=mean_min_distances,
            mode="lines",
            name="Mean Min Goal Distance",
        ),
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

    fig.update_xaxes(title_text="Progress", row=4, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Success", row=2, col=1)
    fig.update_yaxes(title_text="Distance", row=3, col=1)
    fig.update_yaxes(title_text="Loss", row=4, col=1)
    fig.update_layout(
        title="Training Curves",
        height=1100,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    fig.write_html(str(run_dir / "training_curves.html"), include_plotlyjs=True)


def collect_on_policy_rollout(
    env: MathEnv,
    agent: PPOAgent,
    rollout_steps: int,
    device: torch.device,
    observation: np.ndarray,
) -> tuple[OnPolicyBatch, np.ndarray, List[Dict[str, Any]]]:
    """Collect one on-policy rollout batch for PPO."""
    buffer = OnPolicyBuffer(capacity=rollout_steps)
    episode_summaries: List[Dict[str, Any]] = []
    current_observation = observation
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

        buffer.add(
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
            current_observation, _ = env.reset()
        else:
            current_observation = next_observation

    buffer.finalize(
        next_observation=current_observation,
        next_done=last_transition_done,
    )
    return buffer.to_batch(device=device), current_observation, episode_summaries


def summarize_episodes(
    episode_summaries: List[Dict[str, Any]],
) -> tuple[float, float, float, float]:
    """Compute mean reward, length, success, and minimum goal distance."""
    if not episode_summaries:
        return float("nan"), float("nan"), float("nan"), float("nan")

    rewards = [float(ep["return"]) for ep in episode_summaries]
    lengths = [int(ep["length"]) for ep in episode_summaries]
    success = [1.0 if ep["success"] else 0.0 for ep in episode_summaries]
    min_distances = [float(ep["min_goal_distance"]) for ep in episode_summaries]
    return (
        float(np.mean(rewards)),
        float(np.mean(lengths)),
        float(np.mean(success)),
        float(np.mean(min_distances)),
    )


def build_metrics(
    progress: int,
    episode_summaries: List[Dict[str, Any]],
    update_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble one metrics row for logging and plotting."""
    mean_reward, mean_length, success_rate, mean_min_goal_distance = summarize_episodes(
        episode_summaries
    )
    metrics = {
        "progress": progress,
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "mean_min_goal_distance": mean_min_goal_distance,
        "episodes_in_window": len(episode_summaries),
        "policy_loss": float(update_metrics.get("policy_loss", float("nan"))),
        "value_loss": float(update_metrics.get("value_loss", float("nan"))),
    }
    for key, value in update_metrics.items():
        if key not in metrics:
            metrics[key] = value
    return metrics


def log_metrics(prefix: str, metrics: Dict[str, Any]) -> None:
    """Print a compact metrics line."""
    print(
        f"[{prefix} {metrics['progress']:05d}] "
        f"reward={metrics['mean_reward']:.3f} "
        f"length={metrics['mean_length']:.2f} "
        f"success_rate={metrics['success_rate']:.3f} "
        f"min_dist={metrics['mean_min_goal_distance']:.4f} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f}"
    )


def run_ppo_training(
    env: MathEnv,
    agent: PPOAgent,
    config: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
    start_progress: int = 0,
    metrics_history: List[Dict[str, Any]] | None = None,
) -> None:
    """Train PPO using an on-policy rollout buffer."""
    train_cfg = config.get("train", {})
    ppo_cfg = build_ppo_config(config.get("ppo", {}))

    total_updates = int(config.get("ppo", {}).get("total_updates", 200))
    rollout_steps = int(ppo_cfg["rollout_steps"])
    log_interval = int(train_cfg.get("log_interval", 1))

    if total_updates <= 0:
        raise ValueError("ppo.total_updates must be a positive integer.")

    observation, _ = env.reset(seed=int(train_cfg.get("seed", 42)))
    metrics_history = list(metrics_history or [])

    print("=" * 72)
    print("PPO Training On MathEnv")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Total updates: {total_updates}")
    print(f"Resume from update: {start_progress}")
    print(f"Rollout steps: {rollout_steps}")

    for update in range(start_progress + 1, total_updates + 1):
        batch, observation, episode_summaries = collect_on_policy_rollout(
            env=env,
            agent=agent,
            rollout_steps=rollout_steps,
            device=device,
            observation=observation,
        )
        update_metrics = agent.update(batch)
        metrics = build_metrics(update, episode_summaries, update_metrics)
        metrics_history.append(metrics)

        if log_interval > 0 and update % log_interval == 0:
            log_metrics("Update", metrics)

    write_metrics_csv(run_dir, metrics_history)
    save_training_curves(run_dir, metrics_history)
    save_checkpoint(run_dir, "final.pt", agent, config, total_updates, metrics_history[-1])
    print(f"\nTraining complete. Final artifacts saved to: {run_dir}")
    print(f"Metrics CSV: {run_dir / 'metrics.csv'}")
    print(f"Curves HTML: {run_dir / 'training_curves.html'}")


def run_sac_training(
    env: MathEnv,
    agent: SACAgent,
    config: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
    start_progress: int = 0,
    metrics_history: List[Dict[str, Any]] | None = None,
    replay_buffer: ReplayBuffer | None = None,
) -> None:
    """Train SAC using a replay buffer."""
    train_cfg = config.get("train", {})
    sac_cfg = build_sac_config(config.get("sac", {}))

    total_steps = int(config.get("sac", {}).get("total_steps", 50000))
    batch_size = int(sac_cfg["batch_size"])
    buffer_size = int(sac_cfg["buffer_size"])
    learning_starts = int(sac_cfg["learning_starts"])
    updates_per_step = int(sac_cfg["updates_per_step"])
    log_interval_steps = int(sac_cfg["log_interval_steps"])

    if total_steps <= 0:
        raise ValueError("sac.total_steps must be a positive integer.")

    replay_buffer = replay_buffer or ReplayBuffer(
        capacity=buffer_size,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
    )
    observation, _ = env.reset(seed=int(train_cfg.get("seed", 42)))
    recent_episodes: List[Dict[str, Any]] = []
    metrics_history = list(metrics_history or [])
    latest_metrics: Dict[str, Any] = {
        "policy_loss": float("nan"),
        "value_loss": float("nan"),
    }

    print("=" * 72)
    print("SAC Training On MathEnv")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Total steps: {total_steps}")
    print(f"Resume from step: {start_progress}")
    print(f"Replay buffer size: {buffer_size}")

    for step in range(start_progress + 1, total_steps + 1):
        if step < learning_starts:
            action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
        else:
            obs_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            action = agent.act(obs_tensor).squeeze(0).cpu().numpy().astype(np.float32)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(
            observation=observation,
            action=action,
            reward=float(reward),
            next_observation=next_observation,
            done=done,
        )

        if step >= learning_starts and len(replay_buffer) >= batch_size:
            for _ in range(updates_per_step):
                batch: ReplayBatch = replay_buffer.sample(batch_size=batch_size, device=device)
                latest_metrics = agent.update(batch)

        if done:
            if "episode" in info:
                recent_episodes.append(info["episode"])
            observation, _ = env.reset()
        else:
            observation = next_observation

        should_log = step % log_interval_steps == 0 or step == total_steps
        if should_log:
            metrics = build_metrics(step, recent_episodes, latest_metrics)
            metrics_history.append(metrics)

            if log_interval_steps > 0:
                log_metrics("Step", metrics)

            recent_episodes = []

    write_metrics_csv(run_dir, metrics_history)
    save_training_curves(run_dir, metrics_history)
    final_metrics = metrics_history[-1] if metrics_history else latest_metrics
    save_checkpoint(
        run_dir,
        "final.pt",
        agent,
        config,
        total_steps,
        final_metrics,
        extra_state={"replay_buffer": replay_buffer.state_dict()},
    )
    print(f"\nTraining complete. Final artifacts saved to: {run_dir}")
    print(f"Metrics CSV: {run_dir / 'metrics.csv'}")
    print(f"Curves HTML: {run_dir / 'training_curves.html'}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config.get("train", {})
    env_cfg = config.get("env", {})
    planner_cfg = config.get("planner", {})
    rl_cfg = config.get("rl", {})
    robot_cfg = config.get("robot", {})
    disturbance_cfg = config.get("disturbance", {})
    algorithm_cfg = config.get("algorithm", {})
    model_cfg = config.get("model", {})
    algorithm_name = str(algorithm_cfg.get("name", "ppo")).lower()
    resume_checkpoint = resolve_resume_checkpoint(config)
    resume_training = bool(train_cfg.get("resume", False))
    resume_payload: Dict[str, Any] | None = None
    start_progress = 0
    metrics_history: List[Dict[str, Any]] = []

    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = build_device(str(train_cfg.get("device", "cpu")))

    env = MathEnv(
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        algorithm_cfg=algorithm_cfg,
        disturbance_cfg=disturbance_cfg,
    )
    run_dir = build_run_dir(config)
    if resume_checkpoint is not None:
        resume_payload = torch.load(resume_checkpoint, map_location=device)
        if resume_training:
            start_progress = int(resume_payload.get("progress", 0))
            metrics_history = load_metrics_csv(resume_checkpoint.parent)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    if resume_checkpoint is not None:
        if resume_training:
            print(f"Resume training from checkpoint: {resume_checkpoint}")
            print(f"Resume progress: {start_progress}")
        else:
            print(f"Initialize from pretrained checkpoint: {resume_checkpoint}")
            print("Training progress will restart from 0.")
    print(f"Saving outputs to: {run_dir}")

    if algorithm_name == "ppo":
        ppo_cfg = build_ppo_config(config.get("ppo", {}))
        ppo_cfg["gamma"] = float(algorithm_cfg.get("gamma", 0.99))
        agent = PPOAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            model_cfg=model_cfg,
            algorithm_cfg=ppo_cfg,
            device=device,
        )
        if resume_payload is not None and resume_training:
            agent.load_training_state(resume_payload["state_dict"])
        elif resume_payload is not None:
            agent.load_policy_state(resume_payload["state_dict"])
        run_ppo_training(
            env=env,
            agent=agent,
            config=config,
            run_dir=run_dir,
            device=device,
            start_progress=start_progress,
            metrics_history=metrics_history,
        )
    elif algorithm_name == "sac":
        sac_cfg = build_sac_config(config.get("sac", {}))
        sac_cfg["gamma"] = float(algorithm_cfg.get("gamma", 0.99))
        agent = SACAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            model_cfg=model_cfg,
            algorithm_cfg=sac_cfg,
            device=device,
        )
        replay_buffer = ReplayBuffer(
            capacity=int(sac_cfg.get("buffer_size", 100000)),
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
        )
        if resume_payload is not None and resume_training:
            agent.load_training_state(resume_payload["state_dict"])
            extra_state = resume_payload.get("extra_state", {})
            if "replay_buffer" in extra_state:
                replay_buffer.load_state_dict(extra_state["replay_buffer"])
        elif resume_payload is not None:
            agent.load_policy_state(resume_payload["state_dict"])
        run_sac_training(
            env=env,
            agent=agent,
            config=config,
            run_dir=run_dir,
            device=device,
            start_progress=start_progress,
            metrics_history=metrics_history,
            replay_buffer=replay_buffer,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")


if __name__ == "__main__":
    main()
