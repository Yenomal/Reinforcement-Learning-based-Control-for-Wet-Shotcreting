#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training entry for reinforcement learning on the mathematical planning environment."""

from __future__ import annotations

import argparse
import csv
import select
import sys
import termios
import tty
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from plotly.subplots import make_subplots

from .algorithm.lr_schedule import (
    CosineThenSuccessRateTriggeredScheduler,
    OptimizerLRScheduler,
    ScalarScheduler,
    SuccessRateTriggeredScheduler,
)
from .algorithm.ppo import PPOAgent, build_ppo_config, resolve_ppo_std_config
from .algorithm.sac import SACAgent, build_sac_config
from .component.buffer import OnPolicyBatch, OnPolicyBuffer, ReplayBatch, ReplayBuffer
from .config import load_config
from .rl_env.math_env import MathEnv
from .rl_env.train_env import BaseTrainEnv, build_train_env


class InteractiveTrainingControl:
    """Minimal terminal control for pause/resume and graceful exit."""

    def __init__(self) -> None:
        self.enabled = bool(sys.stdin.isatty())
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None
        self.paused = False

    def __enter__(self) -> "InteractiveTrainingControl":
        if not self.enabled:
            return self
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self.enabled or self._fd is None or self._old_settings is None:
            return
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def _poll_chars(self) -> list[str]:
        if not self.enabled:
            return []
        chars: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            char = sys.stdin.read(1)
            if char == "":
                break
            chars.append(char)
        return chars

    def handle_input(self) -> str | None:
        """Return 'quit' when q is pressed, otherwise update pause state."""
        if not self.enabled:
            return None

        for char in self._poll_chars():
            if char == " ":
                self.paused = not self.paused
                if self.paused:
                    print("[Control] 已暂停。按空格继续，按 q 保存并退出。")
                else:
                    print("[Control] 已继续训练。")
            elif char.lower() == "q":
                return "quit"
        return None

    def wait_if_paused(self) -> str | None:
        """Block in a light loop while paused, still allowing quit."""
        while self.paused:
            action = self.handle_input()
            if action == "quit":
                return action
            sleep(0.1)
        return None


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


def save_training_artifacts(
    *,
    run_dir: Path,
    filename: str,
    agent: PPOAgent | SACAgent,
    config: Dict[str, Any],
    progress: int,
    metrics: Dict[str, Any],
    history: List[Dict[str, Any]],
    extra_state: Dict[str, Any] | None = None,
) -> None:
    """Persist checkpoint, csv, and html in one place."""
    write_metrics_csv(run_dir, history)
    save_training_curves(run_dir, history)
    save_checkpoint(
        run_dir=run_dir,
        filename=filename,
        agent=agent,
        config=config,
        progress=progress,
        metrics=metrics,
        extra_state=extra_state,
    )


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

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_training_curves(run_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Generate an HTML dashboard for training and deterministic evaluation metrics."""
    if not history:
        return

    progress = [row["progress"] for row in history]
    batch_rewards = [row.get("batch_reward_mean", float("nan")) for row in history]
    episodes = [row.get("episodes_in_window", 0.0) for row in history]
    success_episodes = [row.get("success_episodes", 0.0) for row in history]
    success_rates = [row.get("success_rate", float("nan")) for row in history]
    det_success_rates = [row.get("det_success_rate", float("nan")) for row in history]
    det_mean_min_goal_distances = [
        row.get("det_mean_min_goal_distance", float("nan")) for row in history
    ]
    policy_losses = [row["policy_loss"] for row in history]
    value_losses = [row["value_loss"] for row in history]
    approx_kls = [row.get("approx_kl", float("nan")) for row in history]
    explained_variances = [row.get("explained_variance", float("nan")) for row in history]
    ppo_stds = [row.get("ppo_std_mean", float("nan")) for row in history]
    ppo_log_stds = [row.get("ppo_log_std_mean", float("nan")) for row in history]
    lrs = [row.get("lr", float("nan")) for row in history]
    actor_lrs = [row.get("actor_lr", float("nan")) for row in history]
    critic_lrs = [row.get("critic_lr", float("nan")) for row in history]
    alpha_lrs = [row.get("alpha_lr", float("nan")) for row in history]

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
    if any(not np.isnan(value) for value in det_success_rates):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=det_success_rates,
                mode="lines+markers",
                name="Det Success Rate",
                connectgaps=True,
            ),
            row=3,
            col=1,
        )
    if any(not np.isnan(value) for value in det_mean_min_goal_distances):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=det_mean_min_goal_distances,
                mode="lines+markers",
                name="Det Mean Min Goal Distance",
                connectgaps=True,
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
    if any(not np.isnan(value) for value in approx_kls):
        fig.add_trace(
            go.Scatter(x=progress, y=approx_kls, mode="lines", name="Approx KL"),
            row=6,
            col=1,
        )
    if any(not np.isnan(value) for value in explained_variances):
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
    if any(not np.isnan(value) for value in ppo_stds):
        fig.add_trace(
            go.Scatter(x=progress, y=ppo_stds, mode="lines", name="PPO Std"),
            row=8,
            col=1,
        )
    if any(not np.isnan(value) for value in ppo_log_stds):
        fig.add_trace(
            go.Scatter(x=progress, y=ppo_log_stds, mode="lines", name="PPO Log Std"),
            row=8,
            col=1,
        )
    if any(not np.isnan(value) for value in lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=lrs, mode="lines", name="LR"),
            row=9,
            col=1,
        )
    if any(not np.isnan(value) for value in actor_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=actor_lrs, mode="lines", name="Actor LR"),
            row=9,
            col=1,
        )
    if any(not np.isnan(value) for value in critic_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=critic_lrs, mode="lines", name="Critic LR"),
            row=9,
            col=1,
        )
    if any(not np.isnan(value) for value in alpha_lrs):
        fig.add_trace(
            go.Scatter(x=progress, y=alpha_lrs, mode="lines", name="Alpha LR"),
            row=9,
            col=1,
        )

    fig.update_xaxes(title_text="Progress", row=9, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Episodes", row=2, col=1)
    fig.update_yaxes(title_text="Success", row=3, col=1)
    fig.update_yaxes(title_text="Distance", row=4, col=1)
    fig.update_yaxes(title_text="Loss", row=5, col=1)
    fig.update_yaxes(title_text="KL", row=6, col=1)
    fig.update_yaxes(title_text="EV", row=7, col=1)
    fig.update_yaxes(title_text="Std", row=8, col=1)
    fig.update_yaxes(title_text="LR", row=9, col=1)
    fig.update_layout(
        title="Training Curves",
        height=2250,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    fig.write_html(str(run_dir / "training_curves.html"), include_plotlyjs=True)


def collect_on_policy_rollout(
    env: BaseTrainEnv,
    agent: PPOAgent,
    rollout_steps: int,
    device: torch.device,
    observation: Any,
) -> tuple[OnPolicyBatch, np.ndarray, List[Dict[str, Any]]]:
    """Collect one on-policy rollout batch for PPO."""
    buffer = OnPolicyBuffer(capacity=rollout_steps)
    episode_summaries: List[Dict[str, Any]] = []
    current_observation = observation
    last_transition_done = np.zeros(env.num_envs, dtype=np.float32)

    for _ in range(rollout_steps):
        obs_tensor = torch.as_tensor(
            current_observation,
            dtype=torch.float32,
            device=device,
        )
        action_tensor, log_prob_tensor, value_tensor = agent.act(obs_tensor)
        action = action_tensor.detach()

        next_observation, rewards, terminated, truncated, infos = env.step(action)
        done = torch.logical_or(
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
        )

        buffer.add(
            observation=current_observation,
            action=action,
            log_prob=log_prob_tensor.detach(),
            reward=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            done=done,
            value=value_tensor.detach(),
        )
        last_transition_done = done.to(dtype=torch.float32)
        current_observation = next_observation
        episode_summaries.extend(info["episode"] for info in infos if "episode" in info)

    buffer.finalize(
        next_observation=current_observation,
        next_done=last_transition_done,
    )
    return buffer.to_batch(device=device), current_observation, episode_summaries


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


def build_action_scale_scheduler(
    rl_cfg: Dict[str, Any],
    *,
    total_progress: int,
) -> ScalarScheduler | None:
    schedule_cfg = dict(rl_cfg.get("action_scale_schedule", {}))
    if not bool(schedule_cfg.get("enable", False)):
        return None

    return ScalarScheduler(
        start_value=float(schedule_cfg.get("start_ratio", 1.0)),
        end_value=float(schedule_cfg.get("end_ratio", 1.0)),
        total_progress=total_progress,
        schedule=str(schedule_cfg.get("schedule", "cosine")),
    )


def resolve_deterministic_eval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve training-time deterministic evaluation config."""
    train_cfg = config.get("train", {})
    det_cfg = dict(train_cfg.get("deterministic_eval", {}))
    default_num_envs = int(train_cfg.get("num_envs", 1))

    return {
        "enable": bool(det_cfg.get("enable", False)),
        "interval_updates": max(int(det_cfg.get("interval_updates", 0)), 1),
        "episodes": max(int(det_cfg.get("episodes", default_num_envs)), 1),
        "num_envs": max(int(det_cfg.get("num_envs", default_num_envs)), 1),
        "seed": int(det_cfg.get("seed", 123)),
    }


class DeterministicEvalRunner:
    """Run fixed-seed deterministic evaluation without auto-reset bias."""

    def __init__(
        self,
        *,
        num_envs: int,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        robot_cfg: Dict[str, Any],
        algorithm_cfg: Dict[str, Any],
        disturbance_cfg: Dict[str, Any],
    ) -> None:
        self.envs = [
            MathEnv(
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                rl_cfg=rl_cfg,
                robot_cfg=robot_cfg,
                algorithm_cfg=algorithm_cfg,
                disturbance_cfg=disturbance_cfg,
            )
            for _ in range(max(int(num_envs), 1))
        ]

    def set_action_scale_ratio(self, ratio: float) -> None:
        for env in self.envs:
            env.set_action_scale_ratio(ratio)

    def evaluate(
        self,
        *,
        agent: PPOAgent,
        device: torch.device,
        episodes: int,
        seed: int,
    ) -> Dict[str, float]:
        episode_summaries: List[Dict[str, Any]] = []
        next_seed = int(seed)

        while len(episode_summaries) < episodes:
            active_count = min(len(self.envs), episodes - len(episode_summaries))
            active_envs = self.envs[:active_count]
            observations: list[np.ndarray] = []
            done_flags = [False] * active_count

            for env in active_envs:
                observation, _ = env.reset(seed=next_seed)
                next_seed += 1
                observations.append(observation)

            while not all(done_flags):
                obs_tensor = torch.as_tensor(
                    np.stack(observations, axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    actions = agent.act_deterministic(obs_tensor).detach().cpu().numpy()

                for env_index, env in enumerate(active_envs):
                    if done_flags[env_index]:
                        continue
                    next_observation, _, terminated, truncated, info = env.step(actions[env_index])
                    observations[env_index] = next_observation
                    if terminated or truncated:
                        done_flags[env_index] = True
                        episode_summaries.append(info["episode"])

        success_values = [1.0 if ep["success"] else 0.0 for ep in episode_summaries]
        min_distances = [float(ep["min_goal_distance"]) for ep in episode_summaries]
        lengths = [float(ep["length"]) for ep in episode_summaries]

        return {
            "det_success_rate": float(np.mean(success_values)) if success_values else float("nan"),
            "det_mean_min_goal_distance": (
                float(np.mean(min_distances)) if min_distances else float("nan")
            ),
            "det_mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        }


def run_deterministic_evaluation(
    *,
    runner: DeterministicEvalRunner,
    agent: PPOAgent,
    device: torch.device,
    episodes: int,
    seed: int,
    action_scale_ratio: float,
) -> Dict[str, float]:
    """Evaluate PPO deterministically on a fixed seed set."""
    runner.set_action_scale_ratio(action_scale_ratio)
    return runner.evaluate(
        agent=agent,
        device=device,
        episodes=episodes,
        seed=seed,
    )


def build_ppo_std_scheduler(
    *,
    agent: PPOAgent,
    ppo_std_cfg: Dict[str, Any],
    total_updates: int,
) -> (
    ScalarScheduler
    | SuccessRateTriggeredScheduler
    | CosineThenSuccessRateTriggeredScheduler
    | None
):
    if agent.std_mode == "cosine_schedule":
        schedule_cfg = dict(ppo_std_cfg.get("schedule", {}))
        return ScalarScheduler(
            start_value=float(
                schedule_cfg.get(
                    "start_log_std",
                    ppo_std_cfg.get("global_log_std", agent.get_log_std_value()),
                )
            ),
            end_value=float(
                schedule_cfg.get(
                    "end_log_std",
                    ppo_std_cfg.get("global_log_std", agent.get_log_std_value()),
                )
            ),
            total_progress=total_updates,
            schedule=str(schedule_cfg.get("schedule", "cosine")),
        )

    if agent.std_mode == "success_rate_triggered":
        trigger_cfg = dict(ppo_std_cfg.get("success_trigger", {}))
        return SuccessRateTriggeredScheduler(
            start_value=float(
                ppo_std_cfg.get("global_log_std", agent.get_log_std_value())
            ),
            min_value=float(
                trigger_cfg.get("min_log_std", ppo_std_cfg.get("min_log_std", -6.0))
            ),
            success_threshold=float(trigger_cfg.get("success_threshold", 0.90)),
            value_step=float(trigger_cfg.get("log_std_step", -0.5)),
            ema_alpha=float(trigger_cfg.get("ema_alpha", 0.1)),
            patience_updates=int(trigger_cfg.get("patience_updates", 10)),
            cooldown_updates=int(trigger_cfg.get("cooldown_updates", 0)),
            min_episodes_in_window=int(trigger_cfg.get("min_episodes_in_window", 0)),
        )

    if agent.std_mode == "cosine_then_success_rate_triggered":
        schedule_cfg = dict(ppo_std_cfg.get("schedule", {}))
        switch_update = int(ppo_std_cfg.get("switch_update", 0))
        cosine_scheduler = ScalarScheduler(
            start_value=float(
                schedule_cfg.get(
                    "start_log_std",
                    ppo_std_cfg.get("global_log_std", agent.get_log_std_value()),
                )
            ),
            end_value=float(
                schedule_cfg.get(
                    "end_log_std",
                    ppo_std_cfg.get("global_log_std", agent.get_log_std_value()),
                )
            ),
            total_progress=max(switch_update, 1),
            schedule=str(schedule_cfg.get("schedule", "cosine")),
        )
        trigger_cfg = dict(ppo_std_cfg.get("success_trigger", {}))
        trigger_scheduler = SuccessRateTriggeredScheduler(
            start_value=float(cosine_scheduler.value(max(switch_update - 1, 0))),
            min_value=float(
                trigger_cfg.get("min_log_std", ppo_std_cfg.get("min_log_std", -6.0))
            ),
            success_threshold=float(trigger_cfg.get("success_threshold", 0.90)),
            value_step=float(trigger_cfg.get("log_std_step", -0.5)),
            ema_alpha=float(trigger_cfg.get("ema_alpha", 0.1)),
            patience_updates=int(trigger_cfg.get("patience_updates", 10)),
            cooldown_updates=int(trigger_cfg.get("cooldown_updates", 0)),
            min_episodes_in_window=int(trigger_cfg.get("min_episodes_in_window", 0)),
        )
        return CosineThenSuccessRateTriggeredScheduler(
            cosine_scheduler=cosine_scheduler,
            trigger_scheduler=trigger_scheduler,
            switch_update=switch_update,
        )

    return None


def run_ppo_training(
    env: BaseTrainEnv,
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
    ppo_std_cfg = resolve_ppo_std_config(config.get("ppo", {}))

    total_updates = int(config.get("ppo", {}).get("total_updates", 200))
    target_rollout_steps = int(ppo_cfg["rollout_steps"])
    rollout_steps = max(target_rollout_steps // env.num_envs, 1)
    rollout_batch_size = rollout_steps * env.num_envs
    log_interval = int(train_cfg.get("log_interval", 1))
    lr_scheduler = OptimizerLRScheduler(
        {"lr": agent.optimizer},
        total_progress=total_updates,
        schedule=str(train_cfg.get("lr_schedule", "none")),
        min_ratio=float(train_cfg.get("lr_min_ratio", 0.1)),
    )
    ppo_std_scheduler = build_ppo_std_scheduler(
        agent=agent,
        ppo_std_cfg=ppo_std_cfg,
        total_updates=total_updates,
    )
    action_scale_scheduler = build_action_scale_scheduler(
        config.get("rl", {}),
        total_progress=total_updates,
    )

    if total_updates <= 0:
        raise ValueError("ppo.total_updates must be a positive integer.")

    observation, _ = env.reset(seed=int(train_cfg.get("seed", 42)))
    metrics_history = list(metrics_history or [])
    latest_metrics: Dict[str, Any] = {
        "progress": start_progress,
        "policy_loss": float("nan"),
        "value_loss": float("nan"),
    }
    current_progress = int(start_progress)

    print("=" * 72)
    print("PPO Training On MathEnv")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Num envs: {env.num_envs}")
    print(f"Total updates: {total_updates}")
    print(f"Resume from update: {start_progress}")
    print(f"Rollout steps per env: {rollout_steps}")
    print(f"Effective rollout batch size: {rollout_batch_size}")
    print(f"PPO std mode: {agent.std_mode}")
    print(f"PPO std scheduler: {'on' if ppo_std_scheduler is not None else 'off'}")
    print(f"Action scale schedule: {'on' if action_scale_scheduler is not None else 'off'}")
    print("Control: 空格暂停/继续，q 保存并退出。")

    det_eval_cfg = resolve_deterministic_eval_config(config)
    det_eval_runner: DeterministicEvalRunner | None = None
    if det_eval_cfg["enable"]:
        det_eval_runner = DeterministicEvalRunner(
            num_envs=int(det_eval_cfg["num_envs"]),
            env_cfg=config.get("env", {}),
            planner_cfg=config.get("planner", {}),
            rl_cfg=config.get("rl", {}),
            robot_cfg=config.get("robot", {}),
            algorithm_cfg=config.get("algorithm", {}),
            disturbance_cfg=config.get("disturbance", {}),
        )
        print(
            "Deterministic eval: "
            f"on, interval={det_eval_cfg['interval_updates']}, "
            f"episodes={det_eval_cfg['episodes']}, num_envs={det_eval_cfg['num_envs']}"
        )
    else:
        print("Deterministic eval: off")

    train_start_time = perf_counter()

    try:
        with InteractiveTrainingControl() as control:
            for update in range(start_progress + 1, total_updates + 1):
                action = control.handle_input()
                if action == "quit":
                    print("\n[Control] 收到退出指令，正在保存当前训练结果...")
                    break
                action = control.wait_if_paused()
                if action == "quit":
                    print("\n[Control] 收到退出指令，正在保存当前训练结果...")
                    break

                lr_metrics = lr_scheduler.step(update - 1)
                terminal_metrics: Dict[str, Any] = {}
                if action_scale_scheduler is not None:
                    action_scale_ratio = action_scale_scheduler.step(update - 1)
                    env.set_action_scale_ratio(action_scale_ratio)
                    terminal_metrics["action_scale"] = env.get_action_scale_ratio()
                std_progress = update - 1
                if isinstance(ppo_std_scheduler, ScalarScheduler):
                    scheduled_log_std = ppo_std_scheduler.step(update - 1)
                    agent.set_log_std_value(scheduled_log_std)
                elif isinstance(ppo_std_scheduler, SuccessRateTriggeredScheduler):
                    scheduled_log_std = ppo_std_scheduler.current()
                    agent.set_log_std_value(scheduled_log_std)
                elif isinstance(ppo_std_scheduler, CosineThenSuccessRateTriggeredScheduler):
                    scheduled_log_std = ppo_std_scheduler.current(std_progress)
                    agent.set_log_std_value(scheduled_log_std)
                    phase_metrics = ppo_std_scheduler.state(std_progress)
                    terminal_metrics["ppo_std_phase"] = phase_metrics.get("ppo_std_phase", "NA")
                batch, observation, episode_summaries = collect_on_policy_rollout(
                    env=env,
                    agent=agent,
                    rollout_steps=rollout_steps,
                    device=device,
                    observation=observation,
                )
                update_metrics = agent.update(batch)
                update_metrics.update(lr_metrics)
                terminal_metrics["ppo_log_std"] = update_metrics.get("ppo_log_std_mean", float("nan"))
                terminal_metrics["ppo_std"] = update_metrics.get("ppo_std_mean", float("nan"))
                elapsed = max(perf_counter() - train_start_time, 1e-6)
                update_metrics["env_steps_per_sec"] = (update * rollout_batch_size) / elapsed
                rollout_metrics = {
                    "batch_reward_mean": float(batch.rewards.mean().detach().cpu().item()),
                    "batch_done_rate": float(batch.dones.mean().detach().cpu().item()),
                }
                metrics = build_metrics(
                    update,
                    episode_summaries,
                    update_metrics,
                    rollout_metrics=rollout_metrics,
                )
                if isinstance(ppo_std_scheduler, SuccessRateTriggeredScheduler):
                    trigger_metrics = ppo_std_scheduler.observe(
                        metrics.get("success_rate", float("nan")),
                        episodes_in_window=int(metrics.get("episodes_in_window", 0)),
                    )
                    metrics.update(trigger_metrics)
                    terminal_metrics["ppo_success_ema"] = trigger_metrics.get(
                        "ppo_success_ema",
                        float("nan"),
                    )
                    terminal_metrics["ppo_std_streak"] = trigger_metrics.get(
                        "ppo_std_streak",
                        float("nan"),
                    )
                    terminal_metrics["ppo_next_log_std"] = trigger_metrics.get(
                        "ppo_next_log_std",
                        float("nan"),
                    )
                    terminal_metrics["ppo_std_cooldown"] = trigger_metrics.get(
                        "ppo_std_cooldown_remaining",
                        float("nan"),
                    )
                elif isinstance(ppo_std_scheduler, CosineThenSuccessRateTriggeredScheduler):
                    if ppo_std_scheduler.in_trigger_phase(std_progress):
                        trigger_metrics = ppo_std_scheduler.observe(
                            metrics.get("success_rate", float("nan")),
                            episodes_in_window=int(metrics.get("episodes_in_window", 0)),
                        )
                    else:
                        trigger_metrics = ppo_std_scheduler.state(std_progress)
                    metrics.update(trigger_metrics)
                    terminal_metrics["ppo_success_ema"] = trigger_metrics.get(
                        "ppo_success_ema",
                        float("nan"),
                    )
                    terminal_metrics["ppo_std_streak"] = trigger_metrics.get(
                        "ppo_std_streak",
                        float("nan"),
                    )
                    terminal_metrics["ppo_next_log_std"] = trigger_metrics.get(
                        "ppo_next_log_std",
                        float("nan"),
                    )
                    terminal_metrics["ppo_std_cooldown"] = trigger_metrics.get(
                        "ppo_std_cooldown_remaining",
                        float("nan"),
                    )
                    terminal_metrics["ppo_std_phase"] = trigger_metrics.get(
                        "ppo_std_phase",
                        "NA",
                    )

                if (
                    det_eval_runner is not None
                    and update % int(det_eval_cfg["interval_updates"]) == 0
                ):
                    det_metrics = run_deterministic_evaluation(
                        runner=det_eval_runner,
                        agent=agent,
                        device=device,
                        episodes=int(det_eval_cfg["episodes"]),
                        seed=int(det_eval_cfg["seed"]),
                        action_scale_ratio=env.get_action_scale_ratio(),
                    )
                    metrics.update(det_metrics)
                    terminal_metrics["det_success_rate"] = det_metrics.get(
                        "det_success_rate",
                        float("nan"),
                    )
                    terminal_metrics["det_min_dist"] = det_metrics.get(
                        "det_mean_min_goal_distance",
                        float("nan"),
                    )

                metrics_history.append(metrics)
                latest_metrics = metrics
                current_progress = update

                if log_interval > 0 and update % log_interval == 0:
                    log_metrics("Update", metrics, terminal_metrics=terminal_metrics)
    finally:
        if det_eval_runner is not None:
            pass

    final_metrics = metrics_history[-1] if metrics_history else latest_metrics
    save_training_artifacts(
        run_dir=run_dir,
        filename="final.pt",
        agent=agent,
        config=config,
        progress=current_progress,
        metrics=final_metrics,
        history=metrics_history,
    )
    print(f"\nTraining complete. Final artifacts saved to: {run_dir}")
    print(f"Metrics CSV: {run_dir / 'metrics.csv'}")
    print(f"Curves HTML: {run_dir / 'training_curves.html'}")


def run_sac_training(
    env: BaseTrainEnv,
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
    lr_scheduler = OptimizerLRScheduler(
        {
            "actor_lr": agent.actor_optimizer,
            "critic_lr": [agent.critic_1_optimizer, agent.critic_2_optimizer],
            "alpha_lr": agent.alpha_optimizer,
        },
        total_progress=total_steps,
        schedule=str(train_cfg.get("lr_schedule", "none")),
        min_ratio=float(train_cfg.get("lr_min_ratio", 0.1)),
    )
    exploration_cfg = dict(sac_cfg.get("exploration_schedule", {}))
    exploration_scheduler = None
    if bool(exploration_cfg.get("enable", False)):
        start_target_entropy = float(
            exploration_cfg.get("start_target_entropy", agent.get_target_entropy())
        )
        end_target_entropy = float(
            exploration_cfg.get("end_target_entropy", agent.get_target_entropy())
        )
        agent.set_target_entropy(start_target_entropy)
        exploration_scheduler = ScalarScheduler(
            start_value=start_target_entropy,
            end_value=end_target_entropy,
            total_progress=total_steps,
            schedule=str(exploration_cfg.get("schedule", "cosine")),
        )
    action_scale_scheduler = build_action_scale_scheduler(
        config.get("rl", {}),
        total_progress=total_steps,
    )

    if total_steps <= 0:
        raise ValueError("sac.total_steps must be a positive integer.")

    replay_buffer = replay_buffer or ReplayBuffer(
        capacity=buffer_size,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        storage_device=device if str(train_cfg.get("env_backend", "classic")).lower() == "torch" else None,
    )
    observation, _ = env.reset(seed=int(train_cfg.get("seed", 42)))
    recent_episodes: List[Dict[str, Any]] = []
    metrics_history = list(metrics_history or [])
    latest_metrics: Dict[str, Any] = {
        "progress": start_progress,
        "policy_loss": float("nan"),
        "value_loss": float("nan"),
    }
    latest_metrics.update(lr_scheduler.step(max(start_progress, 0)))
    window_reward_sum = 0.0
    window_transition_count = 0
    window_done_count = 0

    print("=" * 72)
    print("SAC Training On MathEnv")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Num envs: {env.num_envs}")
    print(f"Total steps: {total_steps}")
    print(f"Resume from step: {start_progress}")
    print(f"Replay buffer size: {buffer_size}")
    print(f"Updates per batched env step: {updates_per_step}")
    print(f"Exploration schedule: {'on' if exploration_scheduler is not None else 'off'}")
    print(f"Action scale schedule: {'on' if action_scale_scheduler is not None else 'off'}")
    print("Control: 空格暂停/继续，q 保存并退出。")

    progress = int(start_progress)
    next_log_step = (
        ((progress // log_interval_steps) + 1) * log_interval_steps
        if log_interval_steps > 0
        else total_steps
    )
    train_start_time = perf_counter()

    with InteractiveTrainingControl() as control:
        while progress < total_steps:
            action_signal = control.handle_input()
            if action_signal == "quit":
                print("\n[Control] 收到退出指令，正在保存当前训练结果...")
                break
            action_signal = control.wait_if_paused()
            if action_signal == "quit":
                print("\n[Control] 收到退出指令，正在保存当前训练结果...")
                break

            terminal_metrics: Dict[str, Any] = {}
            if action_scale_scheduler is not None:
                action_scale_ratio = action_scale_scheduler.step(progress)
                env.set_action_scale_ratio(action_scale_ratio)
                terminal_metrics["action_scale"] = env.get_action_scale_ratio()
            if progress < learning_starts:
                action = torch.empty(
                    (env.num_envs, env.action_dim),
                    dtype=torch.float32,
                    device=device,
                ).uniform_(-1.0, 1.0)
            else:
                obs_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )
                action = agent.act(obs_tensor)

            next_observation, rewards, terminated, truncated, infos = env.step(action)
            done = torch.logical_or(
                torch.as_tensor(terminated, dtype=torch.bool, device=device),
                torch.as_tensor(truncated, dtype=torch.bool, device=device),
            )

            replay_buffer.add_batch(
                observations=observation,
                actions=action,
                rewards=rewards,
                next_observations=next_observation,
                dones=done.to(dtype=torch.float32),
            )
            progress += env.num_envs
            current_progress = min(progress, total_steps)
            latest_metrics.update(lr_scheduler.step(current_progress))
            if exploration_scheduler is not None:
                scheduled_target_entropy = exploration_scheduler.step(current_progress)
                agent.set_target_entropy(scheduled_target_entropy)
                terminal_metrics["sac_target_entropy"] = agent.get_target_entropy()
            window_reward_sum += float(torch.as_tensor(rewards).sum().detach().cpu().item())
            window_transition_count += int(torch.as_tensor(rewards).numel())
            window_done_count += int(done.to(dtype=torch.int32).sum().detach().cpu().item())

            if progress >= learning_starts and len(replay_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    batch: ReplayBatch = replay_buffer.sample(batch_size=batch_size, device=device)
                    latest_metrics = agent.update(batch)

            observation = next_observation
            recent_episodes.extend(info["episode"] for info in infos if "episode" in info)

            should_log = current_progress >= total_steps or (
                log_interval_steps > 0 and current_progress >= next_log_step
            )
            if should_log:
                elapsed = max(perf_counter() - train_start_time, 1e-6)
                latest_metrics["env_steps_per_sec"] = current_progress / elapsed
                rollout_metrics = {
                    "batch_reward_mean": (
                        window_reward_sum / max(window_transition_count, 1)
                    ),
                    "batch_done_rate": (
                        window_done_count / max(window_transition_count, 1)
                    ),
                }
                metrics = build_metrics(
                    current_progress,
                    recent_episodes,
                    latest_metrics,
                    rollout_metrics=rollout_metrics,
                )
                metrics_history.append(metrics)
                latest_metrics = metrics

                if log_interval_steps > 0:
                    log_metrics("Step", metrics, terminal_metrics=terminal_metrics)

                recent_episodes = []
                window_reward_sum = 0.0
                window_transition_count = 0
                window_done_count = 0
                if log_interval_steps > 0:
                    while next_log_step <= current_progress:
                        next_log_step += log_interval_steps

    final_metrics = metrics_history[-1] if metrics_history else latest_metrics
    save_training_artifacts(
        run_dir=run_dir,
        filename="final.pt",
        agent=agent,
        config=config,
        progress=min(progress, total_steps),
        metrics=final_metrics,
        history=metrics_history,
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
    env_backend = str(train_cfg.get("env_backend", "classic")).lower()
    num_envs = int(train_cfg.get("num_envs", 1))

    env = build_train_env(
        backend=env_backend,
        num_envs=num_envs,
        device=device,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        algorithm_cfg=algorithm_cfg,
        disturbance_cfg=disturbance_cfg,
    )
    run_dir = build_run_dir(config)
    try:
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
        print(f"Env backend: {env_backend}")
        print(f"Num envs: {env.num_envs}")

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
                storage_device=device if env_backend == "torch" else None,
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
