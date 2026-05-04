"""Training runner entry points."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from rl_robot.algorithms.buffer import (
    OnPolicyBatch,
    OnPolicyBuffer,
    ReplayBatch,
    ReplayBuffer,
)
from rl_robot.algorithms.lr_schedule import OptimizerLRScheduler, ScalarScheduler
from rl_robot.algorithms.ppo import PPOAgent, build_ppo_config
from rl_robot.algorithms.sac import SACAgent, build_sac_config
from rl_robot.envs.train_env import BaseTrainEnv, build_train_env

from .artifacts import (
    build_run_dir,
    load_metrics_csv,
    resolve_resume_checkpoint,
    save_checkpoint,
    save_training_curves,
    write_metrics_csv,
)
from .eval_hooks import build_action_scale_scheduler
from .metrics import build_metrics, log_metrics


def build_device(requested_device: str) -> torch.device:
    """Resolve the requested compute device."""
    normalized = requested_device.lower()
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    exploration_cfg = dict(ppo_cfg.get("exploration_schedule", {}))
    exploration_scheduler = None
    if bool(exploration_cfg.get("enable", False)):
        default_log_std = agent.get_log_std_value()
        exploration_scheduler = ScalarScheduler(
            start_value=float(
                exploration_cfg.get("start_log_std", default_log_std)
            ),
            end_value=float(
                exploration_cfg.get("end_log_std", default_log_std)
            ),
            total_progress=total_updates,
            schedule=str(exploration_cfg.get("schedule", "cosine")),
        )
    action_scale_scheduler = build_action_scale_scheduler(
        config.get("rl", {}),
        total_progress=total_updates,
    )

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
    print(f"Num envs: {env.num_envs}")
    print(f"Total updates: {total_updates}")
    print(f"Resume from update: {start_progress}")
    print(f"Rollout steps per env: {rollout_steps}")
    print(f"Effective rollout batch size: {rollout_batch_size}")
    print(f"Exploration schedule: {'on' if exploration_scheduler is not None else 'off'}")
    print(f"Action scale schedule: {'on' if action_scale_scheduler is not None else 'off'}")

    train_start_time = perf_counter()

    for update in range(start_progress + 1, total_updates + 1):
        lr_metrics = lr_scheduler.step(update - 1)
        terminal_metrics: Dict[str, Any] = {}
        if action_scale_scheduler is not None:
            action_scale_ratio = action_scale_scheduler.step(update - 1)
            env.set_action_scale_ratio(action_scale_ratio)
            terminal_metrics["action_scale"] = env.get_action_scale_ratio()
        if exploration_scheduler is not None:
            scheduled_log_std = exploration_scheduler.step(update - 1)
            agent.set_log_std_value(scheduled_log_std)
            terminal_metrics["ppo_log_std"] = agent.get_log_std_value()
            terminal_metrics["ppo_std"] = agent.get_std_value()
        batch, observation, episode_summaries = collect_on_policy_rollout(
            env=env,
            agent=agent,
            rollout_steps=rollout_steps,
            device=device,
            observation=observation,
        )
        update_metrics = agent.update(batch)
        update_metrics.update(lr_metrics)
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
        metrics_history.append(metrics)

        if log_interval > 0 and update % log_interval == 0:
            log_metrics("Update", metrics, terminal_metrics=terminal_metrics)

    write_metrics_csv(run_dir, metrics_history)
    save_training_curves(run_dir, metrics_history)
    save_checkpoint(run_dir, "final.pt", agent, config, total_updates, metrics_history[-1])
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
    algorithm_cfg = config.get("algorithm", {})
    sac_cfg = build_sac_config(
        {
            **config.get("sac", {}),
            "gamma": float(algorithm_cfg.get("gamma", 0.99)),
        }
    )

    total_steps = int(sac_cfg["total_steps"])
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

    progress = int(start_progress)
    next_log_step = (
        ((progress // log_interval_steps) + 1) * log_interval_steps
        if log_interval_steps > 0
        else total_steps
    )
    train_start_time = perf_counter()

    while progress < total_steps:
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

            if log_interval_steps > 0:
                log_metrics("Step", metrics, terminal_metrics=terminal_metrics)

            recent_episodes = []
            window_reward_sum = 0.0
            window_transition_count = 0
            window_done_count = 0
            if log_interval_steps > 0:
                while next_log_step <= current_progress:
                    next_log_step += log_interval_steps

    write_metrics_csv(run_dir, metrics_history)
    save_training_curves(run_dir, metrics_history)
    final_metrics = metrics_history[-1] if metrics_history else latest_metrics
    save_checkpoint(
        run_dir,
        "final.pt",
        agent,
        config,
        min(progress, total_steps),
        final_metrics,
        extra_state={"replay_buffer": replay_buffer.state_dict()},
    )
    print(f"\nTraining complete. Final artifacts saved to: {run_dir}")
    print(f"Metrics CSV: {run_dir / 'metrics.csv'}")
    print(f"Curves HTML: {run_dir / 'training_curves.html'}")


def run_training(cfg: Any) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Training config must resolve to a mapping.")

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
            ppo_cfg = build_ppo_config(
                {
                    **config.get("ppo", {}),
                    "gamma": float(algorithm_cfg.get("gamma", 0.99)),
                }
            )
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
            return

        if algorithm_name == "sac":
            sac_cfg = build_sac_config(
                {
                    **config.get("sac", {}),
                    "gamma": float(algorithm_cfg.get("gamma", 0.99)),
                }
            )
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
            return

        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    finally:
        env.close()


__all__ = [
    "build_device",
    "collect_on_policy_rollout",
    "run_ppo_training",
    "run_sac_training",
    "run_training",
]
