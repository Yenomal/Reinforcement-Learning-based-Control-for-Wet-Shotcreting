from __future__ import annotations

from typing import Any, Dict

import torch

from rl_robot.algorithms.lr_schedule import (
    CosineThenSuccessRateTriggeredScheduler,
    ScalarScheduler,
    SuccessRateTriggeredScheduler,
)
from rl_robot.algorithms.ppo import PPOAgent
from rl_robot.envs.train_env import BaseTrainEnv, build_train_env
from rl_robot.training.metrics import summarize_episodes


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
    train_cfg = config.get("train", {})
    det_cfg = dict(train_cfg.get("deterministic_eval", {}))
    default_num_envs = max(int(train_cfg.get("num_envs", 1)), 1)
    return {
        "enable": bool(det_cfg.get("enable", False)),
        "interval_updates": max(int(det_cfg.get("interval_updates", 1)), 1),
        "episodes": max(int(det_cfg.get("episodes", default_num_envs)), 1),
        "num_envs": max(int(det_cfg.get("num_envs", default_num_envs)), 1),
        "seed": int(det_cfg.get("seed", 123)),
        "backend": str(det_cfg.get("backend", train_cfg.get("env_backend", "classic"))),
    }


class DeterministicEvalRunner:
    """Run deterministic PPO evaluation on a dedicated batched environment."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        device: torch.device,
        backend: str,
        num_envs: int,
    ) -> None:
        self.env: BaseTrainEnv = build_train_env(
            backend=backend,
            num_envs=num_envs,
            device=device,
            env_cfg=config.get("env", {}),
            planner_cfg=config.get("planner", {}),
            rl_cfg=config.get("rl", {}),
            robot_cfg=config.get("robot", {}),
            algorithm_cfg=config.get("algorithm", {}),
            disturbance_cfg=config.get("disturbance", {}),
        )

    def set_action_scale_ratio(self, ratio: float) -> None:
        self.env.set_action_scale_ratio(ratio)

    def close(self) -> None:
        self.env.close()

    def evaluate(
        self,
        *,
        agent: PPOAgent,
        device: torch.device,
        episodes: int,
        seed: int,
    ) -> Dict[str, float]:
        total_episodes = max(int(episodes), 1)
        observation, _ = self.env.reset(seed=int(seed))
        episode_summaries = []

        while len(episode_summaries) < total_episodes:
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            )
            action_tensor = agent.act_deterministic(observation_tensor)
            observation, _, _, _, infos = self.env.step(action_tensor)
            for info in infos:
                episode_summary = info.get("episode")
                if episode_summary is None:
                    continue
                episode_summaries.append(episode_summary)
                if len(episode_summaries) >= total_episodes:
                    break

        (
            mean_reward,
            mean_length,
            success_rate,
            mean_min_goal_distance,
            success_episodes,
        ) = summarize_episodes(episode_summaries[:total_episodes])

        return {
            "det_mean_reward": mean_reward,
            "det_mean_length": mean_length,
            "det_success_rate": success_rate,
            "det_mean_min_goal_distance": mean_min_goal_distance,
            "det_success_episodes": float(success_episodes),
            "det_episodes": float(total_episodes),
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
    runner.set_action_scale_ratio(action_scale_ratio)
    return runner.evaluate(
        agent=agent,
        device=device,
        episodes=episodes,
        seed=seed,
    )


def build_ppo_std_scheduler(
    *,
    agent: PPOAgent | None,
    ppo_std_cfg: Dict[str, Any],
    total_updates: int,
) -> (
    ScalarScheduler
    | SuccessRateTriggeredScheduler
    | CosineThenSuccessRateTriggeredScheduler
    | None
):
    mode = str(ppo_std_cfg.get("mode", "global_learned")).lower()
    fallback_log_std = (
        float(agent.get_log_std_value()) if agent is not None else 0.0
    )
    global_log_std = float(ppo_std_cfg.get("global_log_std", fallback_log_std))
    min_log_std = float(ppo_std_cfg.get("min_log_std", global_log_std))
    schedule_cfg = dict(ppo_std_cfg.get("schedule", {}))
    trigger_cfg = dict(ppo_std_cfg.get("success_trigger", {}))

    if mode == "global_fixed" or mode == "global_learned":
        return None

    if mode == "cosine_schedule":
        return ScalarScheduler(
            start_value=float(schedule_cfg.get("start_log_std", global_log_std)),
            end_value=float(schedule_cfg.get("end_log_std", global_log_std)),
            total_progress=total_updates,
            schedule=str(schedule_cfg.get("schedule", "cosine")),
        )

    if mode == "success_rate_triggered":
        return SuccessRateTriggeredScheduler(
            start_value=global_log_std,
            min_value=min_log_std,
            success_threshold=float(trigger_cfg.get("success_threshold", 0.9)),
            value_step=float(trigger_cfg.get("log_std_step", -0.01)),
            ema_alpha=float(trigger_cfg.get("ema_alpha", 1.0)),
            patience_updates=max(int(trigger_cfg.get("patience_updates", 1)), 1),
            cooldown_updates=max(int(trigger_cfg.get("cooldown_updates", 0)), 0),
            min_episodes_in_window=max(
                int(trigger_cfg.get("min_episodes_in_window", 0)),
                0,
            ),
        )

    if mode == "cosine_then_success_rate_triggered":
        cosine_scheduler = ScalarScheduler(
            start_value=float(schedule_cfg.get("start_log_std", global_log_std)),
            end_value=float(schedule_cfg.get("end_log_std", global_log_std)),
            total_progress=total_updates,
            schedule=str(schedule_cfg.get("schedule", "cosine")),
        )
        trigger_scheduler = SuccessRateTriggeredScheduler(
            start_value=float(schedule_cfg.get("end_log_std", global_log_std)),
            min_value=min_log_std,
            success_threshold=float(trigger_cfg.get("success_threshold", 0.9)),
            value_step=float(trigger_cfg.get("log_std_step", -0.01)),
            ema_alpha=float(trigger_cfg.get("ema_alpha", 1.0)),
            patience_updates=max(int(trigger_cfg.get("patience_updates", 1)), 1),
            cooldown_updates=max(int(trigger_cfg.get("cooldown_updates", 0)), 0),
            min_episodes_in_window=max(
                int(trigger_cfg.get("min_episodes_in_window", 0)),
                0,
            ),
        )
        return CosineThenSuccessRateTriggeredScheduler(
            cosine_scheduler=cosine_scheduler,
            trigger_scheduler=trigger_scheduler,
            switch_update=max(int(ppo_std_cfg.get("switch_update", 0)), 0),
        )

    raise ValueError(f"Unsupported PPO std mode: {mode}")


def _build_legacy_ppo_exploration_scheduler(
    *,
    agent: PPOAgent,
    ppo_cfg: Dict[str, Any],
    total_updates: int,
) -> ScalarScheduler | None:
    exploration_cfg = dict(ppo_cfg.get("exploration_schedule", {}))
    if not bool(exploration_cfg.get("enable", False)):
        return None

    default_log_std = agent.get_log_std_value()
    return ScalarScheduler(
        start_value=float(exploration_cfg.get("start_log_std", default_log_std)),
        end_value=float(exploration_cfg.get("end_log_std", default_log_std)),
        total_progress=total_updates,
        schedule=str(exploration_cfg.get("schedule", "cosine")),
    )


def _stateful_scheduler_core(
    scheduler: (
        ScalarScheduler
        | SuccessRateTriggeredScheduler
        | CosineThenSuccessRateTriggeredScheduler
        | None
    ),
) -> SuccessRateTriggeredScheduler | None:
    if isinstance(scheduler, SuccessRateTriggeredScheduler):
        return scheduler
    if isinstance(scheduler, CosineThenSuccessRateTriggeredScheduler):
        return scheduler.trigger_scheduler
    return None


def serialize_ppo_std_scheduler_state(
    scheduler: (
        ScalarScheduler
        | SuccessRateTriggeredScheduler
        | CosineThenSuccessRateTriggeredScheduler
        | None
    ),
) -> Dict[str, Any] | None:
    core = _stateful_scheduler_core(scheduler)
    if core is None:
        return None

    return {
        "current_value": float(core.current_value),
        "success_ema": (
            None if core.success_ema is None else float(core.success_ema)
        ),
        "streak": int(core.streak),
        "stage": int(core.stage),
        "cooldown_remaining": int(core.cooldown_remaining),
    }


def restore_ppo_std_scheduler_state(
    scheduler: (
        ScalarScheduler
        | SuccessRateTriggeredScheduler
        | CosineThenSuccessRateTriggeredScheduler
        | None
    ),
    state: Dict[str, Any] | None,
) -> None:
    if state is None:
        return

    core = _stateful_scheduler_core(scheduler)
    if core is None:
        return

    core.current_value = float(state.get("current_value", core.current_value))
    success_ema = state.get("success_ema", core.success_ema)
    core.success_ema = None if success_ema is None else float(success_ema)
    core.streak = int(state.get("streak", core.streak))
    core.stage = int(state.get("stage", core.stage))
    core.cooldown_remaining = int(
        state.get("cooldown_remaining", core.cooldown_remaining)
    )


def _default_ppo_std_metrics(*, log_std: float, phase: str) -> Dict[str, Any]:
    return {
        "ppo_success_ema": float("nan"),
        "ppo_std_streak": 0.0,
        "ppo_next_log_std": float(log_std),
        "ppo_std_cooldown_remaining": float("nan"),
        "ppo_std_phase": phase,
    }


def get_ppo_std_log_std(
    scheduler: (
        ScalarScheduler
        | SuccessRateTriggeredScheduler
        | CosineThenSuccessRateTriggeredScheduler
    ),
    *,
    progress: int,
) -> float:
    if isinstance(scheduler, CosineThenSuccessRateTriggeredScheduler):
        return float(scheduler.current(progress))
    if isinstance(scheduler, ScalarScheduler):
        return float(scheduler.step(progress))
    return float(scheduler.current())


def observe_ppo_std_scheduler(
    scheduler: (
        ScalarScheduler
        | SuccessRateTriggeredScheduler
        | CosineThenSuccessRateTriggeredScheduler
    ),
    *,
    progress: int,
    success_rate: float,
    episodes_in_window: int,
    current_log_std: float,
) -> Dict[str, Any]:
    if isinstance(scheduler, CosineThenSuccessRateTriggeredScheduler):
        return dict(
            scheduler.observe(
                success_rate,
                progress=progress,
                episodes_in_window=episodes_in_window,
            )
        )
    if isinstance(scheduler, SuccessRateTriggeredScheduler):
        metrics = dict(
            scheduler.observe(
                success_rate,
                episodes_in_window=episodes_in_window,
            )
        )
        metrics["ppo_std_phase"] = "trigger"
        return metrics
    return _default_ppo_std_metrics(
        log_std=current_log_std,
        phase=str(scheduler.schedule),
    )


class PPOStdRuntimeController:
    """Encapsulate PPO std scheduling, logging state, and resume state."""

    def __init__(
        self,
        *,
        agent: PPOAgent,
        ppo_std_scheduler: (
            ScalarScheduler
            | SuccessRateTriggeredScheduler
            | CosineThenSuccessRateTriggeredScheduler
            | None
        ),
        legacy_exploration_scheduler: ScalarScheduler | None,
        ppo_std_mode: str,
    ) -> None:
        self.agent = agent
        self.ppo_std_scheduler = ppo_std_scheduler
        self.legacy_exploration_scheduler = legacy_exploration_scheduler
        self.ppo_std_mode = str(ppo_std_mode)
        self.last_log_std = float(agent.get_log_std_value())

    def before_rollout(self, *, progress: int) -> Dict[str, float]:
        if self.ppo_std_scheduler is not None:
            scheduled_log_std = get_ppo_std_log_std(
                self.ppo_std_scheduler,
                progress=progress,
            )
            self.agent.set_log_std_value(scheduled_log_std)
        elif self.legacy_exploration_scheduler is not None:
            scheduled_log_std = float(self.legacy_exploration_scheduler.step(progress))
            self.agent.set_log_std_value(scheduled_log_std)

        self.last_log_std = float(self.agent.get_log_std_value())
        return {
            "ppo_log_std": self.last_log_std,
            "ppo_std": float(self.agent.get_std_value()),
        }

    def after_rollout(
        self,
        *,
        progress: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.ppo_std_scheduler is not None:
            return observe_ppo_std_scheduler(
                self.ppo_std_scheduler,
                progress=progress,
                success_rate=float(metrics.get("success_rate", float("nan"))),
                episodes_in_window=int(metrics.get("episodes_in_window", 0)),
                current_log_std=self.last_log_std,
            )
        if self.legacy_exploration_scheduler is not None:
            return _default_ppo_std_metrics(
                log_std=self.last_log_std,
                phase=f"legacy_{self.legacy_exploration_scheduler.schedule}",
            )
        return _default_ppo_std_metrics(
            log_std=float(self.agent.get_log_std_value()),
            phase=self.ppo_std_mode,
        )

    def state_dict(self) -> Dict[str, Any] | None:
        scheduler_state = serialize_ppo_std_scheduler_state(self.ppo_std_scheduler)
        if scheduler_state is None:
            return None
        return {"ppo_std_scheduler": scheduler_state}

    def load_state_dict(self, state: Dict[str, Any] | None) -> None:
        if not state:
            return
        restore_ppo_std_scheduler_state(
            self.ppo_std_scheduler,
            state.get("ppo_std_scheduler"),
        )


def build_ppo_std_runtime_controller(
    *,
    agent: PPOAgent,
    ppo_cfg: Dict[str, Any],
    ppo_std_cfg: Dict[str, Any],
    total_updates: int,
) -> PPOStdRuntimeController:
    ppo_std_scheduler = build_ppo_std_scheduler(
        agent=agent,
        ppo_std_cfg=ppo_std_cfg,
        total_updates=total_updates,
    )
    legacy_exploration_scheduler = None
    if ppo_std_scheduler is None:
        legacy_exploration_scheduler = _build_legacy_ppo_exploration_scheduler(
            agent=agent,
            ppo_cfg=ppo_cfg,
            total_updates=total_updates,
        )
    return PPOStdRuntimeController(
        agent=agent,
        ppo_std_scheduler=ppo_std_scheduler,
        legacy_exploration_scheduler=legacy_exploration_scheduler,
        ppo_std_mode=str(ppo_std_cfg.get("mode", "global_learned")),
    )


__all__ = [
    "DeterministicEvalRunner",
    "PPOStdRuntimeController",
    "build_action_scale_scheduler",
    "build_ppo_std_scheduler",
    "build_ppo_std_runtime_controller",
    "get_ppo_std_log_std",
    "observe_ppo_std_scheduler",
    "resolve_deterministic_eval_config",
    "restore_ppo_std_scheduler_state",
    "run_deterministic_evaluation",
    "serialize_ppo_std_scheduler_state",
]
