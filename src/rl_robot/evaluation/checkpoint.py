from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from rl_robot.algorithms.lr_schedule import ScalarScheduler
from rl_robot.algorithms.ppo import PPOAgent, build_ppo_config
from rl_robot.algorithms.sac import SACAgent, build_sac_config
from rl_robot.envs.math_env import MathEnv


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a saved training checkpoint."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_agent_from_checkpoint(
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
) -> PPOAgent | SACAgent:
    """Recreate the configured agent and load trained weights."""
    algorithm_name = str(config.get("algorithm", {}).get("name", "ppo")).lower()

    model_cfg = config.get("model", {})
    env_cfg = config.get("env", {})
    planner_cfg = config.get("planner", {})
    rl_cfg = config.get("rl", {})
    robot_cfg = config.get("robot", {})
    algorithm_cfg = config.get("algorithm", {})
    disturbance_cfg = config.get("disturbance", {})

    env = MathEnv(
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        algorithm_cfg=algorithm_cfg,
        disturbance_cfg=disturbance_cfg,
    )

    if algorithm_name == "ppo":
        ppo_cfg = build_ppo_config(
            {
                **config.get("ppo", {}),
                "gamma": float(config.get("algorithm", {}).get("gamma", 0.99)),
            }
        )
        agent: PPOAgent | SACAgent = PPOAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            model_cfg=model_cfg,
            algorithm_cfg=ppo_cfg,
            device=device,
        )
    elif algorithm_name == "sac":
        sac_cfg = build_sac_config(
            {
                **config.get("sac", {}),
                "gamma": float(config.get("algorithm", {}).get("gamma", 0.99)),
            }
        )
        agent = SACAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            model_cfg=model_cfg,
            algorithm_cfg=sac_cfg,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    agent.load_policy_state(checkpoint["state_dict"])
    return agent


def build_action_scale_scheduler(config: Dict[str, Any]) -> ScalarScheduler | None:
    rl_cfg = config.get("rl", {})
    schedule_cfg = dict(rl_cfg.get("action_scale_schedule", {}))
    if not bool(schedule_cfg.get("enable", False)):
        return None

    algorithm_name = str(config.get("algorithm", {}).get("name", "ppo")).lower()
    if algorithm_name == "ppo":
        total_progress = int(config.get("ppo", {}).get("total_updates", 1))
    else:
        total_progress = int(config.get("sac", {}).get("total_steps", 1))

    return ScalarScheduler(
        start_value=float(schedule_cfg.get("start_ratio", 1.0)),
        end_value=float(schedule_cfg.get("end_ratio", 1.0)),
        total_progress=total_progress,
        schedule=str(schedule_cfg.get("schedule", "cosine")),
    )
