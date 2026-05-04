"""Evaluation runner entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from rl_robot.envs.math_env import MathEnv
from rl_robot.simulation.tunnel.build_tunnel_environment import (
    SurfaceGrid,
    load_surface_grid,
    plotly_to_pybullet,
)

from .checkpoint import (
    build_action_scale_scheduler,
    build_agent_from_checkpoint,
    load_checkpoint,
)
from .renderer import EvalRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO on the math environment.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override, typically final.pt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config override. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional number of evaluation episodes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional evaluation device override.",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=None,
        help="Disable the real-time visualization window.",
    )
    parser.add_argument(
        "--render",
        dest="headless",
        action="store_false",
        help="Force-enable the real-time visualization window.",
    )
    return parser.parse_args()


def _normalize_evaluation_config(cfg: Any) -> Dict[str, Any]:
    if OmegaConf.is_config(cfg):
        config = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, Mapping):
        config = dict(cfg)
    else:
        raise TypeError(
            "Evaluation config must be a mapping or an OmegaConf config object."
        )

    if not isinstance(config, dict):
        raise ValueError("Evaluation config must resolve to a mapping.")
    return config


def _surface_scene_from_rock_env(rock_env: Dict[str, Any]) -> Dict[str, np.ndarray]:
    points_grid = np.asarray(rock_env["points_grid"], dtype=np.float32)
    radius_grid = np.asarray(rock_env["radius_grid"], dtype=np.float32)
    return {
        "x_grid": points_grid[:, :, 0],
        "y_grid": points_grid[:, :, 1],
        "z_grid": points_grid[:, :, 2],
        "color_grid": radius_grid,
    }


def _surface_scene_from_html(html_path: Path) -> Dict[str, np.ndarray]:
    grid: SurfaceGrid = load_surface_grid(html_path)
    x_grid = np.zeros((grid.rows, grid.cols), dtype=np.float32)
    y_grid = np.zeros((grid.rows, grid.cols), dtype=np.float32)
    z_grid = np.zeros((grid.rows, grid.cols), dtype=np.float32)
    color_grid = np.asarray(grid.surfacecolor, dtype=np.float32).reshape(grid.rows, grid.cols)

    for row in range(grid.rows):
        for col in range(grid.cols):
            index = row * grid.cols + col
            x_val, y_val, z_val = plotly_to_pybullet(
                float(grid.x[index]),
                float(grid.y[index]),
                float(grid.z[index]),
            )
            x_grid[row, col] = x_val
            y_grid[row, col] = y_val
            z_grid[row, col] = z_val

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "z_grid": z_grid,
        "color_grid": color_grid,
    }


def load_eval_surface_scene(
    eval_cfg: Dict[str, Any],
    rock_env: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Load the fixed evaluation surface scene, defaulting to rock_env fallback."""
    html_path_raw = str(eval_cfg.get("digital_env_html", "")).strip()
    if not html_path_raw:
        return _surface_scene_from_rock_env(rock_env)

    html_path = Path(html_path_raw)
    if not html_path.exists():
        raise FileNotFoundError(f"Digital environment HTML not found: {html_path}")
    return _surface_scene_from_html(html_path)


def build_device(requested_device: Optional[str]) -> torch.device:
    """Resolve the evaluation device."""
    if requested_device is not None:
        normalized = requested_device.lower()
        if normalized == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_episodes(
    config: Dict[str, Any],
    agent: Any,
    device: torch.device,
    action_scale_scheduler: Any,
) -> None:
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
    if action_scale_scheduler is not None:
        total_progress = int(action_scale_scheduler.total_progress)
        env.set_action_scale_ratio(action_scale_scheduler.step(total_progress))

    eval_cfg = config.get("eval", {})
    checkpoint_path = Path(str(eval_cfg.get("checkpoint", "")))
    total_episodes = int(eval_cfg.get("episodes", 5))
    eval_seed = int(eval_cfg.get("seed", 123))
    render_pause = float(eval_cfg.get("render_pause", 0.05))
    episode_pause = float(eval_cfg.get("episode_pause", 0.8))
    headless = bool(eval_cfg.get("headless", False))
    pybullet_cfg = dict(eval_cfg.get("pybullet", {}))
    enable_pybullet = bool(pybullet_cfg.get("enable", not headless))
    surface_scene = load_eval_surface_scene(eval_cfg=eval_cfg, rock_env=env.rock_env)

    renderer = None
    if not headless:
        renderer = EvalRenderer(
            surface_scene=surface_scene,
            render_pause=render_pause,
            episode_pause=episode_pause,
        )
    player = None
    if enable_pybullet:
        from rl_robot.simulation.robot.pybullet_player import PyBulletRobotPlayer

        player = PyBulletRobotPlayer(
            dt=float(pybullet_cfg.get("dt", 1.0 / 240.0)),
            headless=bool(pybullet_cfg.get("headless", headless)),
            show_tunnel=bool(pybullet_cfg.get("show_tunnel", True)),
            show_plane=bool(pybullet_cfg.get("show_plane", False)),
            robot_position=pybullet_cfg.get("robot_position", (0.0, 0.0, 0.45)),
            robot_yaw_deg=float(pybullet_cfg.get("robot_yaw_deg", 90.0)),
            tunnel_position=pybullet_cfg.get("tunnel_position", (0.0, 0.0, 0.0)),
        )

    returns = []
    lengths = []
    success_flags = []
    min_goal_distances = []

    print("=" * 72)
    print("MathEnv Evaluation")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Episodes: {total_episodes}")
    print(f"Headless: {headless}")
    print(f"PyBullet: {enable_pybullet}")
    print(f"Action scale ratio: {env.get_action_scale_ratio():.6f}")

    for episode_index in range(1, total_episodes + 1):
        observation, info = env.reset(seed=eval_seed + episode_index - 1)
        done = False
        reward = 0.0

        if player is not None:
            player.reset_episode(info["current_q_deg"])
        if renderer is not None and env.current_task is not None:
            renderer.reset_episode(
                episode_index=episode_index,
                total_episodes=total_episodes,
                start_point=env.current_task["start"]["point"],
                goal_point=env.current_task["goal"]["point"],
            )
            if episode_index == 1:
                renderer.wait_for_start_with_idle(
                    on_idle=player.idle if player is not None else None
                )

        while not done:
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            action_tensor = agent.act_deterministic(observation_tensor)
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if player is not None:
                player.update(info["current_q_deg"])
            if renderer is not None:
                renderer.update(
                    episode_index=episode_index,
                    total_episodes=total_episodes,
                    point_path=np.asarray(env.path_points, dtype=np.float32),
                    reward=reward,
                    episode_return=float(env.episode_return),
                    step=int(env.current_step),
                    goal_distance=float(info["goal_distance"]),
                )

        episode_summary = info["episode"]
        returns.append(float(episode_summary["return"]))
        lengths.append(int(episode_summary["length"]))
        success_flags.append(1.0 if episode_summary["success"] else 0.0)
        min_goal_distances.append(float(episode_summary["min_goal_distance"]))

        print(
            f"[Episode {episode_index:03d}] "
            f"reward={episode_summary['return']:.3f} "
            f"length={episode_summary['length']} "
            f"success={episode_summary['success']} "
            f"min_dist={episode_summary['min_goal_distance']:.4f}"
        )

        if renderer is not None:
            renderer.pause_between_episodes(
                episode_index=episode_index,
                total_episodes=total_episodes,
                episode_return=float(episode_summary["return"]),
                success=bool(episode_summary["success"]),
                min_goal_distance=float(episode_summary["min_goal_distance"]),
            )

    mean_reward = float(np.mean(returns)) if returns else float("nan")
    mean_length = float(np.mean(lengths)) if lengths else float("nan")
    success_rate = float(np.mean(success_flags)) if success_flags else float("nan")
    mean_min_goal_distance = (
        float(np.mean(min_goal_distances)) if min_goal_distances else float("nan")
    )

    print("\nEvaluation summary")
    print(f"  Mean reward: {mean_reward:.3f}")
    print(f"  Mean length: {mean_length:.2f}")
    print(f"  Success rate: {success_rate:.3f}")
    print(f"  Mean min goal distance: {mean_min_goal_distance:.4f}")

    if renderer is not None:
        renderer.finalize()
    if player is not None:
        player.close()


def run_evaluation(cfg: Any) -> None:
    config = _normalize_evaluation_config(cfg)
    eval_cfg = config.get("eval", {})
    checkpoint_raw = str(eval_cfg.get("checkpoint", "")).strip()
    if not checkpoint_raw:
        raise ValueError(
            "No checkpoint provided. Set eval.checkpoint in config or pass --checkpoint."
        )

    checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_cfg = config.get("train", {})
    requested_device = train_cfg.get("device", eval_cfg.get("device"))
    device = build_device(str(requested_device) if requested_device is not None else None)
    checkpoint = load_checkpoint(checkpoint_path, device)
    agent = build_agent_from_checkpoint(checkpoint, config, device)
    action_scale_scheduler = build_action_scale_scheduler(config)
    evaluate_episodes(config, agent, device, action_scale_scheduler)


def main() -> None:
    from config import load_config

    args = parse_args()
    config = load_config(args.config)
    eval_cfg = dict(config.get("eval", {}))

    checkpoint_arg = args.checkpoint or eval_cfg.get("checkpoint", "")
    if not checkpoint_arg:
        raise ValueError(
            "No checkpoint provided. Set eval.checkpoint in config or pass --checkpoint."
        )

    checkpoint_path = Path(checkpoint_arg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = build_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if args.config is None:
        checkpoint_config = checkpoint["config"]
        checkpoint_config["eval"] = {
            **checkpoint_config.get("eval", {}),
            **eval_cfg,
        }
        config = checkpoint_config
        eval_cfg = dict(config.get("eval", {}))

    eval_cfg["checkpoint"] = str(checkpoint_path)
    if args.episodes is not None:
        eval_cfg["episodes"] = int(args.episodes)
    if args.device is not None:
        train_cfg = dict(config.get("train", {}))
        train_cfg["device"] = args.device
        config["train"] = train_cfg
    if args.headless is not None:
        eval_cfg["headless"] = bool(args.headless)
    config["eval"] = eval_cfg

    run_evaluation(config)


if __name__ == "__main__":
    main()
