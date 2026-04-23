#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation entry with optional real-time visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from .algorithm.lr_schedule import ScalarScheduler
from .algorithm.ppo import PPOAgent
from .algorithm.sac import SACAgent
from .config import load_config
from .rl_env.math_env import MathEnv
from .rock_3D.robot_4dof.pybullet_player import PyBulletRobotPlayer
from .rock_3D.tools.build_tunnel_environment import (
    SurfaceGrid,
    load_surface_grid,
    plotly_to_pybullet,
)


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


class EvalRenderer:
    """Matplotlib-based real-time renderer for evaluation rollout."""

    def __init__(
        self,
        surface_scene: Dict[str, np.ndarray],
        render_pause: float = 0.05,
        episode_pause: float = 0.8,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from matplotlib.widgets import Button
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for interactive evaluation rendering."
            ) from exc

        self.plt = plt
        self.Button = Button
        self.render_pause = render_pause
        self.episode_pause = episode_pause
        self.started = False

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(bottom=0.14)

        x_grid = surface_scene["x_grid"]
        y_grid = surface_scene["y_grid"]
        z_grid = surface_scene["z_grid"]
        color_grid = surface_scene["color_grid"]

        norm = Normalize(vmin=float(color_grid.min()), vmax=float(color_grid.max()))
        facecolors = cm.get_cmap("terrain")(norm(color_grid))
        self.ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.75,
        )

        self.point_line, = self.ax.plot(
            [], [], [], color="crimson", linewidth=2.5, label="point_path"
        )
        self.current_point_artist, = self.ax.plot(
            [], [], [], marker="o", color="crimson", markersize=6, linestyle=""
        )

        self.start_artist = None
        self.goal_artist = None
        self.start_link = None
        self.goal_link = None
        self.status_text = self.ax.text2D(
            0.02,
            0.98,
            "Waiting to start...",
            transform=self.ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.button_ax = self.fig.add_axes([0.42, 0.03, 0.16, 0.06])
        self.start_button = self.Button(self.button_ax, "Start")
        self.start_button.on_clicked(self._handle_start)

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("MathEnv Evaluation")
        self.ax.legend(loc="upper right")

        x_span = float(x_grid.max() - x_grid.min())
        y_span = float(y_grid.max() - y_grid.min())
        z_span = float(z_grid.max() - z_grid.min())
        self.ax.set_box_aspect((max(x_span, 1e-3), max(y_span, 1e-3), max(z_span, 1e-3)))

        self.fig.tight_layout()
        self.plt.ion()
        self.plt.show(block=False)

    def _handle_start(self, _event: object) -> None:
        """Start the evaluation rollout after the button is pressed."""
        self.started = True
        self.start_button.label.set_text("Running...")
        self.status_text.set_text("Evaluation started.\nPreparing first rollout...")
        self.fig.canvas.draw_idle()

    def wait_for_start(self) -> None:
        """Block in the UI loop until the Start button is pressed."""
        while not self.started:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def wait_for_start_with_idle(
        self,
        on_idle: Optional[Callable[[], None]] = None,
    ) -> None:
        """Block until Start is pressed while optionally ticking other viewers."""
        while not self.started:
            if on_idle is not None:
                on_idle()
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def reset_episode(
        self,
        episode_index: int,
        total_episodes: int,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ) -> None:
        """Clear the old trajectory and mark the new task."""
        self.point_line.set_data_3d([], [], [])
        self.current_point_artist.set_data_3d([], [], [])

        if self.start_artist is not None:
            self.start_artist.remove()
        if self.goal_artist is not None:
            self.goal_artist.remove()
        if self.start_link is not None:
            self.start_link.remove()
            self.start_link = None
        if self.goal_link is not None:
            self.goal_link.remove()
            self.goal_link = None

        self.start_artist = self.ax.scatter(
            [start_point[0]],
            [start_point[1]],
            [start_point[2]],
            c="limegreen",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            label="start",
            depthshade=False,
        )
        self.goal_artist = self.ax.scatter(
            [goal_point[0]],
            [goal_point[1]],
            [goal_point[2]],
            c="gold",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            label="goal",
            depthshade=False,
        )

        self.status_text.set_text(
            f"Episode {episode_index}/{total_episodes}\nWaiting for rollout..."
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.render_pause)

    def update(
        self,
        episode_index: int,
        total_episodes: int,
        point_path: np.ndarray,
        reward: float,
        episode_return: float,
        step: int,
        goal_distance: float,
    ) -> None:
        """Refresh the trajectory lines and overlay metrics."""
        self.point_line.set_data_3d(
            point_path[:, 0], point_path[:, 1], point_path[:, 2]
        )
        self.current_point_artist.set_data_3d(
            [point_path[-1, 0]],
            [point_path[-1, 1]],
            [point_path[-1, 2]],
        )

        self.status_text.set_text(
            "\n".join(
                [
                    f"Episode {episode_index}/{total_episodes}",
                    f"Step: {step}",
                    f"Reward: {reward:.3f}",
                    f"Return: {episode_return:.3f}",
                    f"Goal dist: {goal_distance:.3f}",
                ]
            )
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.render_pause)

    def pause_between_episodes(
        self,
        episode_index: int,
        total_episodes: int,
        episode_return: float,
        success: bool,
        min_goal_distance: float,
    ) -> None:
        """Show one short episode-end status before the next task starts."""
        self.status_text.set_text(
            "\n".join(
                [
                    f"Episode {episode_index}/{total_episodes} finished",
                    f"Return: {episode_return:.3f}",
                    f"Success: {success}",
                    f"Min dist: {min_goal_distance:.4f}",
                ]
            )
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.episode_pause)

    def finalize(self) -> None:
        """Keep the last frame visible after evaluation."""
        self.start_button.label.set_text("Done")
        self.fig.canvas.draw_idle()
        self.plt.ioff()
        self.plt.show()


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
        algorithm_cfg = dict(config.get("ppo", {}))
        algorithm_cfg["gamma"] = float(config.get("algorithm", {}).get("gamma", 0.99))
        agent: PPOAgent | SACAgent = PPOAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            model_cfg=model_cfg,
            algorithm_cfg=algorithm_cfg,
            device=device,
        )
    elif algorithm_name == "sac":
        algorithm_cfg = dict(config.get("sac", {}))
        algorithm_cfg["gamma"] = float(config.get("algorithm", {}).get("gamma", 0.99))
        agent = SACAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            model_cfg=model_cfg,
            algorithm_cfg=algorithm_cfg,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    agent.load_policy_state(checkpoint["state_dict"])
    return agent


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.get("eval", {})

    checkpoint_arg = args.checkpoint or eval_cfg.get("checkpoint", "")
    if not checkpoint_arg:
        raise ValueError(
            "No checkpoint provided. Set eval.checkpoint in config or pass --checkpoint."
        )

    checkpoint_path = Path(checkpoint_arg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = build_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    if args.config is None:
        checkpoint_config = checkpoint["config"]
        checkpoint_config["eval"] = {
            **checkpoint_config.get("eval", {}),
            **eval_cfg,
        }
        config = checkpoint_config
        eval_cfg = config.get("eval", {})

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
    agent = build_agent_from_checkpoint(checkpoint, config, device=device)
    algorithm_name = str(config.get("algorithm", {}).get("name", "ppo")).lower()
    if algorithm_name == "ppo":
        total_progress = int(config.get("ppo", {}).get("total_updates", 1))
    else:
        total_progress = int(config.get("sac", {}).get("total_steps", 1))
    action_scale_scheduler = build_action_scale_scheduler(
        rl_cfg,
        total_progress=total_progress,
    )
    if action_scale_scheduler is not None:
        env.set_action_scale_ratio(action_scale_scheduler.step(total_progress))

    total_episodes = int(args.episodes or eval_cfg.get("episodes", 5))
    eval_seed = int(eval_cfg.get("seed", 123))
    render_pause = float(eval_cfg.get("render_pause", 0.05))
    episode_pause = float(eval_cfg.get("episode_pause", 0.8))
    headless = args.headless
    if headless is None:
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


if __name__ == "__main__":
    main()
