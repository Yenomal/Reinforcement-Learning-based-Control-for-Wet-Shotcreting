#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal-conditioned 3D point environment for tunnel planning."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..component.disturbance import SensorNoise
from ..component.planner import sample_planner_task_from_environment
from ..rock_env.rock_wall import generate_rock_environment


def _as_xyz_array(value: Any) -> np.ndarray:
    """Convert a scalar or length-3 iterable into an xyz array."""
    if isinstance(value, (int, float)):
        scalar = float(value)
        return np.array([scalar, scalar, scalar], dtype=np.float32)

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError("XYZ values must be a scalar or a length-3 array.")
    return array


class MathEnv:
    """A goal-conditioned free-space point environment anchored to the rock wall."""

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        algorithm_cfg: Optional[Dict[str, Any]] = None,
        disturbance_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env_cfg = dict(env_cfg)
        self.planner_cfg = dict(planner_cfg)
        self.rl_cfg = dict(rl_cfg)
        self.algorithm_cfg = dict(algorithm_cfg or {})
        self.disturbance_cfg = dict(disturbance_cfg or {})

        self.rock_env = generate_rock_environment(
            n_theta=int(self.env_cfg.get("n_theta", 200)),
            n_z=int(self.env_cfg.get("n_z", 100)),
            seed=int(self.env_cfg.get("seed", 42)),
        )

        self.retreat_distance = float(self.planner_cfg.get("retreat_distance", 1.0))
        raw_points = np.asarray(self.rock_env["points"], dtype=np.float32)
        workspace_margin = float(
            self.rl_cfg.get("workspace_margin", self.retreat_distance + 0.25)
        )
        self.point_lower = raw_points.min(axis=0) - workspace_margin
        self.point_upper = raw_points.max(axis=0) + workspace_margin
        self.point_span = np.maximum(self.point_upper - self.point_lower, 1e-6).astype(
            np.float32
        )

        action_scale_cfg = self.rl_cfg.get("action_scale")
        if action_scale_cfg is None:
            action_scale_cfg = [
                float(self.rl_cfg.get("action_scale_x", 0.2)),
                float(self.rl_cfg.get("action_scale_y", 0.2)),
                float(self.rl_cfg.get("action_scale_z", 0.2)),
            ]

        self.max_episode_steps = int(self.rl_cfg.get("max_episode_steps", 200))
        self.goal_tolerance = float(
            self.rl_cfg.get(
                "goal_tolerance",
                self.rl_cfg.get("goal_tolerance_xyz", 0.2),
            )
        )
        self.potential_gamma = float(self.algorithm_cfg.get("gamma", 0.99))
        self.action_scale = _as_xyz_array(action_scale_cfg)
        self.progress_reward_weight = float(
            self.rl_cfg.get("progress_reward_weight", 1.0)
        )
        self.success_reward = float(self.rl_cfg.get("success_reward", 20.0))
        self.step_penalty = float(self.rl_cfg.get("step_penalty", 0.01))
        self.action_l2_weight = float(self.rl_cfg.get("action_l2_weight", 0.001))
        self.action_smoothness_weight = float(
            self.rl_cfg.get("action_smoothness_weight", 0.01)
        )
        self.boundary_penalty = float(self.rl_cfg.get("boundary_penalty", 0.1))
        self.sensor_noise = SensorNoise(
            self.disturbance_cfg.get("sensor_noise", {})
        )

        self._rng = np.random.default_rng(int(self.planner_cfg.get("seed", 0)))
        self.observation_dim = 13
        self.action_dim = 3

        self.current_point = np.zeros(3, dtype=np.float32)
        self.goal_point = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.current_step = 0
        self.episode_return = 0.0
        self.previous_goal_distance = 0.0
        self.min_goal_distance = float("inf")
        self.current_task: Optional[Dict[str, Any]] = None
        self.path_points: list[np.ndarray] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new planning episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        sampled_seed = int(self._rng.integers(0, 2**31 - 1))
        self.current_task = sample_planner_task_from_environment(
            self.rock_env,
            seed=sampled_seed,
            retreat_distance=self.retreat_distance,
            margin_ratio=float(self.planner_cfg.get("margin_ratio", 0.05)),
            min_point_distance_ratio=float(
                self.planner_cfg.get("min_start_goal_ratio", 0.30)
            ),
            k_neighbors=int(self.planner_cfg.get("normal_neighbors", 32)),
        )

        self.current_point = self.current_task["start"]["retreated_point"].astype(np.float32)
        self.goal_point = self.current_task["goal"]["retreated_point"].astype(np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_step = 0
        self.episode_return = 0.0
        self.previous_goal_distance = self._goal_distance(self.current_point)
        self.min_goal_distance = self.previous_goal_distance
        self.sensor_noise.reset_episode(self._rng)

        self.path_points = [self.current_point.copy()]

        observation = self._build_observation()
        info = self._build_info(done=False, success=False)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one action."""
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        clipped_action = np.clip(action, -1.0, 1.0)

        proposed_point = self.current_point + clipped_action * self.action_scale
        clipped_point = self._clip_point(proposed_point)
        boundary_hit = not np.allclose(proposed_point, clipped_point)

        self.current_point = clipped_point.astype(np.float32)
        self.current_step += 1

        goal_distance = self._goal_distance(self.current_point)
        progress_reward = self.progress_reward_weight * (
            self.previous_goal_distance - self.potential_gamma * goal_distance
        )
        step_penalty = self.step_penalty
        action_penalty = self.action_l2_weight * float(np.sum(clipped_action**2))
        smoothness_penalty = self.action_smoothness_weight * float(
            np.sum((clipped_action - self.prev_action) ** 2)
        )
        boundary_penalty = self.boundary_penalty if boundary_hit else 0.0

        reward = (
            progress_reward
            - step_penalty
            - action_penalty
            - smoothness_penalty
            - boundary_penalty
        )

        success = goal_distance <= self.goal_tolerance
        terminated = success
        truncated = self.current_step >= self.max_episode_steps and not terminated

        if terminated:
            reward += self.success_reward

        self.episode_return += reward
        self.previous_goal_distance = goal_distance
        self.min_goal_distance = min(self.min_goal_distance, goal_distance)
        self.prev_action = clipped_action.astype(np.float32)
        self.path_points.append(self.current_point.copy())

        observation = self._build_observation()
        info = self._build_info(done=terminated or truncated, success=success)
        info["reward_terms"] = {
            "progress_reward": progress_reward,
            "potential_gamma": self.potential_gamma,
            "step_penalty": step_penalty,
            "action_penalty": action_penalty,
            "smoothness_penalty": smoothness_penalty,
            "boundary_penalty": boundary_penalty,
        }
        return observation, float(reward), terminated, truncated, info

    def _clip_point(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self.point_lower, self.point_upper).astype(np.float32)

    def _goal_distance(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_point - point))

    def _normalize_point(self, point: np.ndarray) -> np.ndarray:
        normalized = (point - self.point_lower) / self.point_span
        return 2.0 * normalized - 1.0

    def _build_observation(self) -> np.ndarray:
        observed_current_point, observed_goal_point = self.sensor_noise.apply(
            self.current_point,
            self.goal_point,
            self._rng,
        )
        current_point_norm = self._normalize_point(observed_current_point)
        goal_point_norm = self._normalize_point(observed_goal_point)
        delta_point_norm = (observed_goal_point - observed_current_point) / self.point_span
        step_ratio = np.array(
            [self.current_step / max(self.max_episode_steps, 1)],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                current_point_norm.astype(np.float32),
                goal_point_norm.astype(np.float32),
                delta_point_norm.astype(np.float32),
                self.prev_action.astype(np.float32),
                step_ratio,
            ]
        )
        return observation.astype(np.float32)

    def _build_info(self, done: bool, success: bool) -> Dict[str, Any]:
        info = {
            "current_point": self.current_point.copy(),
            "goal_point": self.goal_point.copy(),
            "goal_distance": self._goal_distance(self.current_point),
            "episode_step": self.current_step,
            "episode_return": self.episode_return,
            "success": success,
        }

        if self.current_task is not None:
            info["start_surface_point"] = self.current_task["start"]["surface_point"].copy()
            info["goal_surface_point"] = self.current_task["goal"]["surface_point"].copy()

        if done and self.current_task is not None:
            info["episode"] = {
                "start_point": self.current_task["start"]["retreated_point"].copy(),
                "goal_point": self.current_task["goal"]["retreated_point"].copy(),
                "start_surface_point": self.current_task["start"]["surface_point"].copy(),
                "goal_surface_point": self.current_task["goal"]["surface_point"].copy(),
                "point_path": np.asarray(self.path_points, dtype=np.float32),
                "return": float(self.episode_return),
                "length": int(self.current_step),
                "success": bool(success),
                "min_goal_distance": float(self.min_goal_distance),
            }
        return info
