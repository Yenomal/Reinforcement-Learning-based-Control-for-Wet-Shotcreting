#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal-conditioned joint-space environment with FK-based EE tracking."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..component.disturbance import SensorNoise
from ..component.planner import sample_planner_task_from_environment
from ..rock_3D.robot_4dof.kinematics import RobotKinematics, load_robot_kinematics
from ..rock_env.rock_wall import generate_rock_environment


def _as_float_array(value: Any, length: int) -> np.ndarray:
    """Convert a scalar or fixed-length iterable into a float array."""
    if isinstance(value, (int, float)):
        scalar = float(value)
        return np.full(length, scalar, dtype=np.float32)

    array = np.asarray(value, dtype=np.float32).reshape(length)
    return array.astype(np.float32)


class MathEnv:
    """A goal-conditioned joint-space environment aligned with the PyBullet robot."""

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        robot_cfg: Optional[Dict[str, Any]] = None,
        algorithm_cfg: Optional[Dict[str, Any]] = None,
        disturbance_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env_cfg = dict(env_cfg)
        self.planner_cfg = dict(planner_cfg)
        self.rl_cfg = dict(rl_cfg)
        self.robot_cfg = dict(robot_cfg or {})
        self.algorithm_cfg = dict(algorithm_cfg or {})
        self.disturbance_cfg = dict(disturbance_cfg or {})

        self.rock_env = generate_rock_environment(
            n_theta=int(self.env_cfg.get("n_theta", 200)),
            n_z=int(self.env_cfg.get("n_z", 100)),
            seed=int(self.env_cfg.get("seed", 42)),
        )

        self.kinematics: RobotKinematics = load_robot_kinematics(
            self.robot_cfg.get("kinematics_path")
        )
        workspace_padding = float(self.rl_cfg.get("workspace_margin", 0.25))
        self.point_lower, self.point_upper = self.kinematics.estimate_workspace_bounds(
            padding=workspace_padding
        )
        self.point_span = np.maximum(self.point_upper - self.point_lower, 1e-6).astype(
            np.float32
        )

        dof = len(self.kinematics.joint_order)
        max_joint_delta_deg = self.rl_cfg.get(
            "max_joint_delta_deg",
            self.rl_cfg.get("action_scale_joint_deg", 4.0),
        )
        self.max_joint_delta = np.deg2rad(
            _as_float_array(max_joint_delta_deg, dof)
        ).astype(np.float32)

        self.max_episode_steps = int(self.rl_cfg.get("max_episode_steps", 200))
        self.goal_tolerance = float(self.rl_cfg.get("goal_tolerance", 0.2))
        self.potential_gamma = float(self.algorithm_cfg.get("gamma", 0.99))
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
        self.observation_dim = dof + 3 + 3 + 3 + dof + 1
        self.action_dim = dof

        self.current_q = np.zeros(dof, dtype=np.float32)
        self.goal_q = np.zeros(dof, dtype=np.float32)
        self.current_point = np.zeros(3, dtype=np.float32)
        self.goal_point = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(dof, dtype=np.float32)
        self.current_step = 0
        self.episode_return = 0.0
        self.previous_goal_distance = 0.0
        self.min_goal_distance = float("inf")
        self.current_task: Optional[Dict[str, Any]] = None
        self.path_q: list[np.ndarray] = []
        self.path_points: list[np.ndarray] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new planning episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        sampled_seed = int(self._rng.integers(0, 2**31 - 1))
        self.current_task = sample_planner_task_from_environment(
            rock_env=self.rock_env,
            kinematics=self.kinematics,
            planner_cfg=self.planner_cfg,
            rl_cfg=self.rl_cfg,
            seed=sampled_seed,
        )

        self.current_q = self.current_task["start"]["q"].astype(np.float32)
        goal_q_guess = self.current_task["goal"].get("q_guess")
        if goal_q_guess is None:
            self.goal_q = np.full(self.action_dim, np.nan, dtype=np.float32)
        else:
            self.goal_q = np.asarray(goal_q_guess, dtype=np.float32).reshape(self.action_dim)
        self.current_point = self.current_task["start"]["point"].astype(np.float32)
        self.goal_point = self.current_task["goal"]["point"].astype(np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_step = 0
        self.episode_return = 0.0
        self.previous_goal_distance = self._goal_distance(self.current_point)
        self.min_goal_distance = self.previous_goal_distance
        self.sensor_noise.reset_episode(self._rng)

        self.path_q = [self.current_q.copy()]
        self.path_points = [self.current_point.copy()]

        observation = self._build_observation()
        info = self._build_info(done=False, success=False)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one action."""
        normalized_joint_delta = np.asarray(action, dtype=np.float32).reshape(
            self.action_dim
        )
        # PPO now outputs bounded joint-delta commands in [-1, 1]. This clamp
        # is only a safety guard for external callers or numerical spillover.
        bounded_joint_delta = np.clip(normalized_joint_delta, -1.0, 1.0)

        proposed_q = self.current_q + bounded_joint_delta * self.max_joint_delta
        clipped_q = self.kinematics.clip_configuration(proposed_q)
        boundary_hit = not np.allclose(proposed_q, clipped_q)

        self.current_q = clipped_q.astype(np.float32)
        fk = self.kinematics.forward_kinematics(self.current_q)
        self.current_point = fk["tool_tip"].astype(np.float32)
        self.current_step += 1

        goal_distance = self._goal_distance(self.current_point)
        progress_reward = self.progress_reward_weight * (
            self.previous_goal_distance - self.potential_gamma * goal_distance
        )
        step_penalty = self.step_penalty
        action_penalty = self.action_l2_weight * float(np.sum(bounded_joint_delta**2))
        smoothness_penalty = self.action_smoothness_weight * float(
            np.sum((bounded_joint_delta - self.prev_action) ** 2)
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
        self.prev_action = bounded_joint_delta.astype(np.float32)
        self.path_q.append(self.current_q.copy())
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

    def _goal_distance(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_point - point))

    def _normalize_point(self, point: np.ndarray) -> np.ndarray:
        normalized = (point - self.point_lower) / self.point_span
        return (2.0 * normalized - 1.0).astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        observed_current_point, observed_goal_point = self.sensor_noise.apply(
            self.current_point,
            self.goal_point,
            self._rng,
        )
        q_norm = self.kinematics.normalize_configuration(self.current_q)
        current_point_norm = self._normalize_point(observed_current_point)
        goal_point_norm = self._normalize_point(observed_goal_point)
        delta_point_norm = (
            (observed_goal_point - observed_current_point) / self.point_span
        ).astype(np.float32)
        step_ratio = np.array(
            [self.current_step / max(self.max_episode_steps, 1)],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                q_norm.astype(np.float32),
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
            "current_q": self.current_q.copy(),
            "current_q_deg": np.rad2deg(self.current_q).astype(np.float32),
            "current_point": self.current_point.copy(),
            "goal_point": self.goal_point.copy(),
            "goal_distance": self._goal_distance(self.current_point),
            "episode_step": self.current_step,
            "episode_return": self.episode_return,
            "success": success,
        }
        if np.all(np.isfinite(self.goal_q)):
            info["goal_q"] = self.goal_q.copy()
            info["goal_q_deg"] = np.rad2deg(self.goal_q).astype(np.float32)
        if self.current_task is not None:
            info["goal_surface_point"] = self.current_task["goal"]["surface_point"].copy()
            info["goal_surface_normal"] = self.current_task["goal"]["surface_normal"].copy()
            info["goal_spray_angle_deg"] = float(self.current_task["goal"]["spray_angle_deg"])
            if "reachability" in self.current_task["goal"]:
                info["goal_reachability"] = {
                    "reachable": bool(self.current_task["goal"]["reachability"]["reachable"]),
                    "best_distance": float(self.current_task["goal"]["reachability"]["best_distance"]),
                }

        if done and self.current_task is not None:
            info["episode"] = {
                "start_q_deg": self.current_task["start"]["q_deg"].copy(),
                "start_point": self.current_task["start"]["point"].copy(),
                "goal_point": self.current_task["goal"]["point"].copy(),
                "goal_surface_point": self.current_task["goal"]["surface_point"].copy(),
                "goal_surface_normal": self.current_task["goal"]["surface_normal"].copy(),
                "goal_spray_angle_deg": float(self.current_task["goal"]["spray_angle_deg"]),
                "q_path": np.asarray(self.path_q, dtype=np.float32),
                "point_path": np.asarray(self.path_points, dtype=np.float32),
                "return": float(self.episode_return),
                "length": int(self.current_step),
                "success": bool(success),
                "min_goal_distance": float(self.min_goal_distance),
            }
            if np.all(np.isfinite(self.goal_q)):
                info["episode"]["goal_q_deg"] = np.rad2deg(self.goal_q).astype(np.float32)
            if "reachability" in self.current_task["goal"]:
                info["episode"]["goal_reachability"] = {
                    "reachable": bool(self.current_task["goal"]["reachability"]["reachable"]),
                    "best_distance": float(self.current_task["goal"]["reachability"]["best_distance"]),
                }
        return info
