#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal-conditioned mathematical planning environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..component.disturbance import SensorNoise
from ..component.planner import sample_planner_task_from_environment
from ..rock_env.rock_wall import generate_rock_environment, query_surface_state


class MathEnv:
    """A lightweight goal-conditioned environment in the flattened tunnel space."""

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        disturbance_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env_cfg = dict(env_cfg)
        self.planner_cfg = dict(planner_cfg)
        self.rl_cfg = dict(rl_cfg)
        self.disturbance_cfg = dict(disturbance_cfg or {})

        self.rock_env = generate_rock_environment(
            n_theta=int(self.env_cfg.get("n_theta", 200)),
            n_z=int(self.env_cfg.get("n_z", 100)),
            seed=int(self.env_cfg.get("seed", 42)),
        )
        self.noise_gen = self.rock_env["noise_gen"]
        self.u_bounds = tuple(float(v) for v in self.rock_env["u_bounds"])
        self.v_bounds = tuple(float(v) for v in self.rock_env["v_bounds"])
        self.uv_span = np.array(
            [self.u_bounds[1] - self.u_bounds[0], self.v_bounds[1] - self.v_bounds[0]],
            dtype=np.float32,
        )

        self.max_episode_steps = int(self.rl_cfg.get("max_episode_steps", 200))
        self.goal_tolerance_uv = float(self.rl_cfg.get("goal_tolerance_uv", 0.2))
        self.action_scale = np.array(
            [
                float(self.rl_cfg.get("action_scale_u", 0.2)),
                float(self.rl_cfg.get("action_scale_v", 0.2)),
            ],
            dtype=np.float32,
        )
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
        self.observation_dim = 9
        self.action_dim = 2

        self.current_uv = np.zeros(2, dtype=np.float32)
        self.goal_uv = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        self.episode_return = 0.0
        self.previous_goal_distance = 0.0
        self.min_goal_distance = float("inf")
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_surface_state: Optional[Dict[str, Any]] = None
        self.path_uv: list[np.ndarray] = []
        self.path_surface: list[np.ndarray] = []
        self.path_retreated: list[np.ndarray] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new planning episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        sampled_seed = int(self._rng.integers(0, 2**31 - 1))
        self.current_task = sample_planner_task_from_environment(
            self.rock_env,
            seed=sampled_seed,
            retreat_distance=float(self.planner_cfg.get("retreat_distance", 1.0)),
            margin_ratio=float(self.planner_cfg.get("margin_ratio", 0.05)),
            min_uv_distance_ratio=float(
                self.planner_cfg.get("min_start_goal_ratio", 0.30)
            ),
            k_neighbors=int(self.planner_cfg.get("normal_neighbors", 32)),
        )

        self.current_uv = self.current_task["start"]["uv"].astype(np.float32)
        self.goal_uv = self.current_task["goal"]["uv"].astype(np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_step = 0
        self.episode_return = 0.0
        self.previous_goal_distance = self._goal_distance(self.current_uv)
        self.min_goal_distance = self.previous_goal_distance
        self.current_surface_state = self._query_runtime_state(self.current_uv)
        self.sensor_noise.reset_episode(self._rng)

        self.path_uv = [self.current_uv.copy()]
        self.path_surface = [self.current_surface_state["compensated_point"].copy()]
        self.path_retreated = [self.current_surface_state["retreated_point"].copy()]

        observation = self._build_observation()
        info = self._build_info(done=False, success=False)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one action."""
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        clipped_action = np.clip(action, -1.0, 1.0)

        proposed_uv = self.current_uv + clipped_action * self.action_scale
        clipped_uv = self._clip_uv(proposed_uv)
        boundary_hit = not np.allclose(proposed_uv, clipped_uv)

        self.current_uv = clipped_uv.astype(np.float32)
        self.current_step += 1
        self.current_surface_state = self._query_runtime_state(self.current_uv)

        goal_distance = self._goal_distance(self.current_uv)
        progress_reward = self.progress_reward_weight * (
            self.previous_goal_distance - goal_distance
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

        success = goal_distance <= self.goal_tolerance_uv
        terminated = success
        truncated = self.current_step >= self.max_episode_steps and not terminated

        if terminated:
            reward += self.success_reward

        self.episode_return += reward
        self.previous_goal_distance = goal_distance
        self.min_goal_distance = min(self.min_goal_distance, goal_distance)
        self.prev_action = clipped_action.astype(np.float32)

        self.path_uv.append(self.current_uv.copy())
        self.path_surface.append(self.current_surface_state["compensated_point"].copy())
        self.path_retreated.append(self.current_surface_state["retreated_point"].copy())

        observation = self._build_observation()
        info = self._build_info(done=terminated or truncated, success=success)
        info["reward_terms"] = {
            "progress_reward": progress_reward,
            "step_penalty": step_penalty,
            "action_penalty": action_penalty,
            "smoothness_penalty": smoothness_penalty,
            "boundary_penalty": boundary_penalty,
        }
        return observation, float(reward), terminated, truncated, info

    def _clip_uv(self, uv: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(uv[0], self.u_bounds[0], self.u_bounds[1]),
                np.clip(uv[1], self.v_bounds[0], self.v_bounds[1]),
            ],
            dtype=np.float32,
        )

    def _goal_distance(self, uv: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_uv - uv))

    def _normalize_uv(self, uv: np.ndarray) -> np.ndarray:
        lower = np.array([self.u_bounds[0], self.v_bounds[0]], dtype=np.float32)
        normalized = (uv - lower) / np.maximum(self.uv_span, 1e-6)
        return 2.0 * normalized - 1.0

    def _build_observation(self) -> np.ndarray:
        observed_current_uv, observed_goal_uv = self.sensor_noise.apply(
            self.current_uv,
            self.goal_uv,
            self._rng,
        )
        current_uv_norm = self._normalize_uv(observed_current_uv)
        goal_uv_norm = self._normalize_uv(observed_goal_uv)
        delta_uv_norm = (observed_goal_uv - observed_current_uv) / np.maximum(
            self.uv_span, 1e-6
        )
        step_ratio = np.array(
            [self.current_step / max(self.max_episode_steps, 1)],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                current_uv_norm.astype(np.float32),
                goal_uv_norm.astype(np.float32),
                delta_uv_norm.astype(np.float32),
                self.prev_action.astype(np.float32),
                step_ratio,
            ]
        )
        return observation.astype(np.float32)

    def _query_runtime_state(self, uv: np.ndarray) -> Dict[str, Any]:
        surface_state = query_surface_state(
            float(uv[0]),
            float(uv[1]),
            noise_gen=self.noise_gen,
        )
        retreat_distance = float(self.planner_cfg.get("retreat_distance", 1.0))
        surface_state["retreated_point"] = (
            surface_state["compensated_point"] - surface_state["normal"] * retreat_distance
        )
        return surface_state

    def _build_info(self, done: bool, success: bool) -> Dict[str, Any]:
        info = {
            "current_uv": self.current_uv.copy(),
            "goal_uv": self.goal_uv.copy(),
            "goal_distance_uv": self._goal_distance(self.current_uv),
            "episode_step": self.current_step,
            "episode_return": self.episode_return,
            "success": success,
            "surface_point": self.current_surface_state["compensated_point"].copy(),
            "retreated_point": self.current_surface_state["retreated_point"].copy(),
        }
        if done and self.current_task is not None:
            info["episode"] = {
                "start_uv": self.current_task["start"]["uv"].copy(),
                "goal_uv": self.current_task["goal"]["uv"].copy(),
                "uv_path": np.asarray(self.path_uv, dtype=np.float32),
                "surface_path": np.asarray(self.path_surface, dtype=np.float32),
                "retreated_path": np.asarray(self.path_retreated, dtype=np.float32),
                "return": float(self.episode_return),
                "length": int(self.current_step),
                "success": bool(success),
                "min_goal_distance_uv": float(self.min_goal_distance),
            }
        return info
