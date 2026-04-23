#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-batched analytical training environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..component.disturbance import SensorNoise
from ..component.planner import sample_planner_task_from_environment
from ..component.reachability_map import build_or_load_reachability_map
from ..rock_3D.robot_4dof.kinematics import RobotKinematics, load_robot_kinematics
from ..rock_3D.robot_4dof.torch_kinematics import TorchRobotKinematics
from ..rock_env.rock_wall import build_training_rock_environment


SEED_STRIDE = 9973


def _as_float_array(value: Any, length: int) -> np.ndarray:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return np.full(length, scalar, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(length).astype(np.float32)


@dataclass
class EpisodeSummaryState:
    start_q_deg: np.ndarray
    start_point: np.ndarray
    goal_point: np.ndarray


class TorchMathEnv:
    """Torch-batched variant of MathEnv for higher GPU utilization."""

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        robot_cfg: Optional[Dict[str, Any]] = None,
        algorithm_cfg: Optional[Dict[str, Any]] = None,
        disturbance_cfg: Optional[Dict[str, Any]] = None,
        *,
        num_envs: int,
        device: torch.device,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be a positive integer.")

        self.env_cfg = dict(env_cfg)
        self.planner_cfg = dict(planner_cfg)
        self.rl_cfg = dict(rl_cfg)
        self.robot_cfg = dict(robot_cfg or {})
        self.algorithm_cfg = dict(algorithm_cfg or {})
        self.disturbance_cfg = dict(disturbance_cfg or {})
        self.device = device
        self.num_envs = int(num_envs)
        self.dtype = torch.float32

        self.rock_env = build_training_rock_environment(self.env_cfg)
        self.kinematics: RobotKinematics = load_robot_kinematics(
            self.robot_cfg.get("kinematics_path")
        )
        self.reachability_map = build_or_load_reachability_map(
            rock_env=self.rock_env,
            kinematics=self.kinematics,
            env_cfg=self.env_cfg,
            planner_cfg=self.planner_cfg,
            rl_cfg=self.rl_cfg,
            robot_cfg=self.robot_cfg,
            device=self.device,
        )
        self.torch_kinematics = TorchRobotKinematics.from_robot_kinematics(
            self.kinematics,
            device=self.device,
        )

        workspace_padding = float(self.rl_cfg.get("workspace_margin", 0.25))
        point_lower, point_upper = self.kinematics.estimate_workspace_bounds(
            padding=workspace_padding
        )
        point_span = np.maximum(point_upper - point_lower, 1e-6).astype(np.float32)
        self.point_lower = torch.as_tensor(point_lower, dtype=self.dtype, device=self.device)
        self.point_upper = torch.as_tensor(point_upper, dtype=self.dtype, device=self.device)
        self.point_span = torch.as_tensor(point_span, dtype=self.dtype, device=self.device)

        dof = len(self.kinematics.joint_order)
        max_joint_delta_deg = self.rl_cfg.get(
            "max_joint_delta_deg",
            self.rl_cfg.get("action_scale_joint_deg", 4.0),
        )
        self.max_joint_delta = torch.as_tensor(
            np.deg2rad(_as_float_array(max_joint_delta_deg, dof)).astype(np.float32),
            dtype=self.dtype,
            device=self.device,
        )
        self.action_scale_ratio = 1.0

        self.max_episode_steps = int(self.rl_cfg.get("max_episode_steps", 200))
        self.goal_tolerance = float(self.rl_cfg.get("goal_tolerance", 0.2))
        self.progress_reward_weight = float(self.rl_cfg.get("progress_reward_weight", 1.0))
        self.sensor_noises = [
            SensorNoise(self.disturbance_cfg.get("sensor_noise", {}))
            for _ in range(self.num_envs)
        ]
        self.observation_dim = dof + 3 + 3 + 3 + dof + 1
        self.action_dim = dof

        self.rngs = [
            np.random.default_rng(int(self.planner_cfg.get("seed", 0)) + env_index * SEED_STRIDE)
            for env_index in range(self.num_envs)
        ]

        self.current_q = torch.zeros((self.num_envs, dof), dtype=self.dtype, device=self.device)
        self.goal_q = torch.full(
            (self.num_envs, dof),
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        self.current_point = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.goal_point = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.prev_action = torch.zeros((self.num_envs, dof), dtype=self.dtype, device=self.device)
        self.current_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        self.previous_goal_distance = torch.zeros(
            self.num_envs,
            dtype=self.dtype,
            device=self.device,
        )
        self.min_goal_distance = torch.full(
            (self.num_envs,),
            float("inf"),
            dtype=self.dtype,
            device=self.device,
        )
        self.episode_summary_state: List[EpisodeSummaryState | None] = [None] * self.num_envs

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
        indices = list(range(self.num_envs))
        self._reset_indices(indices, base_seed=seed)
        observation = self._build_observation()
        return observation, [{} for _ in range(self.num_envs)]

    def step(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        action_tensor = torch.as_tensor(action, dtype=self.dtype, device=self.device).reshape(
            self.num_envs, self.action_dim
        )
        bounded_joint_delta = torch.clamp(action_tensor, -1.0, 1.0)

        effective_max_joint_delta = self.max_joint_delta * float(self.action_scale_ratio)
        proposed_q = self.current_q + bounded_joint_delta * effective_max_joint_delta
        clipped_q = self.torch_kinematics.clip_configuration(proposed_q)
        boundary_hit = torch.any(torch.abs(proposed_q - clipped_q) > 1e-6, dim=-1)

        previous_goal_distance = self.previous_goal_distance.clone()
        self.current_q = clipped_q
        self.current_point = self.torch_kinematics.forward_kinematics(self.current_q)
        self.current_step = self.current_step + 1

        goal_distance = self._goal_distance(self.current_point)
        progress_reward, previous_phi, current_phi = self._compute_progress_reward(
            previous_goal_distance,
            goal_distance,
        )
        reward = progress_reward

        success = goal_distance <= self.goal_tolerance
        terminated = success
        truncated = (self.current_step >= self.max_episode_steps) & (~terminated)
        done = terminated | truncated

        self.episode_return = self.episode_return + reward
        self.previous_goal_distance = goal_distance
        self.min_goal_distance = torch.minimum(self.min_goal_distance, goal_distance)
        self.prev_action = bounded_joint_delta

        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        done_indices = torch.nonzero(done, as_tuple=False).flatten().tolist()
        if done_indices:
            for env_index in done_indices:
                infos[env_index]["episode"] = self._build_episode_summary(
                    env_index=env_index,
                    success=bool(success[env_index].item()),
                )
            self._reset_indices(done_indices, base_seed=None)

        observation = self._build_observation()
        return observation, reward, terminated, truncated, infos

    def close(self) -> None:
        return None

    def set_action_scale_ratio(self, ratio: float) -> None:
        self.action_scale_ratio = float(max(ratio, 1.0e-6))

    def get_action_scale_ratio(self) -> float:
        return float(self.action_scale_ratio)

    def _reset_indices(self, indices: Sequence[int], base_seed: Optional[int]) -> None:
        for env_index in indices:
            if base_seed is not None:
                self.rngs[env_index] = np.random.default_rng(base_seed + env_index * SEED_STRIDE)
            rng = self.rngs[env_index]
            sampled_seed = int(rng.integers(0, 2**31 - 1))
            current_task = sample_planner_task_from_environment(
                rock_env=self.rock_env,
                kinematics=self.kinematics,
                planner_cfg=self.planner_cfg,
                rl_cfg=self.rl_cfg,
                reachability_map=self.reachability_map,
                seed=sampled_seed,
            )

            current_q = current_task["start"]["q"].astype(np.float32)
            goal_q_guess = current_task["goal"].get("q_guess")
            if goal_q_guess is None:
                goal_q = np.full(self.action_dim, np.nan, dtype=np.float32)
            else:
                goal_q = np.asarray(goal_q_guess, dtype=np.float32).reshape(self.action_dim)

            current_point = current_task["start"]["point"].astype(np.float32)
            goal_point = current_task["goal"]["point"].astype(np.float32)

            self.current_q[env_index] = torch.as_tensor(
                current_q,
                dtype=self.dtype,
                device=self.device,
            )
            self.goal_q[env_index] = torch.as_tensor(
                goal_q,
                dtype=self.dtype,
                device=self.device,
            )
            self.current_point[env_index] = torch.as_tensor(
                current_point,
                dtype=self.dtype,
                device=self.device,
            )
            self.goal_point[env_index] = torch.as_tensor(
                goal_point,
                dtype=self.dtype,
                device=self.device,
            )
            self.prev_action[env_index].zero_()
            self.current_step[env_index] = 0
            self.episode_return[env_index] = 0.0
            initial_distance = float(np.linalg.norm(goal_point - current_point))
            self.previous_goal_distance[env_index] = initial_distance
            self.min_goal_distance[env_index] = initial_distance
            self.sensor_noises[env_index].reset_episode(rng)
            self.episode_summary_state[env_index] = EpisodeSummaryState(
                start_q_deg=current_task["start"]["q_deg"].astype(np.float32),
                start_point=current_point.copy(),
                goal_point=goal_point.copy(),
            )

    def _goal_distance(self, point: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.goal_point - point, dim=-1)

    def _distance_potential(self, distance: torch.Tensor) -> torch.Tensor:
        log_scale = max(self.goal_tolerance, 1e-6)
        phi = torch.log1p(distance / log_scale)
        return phi

    def _compute_progress_reward(
        self,
        previous_goal_distance: torch.Tensor,
        goal_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_phi = self._distance_potential(previous_goal_distance)
        current_phi = self._distance_potential(goal_distance)
        reward = self.progress_reward_weight * (previous_phi - current_phi)
        return reward, previous_phi, current_phi

    def _normalize_point(self, point: torch.Tensor) -> torch.Tensor:
        normalized = (point - self.point_lower) / self.point_span
        return 2.0 * normalized - 1.0

    def _build_observation(self) -> torch.Tensor:
        if all(not sensor_noise.enabled for sensor_noise in self.sensor_noises):
            observed_current_point = self.current_point
            observed_goal_point = self.goal_point
        else:
            current_points: list[torch.Tensor] = []
            goal_points: list[torch.Tensor] = []
            for env_index in range(self.num_envs):
                noisy_current, noisy_goal = self.sensor_noises[env_index].apply(
                    self.current_point[env_index].detach().cpu().numpy(),
                    self.goal_point[env_index].detach().cpu().numpy(),
                    self.rngs[env_index],
                )
                current_points.append(
                    torch.as_tensor(noisy_current, dtype=self.dtype, device=self.device)
                )
                goal_points.append(
                    torch.as_tensor(noisy_goal, dtype=self.dtype, device=self.device)
                )
            observed_current_point = torch.stack(current_points, dim=0)
            observed_goal_point = torch.stack(goal_points, dim=0)

        q_norm = self.torch_kinematics.normalize_configuration(self.current_q)
        current_point_norm = self._normalize_point(observed_current_point)
        goal_point_norm = self._normalize_point(observed_goal_point)
        delta_point_norm = (observed_goal_point - observed_current_point) / self.point_span
        step_ratio = (
            self.current_step.to(self.dtype) / max(float(self.max_episode_steps), 1.0)
        ).unsqueeze(-1)

        observation = torch.cat(
            [
                q_norm,
                current_point_norm,
                goal_point_norm,
                delta_point_norm,
                self.prev_action,
                step_ratio,
            ],
            dim=-1,
        )
        return observation

    def _build_episode_summary(self, env_index: int, success: bool) -> Dict[str, Any]:
        summary_state = self.episode_summary_state[env_index]
        assert summary_state is not None
        return {
            "start_q_deg": summary_state.start_q_deg.copy(),
            "start_point": summary_state.start_point.copy(),
            "goal_point": summary_state.goal_point.copy(),
            "return": float(self.episode_return[env_index].detach().cpu().item()),
            "length": int(self.current_step[env_index].detach().cpu().item()),
            "success": bool(success),
            "min_goal_distance": float(
                self.min_goal_distance[env_index].detach().cpu().item()
            ),
        }
