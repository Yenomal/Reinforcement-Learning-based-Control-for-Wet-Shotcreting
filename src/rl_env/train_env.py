#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified training-environment builders for classic and torch backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .math_env import MathEnv
from .torch_math_env import TorchMathEnv


SEED_STRIDE = 9973


def _seed_for_env(seed: Optional[int], env_index: int) -> Optional[int]:
    if seed is None:
        return None
    return int(seed + env_index * SEED_STRIDE)


class BaseTrainEnv(ABC):
    """Minimal batched environment interface for PPO/SAC training."""

    def __init__(self, num_envs: int, observation_dim: int, action_dim: int) -> None:
        self.num_envs = int(num_envs)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[Any, List[Dict[str, Any]]]:
        """Reset all training environments."""

    @abstractmethod
    def step(
        self, actions: Any
    ) -> tuple[Any, Any, Any, Any, List[Dict[str, Any]]]:
        """Step all environments with batched actions."""

    def close(self) -> None:
        return None


class ClassicTrainEnv(BaseTrainEnv):
    """Single-process batched wrapper around one or more classic MathEnv instances."""

    def __init__(
        self,
        num_envs: int,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        robot_cfg: Optional[Dict[str, Any]] = None,
        algorithm_cfg: Optional[Dict[str, Any]] = None,
        disturbance_cfg: Optional[Dict[str, Any]] = None,
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
            for _ in range(int(num_envs))
        ]
        super().__init__(
            num_envs=len(self.envs),
            observation_dim=self.envs[0].observation_dim,
            action_dim=self.envs[0].action_dim,
        )

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        observations: list[np.ndarray] = []
        infos: List[Dict[str, Any]] = []
        for env_index, env in enumerate(self.envs):
            observation, info = env.reset(seed=_seed_for_env(seed, env_index))
            observations.append(observation)
            infos.append(info)
        return np.stack(observations, axis=0).astype(np.float32), infos

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        batched_actions = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs,
            self.action_dim,
        )
        next_observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: List[Dict[str, Any]] = []

        for env, action in zip(self.envs, batched_actions):
            next_observation, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if done:
                reset_observation, _ = env.reset()
                next_observations.append(reset_observation)
            else:
                next_observations.append(next_observation)
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            infos.append(info)

        return (
            np.stack(next_observations, axis=0).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )


def build_train_env(
    *,
    backend: str,
    num_envs: int,
    device: torch.device,
    env_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    robot_cfg: Optional[Dict[str, Any]] = None,
    algorithm_cfg: Optional[Dict[str, Any]] = None,
    disturbance_cfg: Optional[Dict[str, Any]] = None,
) -> BaseTrainEnv:
    normalized_backend = str(backend).lower()
    if normalized_backend == "classic":
        return ClassicTrainEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            rl_cfg=rl_cfg,
            robot_cfg=robot_cfg,
            algorithm_cfg=algorithm_cfg,
            disturbance_cfg=disturbance_cfg,
        )
    if normalized_backend == "torch":
        return TorchMathEnv(
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            rl_cfg=rl_cfg,
            robot_cfg=robot_cfg,
            algorithm_cfg=algorithm_cfg,
            disturbance_cfg=disturbance_cfg,
            num_envs=num_envs,
            device=device,
        )
    raise ValueError(f"Unsupported training environment backend: {backend}")
