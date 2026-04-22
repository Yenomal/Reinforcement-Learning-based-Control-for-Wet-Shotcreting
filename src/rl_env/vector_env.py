#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vectorized training environments for PPO and SAC."""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from contextlib import suppress
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .math_env import MathEnv


SEED_STRIDE = 9973


def _build_math_env(
    env_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    robot_cfg: Optional[Dict[str, Any]] = None,
    algorithm_cfg: Optional[Dict[str, Any]] = None,
    disturbance_cfg: Optional[Dict[str, Any]] = None,
) -> MathEnv:
    return MathEnv(
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        algorithm_cfg=algorithm_cfg,
        disturbance_cfg=disturbance_cfg,
    )


def _compress_training_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the training-relevant episode summary to reduce IPC overhead."""
    compressed: Dict[str, Any] = {}
    if "episode" in info:
        compressed["episode"] = info["episode"]
    return compressed


def _auto_reset_step(env: MathEnv, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    next_observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    compressed_info = _compress_training_info(info)
    if done:
        reset_observation, _ = env.reset()
        return reset_observation, float(reward), terminated, truncated, compressed_info
    return next_observation, float(reward), terminated, truncated, compressed_info


def _seed_for_env(seed: Optional[int], env_index: int) -> Optional[int]:
    if seed is None:
        return None
    return int(seed + env_index * SEED_STRIDE)


def _subproc_worker(
    remote: Connection,
    parent_remote: Connection,
    env_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    robot_cfg: Optional[Dict[str, Any]],
    algorithm_cfg: Optional[Dict[str, Any]],
    disturbance_cfg: Optional[Dict[str, Any]],
) -> None:
    parent_remote.close()
    env = _build_math_env(
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        algorithm_cfg=algorithm_cfg,
        disturbance_cfg=disturbance_cfg,
    )

    try:
        while True:
            command, data = remote.recv()
            if command == "reset":
                observation, info = env.reset(seed=data)
                remote.send((observation, _compress_training_info(info)))
            elif command == "step":
                remote.send(_auto_reset_step(env, data))
            elif command == "spec":
                remote.send((env.observation_dim, env.action_dim))
            elif command == "close":
                break
            else:
                raise ValueError(f"Unsupported vector-env command: {command}")
    finally:
        with suppress(Exception):
            remote.close()


class BaseVectorEnv(ABC):
    """Minimal vector-environment interface used by the training loops."""

    def __init__(self, num_envs: int, observation_dim: int, action_dim: int) -> None:
        self.num_envs = int(num_envs)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments and return batched observations."""

    @abstractmethod
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments with a batch of actions."""

    @abstractmethod
    def close(self) -> None:
        """Release environment resources."""


class SyncVectorEnv(BaseVectorEnv):
    """Synchronous vector environment executed in the main process."""

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
            _build_math_env(
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
            infos.append(_compress_training_info(info))
        return np.stack(observations, axis=0).astype(np.float32), infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        batched_actions = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, self.action_dim
        )
        next_observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: List[Dict[str, Any]] = []

        for env, action in zip(self.envs, batched_actions):
            next_observation, reward, term, trunc, info = _auto_reset_step(env, action)
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

    def close(self) -> None:
        self.envs = []


class SubprocVectorEnv(BaseVectorEnv):
    """Multiprocess vector environment for faster rollout collection."""

    def __init__(
        self,
        num_envs: int,
        env_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        rl_cfg: Dict[str, Any],
        robot_cfg: Optional[Dict[str, Any]] = None,
        algorithm_cfg: Optional[Dict[str, Any]] = None,
        disturbance_cfg: Optional[Dict[str, Any]] = None,
        start_method: str = "spawn",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be a positive integer.")

        self.num_envs = int(num_envs)
        self.closed = False
        self.ctx = mp.get_context(start_method)
        self.remotes: list[Connection] = []
        self.processes: list[mp.Process] = []

        for _ in range(self.num_envs):
            parent_remote, child_remote = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_subproc_worker,
                args=(
                    child_remote,
                    parent_remote,
                    dict(env_cfg),
                    dict(planner_cfg),
                    dict(rl_cfg),
                    dict(robot_cfg or {}),
                    dict(algorithm_cfg or {}),
                    dict(disturbance_cfg or {}),
                ),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

        self.remotes[0].send(("spec", None))
        observation_dim, action_dim = self.remotes[0].recv()
        super().__init__(
            num_envs=self.num_envs,
            observation_dim=int(observation_dim),
            action_dim=int(action_dim),
        )

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        for env_index, remote in enumerate(self.remotes):
            remote.send(("reset", _seed_for_env(seed, env_index)))
        results = [remote.recv() for remote in self.remotes]
        observations, infos = zip(*results)
        return np.stack(observations, axis=0).astype(np.float32), list(infos)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        batched_actions = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, self.action_dim
        )
        for remote, action in zip(self.remotes, batched_actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]

        next_observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(next_observations, axis=0).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            with suppress(Exception):
                remote.send(("close", None))
        for remote in self.remotes:
            with suppress(Exception):
                remote.close()
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def build_vector_env(
    *,
    num_envs: int,
    backend: str,
    env_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    robot_cfg: Optional[Dict[str, Any]] = None,
    algorithm_cfg: Optional[Dict[str, Any]] = None,
    disturbance_cfg: Optional[Dict[str, Any]] = None,
    start_method: str = "spawn",
) -> BaseVectorEnv:
    normalized_backend = str(backend).lower()
    if int(num_envs) <= 1:
        return SyncVectorEnv(
            num_envs=1,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            rl_cfg=rl_cfg,
            robot_cfg=robot_cfg,
            algorithm_cfg=algorithm_cfg,
            disturbance_cfg=disturbance_cfg,
        )

    if normalized_backend == "sync":
        return SyncVectorEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            rl_cfg=rl_cfg,
            robot_cfg=robot_cfg,
            algorithm_cfg=algorithm_cfg,
            disturbance_cfg=disturbance_cfg,
        )
    if normalized_backend == "subproc":
        return SubprocVectorEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            rl_cfg=rl_cfg,
            robot_cfg=robot_cfg,
            algorithm_cfg=algorithm_cfg,
            disturbance_cfg=disturbance_cfg,
            start_method=start_method,
        )

    raise ValueError(f"Unsupported vector-env backend: {backend}")
