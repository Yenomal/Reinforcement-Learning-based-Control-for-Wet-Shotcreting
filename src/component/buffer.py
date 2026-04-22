#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Buffer utilities for on-policy and off-policy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OnPolicyBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_observation: torch.Tensor
    next_done: torch.Tensor


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class Buffer(ABC):
    """Minimal base interface for all experience buffers."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal storage."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored samples."""


class OnPolicyBuffer(Buffer):
    """Finite buffer used for PPO-style on-policy rollout collection."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        self.capacity = int(capacity)
        self.reset()

    def reset(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []
        self.next_observation: np.ndarray | None = None
        self.next_done: float | None = None

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        if len(self) >= self.capacity:
            raise RuntimeError("OnPolicyBuffer is full. Call reset() before adding more data.")

        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def finalize(self, next_observation: np.ndarray, next_done: bool) -> None:
        self.next_observation = np.asarray(next_observation, dtype=np.float32).copy()
        self.next_done = float(next_done)

    def to_batch(self, device: torch.device) -> OnPolicyBatch:
        if len(self) == 0:
            raise RuntimeError("OnPolicyBuffer is empty.")
        if self.next_observation is None or self.next_done is None:
            raise RuntimeError("OnPolicyBuffer must be finalized before converting to a batch.")

        return OnPolicyBatch(
            observations=torch.as_tensor(
                np.asarray(self.observations, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                np.asarray(self.actions, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            log_probs=torch.as_tensor(
                np.asarray(self.log_probs, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            rewards=torch.as_tensor(
                np.asarray(self.rewards, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            dones=torch.as_tensor(
                np.asarray(self.dones, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            values=torch.as_tensor(
                np.asarray(self.values, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            next_observation=torch.as_tensor(
                self.next_observation,
                dtype=torch.float32,
                device=device,
            ),
            next_done=torch.as_tensor(
                [self.next_done],
                dtype=torch.float32,
                device=device,
            ),
        )

    def __len__(self) -> int:
        return len(self.observations)


class ReplayBuffer(Buffer):
    """Ring buffer for off-policy algorithms such as SAC."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")

        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        self.observations = np.zeros((self.capacity, self.observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.position = 0
        self.size = 0

    def reset(self) -> None:
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = np.asarray(observation, dtype=np.float32)
        self.actions[self.position] = np.asarray(action, dtype=np.float32)
        self.rewards[self.position] = float(reward)
        self.next_observations[self.position] = np.asarray(
            next_observation, dtype=np.float32
        )
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if len(self) < batch_size:
            raise RuntimeError("Not enough samples in ReplayBuffer.")

        indices = np.random.randint(0, self.size, size=int(batch_size))
        return ReplayBatch(
            observations=torch.as_tensor(
                self.observations[indices], dtype=torch.float32, device=device
            ),
            actions=torch.as_tensor(
                self.actions[indices], dtype=torch.float32, device=device
            ),
            rewards=torch.as_tensor(
                self.rewards[indices], dtype=torch.float32, device=device
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices], dtype=torch.float32, device=device
            ),
            dones=torch.as_tensor(
                self.dones[indices], dtype=torch.float32, device=device
            ),
        )

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, object]:
        """Serialize replay-buffer state for checkpointing."""
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "observations": self.observations.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_observations": self.next_observations.copy(),
            "dones": self.dones.copy(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore replay-buffer state from checkpoint data."""
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.observations[:] = np.asarray(state_dict["observations"], dtype=np.float32)
        self.actions[:] = np.asarray(state_dict["actions"], dtype=np.float32)
        self.rewards[:] = np.asarray(state_dict["rewards"], dtype=np.float32)
        self.next_observations[:] = np.asarray(
            state_dict["next_observations"], dtype=np.float32
        )
        self.dones[:] = np.asarray(state_dict["dones"], dtype=np.float32)
