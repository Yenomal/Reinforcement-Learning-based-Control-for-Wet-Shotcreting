#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Buffer utilities for on-policy and off-policy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
        self.observations: list[Any] = []
        self.actions: list[Any] = []
        self.log_probs: list[Any] = []
        self.rewards: list[Any] = []
        self.dones: list[Any] = []
        self.values: list[Any] = []
        self.next_observation: Any | None = None
        self.next_done: Any | None = None

    def add(
        self,
        observation: Any,
        action: Any,
        log_prob: Any,
        reward: Any,
        done: Any,
        value: Any,
    ) -> None:
        if len(self) >= self.capacity:
            raise RuntimeError("OnPolicyBuffer is full. Call reset() before adding more data.")

        self.observations.append(_copy_sample(observation))
        self.actions.append(_copy_sample(action))
        self.log_probs.append(_copy_sample(log_prob))
        self.rewards.append(_copy_sample(reward))
        self.dones.append(_copy_sample(done))
        self.values.append(_copy_sample(value))

    def finalize(self, next_observation: Any, next_done: Any) -> None:
        self.next_observation = _copy_sample(next_observation)
        self.next_done = _copy_sample(next_done)

    def to_batch(self, device: torch.device) -> OnPolicyBatch:
        if len(self) == 0:
            raise RuntimeError("OnPolicyBuffer is empty.")
        if self.next_observation is None or self.next_done is None:
            raise RuntimeError("OnPolicyBuffer must be finalized before converting to a batch.")

        return OnPolicyBatch(
            observations=_to_torch_batch(self.observations, device=device),
            actions=_to_torch_batch(self.actions, device=device),
            log_probs=_to_torch_batch(self.log_probs, device=device),
            rewards=_to_torch_batch(self.rewards, device=device),
            dones=_to_torch_batch(self.dones, device=device),
            values=_to_torch_batch(self.values, device=device),
            next_observation=_to_torch_sample(self.next_observation, device=device),
            next_done=_to_torch_sample(self.next_done, device=device),
        )

    def __len__(self) -> int:
        return len(self.observations)


class ReplayBuffer(Buffer):
    """Ring buffer for off-policy algorithms such as SAC."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        storage_device: torch.device | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")

        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.storage_device = storage_device

        if self.storage_device is None:
            self.observations: np.ndarray | torch.Tensor = np.zeros(
                (self.capacity, self.observation_dim),
                dtype=np.float32,
            )
            self.actions: np.ndarray | torch.Tensor = np.zeros(
                (self.capacity, self.action_dim),
                dtype=np.float32,
            )
            self.rewards: np.ndarray | torch.Tensor = np.zeros(
                (self.capacity,),
                dtype=np.float32,
            )
            self.next_observations: np.ndarray | torch.Tensor = np.zeros(
                (self.capacity, self.observation_dim),
                dtype=np.float32,
            )
            self.dones: np.ndarray | torch.Tensor = np.zeros(
                (self.capacity,),
                dtype=np.float32,
            )
        else:
            self.observations = torch.zeros(
                (self.capacity, self.observation_dim),
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.actions = torch.zeros(
                (self.capacity, self.action_dim),
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.rewards = torch.zeros(
                (self.capacity,),
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.next_observations = torch.zeros(
                (self.capacity, self.observation_dim),
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.dones = torch.zeros(
                (self.capacity,),
                dtype=torch.float32,
                device=self.storage_device,
            )

        self.position = 0
        self.size = 0

    def reset(self) -> None:
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        self.add_batch(
            observations=_ensure_batch(observation, self.observation_dim),
            actions=_ensure_batch(action, self.action_dim),
            rewards=_ensure_vector(np.asarray([reward], dtype=np.float32)),
            next_observations=_ensure_batch(next_observation, self.observation_dim),
            dones=_ensure_vector(np.asarray([done], dtype=np.float32)),
        )

    def add_batch(
        self,
        observations: Any,
        actions: Any,
        rewards: Any,
        next_observations: Any,
        dones: Any,
    ) -> None:
        if self.storage_device is None:
            observations_array = np.asarray(observations, dtype=np.float32).reshape(
                -1, self.observation_dim
            )
            actions_array = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
            rewards_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
            next_observations_array = np.asarray(
                next_observations,
                dtype=np.float32,
            ).reshape(-1, self.observation_dim)
            dones_array = np.asarray(dones, dtype=np.float32).reshape(-1)
        else:
            observations_array = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=self.storage_device,
            ).reshape(-1, self.observation_dim)
            actions_array = torch.as_tensor(
                actions,
                dtype=torch.float32,
                device=self.storage_device,
            ).reshape(-1, self.action_dim)
            rewards_array = torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=self.storage_device,
            ).reshape(-1)
            next_observations_array = torch.as_tensor(
                next_observations,
                dtype=torch.float32,
                device=self.storage_device,
            ).reshape(-1, self.observation_dim)
            dones_array = torch.as_tensor(
                dones,
                dtype=torch.float32,
                device=self.storage_device,
            ).reshape(-1)

        batch_size = observations_array.shape[0]
        if batch_size >= self.capacity:
            observations_array = observations_array[-self.capacity :]
            actions_array = actions_array[-self.capacity :]
            rewards_array = rewards_array[-self.capacity :]
            next_observations_array = next_observations_array[-self.capacity :]
            dones_array = dones_array[-self.capacity :]
            batch_size = self.capacity

        indices = (self.position + np.arange(batch_size)) % self.capacity
        if self.storage_device is None:
            self.observations[indices] = observations_array
            self.actions[indices] = actions_array
            self.rewards[indices] = rewards_array
            self.next_observations[indices] = next_observations_array
            self.dones[indices] = dones_array
        else:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.storage_device)
            self.observations[index_tensor] = observations_array
            self.actions[index_tensor] = actions_array
            self.rewards[index_tensor] = rewards_array
            self.next_observations[index_tensor] = next_observations_array
            self.dones[index_tensor] = dones_array

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if len(self) < batch_size:
            raise RuntimeError("Not enough samples in ReplayBuffer.")

        indices = np.random.randint(0, self.size, size=int(batch_size))
        if self.storage_device is None:
            return ReplayBatch(
                observations=torch.as_tensor(
                    self.observations[indices],
                    dtype=torch.float32,
                    device=device,
                ),
                actions=torch.as_tensor(
                    self.actions[indices],
                    dtype=torch.float32,
                    device=device,
                ),
                rewards=torch.as_tensor(
                    self.rewards[indices],
                    dtype=torch.float32,
                    device=device,
                ),
                next_observations=torch.as_tensor(
                    self.next_observations[indices],
                    dtype=torch.float32,
                    device=device,
                ),
                dones=torch.as_tensor(
                    self.dones[indices],
                    dtype=torch.float32,
                    device=device,
                ),
            )

        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.storage_device)
        return ReplayBatch(
            observations=self.observations[index_tensor].to(device=device),
            actions=self.actions[index_tensor].to(device=device),
            rewards=self.rewards[index_tensor].to(device=device),
            next_observations=self.next_observations[index_tensor].to(device=device),
            dones=self.dones[index_tensor].to(device=device),
        )

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, object]:
        """Serialize replay-buffer state for checkpointing."""
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "storage_device": None if self.storage_device is None else str(self.storage_device),
            "position": self.position,
            "size": self.size,
            "observations": _serialize_storage(self.observations),
            "actions": _serialize_storage(self.actions),
            "rewards": _serialize_storage(self.rewards),
            "next_observations": _serialize_storage(self.next_observations),
            "dones": _serialize_storage(self.dones),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore replay-buffer state from checkpoint data."""
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        if self.storage_device is None:
            self.observations[:] = np.asarray(state_dict["observations"], dtype=np.float32)
            self.actions[:] = np.asarray(state_dict["actions"], dtype=np.float32)
            self.rewards[:] = np.asarray(state_dict["rewards"], dtype=np.float32)
            self.next_observations[:] = np.asarray(
                state_dict["next_observations"], dtype=np.float32
            )
            self.dones[:] = np.asarray(state_dict["dones"], dtype=np.float32)
        else:
            self.observations[:] = torch.as_tensor(
                state_dict["observations"],
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.actions[:] = torch.as_tensor(
                state_dict["actions"],
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.rewards[:] = torch.as_tensor(
                state_dict["rewards"],
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.next_observations[:] = torch.as_tensor(
                state_dict["next_observations"],
                dtype=torch.float32,
                device=self.storage_device,
            )
            self.dones[:] = torch.as_tensor(
                state_dict["dones"],
                dtype=torch.float32,
                device=self.storage_device,
            )


def _copy_sample(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    return np.asarray(value, dtype=np.float32).copy()


def _to_torch_sample(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _to_torch_batch(values: list[Any], device: torch.device) -> torch.Tensor:
    if not values:
        raise RuntimeError("Cannot convert an empty buffer to a torch batch.")
    if torch.is_tensor(values[0]):
        return torch.stack(
            [value.to(device=device, dtype=torch.float32) for value in values],
            dim=0,
        )
    return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device)


def _serialize_storage(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value.copy()


def _ensure_batch(value: Any, width: int) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1, width)


def _ensure_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)
