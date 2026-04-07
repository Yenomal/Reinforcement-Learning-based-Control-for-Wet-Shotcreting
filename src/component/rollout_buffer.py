#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""On-policy rollout buffer for PPO-style algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_observation: torch.Tensor
    next_done: torch.Tensor


class RolloutBuffer:
    """Finite on-policy buffer that is reset before each rollout collection."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")

        self.capacity = int(capacity)
        self.reset()

    def reset(self) -> None:
        """Clear all stored transitions."""
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
        """Append one transition to the rollout buffer."""
        if len(self) >= self.capacity:
            raise RuntimeError("RolloutBuffer is full. Call reset() before adding more data.")

        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def finalize(self, next_observation: np.ndarray, next_done: bool) -> None:
        """Store the bootstrap state used after the rollout ends."""
        self.next_observation = np.asarray(next_observation, dtype=np.float32).copy()
        self.next_done = float(next_done)

    def to_batch(self, device: torch.device) -> RolloutBatch:
        """Convert the rollout contents into tensors for learning."""
        if len(self) == 0:
            raise RuntimeError("RolloutBuffer is empty.")
        if self.next_observation is None or self.next_done is None:
            raise RuntimeError("RolloutBuffer must be finalized before converting to a batch.")

        return RolloutBatch(
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
