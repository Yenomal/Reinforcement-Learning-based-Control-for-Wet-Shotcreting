#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disturbance utilities for observation-side sensor noise."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _as_uv_array(value: Any) -> np.ndarray:
    """Convert a scalar or 2D iterable into a uv-shaped noise array."""
    if isinstance(value, (int, float)):
        return np.array([float(value), float(value)], dtype=np.float32)

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (2,):
        raise ValueError("UV noise parameters must be a scalar or a length-2 array.")
    return array


class SensorNoise:
    """Observation-side sensor noise applied to current and goal uv estimates."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enable", False))

        self.current_uv_step_std = _as_uv_array(
            self.config.get("current_uv_step_std", 0.0)
        )
        self.current_uv_bias_std = _as_uv_array(
            self.config.get("current_uv_bias_std", 0.0)
        )
        self.goal_uv_step_std = _as_uv_array(self.config.get("goal_uv_step_std", 0.0))
        self.goal_uv_bias_std = _as_uv_array(self.config.get("goal_uv_bias_std", 0.0))

        self.current_uv_episode_bias = np.zeros(2, dtype=np.float32)
        self.goal_uv_episode_bias = np.zeros(2, dtype=np.float32)

    def reset_episode(self, rng: np.random.Generator) -> None:
        """Sample one fixed bias for the current episode."""
        if not self.enabled:
            self.current_uv_episode_bias = np.zeros(2, dtype=np.float32)
            self.goal_uv_episode_bias = np.zeros(2, dtype=np.float32)
            return

        self.current_uv_episode_bias = rng.normal(
            loc=0.0,
            scale=self.current_uv_bias_std,
            size=2,
        ).astype(np.float32)
        self.goal_uv_episode_bias = rng.normal(
            loc=0.0,
            scale=self.goal_uv_bias_std,
            size=2,
        ).astype(np.float32)

    def apply(
        self,
        current_uv: np.ndarray,
        goal_uv: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy uv observations while keeping internal state unchanged."""
        current_uv = np.asarray(current_uv, dtype=np.float32)
        goal_uv = np.asarray(goal_uv, dtype=np.float32)

        if not self.enabled:
            return current_uv.copy(), goal_uv.copy()

        current_step_noise = rng.normal(
            loc=0.0,
            scale=self.current_uv_step_std,
            size=2,
        ).astype(np.float32)
        goal_step_noise = rng.normal(
            loc=0.0,
            scale=self.goal_uv_step_std,
            size=2,
        ).astype(np.float32)

        noisy_current_uv = current_uv + self.current_uv_episode_bias + current_step_noise
        noisy_goal_uv = goal_uv + self.goal_uv_episode_bias + goal_step_noise
        return noisy_current_uv, noisy_goal_uv
