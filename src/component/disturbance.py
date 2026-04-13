#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disturbance utilities for observation-side sensor noise."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _as_point_array(value: Any) -> np.ndarray:
    """Convert a scalar or 3D iterable into a point-shaped noise array."""
    if isinstance(value, (int, float)):
        scalar = float(value)
        return np.array([scalar, scalar, scalar], dtype=np.float32)

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError("Point noise parameters must be a scalar or a length-3 array.")
    return array


class SensorNoise:
    """Observation-side sensor noise applied to current and goal point estimates."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enable", False))

        self.current_point_step_std = _as_point_array(
            self.config.get("current_point_step_std", 0.0)
        )
        self.current_point_bias_std = _as_point_array(
            self.config.get("current_point_bias_std", 0.0)
        )
        self.goal_point_step_std = _as_point_array(
            self.config.get("goal_point_step_std", 0.0)
        )
        self.goal_point_bias_std = _as_point_array(
            self.config.get("goal_point_bias_std", 0.0)
        )

        self.current_point_episode_bias = np.zeros(3, dtype=np.float32)
        self.goal_point_episode_bias = np.zeros(3, dtype=np.float32)

    def reset_episode(self, rng: np.random.Generator) -> None:
        """Sample one fixed bias for the current episode."""
        if not self.enabled:
            self.current_point_episode_bias = np.zeros(3, dtype=np.float32)
            self.goal_point_episode_bias = np.zeros(3, dtype=np.float32)
            return

        self.current_point_episode_bias = rng.normal(
            loc=0.0,
            scale=self.current_point_bias_std,
            size=3,
        ).astype(np.float32)
        self.goal_point_episode_bias = rng.normal(
            loc=0.0,
            scale=self.goal_point_bias_std,
            size=3,
        ).astype(np.float32)

    def apply(
        self,
        current_point: np.ndarray,
        goal_point: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy point observations while keeping internal state unchanged."""
        current_point = np.asarray(current_point, dtype=np.float32)
        goal_point = np.asarray(goal_point, dtype=np.float32)

        if not self.enabled:
            return current_point.copy(), goal_point.copy()

        current_step_noise = rng.normal(
            loc=0.0,
            scale=self.current_point_step_std,
            size=3,
        ).astype(np.float32)
        goal_step_noise = rng.normal(
            loc=0.0,
            scale=self.goal_point_step_std,
            size=3,
        ).astype(np.float32)

        noisy_current_point = (
            current_point + self.current_point_episode_bias + current_step_noise
        )
        noisy_goal_point = goal_point + self.goal_point_episode_bias + goal_step_noise
        return noisy_current_point, noisy_goal_point
