#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared learning-rate schedules for PPO and SAC training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict

from torch.optim import Optimizer


class OptimizerLRScheduler:
    """Stateless-on-resume optimizer learning-rate scheduler."""

    def __init__(
        self,
        optimizers: Dict[str, Optimizer | Sequence[Optimizer]],
        *,
        total_progress: int,
        schedule: str = "none",
        min_ratio: float = 0.1,
    ) -> None:
        self.schedule = str(schedule).lower()
        self.min_ratio = float(min_ratio)
        self.total_progress = max(int(total_progress), 1)

        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError("min_ratio must lie in [0, 1].")
        if self.schedule not in {"none", "cosine"}:
            raise ValueError(f"Unsupported lr schedule: {schedule}")

        self.optimizers: Dict[str, list[Optimizer]] = {}
        self.base_lrs: Dict[str, list[list[float]]] = {}

        for name, optimizer_or_group in optimizers.items():
            optimizer_list = (
                list(optimizer_or_group)
                if isinstance(optimizer_or_group, Sequence)
                else [optimizer_or_group]
            )
            if not optimizer_list:
                raise ValueError(f"Optimizer group '{name}' must not be empty.")
            self.optimizers[name] = optimizer_list
            self.base_lrs[name] = [
                [float(group["lr"]) for group in optimizer.param_groups]
                for optimizer in optimizer_list
            ]

    def _scale(self, progress: int) -> float:
        if self.schedule == "none":
            return 1.0

        if self.total_progress <= 1:
            ratio = 1.0
        else:
            ratio = min(max(float(progress) / float(self.total_progress - 1), 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.min_ratio + (1.0 - self.min_ratio) * cosine

    def step(self, progress: int) -> Dict[str, float]:
        scale = self._scale(progress)
        metrics: Dict[str, float] = {}

        for name, optimizer_list in self.optimizers.items():
            for optimizer, optimizer_base_lrs in zip(optimizer_list, self.base_lrs[name]):
                for param_group, base_lr in zip(optimizer.param_groups, optimizer_base_lrs):
                    param_group["lr"] = base_lr * scale
            metrics[name] = float(optimizer_list[0].param_groups[0]["lr"])

        return metrics
