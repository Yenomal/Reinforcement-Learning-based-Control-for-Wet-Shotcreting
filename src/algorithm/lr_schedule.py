#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared scalar and learning-rate schedules for PPO and SAC training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict

from torch.optim import Optimizer


class ScalarScheduler:
    """Stateless scalar scheduler that can drive lr, entropy, or other values."""

    def __init__(
        self,
        *,
        start_value: float,
        end_value: float | None = None,
        total_progress: int,
        schedule: str = "none",
    ) -> None:
        self.start_value = float(start_value)
        self.end_value = (
            float(self.start_value) if end_value is None else float(end_value)
        )
        self.total_progress = max(int(total_progress), 1)
        self.schedule = str(schedule).lower()

        if self.schedule not in {"none", "cosine"}:
            raise ValueError(f"Unsupported scalar schedule: {schedule}")

    def value(self, progress: int) -> float:
        if self.schedule == "none" or self.start_value == self.end_value:
            return self.start_value

        if self.total_progress <= 1:
            ratio = 1.0
        else:
            ratio = min(max(float(progress) / float(self.total_progress - 1), 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.end_value + (self.start_value - self.end_value) * cosine

    def step(self, progress: int) -> float:
        return self.value(progress)


class SuccessRateTriggeredScheduler:
    """Decrease a scalar only after smoothed success stays above a threshold."""

    def __init__(
        self,
        *,
        start_value: float,
        min_value: float,
        success_threshold: float,
        value_step: float,
        ema_alpha: float,
        patience_updates: int,
        cooldown_updates: int,
        min_episodes_in_window: int,
    ) -> None:
        self.current_value = float(start_value)
        self.min_value = float(min_value)
        self.success_threshold = float(success_threshold)
        self.value_step = float(value_step)
        self.ema_alpha = float(ema_alpha)
        self.patience_updates = max(int(patience_updates), 1)
        self.cooldown_updates = max(int(cooldown_updates), 0)
        self.min_episodes_in_window = max(int(min_episodes_in_window), 0)

        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must lie in [0, 1].")
        if self.value_step >= 0.0:
            raise ValueError("value_step must be negative for log_std decay.")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must lie in (0, 1].")
        if self.min_value > self.current_value:
            raise ValueError("min_value must be <= start_value.")

        self.success_ema: float | None = None
        self.streak = 0
        self.stage = 0
        self.cooldown_remaining = 0

    def cooldown_total(self) -> int:
        """Fixed cooldown length after each log_std decay."""
        return self.cooldown_updates

    def current(self) -> float:
        return float(self.current_value)

    def state(self) -> Dict[str, float]:
        return {
            "ppo_success_ema": (
                float("nan") if self.success_ema is None else float(self.success_ema)
            ),
            "ppo_std_streak": float(self.streak),
            "ppo_std_stage": float(self.stage),
            "ppo_next_log_std": float(self.current_value),
            "ppo_std_cooldown_remaining": float(self.cooldown_remaining),
        }

    def observe(
        self,
        success_rate: float,
        *,
        episodes_in_window: int | None = None,
    ) -> Dict[str, float]:
        success_rate = float(success_rate)
        if not math.isfinite(success_rate):
            return self.state()

        if self.success_ema is None:
            self.success_ema = success_rate
        else:
            self.success_ema = (
                self.ema_alpha * success_rate
                + (1.0 - self.ema_alpha) * self.success_ema
            )

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.streak = 0
            return self.state()

        if episodes_in_window is not None and int(episodes_in_window) < self.min_episodes_in_window:
            self.streak = 0
            return self.state()

        can_decrease = self.current_value > (self.min_value + 1.0e-12)
        if self.success_ema >= self.success_threshold and can_decrease:
            self.streak += 1
            if self.streak >= self.patience_updates:
                self.current_value = max(
                    self.current_value + self.value_step,
                    self.min_value,
                )
                self.streak = 0
                self.stage += 1
                self.cooldown_remaining = self.cooldown_total()
        else:
            self.streak = 0

        return self.state()


class CosineThenSuccessRateTriggeredScheduler:
    """Use cosine first, then switch to success-triggered decay."""

    def __init__(
        self,
        *,
        cosine_scheduler: ScalarScheduler,
        trigger_scheduler: SuccessRateTriggeredScheduler,
        switch_update: int,
    ) -> None:
        self.cosine_scheduler = cosine_scheduler
        self.trigger_scheduler = trigger_scheduler
        self.switch_update = max(int(switch_update), 0)

        if self.switch_update > 0:
            handoff_value = self.cosine_scheduler.value(self.switch_update - 1)
            self.trigger_scheduler.current_value = handoff_value

    def in_trigger_phase(self, progress: int) -> bool:
        return int(progress) >= self.switch_update

    def current(self, progress: int) -> float:
        if self.in_trigger_phase(progress):
            return self.trigger_scheduler.current()
        return self.cosine_scheduler.value(progress)

    def state(self, progress: int) -> Dict[str, float | str]:
        if self.in_trigger_phase(progress):
            state = dict(self.trigger_scheduler.state())
            state["ppo_std_phase"] = "trigger"
            return state

        return {
            "ppo_success_ema": float("nan"),
            "ppo_std_streak": 0.0,
            "ppo_std_stage": 0.0,
            "ppo_next_log_std": float(self.cosine_scheduler.value(progress)),
            "ppo_std_cooldown_remaining": float("nan"),
            "ppo_std_phase": "cosine",
        }

    def observe(
        self,
        success_rate: float,
        *,
        episodes_in_window: int | None = None,
    ) -> Dict[str, float | str]:
        state = dict(
            self.trigger_scheduler.observe(
                success_rate,
                episodes_in_window=episodes_in_window,
            )
        )
        state["ppo_std_phase"] = "trigger"
        return state


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

        self.scale_scheduler = ScalarScheduler(
            start_value=1.0,
            end_value=self.min_ratio,
            total_progress=self.total_progress,
            schedule=self.schedule,
        )

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

    def step(self, progress: int) -> Dict[str, float]:
        scale = self.scale_scheduler.step(progress)
        metrics: Dict[str, float] = {}

        for name, optimizer_list in self.optimizers.items():
            for optimizer, optimizer_base_lrs in zip(optimizer_list, self.base_lrs[name]):
                for param_group, base_lr in zip(optimizer.param_groups, optimizer_base_lrs):
                    param_group["lr"] = base_lr * scale
            metrics[name] = float(optimizer_list[0].param_groups[0]["lr"])

        return metrics
