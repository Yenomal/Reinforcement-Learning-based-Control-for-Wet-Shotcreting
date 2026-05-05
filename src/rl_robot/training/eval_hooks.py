from __future__ import annotations

from typing import Any, Dict

from rl_robot.algorithms.lr_schedule import ScalarScheduler


def build_action_scale_scheduler(
    rl_cfg: Dict[str, Any],
    *,
    total_progress: int,
) -> ScalarScheduler | None:
    schedule_cfg = dict(rl_cfg.get("action_scale_schedule", {}))
    if not bool(schedule_cfg.get("enable", False)):
        return None

    return ScalarScheduler(
        start_value=float(schedule_cfg.get("start_ratio", 1.0)),
        end_value=float(schedule_cfg.get("end_ratio", 1.0)),
        total_progress=total_progress,
        schedule=str(schedule_cfg.get("schedule", "cosine")),
    )


__all__ = ["build_action_scale_scheduler"]
