#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task sampler utilities for the joint-space planning stage."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import load_config
from ..rock_3D.robot_4dof.kinematics import RobotKinematics, load_robot_kinematics


def _as_configuration_rad(
    values_deg: np.ndarray | list[float] | tuple[float, ...],
    dof: int,
) -> np.ndarray:
    values_array = np.asarray(values_deg, dtype=np.float32).reshape(dof)
    return np.deg2rad(values_array).astype(np.float32)


def build_task_joint_state(
    kinematics: RobotKinematics,
    q_rad: np.ndarray,
) -> Dict[str, object]:
    """Build one planner state from a joint configuration."""
    q_rad = np.asarray(q_rad, dtype=np.float32).reshape(len(kinematics.joint_order))
    fk = kinematics.forward_kinematics(q_rad)
    joint_positions = fk["joint_positions"]

    return {
        "q": q_rad.copy(),
        "q_deg": np.rad2deg(q_rad).astype(np.float32),
        "point": fk["tool_tip"].copy(),
        "joint_positions": {
            name: joint_positions[name].copy()
            for name in joint_positions
        },
        "joint_axes_world": {
            name: fk["joint_axes_world"][name].copy()
            for name in fk["joint_axes_world"]
        },
    }


def sample_planner_task(
    kinematics: RobotKinematics,
    seed: Optional[int] = None,
    start_configuration_deg: Optional[np.ndarray | list[float] | tuple[float, ...]] = None,
) -> Dict[str, object]:
    """Sample one planner task from robot joint limits."""
    rng = np.random.default_rng(seed)

    if start_configuration_deg is None:
        start_q_rad = kinematics.zero_q_rad.copy()
    else:
        start_q_rad = _as_configuration_rad(
            start_configuration_deg,
            dof=len(kinematics.joint_order),
        )
    start_q_rad = kinematics.clip_configuration(start_q_rad)
    goal_q_rad = kinematics.sample_random_configuration(rng)

    start_state = build_task_joint_state(kinematics, start_q_rad)
    goal_state = build_task_joint_state(kinematics, goal_q_rad)

    return {
        "start": start_state,
        "goal": goal_state,
    }


def sample_planner_task_from_kinematics(
    kinematics: RobotKinematics,
    seed: Optional[int] = None,
    start_configuration_deg: Optional[np.ndarray | list[float] | tuple[float, ...]] = None,
) -> Dict[str, object]:
    """Sample one planner task from a kinematics object."""
    return sample_planner_task(
        kinematics=kinematics,
        seed=seed,
        start_configuration_deg=start_configuration_deg,
    )


def main() -> None:
    """Minimal smoke demo for the joint-space planner task sampler."""
    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    planner_cfg = cfg.get("planner", {})
    rl_cfg = cfg.get("rl", {})

    kinematics = load_robot_kinematics(robot_cfg.get("kinematics_path"))
    task = sample_planner_task_from_kinematics(
        kinematics=kinematics,
        seed=int(planner_cfg.get("seed", 0)),
        start_configuration_deg=rl_cfg.get("initial_configuration_deg"),
    )

    print("Planner task sampler")
    print(f"Start q (deg): {task['start']['q_deg']}")
    print(f"Goal q (deg): {task['goal']['q_deg']}")
    print(f"Start EE: {task['start']['point']}")
    print(f"Goal EE: {task['goal']['point']}")


if __name__ == "__main__":
    main()
