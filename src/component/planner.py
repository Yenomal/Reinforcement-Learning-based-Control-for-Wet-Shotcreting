#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task sampler utilities for wall-constrained shotcrete planning."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import load_config
from ..rock_3D.robot_4dof.kinematics import RobotKinematics, load_robot_kinematics
from ..rock_env.rock_wall import (
    DELTA,
    L_TUNNEL,
    R_BASE,
    clamp_uv,
    generate_rock_environment,
    surface_normal_from_environment,
    surface_point_from_environment,
)


def _as_configuration_rad(
    values_deg: np.ndarray | list[float] | tuple[float, ...],
    dof: int,
) -> np.ndarray:
    values_array = np.asarray(values_deg, dtype=np.float32).reshape(dof)
    return np.deg2rad(values_array).astype(np.float32)


def _math_point_to_world(point_math: np.ndarray, axial_scale: float) -> np.ndarray:
    x_old, y_old, z_old = np.asarray(point_math, dtype=np.float32)
    return np.array([z_old * axial_scale, x_old, y_old], dtype=np.float32)


def _math_vector_to_world(vector_math: np.ndarray, axial_scale: float) -> np.ndarray:
    x_old, y_old, z_old = np.asarray(vector_math, dtype=np.float32)
    return np.array([z_old * axial_scale, x_old, y_old], dtype=np.float32)


def _estimate_surface_normal_world(
    u: float,
    v: float,
    *,
    rock_env: Dict[str, object],
    axial_scale: float,
) -> np.ndarray:
    """Estimate outward surface normal in the robot/PyBullet world frame."""
    normal = surface_normal_from_environment(rock_env, u, v, delta=DELTA)
    theta = u / R_BASE
    radial_math = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
    radial_world = _math_vector_to_world(radial_math, axial_scale=axial_scale)
    radial_world = radial_world / max(np.linalg.norm(radial_world), 1e-8)
    normal = _math_vector_to_world(normal, axial_scale=axial_scale)
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    if float(np.dot(normal, radial_world)) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def _sample_surface_goal(
    rock_env: Dict[str, object],
    rng: np.random.Generator,
    *,
    spray_angle_range_deg: tuple[float, float],
    spray_standoff_distance: float,
    axial_margin_ratio: float,
    axial_scale: float,
) -> Dict[str, object]:
    """Sample one wall point from the allowed wet-spray region and project inward."""
    angle_min_deg, angle_max_deg = spray_angle_range_deg
    spray_angle_deg = float(rng.uniform(angle_min_deg, angle_max_deg))
    theta = float((np.pi * 0.5) + np.deg2rad(spray_angle_deg))
    theta = float(np.clip(theta, 0.0, np.pi))
    u = float(R_BASE * theta)

    margin_v = float(L_TUNNEL * axial_margin_ratio)
    v = float(rng.uniform(margin_v, max(L_TUNNEL - margin_v, margin_v)))

    surface_point_math = surface_point_from_environment(rock_env, u, v)
    surface_point_world = _math_point_to_world(surface_point_math, axial_scale=axial_scale)
    surface_normal_world = _estimate_surface_normal_world(
        u,
        v,
        rock_env=rock_env,
        axial_scale=axial_scale,
    )
    target_point = surface_point_world - surface_normal_world * float(spray_standoff_distance)

    return {
        "surface_u": float(u),
        "surface_v": float(v),
        "spray_angle_deg": spray_angle_deg,
        "surface_point": surface_point_world.astype(np.float32),
        "surface_normal": surface_normal_world.astype(np.float32),
        "point": target_point.astype(np.float32),
    }


def _sample_surface_goal_from_reachability_map(
    reachability_map: Dict[str, object],
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Sample one prevalidated goal directly from the cached reachability map."""
    reachable_indices = np.asarray(reachability_map["reachable_indices"], dtype=np.int64)
    if reachable_indices.size == 0:
        raise ValueError("Reachability map does not contain any reachable planner cells.")

    chosen_index = int(rng.integers(0, reachable_indices.shape[0]))
    row, col = reachable_indices[chosen_index]

    best_q = np.asarray(reachability_map["best_q"], dtype=np.float32)[row, col]
    best_distance = float(np.asarray(reachability_map["best_distance"], dtype=np.float32)[row, col])
    step_lower_bound = float(
        np.asarray(reachability_map["step_lower_bound"], dtype=np.float32)[row, col]
    )

    return {
        "surface_u": float(np.asarray(reachability_map["surface_u_grid"], dtype=np.float32)[row, col]),
        "surface_v": float(np.asarray(reachability_map["surface_v_grid"], dtype=np.float32)[row, col]),
        "spray_angle_deg": float(
            np.asarray(reachability_map["spray_angle_grid"], dtype=np.float32)[row, col]
        ),
        "surface_point": np.asarray(
            reachability_map["surface_point_grid"], dtype=np.float32
        )[row, col].copy(),
        "surface_normal": np.asarray(
            reachability_map["surface_normal_grid"], dtype=np.float32
        )[row, col].copy(),
        "point": np.asarray(reachability_map["goal_point_grid"], dtype=np.float32)[row, col].copy(),
        "q_guess": best_q.copy(),
        "q_guess_deg": np.rad2deg(best_q).astype(np.float32),
        "reachability": {
            "reachable": True,
            "best_distance": best_distance,
            "step_lower_bound": step_lower_bound,
        },
    }


def build_task_joint_state(
    kinematics: RobotKinematics,
    q_rad: np.ndarray,
) -> Dict[str, object]:
    """Build one robot state from a joint configuration."""
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
    rock_env: Dict[str, object],
    kinematics: RobotKinematics,
    planner_cfg: Dict[str, object],
    rl_cfg: Dict[str, object],
    reachability_map: Optional[Dict[str, object]] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Sample one task from the wall spray region."""
    rng = np.random.default_rng(seed)
    dof = len(kinematics.joint_order)

    start_configuration_deg = rl_cfg.get("initial_configuration_deg")
    if start_configuration_deg is None:
        start_q_rad = kinematics.zero_q_rad.copy()
    else:
        start_q_rad = _as_configuration_rad(start_configuration_deg, dof=dof)
    start_q_rad = kinematics.clip_configuration(start_q_rad)
    start_state = build_task_joint_state(kinematics, start_q_rad)

    axial_scale = float(planner_cfg.get("tunnel_axial_scale", 1.5))
    spray_angle_range_deg = tuple(
        float(v) for v in planner_cfg.get("spray_angle_range_deg", (-60.0, 60.0))
    )
    spray_standoff_distance = float(planner_cfg.get("spray_standoff_distance", 1.5))
    axial_margin_ratio = float(planner_cfg.get("axial_margin_ratio", 0.05))
    max_goal_sampling_trials = int(planner_cfg.get("max_goal_sampling_trials", 256))

    goal_state: Optional[Dict[str, object]] = None
    if reachability_map is not None:
        goal_state = _sample_surface_goal_from_reachability_map(reachability_map, rng)

    for _ in range(max_goal_sampling_trials if goal_state is None else 0):
        sampled_goal = _sample_surface_goal(
            rock_env,
            rng,
            spray_angle_range_deg=spray_angle_range_deg,  # type: ignore[arg-type]
            spray_standoff_distance=spray_standoff_distance,
            axial_margin_ratio=axial_margin_ratio,
            axial_scale=axial_scale,
        )

        goal_state = sampled_goal
        break

    if goal_state is None:
        raise ValueError("Failed to sample a valid spray target under the current constraints.")

    return {
        "start": start_state,
        "goal": goal_state,
    }


def sample_planner_task_from_environment(
    rock_env: Dict[str, object],
    kinematics: RobotKinematics,
    planner_cfg: Dict[str, object],
    rl_cfg: Dict[str, object],
    reachability_map: Optional[Dict[str, object]] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Sample one planner task from a wall environment and robot model."""
    return sample_planner_task(
        rock_env=rock_env,
        kinematics=kinematics,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        reachability_map=reachability_map,
        seed=seed,
    )


def main() -> None:
    """Minimal smoke demo for wall-constrained planner task sampling."""
    cfg = load_config()
    env_cfg = cfg.get("env", {})
    robot_cfg = cfg.get("robot", {})
    planner_cfg = cfg.get("planner", {})
    rl_cfg = cfg.get("rl", {})

    rock_env = generate_rock_environment(
        n_theta=int(env_cfg.get("n_theta", 200)),
        n_z=int(env_cfg.get("n_z", 100)),
        seed=int(env_cfg.get("seed", 42)),
    )
    kinematics = load_robot_kinematics(robot_cfg.get("kinematics_path"))
    task = sample_planner_task_from_environment(
        rock_env=rock_env,
        kinematics=kinematics,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        seed=int(planner_cfg.get("seed", 0)),
    )

    print("Planner task sampler")
    print(f"Start q (deg): {task['start']['q_deg']}")
    print(f"Start EE: {task['start']['point']}")
    print(f"Goal surface point: {task['goal']['surface_point']}")
    print(f"Goal target point: {task['goal']['point']}")
    print(f"Spray angle (deg): {task['goal']['spray_angle_deg']:.2f}")
    if "reachability" in task["goal"]:
        print(f"Reachable: {task['goal']['reachability']['reachable']}")
        print(f"Best distance: {task['goal']['reachability']['best_distance']:.4f}")


if __name__ == "__main__":
    main()
