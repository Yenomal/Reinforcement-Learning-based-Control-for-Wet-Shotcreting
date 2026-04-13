#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task sampler utilities for the shotcrete planner stage.

This module does not generate a full path. It only:
- takes a 3D rock-wall point cloud
- randomly samples a valid start/goal pair in 3D
- estimates local normals from the point cloud
- optionally offsets points along the normal for downstream RL usage

The legacy uv-flatten helpers are kept for future analysis, but the main
task-sampling path now works directly in 3D point space.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..config import load_config
from ..rock_env.rock_wall import N_THETA, N_Z, NOISE_SEED, R_BASE, generate_rock_environment


DEFAULT_MARGIN_RATIO = 0.05
DEFAULT_MIN_START_GOAL_RATIO = 0.30
DEFAULT_NORMAL_NEIGHBORS = 32


def flatten_rock_point_cloud(
    points: np.ndarray,
    r_base: float = R_BASE,
) -> Dict[str, np.ndarray]:
    """Flatten a half-cylinder rock-wall point cloud into the (u, v) space."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")

    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    theta = np.clip(theta, 0.0, np.pi)
    u = r_base * theta
    v = points[:, 2].copy()
    uv = np.column_stack([u, v])
    radius = np.linalg.norm(points[:, :2], axis=1)

    return {
        "points": points,
        "uv": uv,
        "u": u,
        "v": v,
        "theta": theta,
        "radius": radius,
        "u_bounds": (float(u.min()), float(u.max())),
        "v_bounds": (float(v.min()), float(v.max())),
    }


def estimate_surface_normal_from_point_cloud(
    points: np.ndarray,
    index: int,
    k_neighbors: int = DEFAULT_NORMAL_NEIGHBORS,
) -> np.ndarray:
    """Estimate one outward-facing surface normal by local PCA."""
    points = np.asarray(points, dtype=float)
    point = points[index]

    k_neighbors = int(max(3, min(k_neighbors, len(points))))
    distances = np.linalg.norm(points - point, axis=1)
    neighbor_indices = np.argpartition(distances, k_neighbors - 1)[:k_neighbors]
    neighbors = points[neighbor_indices]

    centered = neighbors - neighbors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(neighbors) - 1, 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    normal = normal / max(np.linalg.norm(normal), 1e-8)

    radial = np.array([point[0], point[1], 0.0], dtype=float)
    radial_norm = np.linalg.norm(radial)
    if radial_norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    radial = radial / radial_norm
    if np.dot(normal, radial) < 0.0:
        normal = -normal

    return normal


def build_task_point_state(
    points: np.ndarray,
    index: int,
    retreat_distance: float = 1.0,
    k_neighbors: int = DEFAULT_NORMAL_NEIGHBORS,
) -> Dict[str, object]:
    """Build one start/goal state from a sampled 3D point-cloud index."""
    surface_point = points[index]
    normal = estimate_surface_normal_from_point_cloud(
        points,
        index=index,
        k_neighbors=k_neighbors,
    )
    retreated_point = surface_point - normal * retreat_distance

    return {
        "index": int(index),
        "surface_point": surface_point.copy(),
        "normal": normal,
        "retreated_point": retreated_point,
    }


def _sample_valid_index_pair(
    points: np.ndarray,
    rng: np.random.Generator,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    min_point_distance_ratio: float = DEFAULT_MIN_START_GOAL_RATIO,
    max_trials: int = 256,
) -> Tuple[int, int, float]:
    """Sample two point indices with enough distance in 3D space."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    margin = (bbox_max - bbox_min) * margin_ratio
    lower = bbox_min + margin
    upper = bbox_max - margin

    valid_mask = np.all(points >= lower, axis=1) & np.all(points <= upper, axis=1)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 2:
        raise ValueError("Not enough valid points after applying the sampling margin.")

    span = bbox_max - bbox_min
    min_point_distance = float(np.linalg.norm(span) * min_point_distance_ratio)

    for _ in range(max_trials):
        start_index = int(rng.choice(valid_indices))
        goal_index = int(rng.choice(valid_indices))
        if start_index == goal_index:
            continue
        point_distance = np.linalg.norm(points[start_index] - points[goal_index])
        if point_distance >= min_point_distance:
            return start_index, goal_index, min_point_distance

    raise ValueError("Failed to sample a valid start/goal pair from the point cloud.")


def sample_planner_task(
    rock_points: np.ndarray,
    seed: Optional[int] = None,
    retreat_distance: float = 1.0,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    min_point_distance_ratio: float = DEFAULT_MIN_START_GOAL_RATIO,
    k_neighbors: int = DEFAULT_NORMAL_NEIGHBORS,
    r_base: float = R_BASE,
) -> Dict[str, object]:
    """Sample one planner task from a 3D rock-wall point cloud."""
    points = np.asarray(rock_points, dtype=float)
    rng = np.random.default_rng(seed)

    start_index, goal_index, min_point_distance = _sample_valid_index_pair(
        points,
        rng=rng,
        margin_ratio=margin_ratio,
        min_point_distance_ratio=min_point_distance_ratio,
    )

    start_state = build_task_point_state(
        points,
        index=start_index,
        retreat_distance=retreat_distance,
        k_neighbors=k_neighbors,
    )
    goal_state = build_task_point_state(
        points,
        index=goal_index,
        retreat_distance=retreat_distance,
        k_neighbors=k_neighbors,
    )

    return {
        "points": points,
        "start": start_state,
        "goal": goal_state,
        "retreat_distance": float(retreat_distance),
        "min_point_distance": float(min_point_distance),
    }


def sample_planner_task_from_environment(
    rock_env: Dict[str, object],
    seed: Optional[int] = None,
    retreat_distance: float = 1.0,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    min_point_distance_ratio: float = DEFAULT_MIN_START_GOAL_RATIO,
    k_neighbors: int = DEFAULT_NORMAL_NEIGHBORS,
) -> Dict[str, object]:
    """Sample one planner task from a rock environment dictionary."""
    if "points" not in rock_env:
        raise KeyError("rock_env must contain a 'points' field.")

    task = sample_planner_task(
        np.asarray(rock_env["points"], dtype=float),
        seed=seed,
        retreat_distance=retreat_distance,
        margin_ratio=margin_ratio,
        min_point_distance_ratio=min_point_distance_ratio,
        k_neighbors=k_neighbors,
        r_base=float(rock_env.get("R_BASE", R_BASE)),
    )
    task["environment_seed"] = rock_env.get("seed")
    return task


def main() -> None:
    """Minimal smoke demo for the planner task sampler."""
    cfg = load_config()
    env_cfg = cfg.get("env", {})
    planner_cfg = cfg.get("planner", {})

    rock_env = generate_rock_environment(
        n_theta=int(env_cfg.get("n_theta", N_THETA)),
        n_z=int(env_cfg.get("n_z", N_Z)),
        seed=int(env_cfg.get("seed", NOISE_SEED)),
    )
    task = sample_planner_task_from_environment(
        rock_env,
        seed=int(planner_cfg.get("seed", 0)),
        retreat_distance=float(planner_cfg.get("retreat_distance", 1.0)),
        margin_ratio=float(planner_cfg.get("margin_ratio", DEFAULT_MARGIN_RATIO)),
        min_point_distance_ratio=float(
            planner_cfg.get("min_start_goal_ratio", DEFAULT_MIN_START_GOAL_RATIO)
        ),
        k_neighbors=int(planner_cfg.get("normal_neighbors", DEFAULT_NORMAL_NEIGHBORS)),
    )

    print("Planner task sampler")
    print(f"Point count: {len(rock_env['points'])}")
    print(f"Retreat distance: {task['retreat_distance']:.2f} m")
    print(f"Minimum point distance: {task['min_point_distance']:.2f} m")
    print(f"Start surface point: {task['start']['surface_point']}")
    print(f"Goal surface point: {task['goal']['surface_point']}")
    print(f"Start retreated point: {task['start']['retreated_point']}")
    print(f"Goal retreated point: {task['goal']['retreated_point']}")


if __name__ == "__main__":
    main()
