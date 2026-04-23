#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute and cache wall-goal reachability maps for planner sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from ..config import load_config
from ..rock_3D.robot_4dof.kinematics import RobotKinematics, load_robot_kinematics
from ..rock_3D.robot_4dof.torch_kinematics import TorchRobotKinematics
from ..rock_env.rock_wall import (
    L_TUNNEL,
    R_BASE,
    build_training_rock_environment,
    surface_normal_from_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_PATH = "outputs/reachability/reachability_map.npz"
DEFAULT_HTML_PATH = "outputs/reachability/reachability_map.html"
DEFAULT_SAMPLE_COUNT = 32768
DEFAULT_GOAL_CHUNK_SIZE = 512
MAP_VERSION = 1


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _as_float_array(value: Any, length: int) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(length, float(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(length).astype(np.float32)


def _math_points_to_world(points_math: np.ndarray, axial_scale: float) -> np.ndarray:
    points_math = np.asarray(points_math, dtype=np.float32)
    return np.stack(
        [
            points_math[..., 2] * float(axial_scale),
            points_math[..., 0],
            points_math[..., 1],
        ],
        axis=-1,
    ).astype(np.float32)


def _math_vectors_to_world(vectors_math: np.ndarray, axial_scale: float) -> np.ndarray:
    vectors_math = np.asarray(vectors_math, dtype=np.float32)
    return np.stack(
        [
            vectors_math[..., 2] * float(axial_scale),
            vectors_math[..., 0],
            vectors_math[..., 1],
        ],
        axis=-1,
    ).astype(np.float32)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def _compute_distance_tolerance(
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
) -> float:
    planner_tol = float(
        planner_cfg.get(
            "reachability_tolerance",
            rl_cfg.get("goal_tolerance", 0.2),
        )
    )
    goal_tol = float(rl_cfg.get("goal_tolerance", planner_tol))
    return float(min(planner_tol, goal_tol))


def _build_signature(
    env_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    kinematics: RobotKinematics,
) -> Dict[str, Any]:
    kinematics_path = _resolve_project_path(str(robot_cfg.get("kinematics_path", "")))
    kinematics_stat = kinematics_path.stat()

    html_path_raw = str(env_cfg.get("train_rock_env_html", "")).strip()
    html_signature: Dict[str, Any] | None = None
    if html_path_raw:
        html_path = _resolve_project_path(html_path_raw)
        html_stat = html_path.stat()
        html_signature = {
            "path": str(html_path.resolve()),
            "mtime_ns": int(html_stat.st_mtime_ns),
            "size": int(html_stat.st_size),
        }

    max_joint_delta_deg = _as_float_array(
        rl_cfg.get("max_joint_delta_deg", 4.0),
        len(kinematics.joint_order),
    )
    start_q_deg = _as_float_array(
        rl_cfg.get("initial_configuration_deg", np.zeros(len(kinematics.joint_order))),
        len(kinematics.joint_order),
    )

    return {
        "map_version": MAP_VERSION,
        "env_seed": int(env_cfg.get("seed", 42)),
        "n_theta": int(env_cfg.get("n_theta", 200)),
        "n_z": int(env_cfg.get("n_z", 100)),
        "train_rock_env_html": html_signature,
        "spray_angle_range_deg": [
            float(value) for value in planner_cfg.get("spray_angle_range_deg", (-60.0, 60.0))
        ],
        "spray_standoff_distance": float(planner_cfg.get("spray_standoff_distance", 0.5)),
        "axial_margin_ratio": float(planner_cfg.get("axial_margin_ratio", 0.05)),
        "tunnel_axial_scale": float(planner_cfg.get("tunnel_axial_scale", 1.5)),
        "distance_tolerance": _compute_distance_tolerance(planner_cfg, rl_cfg),
        "reachability_map_samples": int(
            planner_cfg.get("reachability_map_samples", DEFAULT_SAMPLE_COUNT)
        ),
        "max_episode_steps": int(rl_cfg.get("max_episode_steps", 200)),
        "max_joint_delta_deg": max_joint_delta_deg.tolist(),
        "initial_configuration_deg": start_q_deg.tolist(),
        "kinematics_path": str(kinematics_path.resolve()),
        "kinematics_mtime_ns": int(kinematics_stat.st_mtime_ns),
        "kinematics_size": int(kinematics_stat.st_size),
    }


def _map_cache_paths(planner_cfg: Dict[str, Any]) -> tuple[Path, Path | None]:
    npz_path_raw = str(planner_cfg.get("reachability_map_path", DEFAULT_MAP_PATH)).strip()
    html_path_raw = str(planner_cfg.get("reachability_map_html", DEFAULT_HTML_PATH)).strip()
    npz_path = _resolve_project_path(npz_path_raw)
    html_path = _resolve_project_path(html_path_raw) if html_path_raw else None
    return npz_path, html_path


def _compute_normals_grid(
    rock_env: Dict[str, Any],
    u_grid: np.ndarray,
    v_grid: np.ndarray,
) -> np.ndarray:
    rows, cols = u_grid.shape
    normals = np.zeros((rows, cols, 3), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            normals[row, col] = surface_normal_from_environment(
                rock_env,
                float(u_grid[row, col]),
                float(v_grid[row, col]),
            )
    return normals


def _sample_joint_configurations(
    kinematics: RobotKinematics,
    sample_count: int,
    seed: int,
    start_q_rad: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled_q = np.stack(
        [kinematics.sample_random_configuration(rng) for _ in range(sample_count)],
        axis=0,
    ).astype(np.float32)

    lower = kinematics.joint_limits_rad[:, 0]
    upper = kinematics.joint_limits_rad[:, 1]
    corners = []
    for flags in range(1 << len(kinematics.joint_order)):
        bits = np.array(
            [(flags >> bit) & 1 for bit in range(len(kinematics.joint_order))],
            dtype=bool,
        )
        corners.append(np.where(bits, upper, lower).astype(np.float32))

    enriched = np.concatenate(
        [
            sampled_q,
            np.asarray(corners, dtype=np.float32),
            kinematics.zero_q_rad.reshape(1, -1).astype(np.float32),
            start_q_rad.reshape(1, -1).astype(np.float32),
        ],
        axis=0,
    )
    return enriched


def _forward_kinematics_batch(
    torch_kinematics: TorchRobotKinematics,
    q_samples: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, q_samples.shape[0], batch_size):
        stop = min(start + batch_size, q_samples.shape[0])
        q_chunk = torch.as_tensor(q_samples[start:stop], dtype=torch.float32, device=device)
        fk_chunk = torch_kinematics.forward_kinematics(q_chunk)
        outputs.append(fk_chunk.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _nearest_joint_samples(
    goal_points: np.ndarray,
    q_samples: np.ndarray,
    fk_points: np.ndarray,
    *,
    device: torch.device,
    goal_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    goal_tensor = torch.as_tensor(goal_points, dtype=torch.float32, device=device)
    fk_tensor = torch.as_tensor(fk_points, dtype=torch.float32, device=device)

    best_distances = np.full(goal_points.shape[0], np.inf, dtype=np.float32)
    best_indices = np.full(goal_points.shape[0], -1, dtype=np.int64)

    for start in range(0, goal_points.shape[0], goal_chunk_size):
        stop = min(start + goal_chunk_size, goal_points.shape[0])
        distance_matrix = torch.cdist(goal_tensor[start:stop], fk_tensor)
        chunk_distances, chunk_indices = torch.min(distance_matrix, dim=1)
        best_distances[start:stop] = chunk_distances.detach().cpu().numpy().astype(np.float32)
        best_indices[start:stop] = chunk_indices.detach().cpu().numpy().astype(np.int64)

    best_q = q_samples[best_indices].astype(np.float32)
    return best_distances, best_q


def _save_reachability_html(map_data: Dict[str, Any], html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    spray_angle_grid = np.asarray(map_data["spray_angle_grid"], dtype=np.float32)
    v_grid = np.asarray(map_data["surface_v_grid"], dtype=np.float32)
    allowed_mask = np.asarray(map_data["allowed_mask"], dtype=bool)
    reachable_mask = np.asarray(map_data["reachable_mask"], dtype=bool)
    best_distance = np.asarray(map_data["best_distance"], dtype=np.float32)
    step_lower_bound = np.asarray(map_data["step_lower_bound"], dtype=np.float32)

    distance_plot = np.where(allowed_mask, best_distance, np.nan).T
    steps_plot = np.where(allowed_mask, step_lower_bound, np.nan).T
    reachable_plot = np.where(allowed_mask, reachable_mask.astype(np.float32), np.nan).T

    x_values = np.asarray(v_grid[:, 0], dtype=np.float32)
    y_values = np.asarray(spray_angle_grid[0], dtype=np.float32)
    reachable_ratio = float(reachable_mask.sum() / max(allowed_mask.sum(), 1))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Best Distance", "Step Lower Bound", "Reachable Mask"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=distance_plot,
            colorscale="Viridis",
            colorbar=dict(title="m", x=0.29),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=steps_plot,
            colorscale="Cividis",
            colorbar=dict(title="steps", x=0.64),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=reachable_plot,
            colorscale=[[0.0, "#c0392b"], [1.0, "#27ae60"]],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="mask", x=0.99),
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Axial Position v", row=1, col=1)
    fig.update_xaxes(title_text="Axial Position v", row=1, col=2)
    fig.update_xaxes(title_text="Axial Position v", row=1, col=3)
    fig.update_yaxes(title_text="Spray Angle (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Spray Angle (deg)", row=1, col=2)
    fig.update_yaxes(title_text="Spray Angle (deg)", row=1, col=3)
    fig.update_layout(
        title=(
            "Planner Reachability Map"
            f" | reachable_ratio={reachable_ratio:.3f}"
            f" | standoff={map_data['metadata']['signature']['spray_standoff_distance']:.2f}m"
        ),
        width=1600,
        height=520,
        margin=dict(l=60, r=60, t=70, b=50),
    )
    fig.write_html(str(html_path), include_plotlyjs=True)


def _materialize_reachability_map(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    map_data = dict(raw_data)
    reachable_mask = np.asarray(map_data["reachable_mask"], dtype=bool)
    map_data["reachable_mask"] = reachable_mask
    map_data["allowed_mask"] = np.asarray(map_data["allowed_mask"], dtype=bool)
    map_data["reachable_indices"] = np.argwhere(reachable_mask).astype(np.int64)
    return map_data


def compute_reachability_map(
    *,
    rock_env: Dict[str, Any],
    kinematics: RobotKinematics,
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    rows, cols = np.asarray(rock_env["u_grid"]).shape
    u_grid = np.asarray(rock_env["u_grid"], dtype=np.float32)
    v_grid = np.asarray(rock_env["v_grid"], dtype=np.float32)
    spray_angle_grid = np.rad2deg((u_grid / R_BASE) - (np.pi * 0.5)).astype(np.float32)

    axial_margin_ratio = float(planner_cfg.get("axial_margin_ratio", 0.05))
    angle_min_deg, angle_max_deg = [
        float(value) for value in planner_cfg.get("spray_angle_range_deg", (-60.0, 60.0))
    ]
    min_v = float(L_TUNNEL * axial_margin_ratio)
    max_v = float(max(L_TUNNEL - min_v, min_v))
    allowed_mask = (
        (spray_angle_grid >= angle_min_deg)
        & (spray_angle_grid <= angle_max_deg)
        & (v_grid >= min_v)
        & (v_grid <= max_v)
    )

    surface_points_math = np.asarray(rock_env["points_grid"], dtype=np.float32)
    surface_normals_math = _compute_normals_grid(rock_env, u_grid=u_grid, v_grid=v_grid)
    axial_scale = float(planner_cfg.get("tunnel_axial_scale", 1.5))
    surface_points_world = _math_points_to_world(surface_points_math, axial_scale=axial_scale)
    surface_normals_world = _normalize_vectors(
        _math_vectors_to_world(surface_normals_math, axial_scale=axial_scale)
    )
    spray_standoff_distance = float(planner_cfg.get("spray_standoff_distance", 0.5))
    goal_points_world = (
        surface_points_world - surface_normals_world * spray_standoff_distance
    ).astype(np.float32)

    start_q_rad = np.deg2rad(
        _as_float_array(
            rl_cfg.get("initial_configuration_deg", kinematics.zero_q_rad),
            len(kinematics.joint_order),
        )
    ).astype(np.float32)
    max_joint_delta = np.deg2rad(
        _as_float_array(
            rl_cfg.get("max_joint_delta_deg", 4.0),
            len(kinematics.joint_order),
        )
    ).astype(np.float32)
    start_point = kinematics.forward_kinematics(start_q_rad)["tool_tip"].astype(np.float32)

    sample_count = int(planner_cfg.get("reachability_map_samples", DEFAULT_SAMPLE_COUNT))
    sample_seed = int(planner_cfg.get("seed", 42))
    q_samples = _sample_joint_configurations(
        kinematics=kinematics,
        sample_count=sample_count,
        seed=sample_seed,
        start_q_rad=start_q_rad,
    )
    torch_kinematics = TorchRobotKinematics.from_robot_kinematics(
        kinematics,
        device=device,
    )
    fk_points = _forward_kinematics_batch(
        torch_kinematics,
        q_samples=q_samples,
        device=device,
    )

    allowed_indices = np.argwhere(allowed_mask)
    candidate_goal_points = goal_points_world[allowed_mask].astype(np.float32)
    goal_chunk_size = int(planner_cfg.get("reachability_map_goal_chunk_size", DEFAULT_GOAL_CHUNK_SIZE))
    best_distances_allowed, best_q_allowed = _nearest_joint_samples(
        candidate_goal_points,
        q_samples=q_samples,
        fk_points=fk_points,
        device=device,
        goal_chunk_size=goal_chunk_size,
    )
    step_lower_bound_allowed = np.max(
        np.abs(best_q_allowed - start_q_rad.reshape(1, -1)) / np.maximum(max_joint_delta, 1e-8),
        axis=1,
    ).astype(np.float32)
    distance_tolerance = _compute_distance_tolerance(planner_cfg, rl_cfg)
    max_episode_steps = float(rl_cfg.get("max_episode_steps", 200))
    reachable_allowed = (
        (best_distances_allowed <= distance_tolerance)
        & (step_lower_bound_allowed <= max_episode_steps)
    )

    best_distance_grid = np.full((rows, cols), np.inf, dtype=np.float32)
    step_lower_bound_grid = np.full((rows, cols), np.inf, dtype=np.float32)
    best_q_grid = np.full((rows, cols, len(kinematics.joint_order)), np.nan, dtype=np.float32)
    reachable_mask = np.zeros((rows, cols), dtype=bool)

    for index, (row, col) in enumerate(allowed_indices):
        best_distance_grid[row, col] = best_distances_allowed[index]
        step_lower_bound_grid[row, col] = step_lower_bound_allowed[index]
        best_q_grid[row, col] = best_q_allowed[index]
        reachable_mask[row, col] = bool(reachable_allowed[index])

    return _materialize_reachability_map(
        {
            "metadata": {},
            "allowed_mask": allowed_mask.astype(bool),
            "reachable_mask": reachable_mask.astype(bool),
            "best_distance": best_distance_grid.astype(np.float32),
            "step_lower_bound": step_lower_bound_grid.astype(np.float32),
            "best_q": best_q_grid.astype(np.float32),
            "surface_point_grid": surface_points_world.astype(np.float32),
            "surface_normal_grid": surface_normals_world.astype(np.float32),
            "goal_point_grid": goal_points_world.astype(np.float32),
            "surface_u_grid": u_grid.astype(np.float32),
            "surface_v_grid": v_grid.astype(np.float32),
            "spray_angle_grid": spray_angle_grid.astype(np.float32),
            "start_point": start_point.astype(np.float32),
        }
    )


def _save_map_npz(map_path: Path, map_data: Dict[str, Any]) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        map_path,
        metadata_json=json.dumps(map_data["metadata"], sort_keys=True, ensure_ascii=True),
        allowed_mask=map_data["allowed_mask"].astype(np.uint8),
        reachable_mask=map_data["reachable_mask"].astype(np.uint8),
        best_distance=map_data["best_distance"].astype(np.float32),
        step_lower_bound=map_data["step_lower_bound"].astype(np.float32),
        best_q=map_data["best_q"].astype(np.float32),
        surface_point_grid=map_data["surface_point_grid"].astype(np.float32),
        surface_normal_grid=map_data["surface_normal_grid"].astype(np.float32),
        goal_point_grid=map_data["goal_point_grid"].astype(np.float32),
        surface_u_grid=map_data["surface_u_grid"].astype(np.float32),
        surface_v_grid=map_data["surface_v_grid"].astype(np.float32),
        spray_angle_grid=map_data["spray_angle_grid"].astype(np.float32),
        start_point=map_data["start_point"].astype(np.float32),
    )


def _load_map_npz(map_path: Path) -> Dict[str, Any]:
    with np.load(map_path, allow_pickle=False) as payload:
        return _materialize_reachability_map(
            {
                "metadata": json.loads(str(payload["metadata_json"].item())),
                "allowed_mask": payload["allowed_mask"].astype(bool),
                "reachable_mask": payload["reachable_mask"].astype(bool),
                "best_distance": payload["best_distance"].astype(np.float32),
                "step_lower_bound": payload["step_lower_bound"].astype(np.float32),
                "best_q": payload["best_q"].astype(np.float32),
                "surface_point_grid": payload["surface_point_grid"].astype(np.float32),
                "surface_normal_grid": payload["surface_normal_grid"].astype(np.float32),
                "goal_point_grid": payload["goal_point_grid"].astype(np.float32),
                "surface_u_grid": payload["surface_u_grid"].astype(np.float32),
                "surface_v_grid": payload["surface_v_grid"].astype(np.float32),
                "spray_angle_grid": payload["spray_angle_grid"].astype(np.float32),
                "start_point": payload["start_point"].astype(np.float32),
            }
        )


def build_or_load_reachability_map(
    *,
    rock_env: Dict[str, Any],
    kinematics: RobotKinematics,
    env_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any] | None:
    if not bool(planner_cfg.get("use_reachability_map", False)):
        return None

    map_path, html_path = _map_cache_paths(planner_cfg)
    signature = _build_signature(
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        kinematics=kinematics,
    )
    force_rebuild = bool(planner_cfg.get("reachability_map_force_rebuild", False))

    if map_path.exists() and not force_rebuild:
        loaded_map = _load_map_npz(map_path)
        if loaded_map["metadata"].get("signature") == signature:
            return loaded_map

    map_data = compute_reachability_map(
        rock_env=rock_env,
        kinematics=kinematics,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        device=device,
    )
    map_data["metadata"] = {
        "signature": signature,
        "summary": {
            "reachable_ratio": float(
                map_data["reachable_mask"].sum() / max(map_data["allowed_mask"].sum(), 1)
            )
        },
    }
    _save_map_npz(map_path, map_data)
    if html_path is not None:
        _save_reachability_html(map_data, html_path)
    return map_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the planner reachability map.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional compute device override, such as cpu or cuda.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding the cached reachability map.",
    )
    return parser.parse_args()


def _resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        normalized = requested_device.lower()
        if normalized == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_cfg = dict(config.get("env", {}))
    planner_cfg = dict(config.get("planner", {}))
    rl_cfg = dict(config.get("rl", {}))
    robot_cfg = dict(config.get("robot", {}))
    planner_cfg["use_reachability_map"] = True
    planner_cfg["reachability_map_force_rebuild"] = bool(args.force)

    rock_env = build_training_rock_environment(env_cfg)
    kinematics = load_robot_kinematics(robot_cfg.get("kinematics_path"))
    device = _resolve_device(args.device)
    reachability_map = build_or_load_reachability_map(
        rock_env=rock_env,
        kinematics=kinematics,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        rl_cfg=rl_cfg,
        robot_cfg=robot_cfg,
        device=device,
    )
    assert reachability_map is not None

    map_path, html_path = _map_cache_paths(planner_cfg)
    print("Reachability map ready")
    print(f"  Device: {device}")
    print(f"  Cache: {map_path}")
    if html_path is not None:
        print(f"  HTML: {html_path}")
    print(
        "  Reachable ratio: "
        f"{reachability_map['metadata']['summary']['reachable_ratio']:.3f}"
    )
    print(
        "  Reachable cells: "
        f"{int(reachability_map['reachable_mask'].sum())}"
    )


if __name__ == "__main__":
    main()
