from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from ...utils.resources import asset_path

DEFAULT_KINEMATICS_ASSET = "robot_4dof/kinematics.yaml"
LEGACY_KINEMATICS_PATHS = {
    "src/rock_3D/robot_4dof/kinematics.yaml",
    "./src/rock_3D/robot_4dof/kinematics.yaml",
}


def _as_vector3(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError("Expected a length-3 vector.")
    return array


def _rotation_x(angle: float) -> np.ndarray:
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float32,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@dataclass
class RobotKinematics:
    joint_order: tuple[str, ...]
    tunnel_axis_world: np.ndarray
    robot_position_world: np.ndarray
    robot_yaw_rad: float
    zero_q_rad: np.ndarray
    joint_limits_rad: np.ndarray
    joint_axes_local: Dict[str, np.ndarray]
    joint_axes_world_zero: Dict[str, np.ndarray]
    link_vectors_local: Dict[str, np.ndarray]
    workspace_samples: int
    workspace_seed: int

    def clip_configuration(self, q_rad: np.ndarray) -> np.ndarray:
        q_rad = np.asarray(q_rad, dtype=np.float32).reshape(len(self.joint_order))
        lower = self.joint_limits_rad[:, 0]
        upper = self.joint_limits_rad[:, 1]
        return np.clip(q_rad, lower, upper).astype(np.float32)

    def normalize_configuration(self, q_rad: np.ndarray) -> np.ndarray:
        q_rad = np.asarray(q_rad, dtype=np.float32).reshape(len(self.joint_order))
        lower = self.joint_limits_rad[:, 0]
        upper = self.joint_limits_rad[:, 1]
        normalized = (q_rad - lower) / np.maximum(upper - lower, 1e-6)
        return (2.0 * normalized - 1.0).astype(np.float32)

    def sample_random_configuration(self, rng: np.random.Generator) -> np.ndarray:
        lower = self.joint_limits_rad[:, 0]
        upper = self.joint_limits_rad[:, 1]
        return rng.uniform(lower, upper).astype(np.float32)

    def forward_kinematics(self, q_rad: np.ndarray) -> Dict[str, Dict[str, np.ndarray] | np.ndarray]:
        q_rad = np.asarray(q_rad, dtype=np.float32).reshape(len(self.joint_order))
        q_yaw, q_shoulder, q_elbow, q_wrist = [float(v) for v in q_rad]

        base_rotation = _rotation_z(self.robot_yaw_rad)
        yaw_rotation = base_rotation @ _rotation_z(q_yaw)
        shoulder_rotation = yaw_rotation @ _rotation_x(-q_shoulder)
        elbow_rotation = shoulder_rotation @ _rotation_x(-q_elbow)
        wrist_rotation = elbow_rotation @ _rotation_x(-q_wrist)

        p_yaw = self.robot_position_world.astype(np.float32)
        p_shoulder = p_yaw + yaw_rotation @ self.link_vectors_local["yaw_to_shoulder"]
        p_elbow = p_shoulder + shoulder_rotation @ self.link_vectors_local["shoulder_to_elbow"]
        p_wrist = p_elbow + elbow_rotation @ self.link_vectors_local["elbow_to_wrist"]
        p_tool_tip = p_wrist + wrist_rotation @ self.link_vectors_local["wrist_to_tool_tip"]

        local_pitch_axis = self.joint_axes_local["shoulder_pitch"]
        return {
            "joint_positions": {
                "yaw": p_yaw.astype(np.float32),
                "shoulder": p_shoulder.astype(np.float32),
                "elbow": p_elbow.astype(np.float32),
                "wrist": p_wrist.astype(np.float32),
            },
            "joint_axes_world": {
                "turret_yaw": (base_rotation @ self.joint_axes_local["turret_yaw"]).astype(
                    np.float32
                ),
                "shoulder_pitch": (yaw_rotation @ local_pitch_axis).astype(np.float32),
                "elbow_pitch": (shoulder_rotation @ local_pitch_axis).astype(np.float32),
                "wrist_pitch": (elbow_rotation @ local_pitch_axis).astype(np.float32),
            },
            "tool_tip": p_tool_tip.astype(np.float32),
        }

    def estimate_workspace_bounds(self, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.workspace_seed)
        points = [self.forward_kinematics(self.zero_q_rad)["tool_tip"]]

        lower = self.joint_limits_rad[:, 0]
        upper = self.joint_limits_rad[:, 1]
        for flags in itertools.product([0, 1], repeat=len(self.joint_order)):
            corner = np.where(np.asarray(flags, dtype=bool), upper, lower).astype(np.float32)
            points.append(self.forward_kinematics(corner)["tool_tip"])

        for _ in range(max(self.workspace_samples, 0)):
            sampled_q = self.sample_random_configuration(rng)
            points.append(self.forward_kinematics(sampled_q)["tool_tip"])

        points_array = np.asarray(points, dtype=np.float32)
        padding_array = np.full(3, float(padding), dtype=np.float32)
        return (
            points_array.min(axis=0) - padding_array,
            points_array.max(axis=0) + padding_array,
        )


def _uses_packaged_kinematics(path: str | Path | None) -> bool:
    if path is None:
        return True

    config_path = Path(path)
    if config_path.exists():
        return False

    normalized = config_path.as_posix()
    return normalized in LEGACY_KINEMATICS_PATHS or config_path.name == "kinematics.yaml"


def load_robot_kinematics(path: str | Path | None = None) -> RobotKinematics:
    if _uses_packaged_kinematics(path):
        with asset_path(DEFAULT_KINEMATICS_ASSET) as packaged_path:
            return load_robot_kinematics(packaged_path)

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    joint_order = tuple(str(name) for name in data["joint_order"])
    zero_q_deg = data["zero_configuration_deg"]
    joint_limits_deg = data["joint_limits_deg"]

    zero_q_rad = np.array(
        [np.deg2rad(float(zero_q_deg[name])) for name in joint_order],
        dtype=np.float32,
    )
    joint_limits_rad = np.array(
        [
            [
                np.deg2rad(float(joint_limits_deg[name][0])),
                np.deg2rad(float(joint_limits_deg[name][1])),
            ]
            for name in joint_order
        ],
        dtype=np.float32,
    )

    joint_axes_local = {
        name: _as_vector3(axis)
        for name, axis in data["joint_axes_local"].items()
    }
    joint_axes_world_zero = {
        name: _as_vector3(axis)
        for name, axis in data["joint_axes_world_zero"].items()
    }
    link_vectors_local = {
        name: _as_vector3(vector)
        for name, vector in data["link_vectors_local"].items()
    }

    workspace_cfg = data.get("workspace_estimation", {})
    default_scene = data["default_scene"]
    frames = data["frames"]

    return RobotKinematics(
        joint_order=joint_order,
        tunnel_axis_world=_as_vector3(frames["tunnel_axis_world"]),
        robot_position_world=_as_vector3(default_scene["robot_position_world"]),
        robot_yaw_rad=float(np.deg2rad(float(default_scene["robot_yaw_deg"]))),
        zero_q_rad=zero_q_rad,
        joint_limits_rad=joint_limits_rad,
        joint_axes_local=joint_axes_local,
        joint_axes_world_zero=joint_axes_world_zero,
        link_vectors_local=link_vectors_local,
        workspace_samples=int(workspace_cfg.get("num_samples", 4096)),
        workspace_seed=int(workspace_cfg.get("seed", 0)),
    )
