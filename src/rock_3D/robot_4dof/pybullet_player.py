from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def require_pybullet():
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 pybullet。请先安装后再运行，例如：\n"
            "  python -m pip install pybullet\n"
        ) from exc
    return p, pybullet_data


def load_tunnel_metadata(root: Path) -> dict | None:
    metadata_path = root.parent / "tunnel_environment" / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class PyBulletRobotPlayer:
    """Simple PyBullet scene player driven by externally provided joint angles."""

    def __init__(
        self,
        *,
        dt: float = 1.0 / 240.0,
        headless: bool = False,
        show_tunnel: bool = True,
        show_plane: bool = False,
        robot_position: Sequence[float] = (0.0, 0.0, 0.45),
        robot_yaw_deg: float = 90.0,
        tunnel_position: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.p, self.pybullet_data = require_pybullet()
        self.dt = float(dt)
        self.headless = bool(headless)
        self.robot_position = [float(v) for v in robot_position]
        self.robot_yaw_deg = float(robot_yaw_deg)
        self.tunnel_position = [float(v) for v in tunnel_position]

        root = Path(__file__).resolve().parent
        urdf_path = root / "shipen_4dof.urdf"
        if not urdf_path.exists():
            raise RuntimeError(
                f"未找到 URDF: {urdf_path}\n请先运行 Blender 资产构建脚本。"
            )
        tunnel_root = root.parent / "tunnel_environment"
        tunnel_urdf_path = tunnel_root / "tunnel_environment.urdf"
        self.tunnel_metadata = load_tunnel_metadata(root)

        client_mode = self.p.DIRECT if self.headless else self.p.GUI
        self.client_id = self.p.connect(client_mode)
        if self.client_id < 0:
            raise RuntimeError("PyBullet 启动失败。")

        if not self.headless:
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        self.p.setAdditionalSearchPath(self.pybullet_data.getDataPath())
        self.p.setGravity(0, 0, 0)
        self.p.setTimeStep(self.dt)

        if show_plane:
            plane_id = self.p.loadURDF("plane.urdf", useFixedBase=True)
            self.p.changeVisualShape(plane_id, -1, rgbaColor=[0.92, 0.92, 0.92, 1.0])

        if show_tunnel:
            if not tunnel_urdf_path.exists():
                raise RuntimeError(
                    f"未找到隧道 URDF: {tunnel_urdf_path}\n"
                    "请先运行 python tools/build_tunnel_environment.py。"
                )
            self.p.loadURDF(
                str(tunnel_urdf_path),
                basePosition=self.tunnel_position,
                baseOrientation=self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                useFixedBase=True,
            )

        self.robot_id = self.p.loadURDF(
            str(urdf_path),
            basePosition=self.robot_position,
            baseOrientation=self.p.getQuaternionFromEuler(
                [0.0, 0.0, math.radians(self.robot_yaw_deg)]
            ),
            useFixedBase=True,
        )

        if not self.headless:
            self._reset_camera()

        self.joint_name_to_index: dict[str, int] = {}
        for joint_index in range(self.p.getNumJoints(self.robot_id)):
            info = self.p.getJointInfo(self.robot_id, joint_index)
            joint_name = info[1].decode("utf-8")
            self.joint_name_to_index[joint_name] = joint_index

        self._sync_once()

    def _reset_camera(self) -> None:
        if self.tunnel_metadata:
            bbox_min = self.tunnel_metadata["mesh"]["bbox_min"]
            bbox_max = self.tunnel_metadata["mesh"]["bbox_max"]
            center = [
                (bbox_min[i] + bbox_max[i]) * 0.5 + self.tunnel_position[i]
                for i in range(3)
            ]
            extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
            max_extent = max(extent)
            target = [
                (center[0] * 0.7) + (self.robot_position[0] * 0.3),
                (center[1] * 0.7) + (self.robot_position[1] * 0.3),
                max(center[2] * 0.55, self.robot_position[2] + 0.8),
            ]
            self.p.resetDebugVisualizerCamera(
                cameraDistance=max_extent * 1.35,
                cameraYaw=38.0,
                cameraPitch=-18.0,
                cameraTargetPosition=target,
            )
        else:
            self.p.resetDebugVisualizerCamera(
                cameraDistance=3.2,
                cameraYaw=130.0,
                cameraPitch=-25.0,
                cameraTargetPosition=[0.6, -0.8, 0.3],
            )

    def _sync_once(self) -> None:
        if self.p.isConnected():
            self.p.stepSimulation()

    def set_joint_positions_deg(self, q_deg: Sequence[float]) -> None:
        joint_names = (
            "turret_yaw",
            "shoulder_pitch",
            "elbow_pitch",
            "wrist_pitch",
        )
        for joint_name, target_deg in zip(joint_names, q_deg):
            joint_index = self.joint_name_to_index[joint_name]
            self.p.resetJointState(
                self.robot_id,
                joint_index,
                targetValue=math.radians(float(target_deg)),
                targetVelocity=0.0,
            )
        self._sync_once()

    def reset_episode(self, start_q_deg: Sequence[float]) -> None:
        self.set_joint_positions_deg(start_q_deg)

    def update(self, current_q_deg: Sequence[float]) -> None:
        self.set_joint_positions_deg(current_q_deg)

    def idle(self) -> None:
        self._sync_once()

    def close(self) -> None:
        if self.p.isConnected():
            self.p.disconnect()
