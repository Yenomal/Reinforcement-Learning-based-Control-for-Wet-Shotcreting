from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the 4-DoF shipen arm in PyBullet.",
    )
    parser.add_argument(
        "--angles",
        nargs=4,
        type=float,
        metavar=("YAW", "SHOULDER", "ELBOW", "WRIST"),
        default=(0.0, 0.0, 0.0, 0.0),
        help="Initial joint targets in degrees.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 240.0,
        help="Simulation step size in seconds.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in DIRECT mode for quick validation without opening the GUI.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Number of simulation steps to run in headless mode.",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Do not load the tunnel environment.",
    )
    parser.add_argument(
        "--show-plane",
        action="store_true",
        help="Load a plane for reference.",
    )
    parser.add_argument(
        "--robot-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.45),
        help="Robot base position in the combined scene.",
    )
    parser.add_argument(
        "--robot-yaw-deg",
        type=float,
        default=90.0,
        help="Whole-robot yaw rotation in degrees around +Z.",
    )
    parser.add_argument(
        "--tunnel-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Tunnel base position in the combined scene.",
    )
    return parser


def require_pybullet():
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 pybullet。请先安装后再运行，例如：\n"
            "  python -m pip install pybullet\n"
        ) from exc
    return p, pybullet_data


def load_tunnel_metadata(root: Path) -> dict | None:
    metadata_path = root.parent / "tunnel_environment" / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    p, pybullet_data = require_pybullet()

    root = Path(__file__).resolve().parent
    urdf_path = root / "shipen_4dof.urdf"
    if not urdf_path.exists():
        raise SystemExit(
            f"未找到 URDF: {urdf_path}\n"
            "请先运行 Blender 资产构建脚本。"
        )
    tunnel_root = root.parent / "tunnel_environment"
    tunnel_urdf_path = tunnel_root / "tunnel_environment.urdf"
    tunnel_metadata = load_tunnel_metadata(root)

    client_mode = p.DIRECT if args.headless else p.GUI
    client = p.connect(client_mode)
    if client < 0:
        raise SystemExit("PyBullet 启动失败。")

    if not args.headless:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setTimeStep(args.dt)

    if args.show_plane:
        plane_id = p.loadURDF("plane.urdf", useFixedBase=True)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.92, 0.92, 0.92, 1.0])

    tunnel_id = None
    if not args.no_tunnel:
        if not tunnel_urdf_path.exists():
            raise SystemExit(
                f"未找到隧道 URDF: {tunnel_urdf_path}\n"
                "请先运行 python tools/build_tunnel_environment.py。"
            )
        tunnel_id = p.loadURDF(
            str(tunnel_urdf_path),
            basePosition=list(args.tunnel_position),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
        )

    robot_id = p.loadURDF(
        str(urdf_path),
        basePosition=list(args.robot_position),
        baseOrientation=p.getQuaternionFromEuler(
            [0.0, 0.0, math.radians(args.robot_yaw_deg)]
        ),
        useFixedBase=True,
    )

    if not args.headless:
        if tunnel_metadata:
            bbox_min = tunnel_metadata["mesh"]["bbox_min"]
            bbox_max = tunnel_metadata["mesh"]["bbox_max"]
            center = [
                (bbox_min[i] + bbox_max[i]) * 0.5 + args.tunnel_position[i]
                for i in range(3)
            ]
            extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
            max_extent = max(extent)
            target = [
                (center[0] * 0.7) + (args.robot_position[0] * 0.3),
                (center[1] * 0.7) + (args.robot_position[1] * 0.3),
                max(center[2] * 0.55, args.robot_position[2] + 0.8),
            ]
            p.resetDebugVisualizerCamera(
                cameraDistance=max_extent * 1.35,
                cameraYaw=38.0,
                cameraPitch=-18.0,
                cameraTargetPosition=target,
            )
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=3.2,
                cameraYaw=130.0,
                cameraPitch=-25.0,
                cameraTargetPosition=[0.6, -0.8, 0.3],
            )

    joint_order = []
    joint_limits_deg = []
    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        joint_name = info[1].decode("utf-8")
        lower = math.degrees(info[8])
        upper = math.degrees(info[9])
        joint_order.append((joint_name, joint_index))
        joint_limits_deg.append((lower, upper))
        p.changeDynamics(robot_id, joint_index, linearDamping=0.0, angularDamping=0.0)

    clamped_targets_deg = []
    sliders = []
    for (joint_name, _), (lower, upper), initial in zip(joint_order, joint_limits_deg, args.angles):
        start = min(max(initial, lower), upper)
        clamped_targets_deg.append(start)
        if not args.headless:
            slider_id = p.addUserDebugParameter(joint_name, lower, upper, start)
            sliders.append(slider_id)

    if tunnel_id is not None and tunnel_metadata is not None:
        print(
            "Loaded tunnel:",
            f"body_id={tunnel_id}",
            f"bbox_min={tunnel_metadata['mesh']['bbox_min']}",
            f"bbox_max={tunnel_metadata['mesh']['bbox_max']}",
        )

    print("Loaded joints:")
    for (joint_name, joint_index), (lower, upper) in zip(joint_order, joint_limits_deg):
        print(f"  {joint_index}: {joint_name} [{lower:.1f}, {upper:.1f}] deg")

    try:
        remaining_steps = args.steps
        while p.isConnected():
            for index, (joint_name, joint_index) in enumerate(joint_order):
                target_deg = clamped_targets_deg[index]
                if not args.headless:
                    target_deg = p.readUserDebugParameter(sliders[index])
                target_rad = math.radians(target_deg)
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_rad,
                    force=200.0,
                    positionGain=0.25,
                    velocityGain=1.0,
                )
            p.stepSimulation()
            if args.headless:
                remaining_steps -= 1
                if remaining_steps <= 0:
                    break
            else:
                time.sleep(args.dt)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
