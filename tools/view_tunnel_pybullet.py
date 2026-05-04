from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time

from rl_robot.utils.resources import asset_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the packaged tunnel environment in PyBullet.",
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
        help="Number of simulation steps in headless mode.",
    )
    parser.add_argument(
        "--show-plane",
        action="store_true",
        help="Load a plane for reference.",
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


def main() -> None:
    args = build_arg_parser().parse_args()
    p, pybullet_data = require_pybullet()

    with ExitStack() as stack:
        metadata_path = stack.enter_context(asset_path("tunnel_environment/metadata.json"))
        urdf_path = stack.enter_context(asset_path("tunnel_environment/tunnel_environment.urdf"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        bbox_min = metadata["mesh"]["bbox_min"]
        bbox_max = metadata["mesh"]["bbox_max"]
        center = [(bbox_min[i] + bbox_max[i]) * 0.5 for i in range(3)]
        extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
        max_extent = max(extent)

        client_mode = p.DIRECT if args.headless else p.GUI
        client = p.connect(client_mode)
        if client < 0:
            raise SystemExit("PyBullet 启动失败。")

        try:
            if not args.headless:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, 0)

            if args.show_plane:
                plane_id = p.loadURDF("plane.urdf", useFixedBase=True)
                p.changeVisualShape(plane_id, -1, rgbaColor=[0.92, 0.92, 0.92, 1.0])

            tunnel_id = p.loadURDF(
                str(urdf_path),
                basePosition=[0.0, 0.0, 0.0],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                useFixedBase=True,
            )

            print(
                "Loaded tunnel mesh:",
                f"vertices={metadata['mesh']['vertex_count']}",
                f"faces={metadata['mesh']['face_count']}",
                f"bbox_min={bbox_min}",
                f"bbox_max={bbox_max}",
                f"body_id={tunnel_id}",
            )

            if not args.headless:
                p.resetDebugVisualizerCamera(
                    cameraDistance=max_extent * 1.2,
                    cameraYaw=35.0,
                    cameraPitch=-15.0,
                    cameraTargetPosition=center,
                )

            remaining = args.steps
            while p.isConnected():
                p.stepSimulation()
                if args.headless:
                    remaining -= 1
                    if remaining <= 0:
                        break
                else:
                    time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            pass
        finally:
            if p.isConnected():
                p.disconnect()


if __name__ == "__main__":
    main()
