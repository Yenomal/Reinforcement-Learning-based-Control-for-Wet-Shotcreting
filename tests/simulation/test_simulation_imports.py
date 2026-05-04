from pathlib import Path

import pytest

from rl_robot.simulation.robot.kinematics import load_robot_kinematics
from rl_robot.simulation.tunnel.rock_wall import build_training_rock_environment
from rl_robot.utils.resources import asset_path


def test_default_robot_kinematics_loads_from_packaged_asset() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        robot = load_robot_kinematics(path)
    assert len(robot.joint_order) == 4


def test_missing_custom_kinematics_path_raises_file_not_found() -> None:
    missing_path = Path("does/not/exist/custom_kinematics.yaml")

    with pytest.raises(FileNotFoundError):
        load_robot_kinematics(missing_path)


def test_missing_custom_default_named_kinematics_path_raises_file_not_found() -> None:
    missing_path = Path("does/not/exist/kinematics.yaml")

    with pytest.raises(FileNotFoundError):
        load_robot_kinematics(missing_path)


def test_legacy_default_train_html_path_loads_packaged_asset() -> None:
    rock_env = build_training_rock_environment(
        {"train_rock_env_html": "src/rock_3D/rock_environment.html"}
    )

    assert rock_env["source"] == "html"


def test_missing_custom_default_named_train_html_path_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        build_training_rock_environment(
            {"train_rock_env_html": "does/not/exist/rock_environment.html"}
        )
