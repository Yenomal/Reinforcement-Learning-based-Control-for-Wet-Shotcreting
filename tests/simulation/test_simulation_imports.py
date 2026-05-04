from rl_robot.simulation.robot.kinematics import load_robot_kinematics
from rl_robot.utils.resources import asset_path


def test_default_robot_kinematics_loads_from_packaged_asset() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        robot = load_robot_kinematics(path)
    assert len(robot.joint_order) == 4
