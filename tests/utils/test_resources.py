from rl_robot.utils.resources import asset_path


def test_default_kinematics_asset_is_packaged() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        assert path.name == "kinematics.yaml"
        assert path.is_file()


def test_default_html_asset_is_packaged() -> None:
    with asset_path("html/rock_environment.html") as path:
        assert path.suffix == ".html"
        assert path.is_file()
