from pathlib import Path

import pytest

from rl_robot.utils.resources import asset_path


def test_default_kinematics_asset_is_packaged() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        assert path.name == "kinematics.yaml"
        assert path.is_file()


def test_default_html_asset_is_packaged() -> None:
    with asset_path("html/rock_environment.html") as path:
        assert path.suffix == ".html"
        assert path.is_file()


@pytest.mark.parametrize("relative_name", ["../__init__.py", "/tmp/outside.txt"])
def test_asset_path_rejects_non_package_relative_paths(relative_name: str) -> None:
    with pytest.raises(ValueError, match="relative package asset path"):
        with asset_path(relative_name):
            pass


def test_distribution_metadata_lists_packaged_assets() -> None:
    sources_manifest = Path("src/rl_robot.egg-info/SOURCES.txt")
    assert sources_manifest.is_file()

    packaged_resources = set(sources_manifest.read_text().splitlines())

    assert "src/rl_robot/assets/robot_4dof/kinematics.yaml" in packaged_resources
    assert "src/rl_robot/assets/html/rock_environment.html" in packaged_resources
