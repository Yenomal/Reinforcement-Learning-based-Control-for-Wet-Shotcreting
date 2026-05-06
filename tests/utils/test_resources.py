import hashlib
import struct
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_robot.utils.resources import asset_path
from rl_robot.simulation.tunnel.build_tunnel_environment import load_surface_grid


def test_default_kinematics_asset_is_packaged() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        assert path.name == "kinematics.yaml"
        assert path.is_file()


def test_default_html_asset_is_packaged() -> None:
    with asset_path("html/rock_environment.html") as path:
        assert path.suffix == ".html"
        assert path.is_file()
        grid = load_surface_grid(path)

    semantic_parts = [str(grid.rows).encode(), str(grid.cols).encode()]
    for values in (grid.x, grid.y, grid.z, grid.surfacecolor):
        semantic_parts.append(struct.pack("<" + "d" * len(values), *values))

    assert hashlib.sha256(b"|".join(semantic_parts)).hexdigest() == (
        "97f05b31d17f9a85e6c943d4af7f79eff5c2253730644335f63770d2da5448dd"
    )


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
