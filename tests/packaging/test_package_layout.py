from importlib import import_module


def test_rl_robot_package_importable() -> None:
    package = import_module("rl_robot")
    assert package.__version__ == "0.1.0"
