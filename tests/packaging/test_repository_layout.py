from pathlib import Path


def test_new_repository_layout_exists() -> None:
    assert Path("src/rl_robot/algorithms").is_dir()
    assert Path("src/rl_robot/conf").is_dir()
    assert Path("src/rl_robot/assets").is_dir()
    assert Path("scripts/train.py").is_file()
    assert Path("tools").is_dir()
