from subprocess import run
import sys


def test_train_script_help() -> None:
    result = run(
        [sys.executable, "scripts/train.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Hydra" in result.stdout or "train" in result.stdout


def test_eval_script_help() -> None:
    result = run(
        [sys.executable, "scripts/eval.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Hydra" in result.stdout or "eval" in result.stdout


def test_build_reachability_map_script_help() -> None:
    result = run(
        [sys.executable, "scripts/build_reachability_map.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "reachability" in result.stdout.lower()


def test_visualize_rock_env_script_help() -> None:
    result = run(
        [sys.executable, "scripts/visualize_rock_env.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "visual" in result.stdout.lower() or "rock" in result.stdout.lower()
