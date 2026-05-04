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
