from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rl_robot.training.artifacts import build_run_dir
from rl_robot.training.runner import build_device, run_training


def test_build_run_dir_uses_algorithm_env_backend(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"name": "ppo"},
            "env": {"name": "math_env"},
            "train": {"env_backend": "torch", "runs_root": str(tmp_path)},
        }
    )
    run_dir = build_run_dir(cfg)
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("ppo_math_env_torch_")


def test_build_device_falls_back_to_cpu() -> None:
    assert build_device("cpu").type == "cpu"


def test_run_training_accepts_plain_dict(monkeypatch, tmp_path) -> None:
    class FakeEnv:
        num_envs = 1

        def close(self) -> None:
            return None

    monkeypatch.setattr("rl_robot.training.runner.build_train_env", lambda **kwargs: FakeEnv())

    cfg = {
        "algorithm": {"name": "invalid"},
        "env": {"name": "math_env"},
        "train": {"runs_root": str(tmp_path), "device": "cpu"},
    }

    with pytest.raises(ValueError, match="Unsupported algorithm: invalid"):
        run_training(cfg)

    run_dirs = list(Path(tmp_path).iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "config.yaml").exists()
