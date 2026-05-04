from omegaconf import OmegaConf

from rl_robot.training.artifacts import build_run_dir
from rl_robot.training.runner import build_device


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
