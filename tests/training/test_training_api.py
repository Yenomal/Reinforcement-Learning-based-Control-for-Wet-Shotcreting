from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
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


def test_run_training_preserves_default_training_semantics(
    monkeypatch, tmp_path
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "rl_robot" / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"train.runs_root={tmp_path}",
                "train.device=cpu",
            ],
        )

    assert cfg.train.lr_schedule == "none"
    assert cfg.train.deterministic_eval.enable is True
    assert cfg.disturbance.sensor_noise.enable is True
    assert cfg.ppo.std.mode == "cosine_schedule"

    captured: dict[str, object] = {}

    class FakeEnv:
        observation_dim = 6
        action_dim = 2

        def __init__(self, num_envs: int) -> None:
            self.num_envs = num_envs

        def close(self) -> None:
            return None

    class FakePPOAgent:
        def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            model_cfg: dict,
            algorithm_cfg: dict,
            device,
        ) -> None:
            captured["agent_init"] = {
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "model_cfg": model_cfg,
                "algorithm_cfg": algorithm_cfg,
                "device": device,
            }

        def load_training_state(self, state_dict: dict) -> None:
            raise AssertionError("resume path should not be used in default semantics test")

        def load_policy_state(self, state_dict: dict) -> None:
            raise AssertionError("checkpoint path should not be used in default semantics test")

    def fake_build_train_env(**kwargs):
        captured["env_build_kwargs"] = kwargs
        return FakeEnv(num_envs=kwargs["num_envs"])

    def fake_run_ppo_training(**kwargs) -> None:
        config = kwargs["config"]
        assert config["train"]["lr_schedule"] == "none"
        assert config["train"]["deterministic_eval"]["enable"] is True
        captured["run_training_kwargs"] = kwargs

    monkeypatch.setattr("rl_robot.training.runner.build_train_env", fake_build_train_env)
    monkeypatch.setattr("rl_robot.training.runner.PPOAgent", FakePPOAgent)
    monkeypatch.setattr("rl_robot.training.runner.run_ppo_training", fake_run_ppo_training)

    run_training(cfg)

    assert "run_training_kwargs" in captured
    env_build_kwargs = captured["env_build_kwargs"]
    assert env_build_kwargs["disturbance_cfg"]["sensor_noise"]["enable"] is True

    agent_init = captured["agent_init"]
    assert agent_init["algorithm_cfg"]["std"]["mode"] == "cosine_schedule"
