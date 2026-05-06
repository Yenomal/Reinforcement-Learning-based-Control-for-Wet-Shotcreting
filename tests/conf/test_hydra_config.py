from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_default_hydra_config_matches_locked_experiment_baseline() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "rl_robot" / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", return_hydra_config=True)

    expected_ppo_std = {
        "mode": "cosine_schedule",
        "global_log_std": -1.0,
        "min_log_std": -10.0,
        "switch_update": 150,
        "schedule": {
            "schedule": "cosine",
            "start_log_std": -1.0,
            "end_log_std": -3.67,
        },
        "success_trigger": {
            "success_threshold": 0.90,
            "log_std_step": -0.01,
            "ema_alpha": 1.0,
            "patience_updates": 20,
            "cooldown_updates": 0,
            "min_episodes_in_window": 0,
        },
    }
    expected_deterministic_eval = {
        "enable": True,
        "interval_updates": 20,
        "episodes": 256,
        "num_envs": 256,
        "seed": 123,
        "backend": "torch",
    }
    raw_deterministic_eval = OmegaConf.to_container(cfg.train.deterministic_eval, resolve=False)

    assert cfg.algorithm.name == "ppo"
    assert cfg.algorithm.gamma == 0.99
    assert cfg.rl.env_name == "math_env"
    assert cfg.robot.kinematics_path == "src/rock_3D/robot_4dof/kinematics.yaml"
    assert OmegaConf.to_container(cfg.disturbance.sensor_noise, resolve=True) == {
        "enable": True,
        "current_point_step_std": 0.01,
        "current_point_bias_std": 0.02,
        "goal_point_step_std": 0.0,
        "goal_point_bias_std": 0.0,
    }
    assert cfg.eval.pybullet.enable is True
    assert cfg.ppo.total_updates == 200
    assert OmegaConf.to_container(cfg.ppo.std, resolve=True) == expected_ppo_std
    assert cfg.sac.total_steps == 2_000_000
    assert cfg.train.lr_schedule == "none"
    assert cfg.train.checkpoint == ""
    assert OmegaConf.to_container(cfg.train.deterministic_eval, resolve=True) == expected_deterministic_eval
    assert raw_deterministic_eval["num_envs"] == "${..num_envs}"
    assert raw_deterministic_eval["backend"] == "${..env_backend}"
    assert cfg.hydra.job.chdir is False
