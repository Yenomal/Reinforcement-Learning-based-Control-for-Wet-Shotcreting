from hydra import compose, initialize_config_module


def test_default_hydra_config_loads() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config", return_hydra_config=True)
    assert cfg.algorithm.name == "ppo"
    assert cfg.rl.env_name == "math_env"
    assert cfg.robot.kinematics_asset == "robot_4dof/kinematics.yaml"
    assert cfg.hydra.job.chdir is False
