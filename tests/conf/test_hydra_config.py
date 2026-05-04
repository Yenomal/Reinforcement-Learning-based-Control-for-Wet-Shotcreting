from hydra import compose, initialize_config_module


def test_default_hydra_config_loads() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config", return_hydra_config=True)
    assert cfg.algorithm.name == "ppo"
    assert cfg.algorithm.gamma == 0.99
    assert cfg.rl.env_name == "math_env"
    assert cfg.robot.kinematics_path == "src/rock_3D/robot_4dof/kinematics.yaml"
    assert cfg.disturbance.sensor_noise.enable is False
    assert cfg.eval.pybullet.enable is True
    assert cfg.ppo.total_updates == 200
    assert cfg.sac.total_steps == 2_000_000
    assert cfg.hydra.job.chdir is False
