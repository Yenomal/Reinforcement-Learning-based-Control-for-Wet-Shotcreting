from hydra import compose, initialize_config_module

from rl_robot.envs.math_env import MathEnv


def test_math_env_reset_step_smoke() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config")
    cfg.planner.use_reachability_map = False
    env = MathEnv(
        env_cfg=cfg.env,
        planner_cfg=cfg.planner,
        rl_cfg=cfg.rl,
        robot_cfg=cfg.robot,
        algorithm_cfg=cfg.algorithm,
        disturbance_cfg=cfg.disturbance,
    )
    observation, _ = env.reset(seed=0)
    next_observation, reward, terminated, truncated, _ = env.step(
        [0.0, 0.0, 0.0, 0.0]
    )
    assert observation.shape == next_observation.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
