from rl_robot.algorithms.ppo import build_ppo_config
from rl_robot.algorithms.sac import build_sac_config
from rl_robot.envs.train_env import build_train_env
from rl_robot.models.mlp import build_state_network
from rl_robot.planning.reachability_map import load_reachability_map


def test_core_modules_import_from_new_package() -> None:
    assert build_ppo_config()["gamma"] == 0.99
    assert "total_steps" in build_sac_config()
    assert callable(build_train_env)
    assert callable(build_state_network)
    assert callable(load_reachability_map)
