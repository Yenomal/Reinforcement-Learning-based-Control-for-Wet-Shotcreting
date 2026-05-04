import importlib
import sys
from pathlib import Path

from rl_robot.algorithms.ppo import build_ppo_config
from rl_robot.algorithms.sac import build_sac_config
from rl_robot.envs.train_env import build_train_env
from rl_robot.models.mlp import build_state_network
from rl_robot.planning.reachability_map import load_reachability_map

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_core_modules_import_from_new_package() -> None:
    assert build_ppo_config()["gamma"] == 0.99
    assert "total_steps" in build_sac_config()
    assert callable(build_train_env)
    assert callable(build_state_network)
    assert callable(load_reachability_map)
    assert importlib.import_module("rl_robot.training").__name__ == "rl_robot.training"
    assert importlib.import_module("src.eval").__name__ == "src.eval"
