# `rl_robot` Project Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前研究仓库重构为可 `pip install -e .` 的 `rl_robot` 包，采用 Hydra 配置、包内资源、外侧薄脚本入口，并拆分过重的训练与评估模块。

**Architecture:** 以 `src/rl_robot/` 为唯一正式包边界，按 `algorithms / envs / planning / models / simulation / training / evaluation / assets / conf / utils` 分层。静态资源通过 `importlib.resources` 暴露，配置通过 `rl_robot.conf` 下的 Hydra config tree 组合，`scripts/*.py` 只负责入口与参数绑定，低频检查脚本移动到 `tools/`。

**Tech Stack:** Python 3.10+, setuptools, Hydra, OmegaConf, PyTorch, NumPy, Plotly, pytest

---

### Task 1: 建立可编辑安装骨架

**Files:**
- Modify: `pyproject.toml`
- Create: `src/rl_robot/__init__.py`
- Create: `src/rl_robot/py.typed`
- Test: `tests/packaging/test_package_layout.py`

- [ ] **Step 1: 写一个会失败的包导入测试**

```python
# tests/packaging/test_package_layout.py
from importlib import import_module


def test_rl_robot_package_importable() -> None:
    package = import_module("rl_robot")
    assert package.__version__ == "0.1.0"
```

- [ ] **Step 2: 运行测试，确认当前还没有 `rl_robot` 包**

Run: `pytest tests/packaging/test_package_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_robot'`

- [ ] **Step 3: 补齐可安装包最小实现**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rl-robot"
version = "0.1.0"
description = "RL environment for wet shotcrete robot research."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "hydra-core>=1.3",
    "matplotlib>=3.8",
    "numpy>=2.0",
    "omegaconf>=2.3",
    "plotly>=5.0",
    "pyyaml>=6.0",
    "torch>=2.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]
noise = ["opensimplex>=0.4.5"]
visualization = ["pybullet>=3.2"]

[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
include = ["rl_robot*"]

[tool.uv]
package = true
```

```python
# src/rl_robot/__init__.py
"""Top-level package for the wet shotcrete RL project."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

```text
# src/rl_robot/py.typed
```

- [ ] **Step 4: 重新安装并确认测试通过**

Run: `python -m pip install -e .`
Expected: editable install succeeds

Run: `pytest tests/packaging/test_package_layout.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add pyproject.toml src/rl_robot/__init__.py src/rl_robot/py.typed tests/packaging/test_package_layout.py
git commit -m "build: create editable rl_robot package skeleton"
```

### Task 2: 打包静态资源并提供统一资源访问 API

**Files:**
- Modify: `pyproject.toml`
- Create: `src/rl_robot/assets/__init__.py`
- Create: `src/rl_robot/utils/__init__.py`
- Create: `src/rl_robot/utils/resources.py`
- Move: `src/rock_3D/robot_4dof/kinematics.yaml`
- Move: `src/rock_3D/robot_4dof/metadata.json`
- Move: `src/rock_3D/robot_4dof/shipen_4dof.urdf`
- Move: `src/rock_3D/robot_4dof/meshes/`
- Move: `src/rock_3D/tunnel_environment/`
- Move: `src/rock_3D/rock_environment.html`
- Test: `tests/utils/test_resources.py`

- [ ] **Step 1: 写一个会失败的资源定位测试**

```python
# tests/utils/test_resources.py
from rl_robot.utils.resources import asset_path


def test_default_kinematics_asset_is_packaged() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        assert path.name == "kinematics.yaml"
        assert path.is_file()


def test_default_html_asset_is_packaged() -> None:
    with asset_path("html/rock_environment.html") as path:
        assert path.suffix == ".html"
        assert path.is_file()
```

- [ ] **Step 2: 运行测试，确认资源 API 还不存在**

Run: `pytest tests/utils/test_resources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_robot.utils'`

- [ ] **Step 3: 移动静态资源并实现 `asset_path()`**

```bash
mkdir -p src/rl_robot/assets/robot_4dof
mkdir -p src/rl_robot/assets/html
git mv src/rock_3D/robot_4dof/kinematics.yaml src/rl_robot/assets/robot_4dof/kinematics.yaml
git mv src/rock_3D/robot_4dof/metadata.json src/rl_robot/assets/robot_4dof/metadata.json
git mv src/rock_3D/robot_4dof/shipen_4dof.urdf src/rl_robot/assets/robot_4dof/shipen_4dof.urdf
git mv src/rock_3D/robot_4dof/meshes src/rl_robot/assets/robot_4dof/meshes
git mv src/rock_3D/tunnel_environment src/rl_robot/assets/tunnel_environment
git mv src/rock_3D/rock_environment.html src/rl_robot/assets/html/rock_environment.html
```

```python
# src/rl_robot/assets/__init__.py
"""Packaged static assets for rl_robot."""
```

```python
# src/rl_robot/utils/__init__.py
from .resources import asset_path

__all__ = ["asset_path"]
```

```python
# src/rl_robot/utils/resources.py
from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator


@contextmanager
def asset_path(relative_name: str) -> Iterator[Path]:
    resource = files("rl_robot.assets").joinpath(relative_name)
    if not resource.exists():
        raise FileNotFoundError(f"Packaged asset not found: {relative_name}")
    with as_file(resource) as resolved:
        yield Path(resolved)
```

```toml
# pyproject.toml
[tool.setuptools.package-data]
rl_robot = [
    "py.typed",
    "assets/**/*.yaml",
    "assets/**/*.json",
    "assets/**/*.urdf",
    "assets/**/*.obj",
    "assets/**/*.html",
]
```

- [ ] **Step 4: 重新运行资源测试**

Run: `pytest tests/utils/test_resources.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add pyproject.toml src/rl_robot/assets src/rl_robot/utils tests/utils/test_resources.py
git commit -m "build: package default simulation assets"
```

### Task 3: 建立 Hydra 配置树

**Files:**
- Create: `src/rl_robot/conf/__init__.py`
- Create: `src/rl_robot/conf/config.yaml`
- Create: `src/rl_robot/conf/algorithm/ppo.yaml`
- Create: `src/rl_robot/conf/algorithm/sac.yaml`
- Create: `src/rl_robot/conf/env/math_env.yaml`
- Create: `src/rl_robot/conf/rl/default.yaml`
- Create: `src/rl_robot/conf/planner/default.yaml`
- Create: `src/rl_robot/conf/model/plain_mlp.yaml`
- Create: `src/rl_robot/conf/model/structured_mlp.yaml`
- Create: `src/rl_robot/conf/train/default.yaml`
- Create: `src/rl_robot/conf/eval/default.yaml`
- Create: `src/rl_robot/conf/robot/robot_4dof.yaml`
- Create: `src/rl_robot/conf/disturbance/sensor_noise.yaml`
- Create: `src/rl_robot/conf/simulation/pybullet.yaml`
- Test: `tests/conf/test_hydra_config.py`

- [ ] **Step 1: 写一个会失败的 Hydra 组合测试**

```python
# tests/conf/test_hydra_config.py
from hydra import compose, initialize_config_module


def test_default_hydra_config_loads() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config")
    assert cfg.algorithm.name == "ppo"
    assert cfg.rl.env_name == "math_env"
    assert cfg.robot.kinematics_asset == "robot_4dof/kinematics.yaml"
    assert cfg.hydra.job.chdir is False
```

- [ ] **Step 2: 运行测试，确认配置树还不存在**

Run: `pytest tests/conf/test_hydra_config.py -v`
Expected: FAIL with `MissingConfigException` or `ModuleNotFoundError`

- [ ] **Step 3: 按组件拆出默认配置**

```python
# src/rl_robot/conf/__init__.py
"""Hydra config package for rl_robot."""
```

```yaml
# src/rl_robot/conf/config.yaml
defaults:
  - algorithm: ppo
  - env: math_env
  - rl: default
  - planner: default
  - model: plain_mlp
  - train: default
  - eval: default
  - robot: robot_4dof
  - disturbance: sensor_noise
  - simulation: pybullet
  - _self_

hydra:
  job:
    chdir: false
```

```yaml
# src/rl_robot/conf/robot/robot_4dof.yaml
kinematics_asset: robot_4dof/kinematics.yaml
```

```yaml
# src/rl_robot/conf/algorithm/ppo.yaml
name: ppo
gamma: 0.99
total_updates: 400
```

```yaml
# src/rl_robot/conf/rl/default.yaml
env_name: math_env
max_episode_steps: 200
success_tolerance: 0.003
reward_distance_scale: 1.0
initial_configuration_deg: [0.0, 0.0, 0.0, 0.0]
max_joint_delta_deg: [4.0, 4.0, 4.0, 4.0]
workspace_margin: 0.25
progress_reward_weight: 1.0
success_reward: 0.0
step_penalty: 0.0
action_l2_weight: 0.0
action_smoothness_weight: 0.0
boundary_penalty: 0.0
```

```yaml
# src/rl_robot/conf/train/default.yaml
seed: 42
device: cuda
env_backend: torch
num_envs: 256
runs_root: outputs/runs
resume: false
checkpoint: ""
```

```yaml
# src/rl_robot/conf/eval/default.yaml
checkpoint: outputs/runs/best/final.pt
episodes: 5
seed: 123
headless: false
render_pause: 0.05
episode_pause: 0.8
```

其余 group 文件按现有 `src/config.yaml` 的现值逐项拆开，字段名保持原语义，不在这一步改研究参数。

- [ ] **Step 4: 重新运行 Hydra 测试**

Run: `pytest tests/conf/test_hydra_config.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/rl_robot/conf tests/conf/test_hydra_config.py
git commit -m "feat: add hydra config tree for rl_robot"
```

### Task 4: 迁移算法、模型、环境与规划模块到正式包路径

**Files:**
- Move: `src/algorithm/` -> `src/rl_robot/algorithms/`
- Move: `src/model/` -> `src/rl_robot/models/`
- Move: `src/component/` -> `src/rl_robot/planning/`
- Move: `src/rl_env/` -> `src/rl_robot/envs/`
- Move: `src/rl_robot/planning/buffer.py` -> `src/rl_robot/algorithms/buffer.py`
- Create: `src/rl_robot/algorithms/__init__.py`
- Create: `src/rl_robot/models/__init__.py`
- Create: `src/rl_robot/planning/__init__.py`
- Create: `src/rl_robot/envs/__init__.py`
- Test: `tests/core/test_core_imports.py`

- [ ] **Step 1: 写一个会失败的核心导入测试**

```python
# tests/core/test_core_imports.py
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
```

- [ ] **Step 2: 运行测试，确认新路径尚未打通**

Run: `pytest tests/core/test_core_imports.py -v`
Expected: FAIL with `ModuleNotFoundError` for `rl_robot.algorithms`

- [ ] **Step 3: 执行目录迁移并修正内部导入**

```bash
git mv src/algorithm src/rl_robot/algorithms
git mv src/model src/rl_robot/models
git mv src/component src/rl_robot/planning
git mv src/rl_env src/rl_robot/envs
git mv src/rl_robot/planning/buffer.py src/rl_robot/algorithms/buffer.py
```

```python
# src/rl_robot/algorithms/__init__.py
from .ppo import PPOAgent, build_ppo_config, resolve_ppo_std_config
from .sac import SACAgent, build_sac_config

__all__ = [
    "PPOAgent",
    "SACAgent",
    "build_ppo_config",
    "build_sac_config",
    "resolve_ppo_std_config",
]
```

```python
# src/rl_robot/envs/__init__.py
from .math_env import MathEnv
from .train_env import BaseTrainEnv, build_train_env

__all__ = ["MathEnv", "BaseTrainEnv", "build_train_env"]
```

```python
# src/rl_robot/planning/__init__.py
from .disturbance import SensorNoise
from .planner import sample_planner_task, sample_planner_task_from_environment
from .reachability_map import build_and_save_reachability_map, load_reachability_map

__all__ = [
    "SensorNoise",
    "sample_planner_task",
    "sample_planner_task_from_environment",
    "build_and_save_reachability_map",
    "load_reachability_map",
]
```

```python
# representative import edits
# src/rl_robot/algorithms/ppo.py
from .buffer import OnPolicyBatch
from ..models.mlp import build_state_network

# src/rl_robot/envs/math_env.py
from ..planning.disturbance import SensorNoise
from ..planning.planner import sample_planner_task_from_environment
from ..planning.reachability_map import load_reachability_map
```

将 `src/rl_robot/algorithms/`, `models/`, `planning/`, `envs/` 下所有相对导入统一改成新包内路径，保证不再引用旧 `src.*` 结构。

- [ ] **Step 4: 重新运行核心导入测试**

Run: `pytest tests/core/test_core_imports.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/rl_robot/algorithms src/rl_robot/models src/rl_robot/planning src/rl_robot/envs tests/core/test_core_imports.py
git commit -m "refactor: migrate core rl modules into rl_robot package"
```

### Task 5: 迁移仿真运行时代码并隔离低频工具

**Files:**
- Create: `src/rl_robot/simulation/__init__.py`
- Create: `src/rl_robot/simulation/robot/__init__.py`
- Create: `src/rl_robot/simulation/tunnel/__init__.py`
- Create: `src/rl_robot/simulation/visualization/__init__.py`
- Move: `src/rock_3D/robot_4dof/kinematics.py`
- Move: `src/rock_3D/robot_4dof/torch_kinematics.py`
- Move: `src/rock_3D/robot_4dof/pybullet_player.py`
- Move: `src/rock_env/rock_wall.py`
- Move: `src/rock_3D/tools/build_tunnel_environment.py`
- Move: `src/rock_3D/tools/check_elbow_assignments.py`
- Move: `src/rock_3D/tools/check_elbow_pivot.py`
- Move: `src/rock_3D/tools/check_wrist_pivot.py`
- Move: `src/rock_3D/robot_4dof/view_pybullet.py`
- Move: `src/rock_3D/tunnel_environment/view_tunnel_pybullet.py`
- Test: `tests/simulation/test_simulation_imports.py`

- [ ] **Step 1: 写一个会失败的仿真导入与资源测试**

```python
# tests/simulation/test_simulation_imports.py
from rl_robot.simulation.robot.kinematics import load_robot_kinematics
from rl_robot.utils.resources import asset_path


def test_default_robot_kinematics_loads_from_packaged_asset() -> None:
    with asset_path("robot_4dof/kinematics.yaml") as path:
        robot = load_robot_kinematics(path)
    assert len(robot.joint_order) == 4
```

- [ ] **Step 2: 运行测试，确认 `simulation` 包还不存在**

Run: `pytest tests/simulation/test_simulation_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_robot.simulation'`

- [ ] **Step 3: 迁移运行时代码，并把检查脚本移动到 `tools/`**

```bash
mkdir -p src/rl_robot/simulation/robot
mkdir -p src/rl_robot/simulation/tunnel
mkdir -p src/rl_robot/simulation/visualization
mkdir -p tools

git mv src/rock_3D/robot_4dof/kinematics.py src/rl_robot/simulation/robot/kinematics.py
git mv src/rock_3D/robot_4dof/torch_kinematics.py src/rl_robot/simulation/robot/torch_kinematics.py
git mv src/rock_3D/robot_4dof/pybullet_player.py src/rl_robot/simulation/robot/pybullet_player.py
git mv src/rock_env/rock_wall.py src/rl_robot/simulation/tunnel/rock_wall.py
git mv src/rock_3D/tools/build_tunnel_environment.py src/rl_robot/simulation/tunnel/build_tunnel_environment.py

git mv src/rock_3D/tools/check_elbow_assignments.py tools/check_elbow_assignments.py
git mv src/rock_3D/tools/check_elbow_pivot.py tools/check_elbow_pivot.py
git mv src/rock_3D/tools/check_wrist_pivot.py tools/check_wrist_pivot.py
git mv src/rock_3D/robot_4dof/view_pybullet.py tools/view_robot_pybullet.py
git mv src/rock_3D/tunnel_environment/view_tunnel_pybullet.py tools/view_tunnel_pybullet.py
```

```python
# src/rl_robot/simulation/robot/__init__.py
from .kinematics import RobotKinematics, load_robot_kinematics
from .pybullet_player import PyBulletRobotPlayer
from .torch_kinematics import TorchRobotKinematics

__all__ = [
    "RobotKinematics",
    "TorchRobotKinematics",
    "PyBulletRobotPlayer",
    "load_robot_kinematics",
]
```

```python
# src/rl_robot/simulation/tunnel/__init__.py
from .build_tunnel_environment import SurfaceGrid, load_surface_grid, plotly_to_pybullet
from .rock_wall import build_training_rock_environment

__all__ = [
    "SurfaceGrid",
    "build_training_rock_environment",
    "load_surface_grid",
    "plotly_to_pybullet",
]
```

```python
# representative import edits
# src/rl_robot/envs/math_env.py
from ..simulation.robot.kinematics import RobotKinematics, load_robot_kinematics
from ..simulation.tunnel.rock_wall import build_training_rock_environment

# src/rl_robot/evaluation/runner.py  # once created in Task 7
from ..simulation.robot.pybullet_player import PyBulletRobotPlayer
from ..simulation.tunnel.build_tunnel_environment import load_surface_grid, plotly_to_pybullet
```

将 `kinematics_path` 风格的配置读取改成包资源读取，不再从 `src/rock_3D/...` 拼接仓库路径。

- [ ] **Step 4: 重新运行仿真测试**

Run: `pytest tests/simulation/test_simulation_imports.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/rl_robot/simulation tools tests/simulation/test_simulation_imports.py
git commit -m "refactor: move simulation runtime into rl_robot package"
```

### Task 6: 拆分训练入口并暴露 `run_training()`

**Files:**
- Create: `src/rl_robot/training/__init__.py`
- Create: `src/rl_robot/training/control.py`
- Create: `src/rl_robot/training/artifacts.py`
- Create: `src/rl_robot/training/metrics.py`
- Create: `src/rl_robot/training/eval_hooks.py`
- Create: `src/rl_robot/training/runner.py`
- Delete after split: `src/train.py`
- Test: `tests/training/test_training_api.py`

- [ ] **Step 1: 写一个会失败的训练 API 测试**

```python
# tests/training/test_training_api.py
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
```

- [ ] **Step 2: 运行测试，确认训练拆分模块还不存在**

Run: `pytest tests/training/test_training_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_robot.training'`

- [ ] **Step 3: 把 `src/train.py` 拆成聚焦模块**

```bash
mkdir -p src/rl_robot/training
git mv src/train.py src/rl_robot/training/runner.py
```

```python
# src/rl_robot/training/__init__.py
from .runner import build_device, run_training

__all__ = ["build_device", "run_training"]
```

```python
# src/rl_robot/training/control.py
from __future__ import annotations

# move the existing InteractiveTrainingControl class here verbatim
```

```python
# src/rl_robot/training/artifacts.py
from __future__ import annotations

# move the following functions from the old src/train.py verbatim:
# - build_run_dir
# - save_checkpoint
# - save_training_artifacts
# - load_metrics_csv
# - resolve_resume_checkpoint
# - write_metrics_csv
# - save_training_curves
```

```python
# src/rl_robot/training/metrics.py
from __future__ import annotations

# move the following functions from the old src/train.py verbatim:
# - summarize_episodes
# - build_metrics
# - _format_metric
# - log_metrics
```

```python
# src/rl_robot/training/eval_hooks.py
from __future__ import annotations

# move the following symbols from the old src/train.py verbatim:
# - build_action_scale_scheduler
# - resolve_deterministic_eval_config
# - DeterministicEvalRunner
# - run_deterministic_evaluation
# - build_ppo_std_scheduler
```

```python
# src/rl_robot/training/runner.py
from __future__ import annotations

from omegaconf import OmegaConf

from .artifacts import build_run_dir
from .control import InteractiveTrainingControl
from .eval_hooks import build_ppo_std_scheduler, run_deterministic_evaluation
from .metrics import build_metrics, log_metrics, summarize_episodes


def run_training(cfg) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    algorithm_name = str(config["algorithm"]["name"]).lower()
    if algorithm_name == "ppo":
        run_ppo_training(config)
        return
    if algorithm_name == "sac":
        run_sac_training(config)
        return
    raise ValueError(f"Unsupported algorithm: {algorithm_name}")
```

在这个任务里，`runner.py` 最终保留：

- `build_device`
- `collect_on_policy_rollout`
- `run_ppo_training`
- `run_sac_training`
- 新增 `run_training`

并把原 `main()` 逻辑改写成 `run_training(cfg)`，不再直接解析 argparse。

- [ ] **Step 4: 重新运行训练 API 测试**

Run: `pytest tests/training/test_training_api.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/rl_robot/training tests/training/test_training_api.py
git commit -m "refactor: split training entry into focused modules"
```

### Task 7: 拆分评估入口并暴露 `run_evaluation()`

**Files:**
- Create: `src/rl_robot/evaluation/__init__.py`
- Create: `src/rl_robot/evaluation/checkpoint.py`
- Create: `src/rl_robot/evaluation/renderer.py`
- Create: `src/rl_robot/evaluation/runner.py`
- Delete after split: `src/eval.py`
- Test: `tests/evaluation/test_evaluation_api.py`

- [ ] **Step 1: 写一个会失败的评估 API 测试**

```python
# tests/evaluation/test_evaluation_api.py
from rl_robot.evaluation.runner import build_device
from rl_robot.evaluation.renderer import EvalRenderer


def test_evaluation_module_exports_runtime_symbols() -> None:
    assert build_device("cpu").type == "cpu"
    assert EvalRenderer.__name__ == "EvalRenderer"
```

- [ ] **Step 2: 运行测试，确认评估包还不存在**

Run: `pytest tests/evaluation/test_evaluation_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_robot.evaluation'`

- [ ] **Step 3: 把 `src/eval.py` 拆成聚焦模块**

```bash
mkdir -p src/rl_robot/evaluation
git mv src/eval.py src/rl_robot/evaluation/runner.py
```

```python
# src/rl_robot/evaluation/__init__.py
from .runner import build_device, run_evaluation

__all__ = ["build_device", "run_evaluation"]
```

```python
# src/rl_robot/evaluation/checkpoint.py
from __future__ import annotations

# move the following functions from the old src/eval.py verbatim:
# - load_checkpoint
# - build_agent_from_checkpoint
# - build_action_scale_scheduler
```

```python
# src/rl_robot/evaluation/renderer.py
from __future__ import annotations

# move the existing EvalRenderer class here verbatim
```

```python
# src/rl_robot/evaluation/runner.py
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from .checkpoint import build_action_scale_scheduler, build_agent_from_checkpoint, load_checkpoint
from .renderer import EvalRenderer


def run_evaluation(cfg) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    checkpoint_path = Path(config["eval"]["checkpoint"])
    device = build_device(config["train"]["device"])
    checkpoint = load_checkpoint(checkpoint_path, device)
    agent = build_agent_from_checkpoint(checkpoint, config, device)
    action_scale_scheduler = build_action_scale_scheduler(config)
    return evaluate_episodes(config, agent, device, action_scale_scheduler)
```

将旧 `eval.py` 里的以下函数保留在 `runner.py`：

- `_surface_scene_from_rock_env`
- `_surface_scene_from_html`
- `load_eval_surface_scene`
- 新增 `evaluate_episodes(config, agent, device, action_scale_scheduler)`，把旧 `main()` 的 rollout 主循环整体搬入这里
- `main()` 的主循环逻辑，改成 `run_evaluation(cfg)`

- [ ] **Step 4: 重新运行评估 API 测试**

Run: `pytest tests/evaluation/test_evaluation_api.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/rl_robot/evaluation tests/evaluation/test_evaluation_api.py
git commit -m "refactor: split evaluation entry into focused modules"
```

### Task 8: 创建正式脚本入口、补 smoke test、移除旧入口

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/eval.py`
- Create: `scripts/build_reachability_map.py`
- Create: `scripts/visualize_rock_env.py`
- Modify: `src/rl_robot/planning/reachability_map.py`
- Create: `src/rl_robot/simulation/visualization/rock_env.py`
- Create: `tests/scripts/test_cli_help.py`
- Create: `tests/envs/test_math_env_smoke.py`
- Create: `tests/packaging/test_legacy_paths_removed.py`
- Modify: `README.md`
- Modify: `docs/process.md`
- Delete: `src/config.py`
- Delete: `src/config.yaml`
- Delete: `src/rock_env/env.py`

- [ ] **Step 1: 写会失败的 CLI 与 smoke tests**

```python
# tests/scripts/test_cli_help.py
from pathlib import Path
from subprocess import run
import sys


def test_train_script_help() -> None:
    result = run([sys.executable, "scripts/train.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hydra" in result.stdout or "train" in result.stdout


def test_eval_script_help() -> None:
    result = run([sys.executable, "scripts/eval.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hydra" in result.stdout or "eval" in result.stdout
```

```python
# tests/envs/test_math_env_smoke.py
from hydra import compose, initialize_config_module

from rl_robot.envs.math_env import MathEnv


def test_math_env_reset_step_smoke() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config")
    env = MathEnv(
        env_cfg=cfg.env,
        planner_cfg=cfg.planner,
        rl_cfg=cfg.rl,
        robot_cfg=cfg.robot,
        algorithm_cfg=cfg.algorithm,
        disturbance_cfg=cfg.disturbance,
    )
    observation, _ = env.reset(seed=0)
    next_observation, reward, terminated, truncated, _ = env.step([0.0, 0.0, 0.0, 0.0])
    assert observation.shape == next_observation.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
```

```python
# tests/packaging/test_legacy_paths_removed.py
from pathlib import Path


def test_legacy_single_config_and_entrypoints_are_removed() -> None:
    assert not Path("src/config.py").exists()
    assert not Path("src/config.yaml").exists()
    assert not Path("src/train.py").exists()
    assert not Path("src/eval.py").exists()
```

- [ ] **Step 2: 运行测试，确认脚本和旧入口清理还没完成**

Run: `pytest tests/scripts/test_cli_help.py tests/envs/test_math_env_smoke.py tests/packaging/test_legacy_paths_removed.py -v`
Expected: FAIL because scripts are missing and legacy files still exist

- [ ] **Step 3: 创建薄脚本、更新文档并移除旧中心入口**

```python
# scripts/train.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rl_robot.training import run_training


@hydra.main(version_base=None, config_path="../src/rl_robot/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
```

```python
# scripts/eval.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rl_robot.evaluation import run_evaluation


@hydra.main(version_base=None, config_path="../src/rl_robot/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
```

```python
# scripts/build_reachability_map.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rl_robot.planning.reachability_map import run_reachability_map


@hydra.main(version_base=None, config_path="../src/rl_robot/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_reachability_map(cfg)


if __name__ == "__main__":
    main()
```

```python
# scripts/visualize_rock_env.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rl_robot.simulation.visualization.rock_env import visualize_rock_environment


@hydra.main(version_base=None, config_path="../src/rl_robot/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    visualize_rock_environment(cfg)


if __name__ == "__main__":
    main()
```

```python
# src/rl_robot/planning/__init__.py
from .reachability_map import build_and_save_reachability_map, load_reachability_map, run_reachability_map
```

```python
# src/rl_robot/planning/reachability_map.py
from __future__ import annotations

from omegaconf import OmegaConf


def run_reachability_map(cfg) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    build_and_save_reachability_map(
        env_cfg=config["env"],
        planner_cfg=config["planner"],
        rl_cfg=config["rl"],
        robot_cfg=config["robot"],
    )
```

```python
# src/rl_robot/simulation/visualization/rock_env.py
from __future__ import annotations

import plotly.graph_objects as go

from ..tunnel.rock_wall import build_training_rock_environment


def create_visualization(rock_env):
    points_grid = rock_env["points_grid"]
    radius_grid = rock_env["radius_grid"]
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=points_grid[:, :, 0],
            y=points_grid[:, :, 1],
            z=points_grid[:, :, 2],
            surfacecolor=radius_grid,
            colorscale="Earth",
            opacity=0.85,
        )
    )
    return figure


def visualize_rock_environment(cfg) -> None:
    rock_env = build_training_rock_environment(cfg.env)
    figure = create_visualization(rock_env)
    figure.write_html(cfg.env.rock_env_html, include_plotlyjs=True, auto_open=False)
```

```bash
git rm src/config.py src/config.yaml src/rock_env/env.py
```

README 中所有 `python -m src.*` 命令统一改成：

```text
python scripts/train.py
python scripts/eval.py
python scripts/build_reachability_map.py
python scripts/visualize_rock_env.py
```

并在 `docs/process.md` 新增一节“新目录结构与入口约定”。

- [ ] **Step 4: 运行脚本帮助与 smoke tests**

Run: `pytest tests/scripts/test_cli_help.py tests/envs/test_math_env_smoke.py tests/packaging/test_legacy_paths_removed.py -v`
Expected: PASS

Run: `pytest tests/packaging tests/utils tests/conf tests/core tests/simulation tests/training tests/evaluation tests/scripts tests/envs -v`
Expected: PASS

- [ ] **Step 5: 提交最终切换**

```bash
git add scripts README.md docs/process.md tests
git commit -m "refactor: switch project to rl_robot package layout"
```

### Task 9: 清理遗留目录并做最终验收

**Files:**
- Delete if empty: `src/algorithm/`
- Delete if empty: `src/component/`
- Delete if empty: `src/model/`
- Delete if empty: `src/rl_env/`
- Delete if empty: `src/rock_env/`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-04-rl-robot-project-refactor-design.md` only if implementation reality requires sync notes

- [ ] **Step 1: 写一个会失败的仓库结构验收测试**

```python
# tests/packaging/test_repository_layout.py
from pathlib import Path


def test_new_repository_layout_exists() -> None:
    assert Path("src/rl_robot/algorithms").is_dir()
    assert Path("src/rl_robot/conf").is_dir()
    assert Path("src/rl_robot/assets").is_dir()
    assert Path("scripts/train.py").is_file()
    assert Path("tools").is_dir()
```

- [ ] **Step 2: 运行测试，确认最终布局还没有完全收口**

Run: `pytest tests/packaging/test_repository_layout.py -v`
Expected: FAIL until the last empty legacy directories and docs updates are done

- [ ] **Step 3: 删除空目录并做最终路径检查**

```bash
find src -maxdepth 1 -type d -empty -delete
find src -maxdepth 2 -type d -name "__pycache__" -prune -o -print
```

README 最终应保留：

```text
1. 安装：python -m pip install -e .
2. 训练：python scripts/train.py
3. 评估：python scripts/eval.py
4. 可达图：python scripts/build_reachability_map.py
5. 岩壁可视化：python scripts/visualize_rock_env.py
```

若实现中出现与设计文档不一致的稳定决策，在 spec 末尾追加一段“实现校准说明”并记录原因。

- [ ] **Step 4: 运行最终验收命令**

Run: `pytest tests/packaging/test_repository_layout.py -v`
Expected: PASS

Run: `python -m pip install -e .`
Expected: editable install succeeds against the final tree

Run: `python -c "import rl_robot; print(rl_robot.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 5: 提交最终收尾**

```bash
git add README.md docs/superpowers/specs/2026-05-04-rl-robot-project-refactor-design.md tests/packaging/test_repository_layout.py
git commit -m "chore: finalize rl_robot repository layout"
```
