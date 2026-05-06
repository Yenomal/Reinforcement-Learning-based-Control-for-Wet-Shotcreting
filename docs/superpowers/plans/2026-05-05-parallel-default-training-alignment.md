# Parallel Default Training Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让当前主分支默认训练行为重新对齐到 `parallel` 最新提交 `bbaadbd` 的实验语义，恢复完整训练观测、默认配置和打包岩壁资源。

**Architecture:** 保留当前 `rl_robot` 包结构和 Hydra 配置树，只回迁 `parallel` 默认实验所依赖的能力。迁移分为四条主线：默认配置回迁、PPO std 与 deterministic eval 回迁、训练产物导出恢复、打包岩壁资源同步与回归验证。

**Tech Stack:** Python, Hydra, PyTorch, Plotly, pytest, importlib.resources, git

---

## 文件结构与职责

- `src/rl_robot/conf/train/default.yaml`
  训练默认参数，恢复 `lr_schedule=none`、`deterministic_eval`、空 checkpoint。
- `src/rl_robot/conf/config.yaml`
  顶层 Hydra 默认树，补回 `ppo.std` 默认配置。
- `src/rl_robot/conf/disturbance/sensor_noise.yaml`
  默认噪声开关，恢复为 `parallel` 默认开启。
- `src/rl_robot/algorithms/lr_schedule.py`
  保留当前标量和学习率调度器，补回 success-triggered std scheduler。
- `src/rl_robot/algorithms/ppo.py`
  恢复 `ppo.std` 配置解析、`std_mode` 语义和 `ppo_std_*` 训练指标。
- `src/rl_robot/training/eval_hooks.py`
  从只处理 action scale 扩展为 deterministic eval 与 PPO std 调度辅助模块。
- `src/rl_robot/training/runner.py`
  在当前拆分后的训练主循环中接回 deterministic eval 和 PPO std 状态记录。
- `src/rl_robot/training/artifacts.py`
  恢复更完整的 `metrics.csv` 字段和 `training_curves.html` 面板。
- `src/rl_robot/assets/html/rock_environment.html`
  同步为 `parallel` 最新岩壁 HTML 快照。
- `tests/conf/test_hydra_config.py`
  覆盖默认配置语义。
- `tests/algorithms/test_lr_schedule.py`
  覆盖 success-triggered std scheduler 和 PPO std 配置解析。
- `tests/training/test_eval_hooks.py`
  覆盖 deterministic eval 配置解析与评估辅助函数。
- `tests/training/test_artifacts.py`
  覆盖完整 CSV 字段和 HTML 面板输出。
- `tests/utils/test_resources.py`
  固定默认打包 HTML 快照哈希，防止主分支默认岩壁环境再次漂移。
- `README.md`
  同步默认训练说明和可达图重建命令。
- `docs/config.md`
  同步当前默认配置字段说明。

### Task 1: 恢复默认配置语义

**Files:**
- Modify: `src/rl_robot/conf/train/default.yaml`
- Modify: `src/rl_robot/conf/config.yaml`
- Modify: `src/rl_robot/conf/disturbance/sensor_noise.yaml`
- Modify: `tests/conf/test_hydra_config.py`

- [ ] **Step 1: 先写失败的 Hydra 默认配置回归测试**

```python
from hydra import compose, initialize_config_module


def test_default_hydra_config_loads() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config", return_hydra_config=True)

    assert cfg.train.lr_schedule == "none"
    assert cfg.train.checkpoint == ""
    assert cfg.train.deterministic_eval.enable is True
    assert cfg.train.deterministic_eval.interval_updates == 20
    assert cfg.train.deterministic_eval.episodes == 256
    assert cfg.train.deterministic_eval.num_envs == 256
    assert cfg.disturbance.sensor_noise.enable is True
    assert cfg.ppo.std.mode == "cosine_schedule"
    assert cfg.ppo.std.schedule.end_log_std == -3.67
```

- [ ] **Step 2: 运行测试，确认当前默认配置与 `parallel` 语义不一致**

Run:

```bash
pytest tests/conf/test_hydra_config.py -v
```

Expected:

```text
FAILED tests/conf/test_hydra_config.py::test_default_hydra_config_loads
E   AssertionError: assert 'cosine' == 'none'
```

- [ ] **Step 3: 最小修改默认配置**

```yaml
# src/rl_robot/conf/train/default.yaml
seed: 42
device: cuda
env_backend: torch
num_envs: 256
deterministic_eval:
  enable: true
  interval_updates: 20
  episodes: 256
  num_envs: 256
  seed: 123
  backend: torch
lr_schedule: none
lr_min_ratio: 0.1
log_interval: 1
runs_root: outputs/runs
resume: false
checkpoint: ""
```

```yaml
# src/rl_robot/conf/disturbance/sensor_noise.yaml
sensor_noise:
  enable: true
  current_point_step_std: 0.01
  current_point_bias_std: 0.02
  goal_point_step_std: 0.0
  goal_point_bias_std: 0.0
```

```yaml
# src/rl_robot/conf/config.yaml
ppo:
  total_updates: 400
  std:
    mode: cosine_schedule
    global_log_std: -1.0
    min_log_std: -10.0
    switch_update: 150
    schedule:
      schedule: cosine
      start_log_std: -1.0
      end_log_std: -3.67
    success_trigger:
      success_threshold: 0.90
      log_std_step: -0.01
      ema_alpha: 1.0
      patience_updates: 20
      cooldown_updates: 0
      min_episodes_in_window: 0
```

- [ ] **Step 4: 重新运行配置测试**

Run:

```bash
pytest tests/conf/test_hydra_config.py -v
```

Expected:

```text
PASSED tests/conf/test_hydra_config.py::test_default_hydra_config_loads
```

- [ ] **Step 5: 提交配置回迁**

```bash
git add \
  src/rl_robot/conf/train/default.yaml \
  src/rl_robot/conf/config.yaml \
  src/rl_robot/conf/disturbance/sensor_noise.yaml \
  tests/conf/test_hydra_config.py
git commit -m "config: restore parallel default training semantics"
```

### Task 2: 恢复 PPO std 调度器与配置解析

**Files:**
- Modify: `src/rl_robot/algorithms/lr_schedule.py`
- Modify: `src/rl_robot/algorithms/ppo.py`
- Create: `tests/algorithms/test_lr_schedule.py`

- [ ] **Step 1: 先写调度器与 PPO std 配置解析回归测试**

```python
import pytest

from rl_robot.algorithms.lr_schedule import (
    CosineThenSuccessRateTriggeredScheduler,
    ScalarScheduler,
    SuccessRateTriggeredScheduler,
)
from rl_robot.algorithms.ppo import resolve_ppo_std_config


def test_success_rate_triggered_scheduler_decreases_after_patience() -> None:
    scheduler = SuccessRateTriggeredScheduler(
        start_value=-3.0,
        min_value=-4.0,
        success_threshold=0.9,
        value_step=-0.1,
        ema_alpha=1.0,
        patience_updates=2,
        cooldown_updates=1,
        min_episodes_in_window=0,
    )

    scheduler.observe(0.95, episodes_in_window=64)
    metrics = scheduler.observe(0.95, episodes_in_window=64)

    assert metrics["ppo_next_log_std"] == pytest.approx(-3.1)
    assert metrics["ppo_std_cooldown_remaining"] == 1.0


def test_cosine_then_trigger_scheduler_reports_phase() -> None:
    cosine = ScalarScheduler(
        start_value=-1.0,
        end_value=-3.0,
        total_progress=10,
        schedule="cosine",
    )
    trigger = SuccessRateTriggeredScheduler(
        start_value=-3.0,
        min_value=-4.0,
        success_threshold=0.9,
        value_step=-0.1,
        ema_alpha=1.0,
        patience_updates=1,
        cooldown_updates=0,
        min_episodes_in_window=0,
    )
    scheduler = CosineThenSuccessRateTriggeredScheduler(
        cosine_scheduler=cosine,
        trigger_scheduler=trigger,
        switch_update=5,
    )

    assert scheduler.state(2)["ppo_std_phase"] == "cosine"
    assert scheduler.observe(0.95, episodes_in_window=32)["ppo_std_phase"] == "trigger"


def test_resolve_ppo_std_config_preserves_nested_std_tree() -> None:
    std_cfg = resolve_ppo_std_config(
        {
            "std": {
                "mode": "cosine_schedule",
                "global_log_std": -1.0,
                "min_log_std": -10.0,
                "schedule": {
                    "schedule": "cosine",
                    "start_log_std": -1.0,
                    "end_log_std": -3.67,
                },
            }
        }
    )

    assert std_cfg["mode"] == "cosine_schedule"
    assert std_cfg["schedule"]["end_log_std"] == pytest.approx(-3.67)
```

- [ ] **Step 2: 运行测试，确认当前 `lr_schedule.py` 和 `ppo.py` 缺少这些语义**

Run:

```bash
pytest tests/algorithms/test_lr_schedule.py -v
```

Expected:

```text
FAILED tests/algorithms/test_lr_schedule.py::test_success_rate_triggered_scheduler_decreases_after_patience
E   ImportError: cannot import name 'SuccessRateTriggeredScheduler'
```

- [ ] **Step 3: 补回 success-triggered scheduler 和 PPO std 解析**

```python
# src/rl_robot/algorithms/lr_schedule.py
class SuccessRateTriggeredScheduler:
    def __init__(
        self,
        *,
        start_value: float,
        min_value: float,
        success_threshold: float,
        value_step: float,
        ema_alpha: float,
        patience_updates: int,
        cooldown_updates: int,
        min_episodes_in_window: int,
    ) -> None:
        self.current_value = float(start_value)
        self.min_value = float(min_value)
        self.success_threshold = float(success_threshold)
        self.value_step = float(value_step)
        self.ema_alpha = float(ema_alpha)
        self.patience_updates = max(int(patience_updates), 1)
        self.cooldown_updates = max(int(cooldown_updates), 0)
        self.min_episodes_in_window = max(int(min_episodes_in_window), 0)
        self.success_ema: float | None = None
        self.streak = 0
        self.stage = 0
        self.cooldown_remaining = 0
```

```python
# src/rl_robot/algorithms/ppo.py
DEFAULT_PPO_CONFIG: Dict[str, Any] = {
    "gamma": 0.99,
    "lr": 3.0e-4,
    "init_log_std": -2.0,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "normalize_value_targets": False,
    "entropy_coef": 0.0,
    "max_grad_norm": 0.5,
    "rollout_steps": 65536,
    "update_epochs": 10,
    "minibatch_size": 512,
    "normalize_advantages": True,
    "std": {
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
    },
}
```

```python
def resolve_ppo_std_config(algorithm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved_cfg = build_ppo_config(algorithm_cfg)
    std_cfg = copy.deepcopy(resolved_cfg["std"])
    std_cfg["mode"] = str(std_cfg.get("mode", "cosine_schedule")).lower()
    std_cfg["global_log_std"] = float(std_cfg.get("global_log_std", resolved_cfg["init_log_std"]))
    std_cfg["min_log_std"] = float(std_cfg.get("min_log_std", -6.0))
    return std_cfg
```

- [ ] **Step 4: 重新运行调度器测试**

Run:

```bash
pytest tests/algorithms/test_lr_schedule.py -v
```

Expected:

```text
PASSED tests/algorithms/test_lr_schedule.py::test_success_rate_triggered_scheduler_decreases_after_patience
PASSED tests/algorithms/test_lr_schedule.py::test_cosine_then_trigger_scheduler_reports_phase
PASSED tests/algorithms/test_lr_schedule.py::test_resolve_ppo_std_config_preserves_nested_std_tree
```

- [ ] **Step 5: 提交调度器与 PPO std 回迁**

```bash
git add \
  src/rl_robot/algorithms/lr_schedule.py \
  src/rl_robot/algorithms/ppo.py \
  tests/algorithms/test_lr_schedule.py
git commit -m "feat: restore parallel ppo std scheduling"
```

### Task 3: 恢复 deterministic evaluation 与训练指标注入

**Files:**
- Modify: `src/rl_robot/training/eval_hooks.py`
- Modify: `src/rl_robot/training/runner.py`
- Create: `tests/training/test_eval_hooks.py`

- [ ] **Step 1: 先写 deterministic eval 辅助函数回归测试**

```python
import torch

from rl_robot.training.eval_hooks import (
    resolve_deterministic_eval_config,
    run_deterministic_evaluation,
)


def test_resolve_deterministic_eval_config_uses_train_num_envs() -> None:
    resolved = resolve_deterministic_eval_config(
        {
            "train": {
                "num_envs": 8,
                "deterministic_eval": {
                    "enable": True,
                    "interval_updates": 20,
                },
            }
        }
    )

    assert resolved["enable"] is True
    assert resolved["episodes"] == 8
    assert resolved["num_envs"] == 8
    assert resolved["seed"] == 123


def test_run_deterministic_evaluation_passes_action_scale_ratio() -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.ratio = None

        def set_action_scale_ratio(self, ratio: float) -> None:
            self.ratio = ratio

        def evaluate(self, *, agent, device, episodes, seed):
            return {
                "det_success_rate": 0.5,
                "det_mean_min_goal_distance": 0.01,
                "det_mean_length": 10.0,
            }

    runner = FakeRunner()
    metrics = run_deterministic_evaluation(
        runner=runner,
        agent=object(),
        device=torch.device("cpu"),
        episodes=16,
        seed=123,
        action_scale_ratio=0.5,
    )

    assert runner.ratio == 0.5
    assert metrics["det_success_rate"] == 0.5
```

- [ ] **Step 2: 运行测试，确认 `eval_hooks.py` 目前只处理 action scale**

Run:

```bash
pytest tests/training/test_eval_hooks.py -v
```

Expected:

```text
FAILED tests/training/test_eval_hooks.py::test_resolve_deterministic_eval_config_uses_train_num_envs
E   ImportError: cannot import name 'resolve_deterministic_eval_config'
```

- [ ] **Step 3: 扩展 `eval_hooks.py`，并在 `runner.py` 接回指标注入**

```python
# src/rl_robot/training/eval_hooks.py
def resolve_deterministic_eval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = config.get("train", {})
    det_cfg = dict(train_cfg.get("deterministic_eval", {}))
    default_num_envs = int(train_cfg.get("num_envs", 1))
    return {
        "enable": bool(det_cfg.get("enable", False)),
        "interval_updates": max(int(det_cfg.get("interval_updates", 0)), 1),
        "episodes": max(int(det_cfg.get("episodes", default_num_envs)), 1),
        "num_envs": max(int(det_cfg.get("num_envs", default_num_envs)), 1),
        "seed": int(det_cfg.get("seed", 123)),
    }
```

```python
# src/rl_robot/training/eval_hooks.py
def run_deterministic_evaluation(
    *,
    runner: DeterministicEvalRunner,
    agent: PPOAgent,
    device: torch.device,
    episodes: int,
    seed: int,
    action_scale_ratio: float,
) -> Dict[str, float]:
    runner.set_action_scale_ratio(action_scale_ratio)
    return runner.evaluate(
        agent=agent,
        device=device,
        episodes=episodes,
        seed=seed,
    )
```

```python
# src/rl_robot/training/runner.py
ppo_std_cfg = resolve_ppo_std_config(config.get("ppo", {}))
ppo_std_scheduler = build_ppo_std_scheduler(
    agent=agent,
    ppo_std_cfg=ppo_std_cfg,
    total_updates=total_updates,
)
det_eval_cfg = resolve_deterministic_eval_config(config)
if det_eval_cfg["enable"] and update % int(det_eval_cfg["interval_updates"]) == 0:
    det_metrics = run_deterministic_evaluation(
        runner=det_eval_runner,
        agent=agent,
        device=device,
        episodes=int(det_eval_cfg["episodes"]),
        seed=int(det_eval_cfg["seed"]),
        action_scale_ratio=env.get_action_scale_ratio(),
    )
    metrics.update(det_metrics)
```

- [ ] **Step 4: 重新运行辅助函数测试**

Run:

```bash
pytest tests/training/test_eval_hooks.py -v
```

Expected:

```text
PASSED tests/training/test_eval_hooks.py::test_resolve_deterministic_eval_config_uses_train_num_envs
PASSED tests/training/test_eval_hooks.py::test_run_deterministic_evaluation_passes_action_scale_ratio
```

- [ ] **Step 5: 提交 deterministic eval 与训练指标回迁**

```bash
git add \
  src/rl_robot/training/eval_hooks.py \
  src/rl_robot/training/runner.py \
  tests/training/test_eval_hooks.py
git commit -m "feat: restore parallel deterministic eval hooks"
```

### Task 4: 恢复完整训练产物导出

**Files:**
- Modify: `src/rl_robot/training/artifacts.py`
- Create: `tests/training/test_artifacts.py`

- [ ] **Step 1: 先写 CSV 字段和 HTML 面板回归测试**

```python
from pathlib import Path

from rl_robot.training.artifacts import save_training_curves, write_metrics_csv


def test_write_metrics_csv_preserves_parallel_training_columns(tmp_path: Path) -> None:
    history = [
        {
            "progress": 20,
            "batch_reward_mean": 0.1,
            "episodes_in_window": 32,
            "success_episodes": 16,
            "success_rate": 0.5,
            "det_success_rate": 0.25,
            "det_mean_min_goal_distance": 0.01,
            "approx_kl": 0.02,
            "explained_variance": 0.99,
            "ppo_success_ema": 0.5,
            "ppo_std_streak": 2.0,
            "ppo_next_log_std": -3.0,
            "ppo_std_cooldown_remaining": 0.0,
            "ppo_std_phase": "trigger",
            "ppo_std_mean": 0.03,
            "ppo_log_std_mean": -3.5,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "lr": 3e-4,
            "env_steps_per_sec": 1234.0,
        }
    ]

    write_metrics_csv(tmp_path, history)
    content = (tmp_path / "metrics.csv").read_text(encoding="utf-8")

    assert "det_success_rate" in content
    assert "det_mean_min_goal_distance" in content
    assert "ppo_std_phase" in content
    assert "ppo_log_std_mean" in content


def test_save_training_curves_writes_parallel_panels(tmp_path: Path) -> None:
    history = [
        {
            "progress": 20,
            "batch_reward_mean": 0.1,
            "episodes_in_window": 32,
            "success_episodes": 16,
            "success_rate": 0.5,
            "det_success_rate": 0.25,
            "det_mean_min_goal_distance": 0.01,
            "approx_kl": 0.02,
            "explained_variance": 0.99,
            "ppo_std_mean": 0.03,
            "ppo_log_std_mean": -3.5,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "lr": 3e-4,
        }
    ]

    save_training_curves(tmp_path, history)
    html = (tmp_path / "training_curves.html").read_text(encoding="utf-8")

    assert "Deterministic Distance" in html
    assert "Approx KL" in html
    assert "Explained Variance" in html
    assert "PPO Std" in html
```

- [ ] **Step 2: 运行测试，确认当前导出逻辑已被精简**

Run:

```bash
pytest tests/training/test_artifacts.py -v
```

Expected:

```text
FAILED tests/training/test_artifacts.py::test_write_metrics_csv_preserves_parallel_training_columns
E   AssertionError: assert 'det_success_rate' in 'progress,batch_reward_mean,...'
```

- [ ] **Step 3: 恢复 `parallel` 风格的字段和曲线面板**

```python
# src/rl_robot/training/artifacts.py
preferred_fields = [
    "progress",
    "batch_reward_mean",
    "episodes_in_window",
    "success_episodes",
    "success_rate",
    "det_success_rate",
    "det_mean_min_goal_distance",
    "approx_kl",
    "explained_variance",
    "ppo_success_ema",
    "ppo_std_streak",
    "ppo_next_log_std",
    "ppo_std_cooldown_remaining",
    "ppo_std_phase",
    "policy_loss",
    "value_loss",
    "ppo_std_mean",
    "ppo_log_std_mean",
    "lr",
    "actor_lr",
    "critic_lr",
    "alpha_lr",
    "env_steps_per_sec",
]
```

```python
# src/rl_robot/training/artifacts.py
fig = make_subplots(
    rows=9,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Batch Reward",
        "Episodes",
        "Success Rate",
        "Deterministic Distance",
        "Loss",
        "Approx KL",
        "Explained Variance",
        "PPO Std",
        "Learning Rate",
    ),
)
```

- [ ] **Step 4: 重新运行导出测试**

Run:

```bash
pytest tests/training/test_artifacts.py -v
```

Expected:

```text
PASSED tests/training/test_artifacts.py::test_write_metrics_csv_preserves_parallel_training_columns
PASSED tests/training/test_artifacts.py::test_save_training_curves_writes_parallel_panels
```

- [ ] **Step 5: 提交训练产物导出恢复**

```bash
git add \
  src/rl_robot/training/artifacts.py \
  tests/training/test_artifacts.py
git commit -m "feat: restore parallel training artifact fields"
```

### Task 5: 同步默认岩壁资源并更新文档

**Files:**
- Modify: `src/rl_robot/assets/html/rock_environment.html`
- Modify: `tests/utils/test_resources.py`
- Modify: `README.md`
- Modify: `docs/config.md`

- [ ] **Step 1: 先写默认打包岩壁快照固定测试**

```python
import hashlib

from rl_robot.utils.resources import asset_path


def test_default_html_asset_matches_parallel_latest_snapshot() -> None:
    with asset_path("html/rock_environment.html") as path:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest == "885d747f31409458b5e7e916809baae3c6a3a2e6eff75cf5d94f65735da85fe2"
```

- [ ] **Step 2: 运行测试，确认当前打包岩壁资源与 `parallel` 最新快照不一致**

Run:

```bash
pytest tests/utils/test_resources.py::test_default_html_asset_matches_parallel_latest_snapshot -v
```

Expected:

```text
FAILED tests/utils/test_resources.py::test_default_html_asset_matches_parallel_latest_snapshot
E   AssertionError: assert 'a83aef41...' == '885d747f...'
```

- [ ] **Step 3: 同步打包岩壁 HTML，更新 README 和配置文档**

Run:

```bash
git show parallel:src/rock_3D/rock_environment.html > src/rl_robot/assets/html/rock_environment.html
```

```md
<!-- README.md -->
- 训练默认对齐 `parallel` 最新实验语义。
- 若修改默认岩壁 HTML，请先重新生成可达图：
  `python scripts/build_reachability_map.py --force --device cuda`
```

```md
<!-- docs/config.md -->
- `train.lr_schedule` 默认值为 `none`
- `train.deterministic_eval.*` 为默认开启
- `ppo.std.*` 为默认训练行为的一部分
- `disturbance.sensor_noise.enable` 默认值为 `true`
```

- [ ] **Step 4: 运行资源测试并重建可达图**

Run:

```bash
pytest tests/utils/test_resources.py::test_default_html_asset_matches_parallel_latest_snapshot -v
PYTHONPATH=src python scripts/build_reachability_map.py --force --device cuda
```

Expected:

```text
PASSED tests/utils/test_resources.py::test_default_html_asset_matches_parallel_latest_snapshot
Reachability map ready
```

- [ ] **Step 5: 提交默认资源与文档同步**

```bash
git add \
  src/rl_robot/assets/html/rock_environment.html \
  tests/utils/test_resources.py \
  README.md \
  docs/config.md \
  outputs/reachability/reachability_map.npz \
  outputs/reachability/reachability_map.html
git commit -m "chore: align packaged training assets with parallel defaults"
```

### Task 6: 端到端验证当前默认训练语义

**Files:**
- Modify: `tests/training/test_training_api.py`

- [ ] **Step 1: 先写训练入口默认语义回归测试**

```python
from hydra import compose, initialize_config_module


def test_default_training_semantics_match_parallel_baseline() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config")

    assert cfg.train.lr_schedule == "none"
    assert cfg.train.deterministic_eval.enable is True
    assert cfg.disturbance.sensor_noise.enable is True
    assert cfg.ppo.std.mode == "cosine_schedule"
```

- [ ] **Step 2: 运行测试，确保回归点都被现有测试捕获**

Run:

```bash
pytest tests/training/test_training_api.py::test_default_training_semantics_match_parallel_baseline -v
```

Expected:

```text
PASSED tests/training/test_training_api.py::test_default_training_semantics_match_parallel_baseline
```

- [ ] **Step 3: 运行聚合测试集**

Run:

```bash
pytest \
  tests/conf/test_hydra_config.py \
  tests/algorithms/test_lr_schedule.py \
  tests/training/test_eval_hooks.py \
  tests/training/test_artifacts.py \
  tests/training/test_training_api.py \
  tests/utils/test_resources.py -v
```

Expected:

```text
============================= test session starts =============================
... collected N items
... passed
```

- [ ] **Step 4: 用默认配置跑一次短训练 smoke test**

Run:

```bash
PYTHONPATH=src python scripts/train.py ppo.total_updates=2 train.device=cpu train.num_envs=4 train.log_interval=1
```

Expected:

```text
PPO Training On MathEnv
[Update 00001] ...
[Update 00002] ...
Training complete. Final artifacts saved to: outputs/runs/...
```

- [ ] **Step 5: 提交端到端验证与收尾**

```bash
git add tests/training/test_training_api.py
git commit -m "test: cover restored parallel training defaults"
```
