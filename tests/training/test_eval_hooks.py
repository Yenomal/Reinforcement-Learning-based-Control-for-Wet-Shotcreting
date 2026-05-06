import importlib
import importlib.util
import sys
from pathlib import Path

import torch


def _ensure_local_rl_robot_package() -> None:
    try:
        importlib.import_module("rl_robot")
        return
    except ModuleNotFoundError:
        pass

    package_root = Path(__file__).resolve().parents[2] / "src" / "rl_robot"
    spec = importlib.util.spec_from_file_location(
        "rl_robot",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load local rl_robot package for tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_robot"] = module
    spec.loader.exec_module(module)


_ensure_local_rl_robot_package()

from rl_robot.algorithms.lr_schedule import (  # noqa: E402
    CosineThenSuccessRateTriggeredScheduler,
    ScalarScheduler,
    SuccessRateTriggeredScheduler,
)
from rl_robot.training.eval_hooks import (  # noqa: E402
    PPOStdRuntimeController,
    build_ppo_std_scheduler,
    build_ppo_std_runtime_controller,
    resolve_deterministic_eval_config,
    restore_ppo_std_scheduler_state,
    run_deterministic_evaluation,
    serialize_ppo_std_scheduler_state,
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
            assert device.type == "cpu"
            assert episodes == 16
            assert seed == 123
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


def test_build_ppo_std_scheduler_returns_scheduler_for_each_mode() -> None:
    scheduler = build_ppo_std_scheduler(
        agent=None,
        ppo_std_cfg={
            "mode": "cosine_schedule",
            "global_log_std": -1.0,
            "schedule": {
                "schedule": "cosine",
                "start_log_std": -1.0,
                "end_log_std": -3.0,
            },
        },
        total_updates=20,
    )
    assert isinstance(scheduler, ScalarScheduler)

    scheduler = build_ppo_std_scheduler(
        agent=None,
        ppo_std_cfg={
            "mode": "success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 1.0,
                "patience_updates": 2,
                "cooldown_updates": 0,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=20,
    )
    assert isinstance(scheduler, SuccessRateTriggeredScheduler)

    scheduler = build_ppo_std_scheduler(
        agent=None,
        ppo_std_cfg={
            "mode": "cosine_then_success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "switch_update": 5,
            "schedule": {
                "schedule": "cosine",
                "start_log_std": -1.0,
                "end_log_std": -3.0,
            },
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 1.0,
                "patience_updates": 2,
                "cooldown_updates": 0,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=20,
    )
    assert isinstance(scheduler, CosineThenSuccessRateTriggeredScheduler)

    scheduler = build_ppo_std_scheduler(
        agent=None,
        ppo_std_cfg={
            "mode": "global_learned",
            "global_log_std": -1.0,
        },
        total_updates=20,
    )
    assert scheduler is None


class _FakeAgent:
    def __init__(self, log_std: float = -1.0) -> None:
        self.log_std = float(log_std)

    def set_log_std_value(self, value: float) -> None:
        self.log_std = float(value)

    def get_log_std_value(self) -> float:
        return float(self.log_std)

    def get_std_value(self) -> float:
        return float(torch.exp(torch.tensor(self.log_std)).item())


def test_stateful_ppo_std_scheduler_state_round_trip() -> None:
    scheduler = build_ppo_std_scheduler(
        agent=None,
        ppo_std_cfg={
            "mode": "success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 0.5,
                "patience_updates": 2,
                "cooldown_updates": 1,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=20,
    )
    assert isinstance(scheduler, SuccessRateTriggeredScheduler)
    scheduler.observe(0.95, episodes_in_window=32)
    scheduler.observe(0.95, episodes_in_window=32)

    state = serialize_ppo_std_scheduler_state(scheduler)

    restored = build_ppo_std_scheduler(
        agent=None,
        ppo_std_cfg={
            "mode": "success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 0.5,
                "patience_updates": 2,
                "cooldown_updates": 1,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=20,
    )
    assert isinstance(restored, SuccessRateTriggeredScheduler)
    restore_ppo_std_scheduler_state(restored, state)

    assert restored.current() == scheduler.current()
    assert restored.success_ema == scheduler.success_ema
    assert restored.streak == scheduler.streak
    assert restored.stage == scheduler.stage
    assert restored.cooldown_remaining == scheduler.cooldown_remaining


def test_runtime_controller_prefers_ppo_std_scheduler_over_legacy_exploration() -> None:
    controller = build_ppo_std_runtime_controller(
        agent=_FakeAgent(),
        ppo_cfg={
            "exploration_schedule": {
                "enable": True,
                "schedule": "cosine",
                "start_log_std": -1.0,
                "end_log_std": -3.0,
            }
        },
        ppo_std_cfg={
            "mode": "success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 1.0,
                "patience_updates": 2,
                "cooldown_updates": 0,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=20,
    )

    assert isinstance(controller, PPOStdRuntimeController)
    assert controller.ppo_std_scheduler is not None
    assert controller.legacy_exploration_scheduler is None


def test_runtime_controller_uses_legacy_exploration_when_no_ppo_std_scheduler() -> None:
    controller = build_ppo_std_runtime_controller(
        agent=_FakeAgent(),
        ppo_cfg={
            "exploration_schedule": {
                "enable": True,
                "schedule": "cosine",
                "start_log_std": -1.0,
                "end_log_std": -3.0,
            }
        },
        ppo_std_cfg={
            "mode": "global_learned",
            "global_log_std": -1.0,
        },
        total_updates=20,
    )

    assert controller.ppo_std_scheduler is None
    assert isinstance(controller.legacy_exploration_scheduler, ScalarScheduler)


def test_runtime_controller_serializes_and_restores_scheduler_state() -> None:
    agent = _FakeAgent()
    controller = build_ppo_std_runtime_controller(
        agent=agent,
        ppo_cfg={
            "exploration_schedule": {
                "enable": False,
            }
        },
        ppo_std_cfg={
            "mode": "cosine_then_success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "switch_update": 2,
            "schedule": {
                "schedule": "cosine",
                "start_log_std": -1.0,
                "end_log_std": -2.0,
            },
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 1.0,
                "patience_updates": 2,
                "cooldown_updates": 1,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=10,
    )

    controller.before_rollout(progress=2)
    metrics = {"success_rate": 0.95, "episodes_in_window": 32}
    controller.after_rollout(progress=2, metrics=metrics)
    controller.before_rollout(progress=3)
    controller.after_rollout(progress=3, metrics=metrics)
    state = controller.state_dict()

    restored = build_ppo_std_runtime_controller(
        agent=_FakeAgent(),
        ppo_cfg={
            "exploration_schedule": {
                "enable": False,
            }
        },
        ppo_std_cfg={
            "mode": "cosine_then_success_rate_triggered",
            "global_log_std": -1.0,
            "min_log_std": -4.0,
            "switch_update": 2,
            "schedule": {
                "schedule": "cosine",
                "start_log_std": -1.0,
                "end_log_std": -2.0,
            },
            "success_trigger": {
                "success_threshold": 0.9,
                "log_std_step": -0.1,
                "ema_alpha": 1.0,
                "patience_updates": 2,
                "cooldown_updates": 1,
                "min_episodes_in_window": 0,
            },
        },
        total_updates=10,
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
