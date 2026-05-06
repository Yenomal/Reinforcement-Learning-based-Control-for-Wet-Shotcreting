import importlib
import importlib.util
import math
import sys
import copy
from pathlib import Path

import pytest
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

lr_schedule_module = importlib.import_module("rl_robot.algorithms.lr_schedule")
ppo_module = importlib.import_module("rl_robot.algorithms.ppo")

CosineThenSuccessRateTriggeredScheduler = (
    lr_schedule_module.CosineThenSuccessRateTriggeredScheduler
)
ScalarScheduler = lr_schedule_module.ScalarScheduler
SuccessRateTriggeredScheduler = lr_schedule_module.SuccessRateTriggeredScheduler
PPOAgent = ppo_module.PPOAgent
build_ppo_config = ppo_module.build_ppo_config
resolve_ppo_std_config = ppo_module.resolve_ppo_std_config


def _build_agent(*, std_mode: str) -> PPOAgent:
    return PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg={
            "std": {
                "mode": std_mode,
                "global_log_std": -1.0,
                "min_log_std": -10.0,
            }
        },
        device=torch.device("cpu"),
    )


def test_success_rate_triggered_scheduler_applies_patience_and_cooldown() -> None:
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

    first_metrics = scheduler.observe(0.95, episodes_in_window=64)
    second_metrics = scheduler.observe(0.95, episodes_in_window=64)
    cooldown_metrics = scheduler.observe(0.95, episodes_in_window=64)
    resumed_metrics = scheduler.observe(0.95, episodes_in_window=64)

    assert first_metrics["ppo_next_log_std"] == pytest.approx(-3.0)
    assert second_metrics["ppo_next_log_std"] == pytest.approx(-3.1)
    assert second_metrics["ppo_std_cooldown_remaining"] == pytest.approx(1.0)
    assert cooldown_metrics["ppo_std_streak"] == pytest.approx(0.0)
    assert cooldown_metrics["ppo_std_cooldown_remaining"] == pytest.approx(0.0)
    assert resumed_metrics["ppo_std_streak"] == pytest.approx(1.0)


def test_success_rate_triggered_scheduler_ignores_short_windows_before_ema_update() -> None:
    scheduler = SuccessRateTriggeredScheduler(
        start_value=-3.0,
        min_value=-4.0,
        success_threshold=0.8,
        value_step=-0.1,
        ema_alpha=0.5,
        patience_updates=2,
        cooldown_updates=1,
        min_episodes_in_window=10,
    )

    ignored_metrics = scheduler.observe(1.0, episodes_in_window=4)
    first_valid_metrics = scheduler.observe(0.81, episodes_in_window=10)
    second_valid_metrics = scheduler.observe(0.81, episodes_in_window=10)

    assert math.isnan(ignored_metrics["ppo_success_ema"])
    assert ignored_metrics["ppo_std_streak"] == pytest.approx(0.0)
    assert ignored_metrics["ppo_std_cooldown_remaining"] == pytest.approx(0.0)
    assert first_valid_metrics["ppo_success_ema"] == pytest.approx(0.81)
    assert first_valid_metrics["ppo_std_streak"] == pytest.approx(1.0)
    assert second_valid_metrics["ppo_next_log_std"] == pytest.approx(-3.1)


def test_cosine_then_trigger_scheduler_waits_until_switch_update() -> None:
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
        patience_updates=2,
        cooldown_updates=0,
        min_episodes_in_window=0,
    )
    scheduler = CosineThenSuccessRateTriggeredScheduler(
        cosine_scheduler=cosine,
        trigger_scheduler=trigger,
        switch_update=5,
    )
    handoff_value = cosine.value(4)

    pre_switch_metrics = scheduler.observe(
        0.95,
        progress=2,
        episodes_in_window=32,
    )
    first_trigger_metrics = scheduler.observe(
        0.95,
        progress=5,
        episodes_in_window=32,
    )
    second_trigger_metrics = scheduler.observe(
        0.95,
        progress=6,
        episodes_in_window=32,
    )

    assert pre_switch_metrics["ppo_std_phase"] == "cosine"
    assert math.isnan(pre_switch_metrics["ppo_success_ema"])
    assert pre_switch_metrics["ppo_next_log_std"] == pytest.approx(cosine.value(2))
    assert first_trigger_metrics["ppo_std_phase"] == "trigger"
    assert first_trigger_metrics["ppo_success_ema"] == pytest.approx(0.95)
    assert first_trigger_metrics["ppo_std_streak"] == pytest.approx(1.0)
    assert first_trigger_metrics["ppo_next_log_std"] == pytest.approx(handoff_value)
    assert second_trigger_metrics["ppo_next_log_std"] == pytest.approx(handoff_value - 0.1)


def test_cosine_then_trigger_scheduler_requires_progress_for_observe() -> None:
    scheduler = CosineThenSuccessRateTriggeredScheduler(
        cosine_scheduler=ScalarScheduler(
            start_value=-1.0,
            end_value=-3.0,
            total_progress=10,
            schedule="cosine",
        ),
        trigger_scheduler=SuccessRateTriggeredScheduler(
            start_value=-3.0,
            min_value=-4.0,
            success_threshold=0.9,
            value_step=-0.1,
            ema_alpha=1.0,
            patience_updates=2,
            cooldown_updates=0,
            min_episodes_in_window=0,
        ),
        switch_update=5,
    )

    with pytest.raises(ValueError, match="progress"):
        scheduler.observe(0.95, episodes_in_window=32)


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


def test_resolve_ppo_std_config_keeps_partial_global_log_std_override_in_sync() -> None:
    algorithm_cfg = {
        "std": {
            "mode": "cosine_schedule",
            "global_log_std": -2.0,
        }
    }
    std_cfg = resolve_ppo_std_config(algorithm_cfg)
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=algorithm_cfg,
        device=torch.device("cpu"),
    )

    assert std_cfg["global_log_std"] == pytest.approx(-2.0)
    assert std_cfg["schedule"]["start_log_std"] == pytest.approx(-2.0)
    assert agent.get_log_std_value() == pytest.approx(-2.0)


def test_resolve_ppo_std_config_keeps_partial_schedule_start_override_in_sync() -> None:
    algorithm_cfg = {
        "std": {
            "mode": "cosine_schedule",
            "schedule": {
                "start_log_std": -2.25,
            },
        }
    }
    std_cfg = resolve_ppo_std_config(algorithm_cfg)
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=algorithm_cfg,
        device=torch.device("cpu"),
    )

    assert std_cfg["schedule"]["start_log_std"] == pytest.approx(-2.25)
    assert std_cfg["global_log_std"] == pytest.approx(-2.25)
    assert agent.get_log_std_value() == pytest.approx(-2.25)


def test_resolve_ppo_std_config_locks_parallel_default_baseline() -> None:
    ppo_cfg = build_ppo_config()
    std_cfg = resolve_ppo_std_config()
    expected = {
        "global_log_std": -1.0,
        "min_log_std": -10.0,
        "switch_update": 150,
        "schedule_end_log_std": -3.67,
        "success_trigger": {
            "success_threshold": 0.90,
            "log_std_step": -0.01,
            "ema_alpha": 1.0,
            "patience_updates": 20,
            "cooldown_updates": 0,
            "min_episodes_in_window": 0,
        },
    }

    assert ppo_cfg["init_log_std"] == pytest.approx(expected["global_log_std"])
    assert ppo_cfg["exploration_schedule"]["start_log_std"] == pytest.approx(
        expected["global_log_std"]
    )
    assert ppo_cfg["exploration_schedule"]["end_log_std"] == pytest.approx(
        expected["schedule_end_log_std"]
    )
    assert std_cfg["global_log_std"] == pytest.approx(expected["global_log_std"])
    assert std_cfg["min_log_std"] == pytest.approx(expected["min_log_std"])
    assert std_cfg["switch_update"] == expected["switch_update"]
    assert std_cfg["schedule"]["end_log_std"] == pytest.approx(
        expected["schedule_end_log_std"]
    )
    assert std_cfg["success_trigger"] == expected["success_trigger"]


@pytest.mark.parametrize(
    ("raw_cfg", "expected_mode", "expected_log_std"),
    [
        ({"init_log_std": -2.75}, "cosine_schedule", -2.75),
        (
            {"exploration_schedule": {"start_log_std": -2.5}},
            "cosine_schedule",
            -2.5,
        ),
        (
            {
                "init_log_std": -2.75,
                "exploration_schedule": {"start_log_std": -2.5},
            },
            "cosine_schedule",
            -2.5,
        ),
        (
            {"exploration_schedule": {"enable": False}},
            "global_learned",
            -1.0,
        ),
    ],
)
def test_legacy_std_inputs_stay_consistent_across_build_resolve_and_agent(
    raw_cfg: dict[str, object],
    expected_mode: str,
    expected_log_std: float,
) -> None:
    built_cfg = build_ppo_config(raw_cfg)
    std_from_raw = resolve_ppo_std_config(raw_cfg)
    std_from_built = resolve_ppo_std_config(built_cfg)
    agent_from_raw = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=raw_cfg,
        device=torch.device("cpu"),
    )
    agent_from_built = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=built_cfg,
        device=torch.device("cpu"),
    )

    assert built_cfg["std"]["mode"] == expected_mode
    assert built_cfg["init_log_std"] == pytest.approx(expected_log_std)
    assert built_cfg["std"]["global_log_std"] == pytest.approx(expected_log_std)
    assert std_from_raw["mode"] == expected_mode
    assert std_from_built["mode"] == expected_mode
    assert std_from_raw["global_log_std"] == pytest.approx(expected_log_std)
    assert std_from_built["global_log_std"] == pytest.approx(expected_log_std)
    assert agent_from_raw.std_mode == expected_mode
    assert agent_from_built.std_mode == expected_mode
    assert agent_from_raw.get_log_std_value() == pytest.approx(expected_log_std)
    assert agent_from_built.get_log_std_value() == pytest.approx(expected_log_std)
    if expected_mode == "global_learned":
        assert agent_from_raw.log_std.requires_grad is True
        assert agent_from_built.log_std.requires_grad is True


def test_resolve_ppo_std_config_keeps_legacy_init_log_std_override() -> None:
    std_cfg = resolve_ppo_std_config({"init_log_std": -2.75})
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg={"init_log_std": -2.75},
        device=torch.device("cpu"),
    )

    assert std_cfg["global_log_std"] == pytest.approx(-2.75)
    assert agent.get_log_std_value() == pytest.approx(-2.75)


def test_resolve_ppo_std_config_keeps_legacy_schedule_start_log_std_in_sync() -> None:
    algorithm_cfg = {
        "exploration_schedule": {
            "enable": True,
            "start_log_std": -2.5,
        }
    }
    std_cfg = resolve_ppo_std_config(algorithm_cfg)
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=algorithm_cfg,
        device=torch.device("cpu"),
    )

    assert std_cfg["global_log_std"] == pytest.approx(-2.5)
    assert std_cfg["schedule"]["start_log_std"] == pytest.approx(-2.5)
    assert agent.get_log_std_value() == pytest.approx(-2.5)


def test_legacy_schedule_disable_resolves_to_global_learned() -> None:
    algorithm_cfg = {
        "exploration_schedule": {
            "enable": False,
        }
    }
    std_cfg = resolve_ppo_std_config(algorithm_cfg)
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=algorithm_cfg,
        device=torch.device("cpu"),
    )

    assert std_cfg["mode"] == "global_learned"
    assert agent.std_mode == "global_learned"
    assert agent.log_std.requires_grad is True


def test_build_ppo_config_legacy_disable_updates_std_mode() -> None:
    ppo_cfg = build_ppo_config(
        {
            "exploration_schedule": {
                "enable": False,
            }
        }
    )

    assert ppo_cfg["exploration_schedule"]["enable"] is False
    assert ppo_cfg["std"]["mode"] == "global_learned"


def test_build_ppo_config_is_idempotent_for_legacy_disable() -> None:
    raw_cfg = {
        "exploration_schedule": {
            "enable": False,
        }
    }

    first = build_ppo_config(raw_cfg)
    second = build_ppo_config(first)

    assert second["std"] == first["std"]
    assert second["exploration_schedule"] == first["exploration_schedule"]


def test_legacy_schedule_partial_override_inherits_default_enable() -> None:
    algorithm_cfg = {
        "exploration_schedule": {
            "start_log_std": -2.5,
        }
    }
    ppo_cfg = build_ppo_config(algorithm_cfg)
    std_cfg = resolve_ppo_std_config(algorithm_cfg)
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=algorithm_cfg,
        device=torch.device("cpu"),
    )

    assert ppo_cfg["exploration_schedule"]["enable"] is True
    assert ppo_cfg["init_log_std"] == pytest.approx(-2.5)
    assert std_cfg["mode"] == "cosine_schedule"
    assert std_cfg["global_log_std"] == pytest.approx(-2.5)
    assert std_cfg["schedule"]["start_log_std"] == pytest.approx(-2.5)
    assert agent.std_mode == "cosine_schedule"
    assert agent.get_log_std_value() == pytest.approx(-2.5)


def test_build_ppo_config_legacy_start_override_updates_std_global_log_std() -> None:
    ppo_cfg = build_ppo_config(
        {
            "exploration_schedule": {
                "start_log_std": -2.5,
            }
        }
    )

    assert ppo_cfg["exploration_schedule"]["enable"] is True
    assert ppo_cfg["std"]["mode"] == "cosine_schedule"
    assert ppo_cfg["std"]["global_log_std"] == pytest.approx(-2.5)
    assert ppo_cfg["std"]["schedule"]["start_log_std"] == pytest.approx(-2.5)


def test_build_ppo_config_legacy_schedule_fields_sync_into_std_schedule() -> None:
    raw_cfg = {
        "exploration_schedule": {
            "schedule": "cosine",
            "start_log_std": -2.5,
            "end_log_std": -3.0,
        }
    }
    ppo_cfg = build_ppo_config(raw_cfg)

    assert ppo_cfg["std"]["mode"] == "cosine_schedule"
    assert ppo_cfg["std"]["global_log_std"] == pytest.approx(-2.5)
    assert ppo_cfg["std"]["schedule"]["schedule"] == "cosine"
    assert ppo_cfg["std"]["schedule"]["start_log_std"] == pytest.approx(-2.5)
    assert ppo_cfg["std"]["schedule"]["end_log_std"] == pytest.approx(-3.0)


def test_agent_std_init_is_idempotent_for_built_legacy_schedule_config() -> None:
    raw_cfg = {
        "exploration_schedule": {
            "start_log_std": -2.5,
        }
    }
    built_cfg = build_ppo_config(raw_cfg)
    agent_from_raw = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=raw_cfg,
        device=torch.device("cpu"),
    )
    agent_from_built = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=built_cfg,
        device=torch.device("cpu"),
    )

    assert built_cfg["init_log_std"] == pytest.approx(-2.5)
    assert agent_from_raw.get_log_std_value() == pytest.approx(-2.5)
    assert agent_from_built.get_log_std_value() == pytest.approx(-2.5)


def test_build_ppo_config_is_idempotent_for_legacy_schedule_overrides() -> None:
    raw_cfg = {
        "exploration_schedule": {
            "schedule": "cosine",
            "start_log_std": -2.5,
            "end_log_std": -3.0,
        }
    }

    first = build_ppo_config(raw_cfg)
    second = build_ppo_config(first)

    assert second["init_log_std"] == pytest.approx(first["init_log_std"])
    assert second["std"] == first["std"]
    assert second["exploration_schedule"] == first["exploration_schedule"]


def test_build_ppo_config_disables_legacy_exploration_when_std_is_explicit() -> None:
    ppo_cfg = build_ppo_config(
        {
            "std": {
                "mode": "global_fixed",
                "global_log_std": -2.0,
            }
        }
    )

    assert ppo_cfg["std"]["mode"] == "global_fixed"
    assert ppo_cfg["std"]["global_log_std"] == pytest.approx(-2.0)
    assert ppo_cfg["exploration_schedule"]["enable"] is False
    assert ppo_cfg["exploration_schedule"]["start_log_std"] == pytest.approx(-2.0)
    assert ppo_cfg["exploration_schedule"]["end_log_std"] == pytest.approx(-2.0)


def test_build_ppo_config_syncs_public_legacy_schedule_with_explicit_std_start() -> None:
    ppo_cfg = build_ppo_config(
        {
            "std": {
                "mode": "cosine_schedule",
                "global_log_std": -2.0,
            }
        }
    )

    assert ppo_cfg["std"]["global_log_std"] == pytest.approx(-2.0)
    assert ppo_cfg["std"]["schedule"]["start_log_std"] == pytest.approx(-2.0)
    assert ppo_cfg["exploration_schedule"]["start_log_std"] == pytest.approx(-2.0)
    assert ppo_cfg["exploration_schedule"]["end_log_std"] == pytest.approx(
        ppo_cfg["std"]["schedule"]["end_log_std"]
    )


def test_explicit_std_round_trip_stays_stable_across_build_resolve_and_agent() -> None:
    raw_cfg = {
        "std": {
            "mode": "global_fixed",
            "global_log_std": -2.0,
        }
    }

    first = build_ppo_config(raw_cfg)
    second = build_ppo_config(first)
    resolved = resolve_ppo_std_config(first)
    agent = PPOAgent(
        observation_dim=3,
        action_dim=2,
        model_cfg={
            "type": "plain_mlp",
            "hidden_sizes": [8],
            "activation": "tanh",
        },
        algorithm_cfg=first,
        device=torch.device("cpu"),
    )

    assert first["std"]["mode"] == "global_fixed"
    assert first["std"]["global_log_std"] == pytest.approx(-2.0)
    assert second["std"] == first["std"]
    assert second["exploration_schedule"] == first["exploration_schedule"]
    assert resolved["mode"] == "global_fixed"
    assert resolved["global_log_std"] == pytest.approx(-2.0)
    assert agent.std_mode == "global_fixed"
    assert agent.get_log_std_value() == pytest.approx(-2.0)


def test_build_ppo_config_keeps_init_log_std_in_sync_with_schedule_start_override() -> None:
    ppo_cfg = build_ppo_config(
        {
            "std": {
                "mode": "cosine_schedule",
                "schedule": {
                    "start_log_std": -2.4,
                },
            }
        }
    )

    assert ppo_cfg["std"]["schedule"]["start_log_std"] == pytest.approx(-2.4)
    assert ppo_cfg["std"]["global_log_std"] == pytest.approx(-2.4)
    assert ppo_cfg["init_log_std"] == pytest.approx(-2.4)


def test_cosine_then_trigger_scheduler_clamps_handoff_to_min_value() -> None:
    scheduler = CosineThenSuccessRateTriggeredScheduler(
        cosine_scheduler=ScalarScheduler(
            start_value=-1.0,
            end_value=-5.0,
            total_progress=10,
            schedule="cosine",
        ),
        trigger_scheduler=SuccessRateTriggeredScheduler(
            start_value=-2.0,
            min_value=-2.0,
            success_threshold=0.9,
            value_step=-0.1,
            ema_alpha=1.0,
            patience_updates=2,
            cooldown_updates=0,
            min_episodes_in_window=0,
        ),
        switch_update=10,
    )

    trigger_state = scheduler.state(10)

    assert trigger_state["ppo_std_phase"] == "trigger"
    assert trigger_state["ppo_next_log_std"] == pytest.approx(-2.0)


def test_load_training_state_skips_incompatible_optimizer_state_across_std_modes() -> None:
    learned_agent = _build_agent(std_mode="global_learned")
    fixed_agent = _build_agent(std_mode="global_fixed")

    with torch.no_grad():
        learned_agent.log_std.fill_(-2.5)

    learned_agent.optimizer.zero_grad()
    learned_agent.log_std.sum().backward()
    learned_agent.optimizer.step()
    checkpoint = learned_agent.state_dict()
    expected_log_std = float(checkpoint["log_std"].mean().item())

    fixed_agent.load_training_state(checkpoint)

    assert fixed_agent.get_log_std_value() == pytest.approx(expected_log_std)
    assert fixed_agent.optimizer.state_dict()["state"] == {}


def test_load_training_state_allows_compatible_optimizer_across_fixed_std_modes() -> None:
    source_agent = _build_agent(std_mode="global_fixed")
    target_agent = _build_agent(std_mode="success_rate_triggered")

    source_agent.optimizer.zero_grad()
    next(iter(source_agent.actor.parameters())).sum().backward()
    source_agent.optimizer.step()
    checkpoint = source_agent.state_dict()

    target_agent.load_training_state(checkpoint)

    assert (
        target_agent.optimizer.state_dict()["param_groups"]
        == source_agent.optimizer.state_dict()["param_groups"]
    )
    assert (
        target_agent.optimizer.state_dict()["state"].keys()
        == source_agent.optimizer.state_dict()["state"].keys()
    )


def test_load_training_state_accepts_legacy_global_learned_optimizer_order() -> None:
    source_agent = _build_agent(std_mode="global_learned")
    target_agent = _build_agent(std_mode="global_learned")

    source_agent.optimizer.zero_grad()
    source_agent.log_std.sum().backward()
    source_agent.optimizer.step()

    checkpoint = source_agent.state_dict()
    legacy_checkpoint = copy.deepcopy(checkpoint)
    legacy_checkpoint.pop("std_mode", None)
    source_optimizer_state = source_agent.optimizer.state_dict()
    param_count = len(source_optimizer_state["param_groups"][0]["params"])
    assert set(source_optimizer_state["state"].keys()) == {param_count - 1}

    target_agent.load_training_state(legacy_checkpoint)

    assert target_agent.get_log_std_value() == pytest.approx(
        source_agent.get_log_std_value()
    )
    assert (
        target_agent.optimizer.state_dict()["param_groups"]
        == source_agent.optimizer.state_dict()["param_groups"]
    )
    assert (
        target_agent.optimizer.state_dict()["state"].keys()
        == source_agent.optimizer.state_dict()["state"].keys()
    )
