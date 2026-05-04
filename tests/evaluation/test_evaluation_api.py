from pathlib import Path

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from rl_robot.evaluation import run_evaluation
from rl_robot.evaluation import runner as evaluation_runner
from rl_robot.evaluation.renderer import EvalRenderer
from rl_robot.evaluation.runner import build_device, load_eval_surface_scene


def test_evaluation_module_exports_runtime_symbols() -> None:
    assert build_device("cpu").type == "cpu"
    assert EvalRenderer.__name__ == "EvalRenderer"
    assert callable(run_evaluation)


def test_run_evaluation_accepts_plain_dict(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}

    def fake_load_checkpoint(path: Path, device: object) -> dict[str, object]:
        captured["checkpoint_path"] = path
        captured["device"] = device
        return {"state_dict": {}, "config": {}}

    def fake_build_agent_from_checkpoint(
        checkpoint: dict[str, object],
        config: dict[str, object],
        device: object,
    ) -> object:
        captured["agent_config"] = config
        return object()

    def fake_build_action_scale_scheduler(config: dict[str, object]) -> None:
        captured["scheduler_config"] = config
        return None

    def fake_evaluate_episodes(
        config: dict[str, object],
        agent: object,
        device: object,
        action_scale_scheduler: object,
    ) -> None:
        captured["evaluate_config"] = config
        captured["evaluate_device"] = device
        captured["evaluate_scheduler"] = action_scale_scheduler

    monkeypatch.setattr(evaluation_runner, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        evaluation_runner,
        "build_agent_from_checkpoint",
        fake_build_agent_from_checkpoint,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "build_action_scale_scheduler",
        fake_build_action_scale_scheduler,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "evaluate_episodes",
        fake_evaluate_episodes,
    )

    config = {
        "train": {"device": "cpu"},
        "eval": {"checkpoint": str(checkpoint_path)},
    }

    run_evaluation(config)

    assert captured["checkpoint_path"] == checkpoint_path
    assert captured["device"] == captured["evaluate_device"]
    assert captured["agent_config"] == config
    assert captured["scheduler_config"] == config
    assert captured["evaluate_config"] == config
    assert captured["evaluate_scheduler"] is None
    assert "load_config" not in evaluation_runner.__dict__


def test_default_eval_config_loads_packaged_surface_scene() -> None:
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name="config")

    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)

    eval_cfg = dict(config["eval"])
    scene = load_eval_surface_scene(eval_cfg=eval_cfg, rock_env={})

    assert eval_cfg["digital_env_html"]
    assert scene["x_grid"].shape == scene["y_grid"].shape
    assert scene["x_grid"].shape == scene["z_grid"].shape
    assert scene["x_grid"].shape == scene["color_grid"].shape
