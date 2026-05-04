from rl_robot.evaluation.runner import build_device
from rl_robot.evaluation.renderer import EvalRenderer


def test_evaluation_module_exports_runtime_symbols() -> None:
    assert build_device("cpu").type == "cpu"
    assert EvalRenderer.__name__ == "EvalRenderer"
