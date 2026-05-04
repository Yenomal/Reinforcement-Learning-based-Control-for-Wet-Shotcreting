from pathlib import Path


def test_legacy_single_config_and_entrypoints_are_removed() -> None:
    assert not Path("src/config.py").exists()
    assert not Path("src/config.yaml").exists()
    assert not Path("src/train.py").exists()
    assert not Path("src/eval.py").exists()
