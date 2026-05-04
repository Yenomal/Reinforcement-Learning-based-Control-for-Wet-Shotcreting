#!/usr/bin/env python3
"""Rock environment visualization script entrypoint."""

from __future__ import annotations

import argparse
from typing import Sequence

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from rl_robot.simulation.visualization.rock_env import run_rock_env_visualization


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the rock environment with Hydra config.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name inside rl_robot.conf.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides such as env.seed=0 env.n_theta=64.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))
    run_rock_env_visualization(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
