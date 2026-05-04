#!/usr/bin/env python3
"""Reachability map build script entrypoint."""

from __future__ import annotations

import argparse
from typing import Sequence

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from rl_robot.planning.reachability_map import run_reachability_map


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the planner reachability map.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name inside rl_robot.conf.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional compute device override, such as cpu or cuda.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding the cached reachability map.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with initialize_config_module(version_base=None, config_module="rl_robot.conf"):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))
    run_reachability_map(
        OmegaConf.to_container(cfg, resolve=True),
        requested_device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
