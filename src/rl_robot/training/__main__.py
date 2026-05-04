"""CLI wrapper for training package entry."""

from __future__ import annotations

import argparse

from src.config import load_config

from . import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on the math environment.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
