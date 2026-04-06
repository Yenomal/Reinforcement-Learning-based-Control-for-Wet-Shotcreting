#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the project config from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return data
