from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator


@contextmanager
def asset_path(relative_name: str) -> Iterator[Path]:
    resource = files("rl_robot.assets").joinpath(relative_name)
    if not resource.exists():
        raise FileNotFoundError(f"Packaged asset not found: {relative_name}")
    with as_file(resource) as resolved:
        yield Path(resolved)
