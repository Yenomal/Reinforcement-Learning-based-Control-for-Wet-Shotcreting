from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterator


def _validate_relative_asset_path(relative_name: str) -> PurePosixPath:
    resource_path = PurePosixPath(relative_name)
    if relative_name == "" or resource_path.is_absolute() or ".." in resource_path.parts:
        raise ValueError(
            f"Asset path must be a relative package asset path: {relative_name}"
        )
    return resource_path


@contextmanager
def asset_path(relative_name: str) -> Iterator[Path]:
    resource_path = _validate_relative_asset_path(relative_name)
    resource = files("rl_robot.assets").joinpath(*resource_path.parts)
    if not resource.exists():
        raise FileNotFoundError(f"Packaged asset not found: {relative_name}")
    with as_file(resource) as resolved:
        yield Path(resolved)
