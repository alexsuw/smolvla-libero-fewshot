"""Platform-independent path resolution from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    datasets_dir: Path
    runs_dir: Path
    checkpoints_dir: Path
    cache_dir: Path
    scratch_dir: Path
    object_uri: str | None


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return Path(value).expanduser().resolve()


def resolve_paths() -> ProjectPaths:
    """Resolve runtime roots without embedding host-specific locations."""

    project_root = Path(os.environ.get("VLA_PROJECT_ROOT", Path.cwd())).resolve()
    return ProjectPaths(
        project_root=project_root,
        data_root=_required_path("VLA_DATA_ROOT"),
        datasets_dir=_required_path("VLA_DATASETS_DIR"),
        runs_dir=_required_path("VLA_RUNS_DIR"),
        checkpoints_dir=_required_path("VLA_CHECKPOINTS_DIR"),
        cache_dir=_required_path("VLA_CACHE_DIR"),
        scratch_dir=_required_path("VLA_SCRATCH_DIR"),
        object_uri=os.environ.get("VLA_OBJECT_URI"),
    )
