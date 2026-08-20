"""Platform-independent path resolution from environment variables."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from vla_fewshot.config import PlatformConfig


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
    durability_backend: str
    ephemeral: bool


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return Path(value).expanduser().resolve()


def _path_from_env_or_default(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _drive_is_durable(
    data_root: Path,
    mount_root: str | None,
) -> bool:
    if mount_root is None:
        return False
    mount = Path(mount_root).expanduser().resolve()
    try:
        data_root.relative_to(mount)
    except ValueError:
        return False
    return mount.exists() and os.path.ismount(mount)


def resolve_paths(platform: PlatformConfig | None = None) -> ProjectPaths:
    """Resolve runtime roots without embedding host-specific locations."""

    project_root = Path(os.environ.get("VLA_PROJECT_ROOT", Path.cwd())).resolve()
    if platform is None:
        return ProjectPaths(
            project_root=project_root,
            data_root=_required_path("VLA_DATA_ROOT"),
            datasets_dir=_required_path("VLA_DATASETS_DIR"),
            runs_dir=_required_path("VLA_RUNS_DIR"),
            checkpoints_dir=_required_path("VLA_CHECKPOINTS_DIR"),
            cache_dir=_required_path("VLA_CACHE_DIR"),
            scratch_dir=_required_path("VLA_SCRATCH_DIR"),
            object_uri=os.environ.get("VLA_OBJECT_URI"),
            durability_backend="environment",
            ephemeral=True,
        )

    data_root = _path_from_env_or_default(
        "VLA_DATA_ROOT",
        Path(platform.storage.data_root_default),
    )
    scratch_dir = _path_from_env_or_default(
        "VLA_SCRATCH_DIR",
        Path(platform.storage.scratch_root_default),
    )
    if platform.storage.durability_backend == "google_drive":
        durable = _drive_is_durable(
            data_root,
            platform.storage.drive_mount_root,
        )
    else:
        durable = platform.storage.durable and data_root != scratch_dir

    return ProjectPaths(
        project_root=project_root,
        data_root=data_root,
        datasets_dir=_path_from_env_or_default(
            "VLA_DATASETS_DIR",
            data_root / "datasets",
        ),
        runs_dir=_path_from_env_or_default("VLA_RUNS_DIR", data_root / "runs"),
        checkpoints_dir=_path_from_env_or_default(
            "VLA_CHECKPOINTS_DIR",
            data_root / "checkpoints",
        ),
        cache_dir=_path_from_env_or_default("VLA_CACHE_DIR", data_root / "cache"),
        scratch_dir=scratch_dir,
        object_uri=os.environ.get("VLA_OBJECT_URI"),
        durability_backend=platform.storage.durability_backend,
        ephemeral=not durable,
    )


def ensure_runtime_directories(paths: ProjectPaths) -> None:
    """Create only the exact configured roots; never remove existing content."""

    for path in (
        paths.data_root,
        paths.datasets_dir,
        paths.runs_dir,
        paths.checkpoints_dir,
        paths.cache_dir,
        paths.scratch_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def disk_free_gb(path: Path) -> float:
    """Return free space using the nearest existing ancestor."""

    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free / (1024**3)


def runtime_environment(paths: ProjectPaths) -> dict[str, str]:
    """Return non-secret environment values consumed by project commands."""

    return {
        "VLA_PROJECT_ROOT": str(paths.project_root),
        "VLA_DATA_ROOT": str(paths.data_root),
        "VLA_DATASETS_DIR": str(paths.datasets_dir),
        "VLA_RUNS_DIR": str(paths.runs_dir),
        "VLA_CHECKPOINTS_DIR": str(paths.checkpoints_dir),
        "VLA_CACHE_DIR": str(paths.cache_dir),
        "VLA_SCRATCH_DIR": str(paths.scratch_dir),
        "HF_HOME": str(paths.cache_dir / "huggingface"),
        "TORCH_HOME": str(paths.cache_dir / "torch"),
        "MUJOCO_GL": "egl",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    }
