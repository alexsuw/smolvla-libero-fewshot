"""Shared bootstrap implementation for Colab and Linux GPU VMs."""

from __future__ import annotations

import importlib.util
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from vla_fewshot.config import PlatformConfig, RevisionsConfig, load_config
from vla_fewshot.paths import (
    ProjectPaths,
    ensure_runtime_directories,
    resolve_paths,
    runtime_environment,
)
from vla_fewshot.reproducibility import (
    atomic_write_json,
    atomic_write_text,
    write_environment_bundle,
)
from vla_fewshot.revisions import validate_revisions


def prepare_libero_assets(
    *,
    paths: ProjectPaths,
    revisions: RevisionsConfig,
) -> Path:
    """Download pinned assets and create LIBERO's config without prompting."""

    from huggingface_hub import snapshot_download

    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("hf-libero is not installed")
    libero_dir = Path(spec.origin).resolve().parent / "libero"
    asset_dir = (
        paths.cache_dir
        / "libero-assets"
        / revisions.source.libero_assets_revision
    )
    snapshot_download(
        repo_id=revisions.source.libero_assets_repo_id,
        repo_type="dataset",
        revision=revisions.source.libero_assets_revision,
        local_dir=asset_dir,
    )
    expected = {
        "assets": str(asset_dir),
        "bddl_files": str(libero_dir / "bddl_files"),
        "datasets": str(libero_dir.parent / "datasets"),
        "init_states": str(libero_dir / "init_files"),
    }
    config_path = Path.home() / ".libero" / "config.yaml"
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"{config_path} already exists with different paths; "
                "refusing to overwrite it"
            )
    else:
        atomic_write_text(
            config_path,
            yaml.safe_dump(expected, sort_keys=True),
        )
    return config_path


def bootstrap_environment(
    *,
    platform_config_path: Path,
    revisions_config_path: Path,
    lock_path: Path,
    output_dir: Path | None,
    command: list[str] | None = None,
) -> tuple[Path, bool]:
    platform_config = load_config(platform_config_path)
    revisions = load_config(revisions_config_path)
    if not isinstance(platform_config, PlatformConfig):
        raise TypeError(f"{platform_config_path} is not a platform config")
    if not isinstance(revisions, RevisionsConfig):
        raise TypeError(f"{revisions_config_path} is not a revisions config")

    paths = resolve_paths(platform_config)
    ensure_runtime_directories(paths)
    environment = runtime_environment(paths)
    os.environ.update(environment)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir or paths.data_root / "bootstrap" / timestamp

    prepare_libero_assets(paths=paths, revisions=revisions)
    upstream = validate_revisions(
        revisions=revisions,
        lock_path=lock_path,
        require_installed=True,
        check_remote=True,
    )
    write_environment_bundle(
        output_dir=destination,
        paths=paths,
        platform_config=platform_config,
        revisions=revisions,
        upstream_revisions=upstream,
        command=command or sys.argv,
    )
    env_lines = [
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(environment.items())
    ]
    atomic_write_text(destination / "runtime.env", "\n".join(env_lines) + "\n")
    result = {
        "schema_version": 1,
        "output_dir": str(destination),
        "platform": platform_config.name,
        "ephemeral": paths.ephemeral,
        "revisions_valid": upstream["acceptance_complete"],
        "acceptance_complete": (
            upstream["acceptance_complete"]
            and (not platform_config.storage.require_verified_backup or not paths.ephemeral)
        ),
        "next_command": f"source {destination / 'runtime.env'}",
    }
    atomic_write_json(destination / "bootstrap_result.json", result)
    return destination, bool(result["acceptance_complete"])
