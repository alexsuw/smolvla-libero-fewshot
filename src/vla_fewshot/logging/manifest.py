"""Training run identity and immutable manifest.json."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any, Mapping

from vla_fewshot.config import TrainConfig
from vla_fewshot.reproducibility import (
    _git_state,
    atomic_write_json,
    atomic_write_text,
    redact_text,
)
from vla_fewshot.storage.layout import MANIFEST_NAME, RESOLVED_CONFIG_NAME

import yaml


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_sha7(project_root: Path) -> str:
    state = _git_state(project_root)
    commit = state.get("commit")
    if not isinstance(commit, str) or len(commit) < 7:
        return "unknown"
    return commit[:7]


def suite_token(suite: str) -> str:
    return suite.replace("_", "")


def n_demos_token(config: TrainConfig) -> str:
    if config.dataset.episodes == "all":
        return "nall"
    if config.dataset.max_episodes is not None:
        return f"n{config.dataset.max_episodes:02d}"
    return "nsel"


def build_run_id(
    config: TrainConfig,
    *,
    project_root: Path,
    created_at: str | None = None,
) -> str:
    stamp = created_at or utc_timestamp()
    seed = f"s{config.training.seed}"
    task_or_suite = suite_token(config.dataset.suite)
    return "__".join(
        [
            config.stage,
            config.method,
            task_or_suite,
            n_demos_token(config),
            seed,
            stamp,
            f"g{git_sha7(project_root)}",
        ]
    )


def write_resolved_config(path: Path, config: TrainConfig) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True, allow_unicode=True),
        overwrite=path.exists(),
    )


def new_training_manifest(
    *,
    run_id: str,
    config: TrainConfig,
    command: list[str],
    project_root: Path,
    config_path: Path,
    profile: str,
) -> dict[str, Any]:
    git = _git_state(project_root)
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "stage": config.stage,
        "method": config.method,
        "status": "running",
        "profile": profile,
        "created_at_utc": now,
        "started_at_utc": now,
        "finished_at_utc": None,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "command": [redact_text(part) for part in command],
        "config_resolved_path": RESOLVED_CONFIG_NAME,
        "config_source_path": str(config_path),
        "dataset_repo_id": config.dataset.repo_id,
        "dataset_revision": config.dataset.revision,
        "suite": config.dataset.suite,
        "task_slug": None,
        "task_text": None,
        "episode_ids": [],
        "n_demos": config.dataset.max_episodes,
        "train_seed": config.training.seed,
        "base_checkpoint_uri": f"{config.model.repo_id}@{config.model.revision}",
        "base_checkpoint_sha256": config.model.revision,
        "model_revision": config.model.revision,
        "trainable_parameter_count": 0,
        "total_parameter_count": 0,
        "hardware": {
            "profile": profile,
            "device": "cpu" if profile == "static" else "cuda",
        },
        "versions": {
            "python": sys.version.split()[0],
            "checkpoint_format_version": 1,
        },
        "final_checkpoint_uri": None,
        "failure": None,
        "wandb_enabled": False,
        "resolved_precision": "fp32",
    }


def write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / MANIFEST_NAME, dict(manifest), overwrite=True)


def update_manifest(run_dir: Path, **fields: Any) -> dict[str, Any]:
    path = run_dir / MANIFEST_NAME
    current = json_load(path)
    current.update(fields)
    write_manifest(run_dir, current)
    return current


def json_load(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def mark_failed(run_dir: Path, error: BaseException) -> None:
    update_manifest(
        run_dir,
        status="failed",
        finished_at_utc=datetime.now(UTC).isoformat(),
        failure={"type": type(error).__name__, "message": redact_text(str(error))},
    )


def mark_interrupted(run_dir: Path) -> None:
    update_manifest(
        run_dir,
        status="interrupted",
        finished_at_utc=datetime.now(UTC).isoformat(),
    )


def mark_completed(run_dir: Path, *, final_checkpoint_uri: str) -> None:
    update_manifest(
        run_dir,
        status="completed",
        finished_at_utc=datetime.now(UTC).isoformat(),
        final_checkpoint_uri=final_checkpoint_uri,
    )
