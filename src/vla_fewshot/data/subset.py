"""Logical episode subsets with nested few-shot prefixes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.data.expected import PREFIX_BUDGETS, TARGET_SUITE
from vla_fewshot.data.splits import TargetSplits, load_target_splits
from vla_fewshot.reproducibility import atomic_write_json


SUBSET_MANIFEST = "subset_manifest.json"


def nested_ids(ids_25: list[int], n_demos: int) -> list[int]:
    if n_demos not in PREFIX_BUDGETS:
        raise ValueError(f"n_demos must be one of {PREFIX_BUDGETS}")
    ids = ids_25[:n_demos]
    if n_demos >= 2:
        assert ids[:1] == ids_25[:1]
    if n_demos >= 5:
        assert ids[:2] == ids_25[:2]
    if n_demos >= 10:
        assert ids[:5] == ids_25[:5]
    if n_demos == 25:
        assert ids[:10] == ids_25[:10]
        assert len(set(ids)) == 25
    return ids


def subset_identity(
    *,
    repo_id: str,
    revision: str,
    task_slug: str,
    n_demos: int,
    episode_ids: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "logical",
        "dataset_repo_id": repo_id,
        "dataset_revision": revision,
        "suite": TARGET_SUITE,
        "task_slug": task_slug,
        "n_demos": n_demos,
        "episode_ids": episode_ids,
        "parent_revision": revision,
        "copies_videos": False,
        "stats_scope": "selected_train_episodes_only",
    }


def materialize_logical_subset(
    *,
    output_dir: Path,
    repo_id: str,
    revision: str,
    task_slug: str,
    n_demos: int,
    splits: TargetSplits,
) -> dict[str, Any]:
    """Write an immutable logical subset manifest. Videos are not copied."""

    if output_dir.exists():
        existing_path = output_dir / SUBSET_MANIFEST
        if not existing_path.exists():
            raise FileExistsError(f"{output_dir} exists without a subset manifest")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        expected_ids = nested_ids(splits.tasks[task_slug].episode_ids_first_25, n_demos)
        if existing.get("episode_ids") != expected_ids:
            raise FileExistsError(
                f"{output_dir} already stores a different subset; refusing to overwrite"
            )
        return existing

    episode_ids = nested_ids(splits.tasks[task_slug].episode_ids_first_25, n_demos)
    manifest = subset_identity(
        repo_id=repo_id,
        revision=revision,
        task_slug=task_slug,
        n_demos=n_demos,
        episode_ids=episode_ids,
    )
    manifest["created_at_utc"] = datetime.now(UTC).isoformat()
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / SUBSET_MANIFEST, manifest)
    return manifest


def load_or_create_logical_subset(
    *,
    output_dir: Path,
    splits_path: Path,
    repo_id: str,
    revision: str,
    task_slug: str,
    n_demos: int,
) -> dict[str, Any]:
    return materialize_logical_subset(
        output_dir=output_dir,
        repo_id=repo_id,
        revision=revision,
        task_slug=task_slug,
        n_demos=n_demos,
        splits=load_target_splits(splits_path),
    )
