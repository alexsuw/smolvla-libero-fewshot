"""Fail-closed no-target-leakage checks for train and report commands."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import DataConfig
from vla_fewshot.data.expected import SEEN_SUITE, TARGET_SUITE, TARGET_TASKS
from vla_fewshot.data.metadata import load_suite_metadata
from vla_fewshot.data.splits import TargetSplits, load_target_splits
from vla_fewshot.data.task_text import task_text_matches


class LeakageError(RuntimeError):
    """Raised when a required leakage check fails."""


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "required": True,
        "detail": detail,
    }


def leakage_checks(
    *,
    revision_root: Path,
    data_config: DataConfig,
    splits: TargetSplits,
    stage: str | None = None,
    task_slug: str | None = None,
    extra_episode_ids: list[int] | None = None,
    extra_suite: str | None = None,
) -> dict[str, Any]:
    seen = load_suite_metadata(revision_root, SEEN_SUITE)
    target = load_suite_metadata(revision_root, TARGET_SUITE)
    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "dataset_revision",
            splits.dataset_revision == data_config.dataset.revision,
            (
                f"split={splits.dataset_revision}, "
                f"config={data_config.dataset.revision}"
            ),
        )
    )
    checks.append(
        _check(
            "dataset_repo",
            splits.dataset_repo_id == data_config.dataset.repo_id,
            f"split={splits.dataset_repo_id}, config={data_config.dataset.repo_id}",
        )
    )

    for slug, spec in TARGET_TASKS.items():
        text = str(spec["task_text"])
        in_seen = seen.contains_task_text(text)
        checks.append(
            _check(
                f"{slug}:absent_from_seen",
                not in_seen,
                f"present_in_{SEEN_SUITE}={in_seen}",
            )
        )
        tracked = splits.tasks[slug]
        checks.append(
            _check(
                f"{slug}:text_matches_config",
                task_text_matches(tracked.task_text, data_config.targets[slug].task_text),
                "split and data.yaml task texts match",
            )
        )
        ids = target.episode_ids_for_task(text)
        checks.append(
            _check(
                f"{slug}:ids_are_prefix",
                ids[:25] == tracked.episode_ids_first_25,
                "tracked first-25 IDs match metadata order",
            )
        )
        checks.append(
            _check(
                f"{slug}:ids_unique",
                len(set(tracked.episode_ids_first_25)) == 25,
                "first-25 episode IDs are unique",
            )
        )

    if extra_episode_ids:
        if extra_suite == TARGET_SUITE and stage == "seen":
            checks.append(
                _check(
                    "seen_training_excludes_target_episodes",
                    False,
                    "target episodes were supplied to a seen-stage run",
                )
            )
        if task_slug:
            allowed = set(splits.tasks[task_slug].episode_ids_first_25)
            unknown = sorted(set(extra_episode_ids) - allowed)
            checks.append(
                _check(
                    "target_ids_are_tracked_prefix",
                    not unknown,
                    f"untracked episode ids={unknown}",
                )
            )

    if stage == "seen" and extra_suite == TARGET_SUITE:
        checks.append(
            _check(
                "seen_suite_is_libero_90",
                False,
                f"{TARGET_SUITE} cannot be used for seen training",
            )
        )

    failed = [check for check in checks if check["status"] != "pass"]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "task_slug": task_slug,
        "checks": checks,
        "acceptance_complete": not failed,
    }


def assert_no_leakage(
    *,
    revision_root: Path,
    data_config: DataConfig,
    splits_path: Path,
    stage: str | None = None,
    task_slug: str | None = None,
    extra_episode_ids: list[int] | None = None,
    extra_suite: str | None = None,
) -> dict[str, Any]:
    report = leakage_checks(
        revision_root=revision_root,
        data_config=data_config,
        splits=load_target_splits(splits_path),
        stage=stage,
        task_slug=task_slug,
        extra_episode_ids=extra_episode_ids,
        extra_suite=extra_suite,
    )
    if not report["acceptance_complete"]:
        failed = [
            check["name"] for check in report["checks"] if check["status"] != "pass"
        ]
        raise LeakageError(f"target leakage checks failed: {failed}")
    return report
