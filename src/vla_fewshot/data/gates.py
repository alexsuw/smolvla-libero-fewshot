"""Automatic no-leakage gate used by train and report commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vla_fewshot.data.cli import load_data_config, revision_root_from_args
from vla_fewshot.data.leakage import assert_no_leakage


def maybe_assert_no_leakage(
    *,
    config_path: Path,
    splits_path: Path,
    output_root: Path | None,
    stage: str,
    task_slug: str | None = None,
    extra_episode_ids: list[int] | None = None,
    extra_suite: str | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    """Fail closed when pinned metadata is present.

    Until a train/report command allocates compute, missing local metadata is
    skipped so CLI stubs stay non-mutating. Real training must pass
    ``required=True``.
    """

    try:
        data_config = load_data_config(config_path)
        root = revision_root_from_args(
            data_config=data_config,
            output_root=output_root,
        )
    except (RuntimeError, SystemExit, FileNotFoundError):
        if required:
            raise
        return None

    seen_info = root / "libero_90" / "meta" / "info.json"
    target_info = root / "libero_goal" / "meta" / "info.json"
    if not seen_info.exists() or not target_info.exists():
        if required:
            raise FileNotFoundError(
                f"pinned suite metadata is missing under {root}; "
                "run scripts/download_dataset.py first"
            )
        return None

    return assert_no_leakage(
        revision_root=root,
        data_config=data_config,
        splits_path=splits_path,
        stage=stage,
        task_slug=task_slug,
        extra_episode_ids=extra_episode_ids,
        extra_suite=extra_suite,
    )
