"""Revision-safe local dataset layout. Paths never omit the full SHA."""

from __future__ import annotations

import os
from pathlib import Path


METADATA_ALLOW_PATTERNS = (
    "README.md",
    ".gitattributes",
    "libero_90/meta/**",
    "libero_goal/meta/**",
)

VIDEO_IGNORE_PATTERNS = ("**/*.mp4", "**/*.avi", "**/*.mov", "**/*.swp", "**/Untitled")


def repo_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "_")


def dataset_revision_root(datasets_dir: Path, repo_id: str, revision: str) -> Path:
    """Return `<datasets>/<repo_id>/<40-char revision>/`."""

    if len(revision) != 40:
        raise ValueError(f"dataset revision must be a 40-character SHA, got {revision!r}")
    return Path(datasets_dir).expanduser().resolve() / repo_dirname(repo_id) / revision


def suite_root(revision_root: Path, suite: str) -> Path:
    return Path(revision_root) / suite


def metadata_root(revision_root: Path, suite: str) -> Path:
    return suite_root(revision_root, suite) / "meta"


def resolve_datasets_dir(output_root: str | Path | None = None) -> Path:
    """Resolve the datasets root from a CLI override or VLA_DATASETS_DIR."""

    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    env = os.environ.get("VLA_DATASETS_DIR")
    if not env:
        raise RuntimeError(
            "set --output-root or VLA_DATASETS_DIR; dataset files must stay "
            "outside the Git worktree. no GPU training was started."
        )
    return Path(env).expanduser().resolve()


def metadata_allow_patterns(*suites: str) -> list[str]:
    if not suites:
        return list(METADATA_ALLOW_PATTERNS)
    patterns = ["README.md", ".gitattributes"]
    for suite in suites:
        patterns.append(f"{suite}/meta/**")
    return patterns
