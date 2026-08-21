"""Pinned, resume-safe dataset download. Never overwrites another revision."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi

from vla_fewshot.data.layout import (
    VIDEO_IGNORE_PATTERNS,
    dataset_revision_root,
    metadata_allow_patterns,
)
from vla_fewshot.reproducibility import atomic_write_json, redact_text


MANIFEST_NAME = "download_manifest.json"


def _snapshot_download(**kwargs: Any) -> Any:
    """Lazy Hub import so `--help` works without the data extra."""

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required for dataset download. "
            "Install with: uv sync --frozen --extra data"
        ) from error
    return snapshot_download(**kwargs)


def _existing_revision(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        revision_path = root / "REVISION"
        if revision_path.exists():
            return revision_path.read_text(encoding="utf-8").strip()
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = payload.get("dataset_revision")
    return str(revision) if revision else None


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".cache/"):
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
            }
        )
    return records


def download_dataset(
    *,
    repo_id: str,
    revision: str,
    datasets_dir: Path,
    suites: tuple[str, ...] = ("libero_90", "libero_goal"),
    metadata_only: bool = True,
    include_videos: bool = False,
    include_actions: bool = False,
) -> dict[str, Any]:
    """Download pinned files into a revision-encoded directory."""

    if include_videos and metadata_only:
        raise ValueError("metadata_only and include_videos cannot both be true")
    if include_videos and len(suites) != 1:
        raise ValueError("video download requires exactly one --suite")

    root = dataset_revision_root(datasets_dir, repo_id, revision)
    existing = _existing_revision(root)
    if existing and existing != revision:
        raise FileExistsError(
            f"{root} already stores revision {existing}; refusing to overwrite "
            f"with {revision}"
        )
    root.mkdir(parents=True, exist_ok=True)

    allow_patterns = metadata_allow_patterns(*suites)
    ignore_patterns = list(VIDEO_IGNORE_PATTERNS)
    if include_actions:
        allow_patterns.extend(f"{suite}/data/**" for suite in suites)
    if include_videos:
        allow_patterns = [f"{suites[0]}/**"]
        ignore_patterns = ["**/*.swp", "**/Untitled"]

    _snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    files = _file_inventory(root)
    video_files = [item for item in files if item["path"].endswith(".mp4")]
    if metadata_only and video_files:
        raise RuntimeError(
            f"metadata-only download produced {len(video_files)} video files"
        )

    required_suffixes = []
    for suite in suites:
        required_suffixes.extend(
            [
                f"{suite}/meta/info.json",
                f"{suite}/meta/stats.json",
                f"{suite}/meta/tasks.parquet",
            ]
        )
    missing = [name for name in required_suffixes if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"download incomplete, missing {missing}")
    if include_actions:
        missing_data = [
            suite
            for suite in suites
            if not any((root / suite / "data").rglob("*.parquet"))
        ]
        if missing_data:
            raise FileNotFoundError(
                f"action download incomplete, missing data parquet for {missing_data}"
            )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_repo_id": repo_id,
        "dataset_revision": revision,
        "suites": list(suites),
        "metadata_only": metadata_only,
        "include_videos": include_videos,
        "include_actions": include_actions,
        "local_root": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    manifest_path = root / MANIFEST_NAME
    atomic_write_json(manifest_path, manifest, overwrite=manifest_path.exists())
    revision_path = root / "REVISION"
    revision_path.write_text(f"{revision}\n", encoding="utf-8")
    return {
        "local_root": str(root),
        "dataset_revision": revision,
        "file_count": len(files),
        "total_bytes": manifest["total_bytes"],
        "metadata_only": metadata_only,
        "notes": redact_text("download completed"),
    }
