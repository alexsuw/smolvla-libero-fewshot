"""Dry-run-first local artifact mirroring. Never deletes destination files."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.checksums import sha256_file
from vla_fewshot.storage.layout import BACKUP_STATUS_NAME


@dataclass(frozen=True)
class SyncPlanItem:
    relative: str
    source: str
    destination: str
    action: str
    sha256: str
    size_bytes: int


def _iter_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def plan_local_mirror(source: Path, destination: Path) -> list[SyncPlanItem]:
    """Describe a non-destructive copy from source into destination."""

    if not source.is_dir():
        raise FileNotFoundError(f"sync source is not a directory: {source}")
    items: list[SyncPlanItem] = []
    for path in _iter_files(source):
        relative = path.relative_to(source).as_posix()
        target = destination / relative
        digest = sha256_file(path)
        size = path.stat().st_size
        if not target.exists():
            action = "copy"
        elif sha256_file(target) == digest:
            action = "skip_identical"
        else:
            action = "conflict"
        items.append(
            SyncPlanItem(
                relative=relative,
                source=str(path),
                destination=str(target),
                action=action,
                sha256=digest,
                size_bytes=size,
            )
        )
    return items


def execute_local_mirror(
    source: Path,
    destination: Path,
    *,
    execute: bool = False,
) -> dict[str, object]:
    """Copy missing files. Refuse conflicting overwrites. Never delete."""

    plan = plan_local_mirror(source, destination)
    conflicts = [item for item in plan if item.action == "conflict"]
    if conflicts:
        names = ", ".join(item.relative for item in conflicts)
        raise FileExistsError(
            f"refusing to overwrite different destination files: {names}"
        )
    copied = 0
    skipped = 0
    if execute:
        destination.mkdir(parents=True, exist_ok=True)
        for item in plan:
            if item.action == "skip_identical":
                skipped += 1
                continue
            target = Path(item.destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = Path(item.source).read_bytes()
            temporary = target.with_name(f".{target.name}.tmp-sync")
            try:
                with temporary.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            copied += 1
    else:
        copied = sum(1 for item in plan if item.action == "copy")
        skipped = sum(1 for item in plan if item.action == "skip_identical")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dry_run": not execute,
        "source": str(source),
        "destination": str(destination),
        "copied": copied,
        "skipped_identical": skipped,
        "conflicts": 0,
        "deleted": 0,
        "items": [asdict(item) for item in plan],
    }
    if execute:
        atomic_write_json(
            destination / BACKUP_STATUS_NAME,
            {
                **{key: value for key, value in report.items() if key != "items"},
                "verified": True,
                "item_count": len(plan),
            },
            overwrite=True,
        )
    return report
