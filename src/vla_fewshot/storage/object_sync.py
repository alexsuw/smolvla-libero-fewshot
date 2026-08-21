"""Checksummed object-storage sync. Dry-run by default. Never deletes."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.reproducibility import atomic_write_json, redact_text
from vla_fewshot.storage.checksums import sha256_bytes, sha256_file
from vla_fewshot.storage.layout import BACKUP_STATUS_NAME
from vla_fewshot.storage.object_store import ObjectStore, ObjectStoreError, open_object_store
from vla_fewshot.storage.uri import join_key, parse_object_uri

COMPLETED_NAME = "COMPLETED.json"
TMP_PREFIX = "_tmp"


@dataclass(frozen=True)
class ObjectPlanItem:
    relative: str
    source: str
    object_key: str
    tmp_key: str
    action: str
    sha256: str
    size_bytes: int


def _iter_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def _destination_action(store: ObjectStore, key: str, digest: str) -> str:
    if not store.exists(key):
        return "upload"
    remote = store.get_bytes(key)
    if sha256_bytes(remote) == digest:
        return "skip_identical"
    return "conflict"


def plan_object_sync(
    source: Path,
    store: ObjectStore,
    *,
    sync_id: str | None = None,
) -> list[ObjectPlanItem]:
    """Describe a non-destructive upload. Conflicts are reported, not overwritten."""

    if not source.is_dir():
        raise FileNotFoundError(f"sync source is not a directory: {source}")
    token = sync_id or uuid.uuid4().hex
    items: list[ObjectPlanItem] = []
    for path in _iter_files(source):
        relative = path.relative_to(source).as_posix()
        if Path(relative).name == COMPLETED_NAME:
            continue
        digest = sha256_file(path)
        items.append(
            ObjectPlanItem(
                relative=relative,
                source=str(path),
                object_key=relative,
                tmp_key=join_key(TMP_PREFIX, token, relative),
                action=_destination_action(store, relative, digest),
                sha256=digest,
                size_bytes=path.stat().st_size,
            )
        )
    return items


def plan_object_sync_offline(source: Path) -> list[ObjectPlanItem]:
    """Dry-run plan when the remote backend is not opened (s3 without boto3)."""

    if not source.is_dir():
        raise FileNotFoundError(f"sync source is not a directory: {source}")
    items: list[ObjectPlanItem] = []
    for path in _iter_files(source):
        relative = path.relative_to(source).as_posix()
        if Path(relative).name == COMPLETED_NAME:
            continue
        items.append(
            ObjectPlanItem(
                relative=relative,
                source=str(path),
                object_key=relative,
                tmp_key=join_key(TMP_PREFIX, "dry-run", relative),
                action="upload",
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return items


def execute_object_sync(
    source: Path,
    uri: str,
    *,
    execute: bool = False,
    backup_status_path: Path | None = None,
) -> dict[str, Any]:
    """Upload through a temporary prefix, verify checksums, then publish.

    Default is dry-run. Existing destination objects with a different SHA-256
    are refused. Temporary prefixes are left in place. Nothing is deleted.
    """

    location = parse_object_uri(uri)
    store: ObjectStore | None = None
    if execute or location.scheme == "file":
        store = open_object_store(uri)
        items = plan_object_sync(source, store)
    else:
        items = plan_object_sync_offline(source)

    conflicts = [item for item in items if item.action == "conflict"]
    if execute and conflicts:
        names = ", ".join(item.relative for item in conflicts)
        raise FileExistsError(
            f"refusing to overwrite different destination objects: {names}"
        )

    copied = 0
    skipped = 0
    if execute:
        assert store is not None
        for item in items:
            if item.action == "skip_identical":
                skipped += 1
                continue
            payload = Path(item.source).read_bytes()
            store.put_bytes(item.tmp_key, payload)
            remote = store.get_bytes(item.tmp_key)
            if sha256_bytes(remote) != item.sha256 or len(remote) != item.size_bytes:
                raise ObjectStoreError(
                    f"temporary object checksum mismatch for {item.relative}"
                )
            store.put_bytes(item.object_key, payload)
            published = store.get_bytes(item.object_key)
            if sha256_bytes(published) != item.sha256:
                raise ObjectStoreError(
                    f"published object checksum mismatch for {item.relative}"
                )
            copied += 1
        completed = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": str(source),
            "object_uri": redact_text(uri),
            "item_count": len(items),
            "checksums": {item.relative: item.sha256 for item in items},
            "sizes": {item.relative: item.size_bytes for item in items},
            "deleted": 0,
        }
        store.put_bytes(
            COMPLETED_NAME,
            json.dumps(completed, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
    else:
        copied = sum(1 for item in items if item.action == "upload")
        skipped = sum(1 for item in items if item.action == "skip_identical")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dry_run": not execute,
        "source": str(source),
        "destination": redact_text(uri),
        "scheme": location.scheme,
        "copied": copied,
        "skipped_identical": skipped,
        "conflicts": len(conflicts),
        "deleted": 0,
        "verified": bool(execute),
        "pid": os.getpid(),
        "items": [asdict(item) for item in items],
    }
    if execute:
        status_path = backup_status_path or (source / BACKUP_STATUS_NAME)
        atomic_write_json(
            status_path,
            {
                **{key: value for key, value in report.items() if key != "items"},
                "completed_marker": COMPLETED_NAME,
                "item_count": len(items),
            },
            overwrite=True,
        )
    return report


def verify_completed_backup(
    store: ObjectStore,
    *,
    source: Path | None = None,
) -> dict[str, Any]:
    """Re-download COMPLETED.json and confirm every listed checksum."""

    if not store.exists(COMPLETED_NAME):
        raise ObjectStoreError("remote COMPLETED.json is missing")
    completed = json.loads(store.get_bytes(COMPLETED_NAME).decode("utf-8"))
    checksums = completed["checksums"]
    mismatches: list[str] = []
    for relative, expected in checksums.items():
        payload = store.get_bytes(relative)
        if sha256_bytes(payload) != expected:
            mismatches.append(relative)
        if source is not None:
            local = source / relative
            if local.is_file() and sha256_file(local) != expected:
                mismatches.append(f"local:{relative}")
    if mismatches:
        raise ObjectStoreError(f"backup checksum mismatch: {mismatches}")
    return {
        "verified": True,
        "item_count": len(checksums),
        "completed_created_at_utc": completed.get("created_at_utc"),
    }
