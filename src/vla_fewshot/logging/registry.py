"""Deterministic registry.csv generated from immutable run manifests."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

from vla_fewshot.logging.manifest import json_load
from vla_fewshot.reproducibility import atomic_write_text
from vla_fewshot.storage.layout import MANIFEST_NAME

REGISTRY_COLUMNS = (
    "run_id",
    "stage",
    "method",
    "status",
    "manifest_path",
    "git_commit",
    "train_seed",
    "final_checkpoint_uri",
)


def discover_manifests(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        path
        for path in runs_root.rglob(MANIFEST_NAME)
        if path.is_file()
        and path.parent.name != "checkpoints"
        and "backup" not in path.parts
    )


def manifest_to_row(path: Path) -> dict[str, Any]:
    payload = json_load(path)
    return {
        "run_id": payload.get("run_id", ""),
        "stage": payload.get("stage", ""),
        "method": payload.get("method", ""),
        "status": payload.get("status", ""),
        "manifest_path": str(path),
        "git_commit": payload.get("git_commit") or "",
        "train_seed": payload.get("train_seed", ""),
        "final_checkpoint_uri": payload.get("final_checkpoint_uri") or "",
    }


def build_registry(runs_root: Path) -> list[dict[str, Any]]:
    rows = [manifest_to_row(path) for path in discover_manifests(runs_root)]
    rows.sort(key=lambda row: str(row["run_id"]))
    return rows


def write_registry_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in REGISTRY_COLUMNS})
    atomic_write_text(path, stream.getvalue(), overwrite=True)
