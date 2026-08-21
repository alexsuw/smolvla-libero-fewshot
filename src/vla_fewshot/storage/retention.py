"""Inventory-only retention. Deletion is never performed by this module."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.storage.layout import CHECKPOINT_COMPLETED_NAME


def inventory_checkpoints(root: Path) -> dict[str, object]:
    """List checkpoint directories under a run or runs root. Never delete."""

    candidates: list[dict[str, object]] = []
    if root.exists():
        for completed in sorted(root.rglob(CHECKPOINT_COMPLETED_NAME)):
            directory = completed.parent
            candidates.append(
                {
                    "path": str(directory),
                    "completed": True,
                    "backed_up": False,
                    "referenced": True,
                    "action": "keep",
                    "reason": "unbacked or referenced checkpoints are never touched",
                }
            )
        for tmp_dir in sorted(root.rglob("step_*.tmp-*")):
            if tmp_dir.is_dir():
                candidates.append(
                    {
                        "path": str(tmp_dir),
                        "completed": False,
                        "backed_up": False,
                        "referenced": False,
                        "action": "keep",
                        "reason": "incomplete .tmp directories stay as forensic artifacts",
                    }
                )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "delete_enabled": False,
        "candidates": candidates,
        "note": "prune is inventory-only; no deletions are performed",
    }
