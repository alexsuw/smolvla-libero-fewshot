"""Crash-tolerant rollouts.jsonl with unique-key resume and conflict refusal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from vla_fewshot.logging.csv_logger import repair_trailing_line
from vla_fewshot.evaluation.protocol import (
    ProtocolError,
    UNIQUE_KEY_FIELDS,
    rollout_key_from_record,
)
from vla_fewshot.reproducibility import redact_text

CONFLICT_FIELDS = (
    "success",
    "terminated",
    "truncated",
    "episode_length",
    "instruction_text_used",
    "checkpoint_sha256",
    "initial_state_fingerprint",
    "protocol_id",
    "task_slug",
    "eval_seed",
    "instruction_condition",
)


class DuplicateConflictError(ProtocolError):
    """Same unique key already exists with a different outcome."""


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ProtocolError(f"{path}:{line_no} is not a JSON object")
        records.append(payload)
    return records


class RolloutStore:
    """Append-only JSONL. Resume skips identical keys; conflicts fail closed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_trailing_line(self.path)
        if not self.path.exists():
            self.path.touch()
        self._records: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record in _parse_jsonl(self.path):
            key = rollout_key_from_record(record)
            if key in self._records:
                raise DuplicateConflictError(
                    f"rollouts.jsonl already contains a duplicate unique key: {key}"
                )
            self._records[key] = record

    def __len__(self) -> int:
        return len(self._records)

    def completed_keys(self) -> set[tuple[Any, ...]]:
        return set(self._records)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        return self._records.get(key)

    def append(self, record: Mapping[str, Any]) -> str:
        """Write one record. Returns 'written' or 'skipped'."""

        payload = dict(record)
        missing = [field for field in UNIQUE_KEY_FIELDS if field not in payload]
        if missing:
            raise ProtocolError(f"rollout record missing unique-key fields: {missing}")
        key = rollout_key_from_record(payload)
        existing = self._records.get(key)
        if existing is not None:
            if _conflict(existing, payload):
                raise DuplicateConflictError(
                    "refusing conflicting duplicate rollout "
                    f"key={key} existing={ {f: existing.get(f) for f in CONFLICT_FIELDS} }"
                )
            return "skipped"
        line = redact_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[key] = payload
        return "written"


def _conflict(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> bool:
    for field in CONFLICT_FIELDS:
        if existing.get(field) != incoming.get(field):
            return True
    return False


def remaining_keys(
    planned: Iterable[tuple[Any, ...]],
    completed: Iterable[tuple[Any, ...]],
) -> list[tuple[Any, ...]]:
    done = set(completed)
    return [key for key in planned if key not in done]
