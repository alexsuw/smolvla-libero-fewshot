"""JSONL event log with crash-tolerant append."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from vla_fewshot.logging.csv_logger import repair_trailing_line
from vla_fewshot.reproducibility import redact_text


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_trailing_line(self.path)
        if not self.path.exists():
            self.path.touch()

    def emit(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        record = {
            "wall_time_utc": datetime.now(UTC).isoformat(),
            "event": event,
            **dict(payload or {}),
        }
        line = redact_text(json.dumps(record, sort_keys=True, ensure_ascii=False))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
