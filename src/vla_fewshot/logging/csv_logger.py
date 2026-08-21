"""Crash-tolerant CSV metrics logger."""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

METRICS_COLUMNS = (
    "wall_time_utc",
    "elapsed_seconds",
    "global_step",
    "samples_seen",
    "epoch_fraction",
    "loss",
    "learning_rate",
    "grad_norm",
    "optimizer_step_skipped",
    "samples_per_second",
    "data_time_seconds",
    "step_time_seconds",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
)


def repair_trailing_line(path: Path) -> bool:
    """Drop a malformed unterminated last line. Returns True if truncated."""

    if not path.exists():
        return False
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return False
    last_newline = payload.rfind(b"\n")
    repaired = payload[: last_newline + 1] if last_newline >= 0 else b""
    with path.open("wb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())
    return True


class CsvMetricsLogger:
    """Append-only metrics.csv with a single header and fsync."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_trailing_line(self.path)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(METRICS_COLUMNS)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            header = self.path.read_text(encoding="utf-8").splitlines()[0]
            expected = ",".join(METRICS_COLUMNS)
            if header != expected:
                raise ValueError(
                    f"metrics.csv header mismatch: {header!r} != {expected!r}"
                )

    def append(self, row: Mapping[str, object]) -> None:
        missing = [column for column in METRICS_COLUMNS if column not in row]
        if missing:
            raise KeyError(f"metrics row missing columns: {missing}")
        values = [row[column] for column in METRICS_COLUMNS]
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(values)
            handle.flush()
            os.fsync(handle.fileno())

    def row_count(self) -> int:
        text = self.path.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        return max(0, len(lines) - 1)


def metrics_row(
    *,
    elapsed_seconds: float,
    global_step: int,
    samples_seen: int,
    epoch_fraction: float,
    loss: float,
    learning_rate: float,
    grad_norm: float,
    optimizer_step_skipped: int = 0,
    samples_per_second: float = 0.0,
    data_time_seconds: float = 0.0,
    step_time_seconds: float = 0.0,
    gpu_memory_allocated_mb: float = 0.0,
    gpu_memory_reserved_mb: float = 0.0,
    wall_time_utc: str | None = None,
) -> dict[str, object]:
    return {
        "wall_time_utc": wall_time_utc or datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "epoch_fraction": epoch_fraction,
        "loss": loss,
        "learning_rate": learning_rate,
        "grad_norm": grad_norm,
        "optimizer_step_skipped": optimizer_step_skipped,
        "samples_per_second": samples_per_second,
        "data_time_seconds": data_time_seconds,
        "step_time_seconds": step_time_seconds,
        "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
        "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
    }


def read_metric_column(path: Path, column: str) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row[column]) for row in rows]
