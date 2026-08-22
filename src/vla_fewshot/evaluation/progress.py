"""Percent and ETA for eval rollouts. Rate uses this session's finished items."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.reporting.constants import TARGET_TASK_SLUGS


@dataclass(frozen=True)
class EvalProgress:
    completed: int
    planned: int
    elapsed_seconds: float
    session_completed: int
    label: str | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.planned - self.completed)

    @property
    def fraction(self) -> float:
        if self.planned <= 0:
            return 0.0
        return min(1.0, self.completed / self.planned)

    @property
    def seconds_per_item(self) -> float | None:
        if self.session_completed <= 0 or self.elapsed_seconds <= 0:
            return None
        return self.elapsed_seconds / self.session_completed

    @property
    def eta_seconds(self) -> float | None:
        rate = self.seconds_per_item
        if rate is None:
            return None
        return rate * self.remaining


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_eval_progress(progress: EvalProgress) -> str:
    percent = 100.0 * progress.fraction
    rate = progress.seconds_per_item
    rate_text = f"{rate:.1f}s/rollout" if rate is not None else "rate n/a"
    eta = format_duration(progress.eta_seconds)
    label = f"  {progress.label}" if progress.label else ""
    return (
        f"[eval {progress.completed}/{progress.planned} {percent:5.1f}%] "
        f"elapsed {format_duration(progress.elapsed_seconds)}  "
        f"eta ~{eta}  {rate_text}{label}"
    )


def progress_from_counts(
    *,
    completed: int,
    planned: int,
    elapsed_seconds: float,
    session_completed: int,
    label: str | None = None,
) -> EvalProgress:
    return EvalProgress(
        completed=max(0, int(completed)),
        planned=max(0, int(planned)),
        elapsed_seconds=max(0.0, float(elapsed_seconds)),
        session_completed=max(0, int(session_completed)),
        label=label,
    )


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def load_eval_root_records(eval_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return records
    for jsonl in sorted(eval_root.rglob("rollouts.jsonl")):
        if "report" in jsonl.parts:
            continue
        records.extend(RolloutStore(jsonl).records())
    records.sort(key=lambda item: str(item.get("created_at_utc") or ""))
    return records


def _manifests(eval_root: Path) -> list[dict[str, Any]]:
    if not eval_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_root.rglob("manifest.json")):
        if "report" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def infer_planned_rollouts(eval_root: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    manifests = _manifests(eval_root)
    stage = None
    if rows:
        stage = str(rows[0].get("stage") or "")
    elif manifests:
        stage = str(manifests[0].get("stage") or "")
    per_cell = [int(item["planned"]) for item in manifests if item.get("planned") is not None]
    if stage in {"language_control", "zero_shot"}:
        if per_cell:
            return max(per_cell) * len(TARGET_TASK_SLUGS)
        seeds = 20
        conditions = 2 if stage == "language_control" else 1
        return len(TARGET_TASK_SLUGS) * seeds * conditions
    if per_cell:
        return sum(per_cell)
    return len(rows)


def progress_from_eval_root(
    eval_root: Path,
    *,
    planned: int | None = None,
    now: datetime | None = None,
) -> EvalProgress:
    records = load_eval_root_records(eval_root)
    planned_n = planned if planned is not None else infer_planned_rollouts(eval_root, records)
    if not records:
        return progress_from_counts(
            completed=0,
            planned=planned_n,
            elapsed_seconds=0.0,
            session_completed=0,
        )
    times = [_parse_utc(str(item.get("created_at_utc") or "")) for item in records]
    known = [item for item in times if item is not None]
    current = now or datetime.now(UTC)
    elapsed = (current - known[0]).total_seconds() if known else 0.0
    last = records[-1]
    label = (
        f"{last.get('task_slug')} seed={last.get('eval_seed')} "
        f"{last.get('instruction_condition')} ok={last.get('success')}"
    )
    return progress_from_counts(
        completed=len(records),
        planned=planned_n,
        elapsed_seconds=elapsed,
        session_completed=len(records),
        label=label,
    )
