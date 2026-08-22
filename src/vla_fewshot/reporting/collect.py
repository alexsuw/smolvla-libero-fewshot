"""Walk rollouts.jsonl trees, drop dev/static rows, write results_long.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from vla_fewshot.evaluation.metrics import cell_summary
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.reporting.constants import (
    COST_CURVE_N,
    EXCLUDED_PROTOCOL_IDS,
    EXCLUDED_PROTOCOL_PREFIXES,
    LONG_COLUMNS,
    REPORT_PROTOCOL_IDS,
    TARGET_METHODS,
    TARGET_TASK_SLUGS,
    TRAIN_SEEDS,
    ZERO_SHOT_PROTOCOL_ID,
)


def is_reportable_protocol(protocol_id: str) -> bool:
    if protocol_id in EXCLUDED_PROTOCOL_IDS:
        return False
    if protocol_id.startswith(EXCLUDED_PROTOCOL_PREFIXES):
        return False
    return protocol_id in REPORT_PROTOCOL_IDS


def find_rollout_files(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.rglob("rollouts.jsonl") if path.is_file())


def load_reportable_records(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in find_rollout_files(runs_root):
        for record in RolloutStore(path).records():
            protocol_id = str(record.get("protocol_id", ""))
            if not is_reportable_protocol(protocol_id):
                continue
            records.append(record)
    records.sort(
        key=lambda item: (
            str(item.get("method")),
            str(item.get("task_slug")),
            int(item.get("n_demos") or 0),
            str(item.get("train_seed")),
            int(item.get("eval_seed")),
            str(item.get("instruction_condition")),
            str(item.get("protocol_id")),
        )
    )
    return records


def records_to_long_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append({column: record.get(column) for column in LONG_COLUMNS})
    return rows


def write_results_long(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LONG_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_results_long(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_cells(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("method"),
            record.get("task_slug"),
            int(record.get("n_demos") or 0),
            record.get("train_seed"),
            record.get("instruction_condition"),
            record.get("protocol_id"),
            record.get("checkpoint_sha256"),
        )
        grouped[key].append(record)
    return dict(grouped)


def summarize_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for key, cell_records in sorted(group_cells(records).items(), key=lambda item: item[0]):
        method, task_slug, n_demos, train_seed, _condition, protocol_id, digest = key
        seed = None if train_seed in (None, "", "None") else int(train_seed)
        summary = cell_summary(
            method=str(method),
            task_slug=str(task_slug),
            n_demos=int(n_demos),
            train_seed=seed,
            records=cell_records,
            checkpoint_sha256=str(digest),
            protocol_id=str(protocol_id),
        )
        summary["instruction_condition"] = key[4]
        summaries.append(summary)
    return summaries


def expected_final_cells() -> list[dict[str, Any]]:
    """The baseline grid the spec requires before a complete report."""

    cells: list[dict[str, Any]] = []
    for task_slug in TARGET_TASK_SLUGS:
        cells.append(
            {
                "method": "zero_shot",
                "task_slug": task_slug,
                "n_demos": 0,
                "train_seed": None,
                "protocol_id": ZERO_SHOT_PROTOCOL_ID,
            }
        )
        for method in TARGET_METHODS:
            for n_demos in COST_CURVE_N[1:]:
                for seed in TRAIN_SEEDS:
                    cells.append(
                        {
                            "method": method,
                            "task_slug": task_slug,
                            "n_demos": n_demos,
                            "train_seed": seed,
                            "protocol_id": "final_v1",
                        }
                    )
    return cells


def missing_final_cells(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = {
        (
            item["method"],
            item["task_slug"],
            int(item["n_demos"]),
            item["train_seed"],
            item["protocol_id"],
        )
        for item in summaries
        if item.get("n_rollouts", 0) > 0
    }
    missing: list[dict[str, Any]] = []
    for cell in expected_final_cells():
        key = (
            cell["method"],
            cell["task_slug"],
            cell["n_demos"],
            cell["train_seed"],
            cell["protocol_id"],
        )
        if key not in present:
            missing.append(cell)
    return missing


def collect_rollouts(
    runs_root: Path,
    *,
    output_dir: Path,
    allow_incomplete: bool = True,
) -> dict[str, Any]:
    records = load_reportable_records(runs_root)
    rows = records_to_long_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "results_long.csv"
    write_results_long(long_path, rows)
    summaries = summarize_cells(records)
    missing = missing_final_cells(summaries)
    report = {
        "n_records": len(records),
        "n_cells": len(summaries),
        "n_missing_final_cells": len(missing),
        "complete": not missing,
        "results_long": str(long_path),
    }
    if missing and not allow_incomplete:
        raise IncompleteGridError(
            f"final grid is incomplete; missing {len(missing)} cells"
        )
    return report


class IncompleteGridError(RuntimeError):
    """Raised when required final-eval cells are absent."""
