"""Deterministic CSV tables rebuilt from results_long.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from vla_fewshot.evaluation.metrics import wilson_interval
from vla_fewshot.reporting.collect import read_results_long, summarize_cells
from vla_fewshot.reporting.constants import COST_CURVE_N


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _as_int(value: Any) -> int:
    if value in (None, "", "None"):
        return 0
    return int(value)


def _as_success(value: Any) -> int:
    if value in (True, "True", "true", "1", 1):
        return 1
    return 0


def records_from_long(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_results_long(path):
        rows.append(
            {
                **row,
                "n_demos": _as_int(row.get("n_demos")),
                "eval_seed": _as_int(row.get("eval_seed")),
                "success": _as_success(row.get("success")),
                "train_seed": None
                if row.get("train_seed") in (None, "", "None")
                else int(row["train_seed"]),
            }
        )
    return rows


def write_report_tables(long_path: Path, tables_dir: Path) -> dict[str, Path]:
    records = records_from_long(long_path)
    summaries = summarize_cells(records)
    tables_dir.mkdir(parents=True, exist_ok=True)

    main_rows = _main_results(records)
    _write_csv(
        tables_dir / "main_results.csv",
        [
            "task_slug",
            "method",
            "n_demos",
            "mean_success",
            "wilson_ci_low",
            "wilson_ci_high",
            "n_rollouts",
            "n_successes",
        ],
        main_rows,
    )
    _write_csv(
        tables_dir / "per_seed_results.csv",
        [
            "method",
            "task_slug",
            "n_demos",
            "train_seed",
            "success_rate",
            "wilson_ci_low",
            "wilson_ci_high",
            "n_rollouts",
            "n_successes",
            "checkpoint_sha256",
            "protocol_id",
            "instruction_condition",
        ],
        summaries,
    )
    zero = [
        row
        for row in main_rows
        if int(row["n_demos"]) == 0 and row["method"] in {"zero_shot", "seen"}
    ]
    _write_csv(
        tables_dir / "zero_shot.csv",
        [
            "task_slug",
            "method",
            "n_demos",
            "mean_success",
            "wilson_ci_low",
            "wilson_ci_high",
            "n_rollouts",
            "n_successes",
        ],
        zero,
    )
    language = [
        item
        for item in summaries
        if item["protocol_id"] == "language_control_v1"
    ]
    _write_csv(
        tables_dir / "language_control.csv",
        [
            "method",
            "task_slug",
            "n_demos",
            "train_seed",
            "success_rate",
            "wilson_ci_low",
            "wilson_ci_high",
            "n_rollouts",
            "n_successes",
            "checkpoint_sha256",
            "protocol_id",
            "instruction_condition",
        ],
        language,
    )
    provenance = _checkpoint_rows(records)
    _write_csv(
        tables_dir / "checkpoint_provenance.csv",
        ["checkpoint_sha256", "methods", "n_rollouts"],
        provenance,
    )
    _write_csv(
        tables_dir / "compute.csv",
        [
            "method",
            "trainable_params",
            "total_params",
            "peak_vram_mb",
            "training_gpu_hours",
            "wall_time",
            "effective_batch",
            "steps",
        ],
        [],
    )
    return {
        "main_results": tables_dir / "main_results.csv",
        "per_seed_results": tables_dir / "per_seed_results.csv",
        "zero_shot": tables_dir / "zero_shot.csv",
        "language_control": tables_dir / "language_control.csv",
        "checkpoint_provenance": tables_dir / "checkpoint_provenance.csv",
        "compute": tables_dir / "compute.csv",
    }


def _main_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("protocol_id") == "language_control_v1":
            continue
        if record.get("instruction_condition") not in {None, "", "correct"}:
            continue
        key = (
            str(record.get("task_slug")),
            str(record.get("method")),
            int(record.get("n_demos") or 0),
        )
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for (task_slug, method, n_demos), cell in sorted(grouped.items()):
        successes = sum(int(item["success"]) for item in cell)
        n = len(cell)
        rate, low, high = wilson_interval(successes, n) if n else (0.0, 0.0, 0.0)
        rows.append(
            {
                "task_slug": task_slug,
                "method": method,
                "n_demos": n_demos,
                "mean_success": rate,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "n_rollouts": n,
                "n_successes": successes,
            }
        )
    return rows


def _checkpoint_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        digest = str(record.get("checkpoint_sha256") or "")
        if not digest:
            continue
        grouped[digest].add(str(record.get("method")))
        counts[digest] += 1
    rows = []
    for digest in sorted(grouped):
        rows.append(
            {
                "checkpoint_sha256": digest,
                "methods": ",".join(sorted(grouped[digest])),
                "n_rollouts": counts[digest],
            }
        )
    return rows


def cost_curve_points(records: list[dict[str, Any]]) -> dict[str, list[tuple[int, float, float, float]]]:
    """Macro mean across tasks for each method at N in {0,5,10,25}."""

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("protocol_id") == "language_control_v1":
            continue
        if record.get("instruction_condition") not in {None, "", "correct"}:
            continue
        key = (
            str(record.get("method")),
            str(record.get("task_slug")),
            int(record.get("n_demos") or 0),
        )
        grouped[key].append(record)

    task_rates: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (method, _task, n_demos), cell in grouped.items():
        if n_demos not in COST_CURVE_N:
            continue
        n = len(cell)
        if not n:
            continue
        task_rates[(method, n_demos)].append(
            sum(int(item["success"]) for item in cell) / n
        )
    points: dict[str, list[tuple[int, float, float, float]]] = {}
    for method in sorted({key[0] for key in task_rates}):
        series: list[tuple[int, float, float, float]] = []
        for n_demos in COST_CURVE_N:
            rates = task_rates.get((method, n_demos), [])
            if not rates:
                continue
            mean = sum(rates) / len(rates)
            series.append((n_demos, mean, min(rates), max(rates)))
        points[method] = series
    return points
