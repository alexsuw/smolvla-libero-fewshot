"""Export zero-shot JSONL cells for the report. Does not delete videos."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.evaluation.metrics import wilson_interval
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.reporting.constants import LONG_COLUMNS, TARGET_TASK_SLUGS

ZERO_SHOT_EXPORT_COLUMNS = LONG_COLUMNS + (
    "video_uri",
    "trace_uri",
    "episode_length",
    "wall_time_seconds",
    "task_text",
    "instruction_text_used",
    "failure_category",
    "created_at_utc",
    "checkpoint_uri",
    "rollout_index",
)


def export_zero_shot_report(
    eval_root: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    eval_root = eval_root.resolve()
    output_dir = (output_dir or (eval_root / "report")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_zero_shot_records(eval_root)
    cells = summarize_zero_shot_cells(records)
    long_path = output_dir / "results_long.csv"
    cell_path = output_dir / "summary_by_task.csv"
    markdown_path = output_dir / "summary.md"
    _write_csv(long_path, records, fieldnames=list(ZERO_SHOT_EXPORT_COLUMNS))
    _write_csv(
        cell_path,
        cells,
        fieldnames=[
            "task_slug",
            "successes",
            "n",
            "success_rate",
            "wilson_low",
            "wilson_high",
            "checkpoint_sha256",
        ],
    )
    markdown_path.write_text(
        render_zero_shot_markdown(eval_root=eval_root, records=records, cells=cells),
        encoding="utf-8",
    )
    return {
        "results_long": long_path,
        "summary_by_task": cell_path,
        "summary_markdown": markdown_path,
    }


def load_zero_shot_records(eval_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return records
    for jsonl in sorted(eval_root.rglob("rollouts.jsonl")):
        if "report" in jsonl.parts:
            continue
        for record in RolloutStore(jsonl).records():
            if str(record.get("stage")) != "zero_shot":
                continue
            row = {column: record.get(column) for column in ZERO_SHOT_EXPORT_COLUMNS}
            records.append(row)
    records.sort(
        key=lambda item: (
            str(item.get("task_slug") or ""),
            int(item.get("eval_seed") or 0),
        )
    )
    return records


def summarize_zero_shot_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        slug = str(record.get("task_slug") or "")
        by_task.setdefault(slug, []).append(record)
    rows: list[dict[str, Any]] = []
    slugs = [slug for slug in TARGET_TASK_SLUGS if slug in by_task]
    slugs.extend(slug for slug in sorted(by_task) if slug not in TARGET_TASK_SLUGS)
    for slug in slugs:
        items = by_task[slug]
        if not items:
            continue
        n = len(items)
        successes = sum(int(item.get("success") or 0) for item in items)
        rate, low, high = wilson_interval(successes, n)
        hashes = {str(item.get("checkpoint_sha256") or "") for item in items}
        rows.append(
            {
                "task_slug": slug,
                "successes": successes,
                "n": n,
                "success_rate": round(rate, 6),
                "wilson_low": round(low, 6),
                "wilson_high": round(high, 6),
                "checkpoint_sha256": hashes.pop() if len(hashes) == 1 else ",".join(sorted(hashes)),
            }
        )
    return rows


def render_zero_shot_markdown(
    *,
    eval_root: Path,
    records: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> str:
    generated = datetime.now(UTC).isoformat()
    total_n = sum(int(row["n"]) for row in cells)
    total_s = sum(int(row["successes"]) for row in cells)
    mean = (total_s / total_n) if total_n else 0.0
    lines = [
        "# Zero-shot export",
        "",
        f"- generated_at_utc: `{generated}`",
        f"- eval_root: `{eval_root}`",
        f"- rollouts: **{len(records)}**",
        f"- overall: **{total_s}/{total_n}** ({mean:.3f})",
        "- protocol: `final_v1`, n_demos=0, frozen seen checkpoint, no target training",
        "",
        "## Per-task success",
        "",
        "| task_slug | successes | n | success_rate | wilson_95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in cells:
        lines.append(
            f"| {row['task_slug']} | {row['successes']}/{row['n']} | {row['n']} | "
            f"{float(row['success_rate']):.3f} | "
            f"[{float(row['wilson_low']):.3f}, {float(row['wilson_high']):.3f}] |"
        )
    if not cells:
        lines.append("|  |  |  |  |  |")
    lines.extend(["", f"- checkpoint_sha256: `{cells[0]['checkpoint_sha256'] if cells else ''}`", ""])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
