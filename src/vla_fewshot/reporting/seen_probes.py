"""Export every seen-probe JSONL row for the paper, including leftover cells."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.evaluation.metrics import wilson_interval
from vla_fewshot.evaluation.select import STAGED_PROBE_STEPS, parse_step_directory_name
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.reporting.constants import PROBE_EXPORT_COLUMNS
from vla_fewshot.storage.layout import step_directory_name


def export_seen_probe_report(
    probe_root: Path,
    *,
    output_dir: Path | None = None,
    pool_steps: tuple[int, ...] = STAGED_PROBE_STEPS,
) -> dict[str, Path]:
    """Write CSV + markdown under `probe_root/report/` without deleting videos/JSONL."""

    probe_root = probe_root.resolve()
    output_dir = (output_dir or (probe_root / "report")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = set(pool_steps)
    records = load_probe_records(probe_root, pool_steps=pool)
    cells = summarize_probe_cells(records)
    long_path = output_dir / "results_long.csv"
    cell_path = output_dir / "summary_by_cell.csv"
    markdown_path = output_dir / "summary.md"
    rollouts_md = output_dir / "rollouts.md"
    write_csv(long_path, records, fieldnames=list(PROBE_EXPORT_COLUMNS))
    write_csv(
        cell_path,
        cells,
        fieldnames=[
            "step",
            "in_selection_pool",
            "task_slug",
            "successes",
            "n",
            "success_rate",
            "wilson_low",
            "wilson_high",
        ],
    )
    markdown_path.write_text(
        render_summary_markdown(
            probe_root=probe_root,
            pool_steps=pool_steps,
            records=records,
            cells=cells,
        ),
        encoding="utf-8",
    )
    rollouts_md.write_text(render_rollouts_markdown(records), encoding="utf-8")
    return {
        "results_long": long_path,
        "summary_by_cell": cell_path,
        "summary_markdown": markdown_path,
        "rollouts_markdown": rollouts_md,
    }


def load_probe_records(
    probe_root: Path,
    *,
    pool_steps: set[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not probe_root.is_dir():
        return records
    for jsonl in sorted(probe_root.rglob("rollouts.jsonl")):
        if "report" in jsonl.parts:
            continue
        step = _step_from_jsonl(jsonl, probe_root)
        if step is None:
            continue
        in_pool = step in pool_steps
        for record in RolloutStore(jsonl).records():
            row = {column: record.get(column) for column in PROBE_EXPORT_COLUMNS}
            row["step"] = step
            row["in_selection_pool"] = in_pool
            records.append(row)
    records.sort(
        key=lambda item: (
            int(item["step"]),
            str(item.get("task_slug") or ""),
            int(item.get("eval_seed") or 0),
            str(item.get("instruction_condition") or ""),
        )
    )
    return records


def summarize_probe_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                int(record["step"]),
                str(record.get("task_slug") or ""),
                bool(record.get("in_selection_pool")),
            )
        ].append(record)
    rows: list[dict[str, Any]] = []
    for (step, slug, in_pool), items in sorted(grouped.items()):
        n = len(items)
        successes = sum(int(item.get("success") or 0) for item in items)
        rate, low, high = wilson_interval(successes, n)
        rows.append(
            {
                "step": step,
                "in_selection_pool": in_pool,
                "task_slug": slug,
                "successes": successes,
                "n": n,
                "success_rate": round(rate, 6),
                "wilson_low": round(low, 6),
                "wilson_high": round(high, 6),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def render_summary_markdown(
    *,
    probe_root: Path,
    pool_steps: tuple[int, ...],
    records: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> str:
    generated = datetime.now(UTC).isoformat()
    pool_rows = [row for row in cells if row["in_selection_pool"]]
    leftover_rows = [row for row in cells if not row["in_selection_pool"]]
    staged = probe_root / "staged_selection.json"
    lines = [
        "# Seen-probe export",
        "",
        f"- generated_at_utc: `{generated}`",
        f"- probe_root: `{probe_root}`",
        f"- rollouts: **{len(records)}**",
        f"- selection pool steps: {', '.join(str(step) for step in pool_steps)}",
        "- leftover interrupt/5k cells are kept and listed below; they are not used to freeze the seen checkpoint",
        "- paste `summary_by_cell.csv` / the tables here; per-rollout rows are in `results_long.csv` and `rollouts.md`",
        "",
        "## Selection-pool cells",
        "",
        _cell_table(pool_rows),
        "",
        "## Leftover cells (kept, ignored for freeze)",
        "",
        _cell_table(leftover_rows) if leftover_rows else "_none_",
        "",
    ]
    if staged.is_file():
        payload = json.loads(staged.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Staged selection JSON",
                "",
                "```json",
                json.dumps(payload, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def render_rollouts_markdown(records: list[dict[str, Any]]) -> str:
    headers = [
        "step",
        "pool",
        "task_slug",
        "eval_seed",
        "success",
        "episode_length",
        "video_uri",
        "checkpoint_sha256",
    ]
    lines = [
        "# Seen-probe rollouts",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(record.get("step")),
                    "yes" if record.get("in_selection_pool") else "no",
                    _md(record.get("task_slug")),
                    _md(record.get("eval_seed")),
                    _md(record.get("success")),
                    _md(record.get("episode_length")),
                    _md(record.get("video_uri")),
                    _md(record.get("checkpoint_sha256")),
                ]
            )
            + " |"
        )
    if len(records) == 0:
        lines.append("|  |  |  |  |  |  |  |  |")
    return "\n".join(lines) + "\n"


def _cell_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_none_"
    headers = [
        "step",
        "task_slug",
        "successes",
        "n",
        "success_rate",
        "wilson_95",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    step_directory_name(int(row["step"])),
                    _md(row["task_slug"]),
                    f"{row['successes']}/{row['n']}",
                    _md(row["n"]),
                    f"{float(row['success_rate']):.3f}",
                    f"[{float(row['wilson_low']):.3f}, {float(row['wilson_high']):.3f}]",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _step_from_jsonl(jsonl: Path, probe_root: Path) -> int | None:
    try:
        relative = jsonl.relative_to(probe_root)
    except ValueError:
        return parse_step_directory_name(jsonl.parent.parent.name)
    for part in relative.parts:
        step = parse_step_directory_name(part)
        if step is not None:
            return step
    return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
