"""Export language-control JSONL cells and paired divergence."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.evaluation.metrics import wilson_interval
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.reporting.constants import LONG_COLUMNS, TARGET_TASK_SLUGS

LANGUAGE_CONTROL_EXPORT_COLUMNS = LONG_COLUMNS + (
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
    "init_state_id",
    "inference_seed",
    "normalization_suite",
    "normalization_stats_sha256",
    "initial_state_fingerprint",
)

PAIR_COLUMNS = (
    "task_slug",
    "eval_seed",
    "checkpoint_sha256",
    "fingerprint",
    "correct_success",
    "wrong_success",
    "action_l2_divergence",
    "action_cosine_divergence",
    "correct_video_uri",
    "wrong_video_uri",
)


def export_language_control_report(
    eval_root: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    eval_root = eval_root.resolve()
    output_dir = (output_dir or (eval_root / "report")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_language_control_records(eval_root)
    pairs = load_language_pairs(eval_root)
    cells = summarize_language_control_cells(records)
    long_path = output_dir / "results_long.csv"
    cell_path = output_dir / "summary_by_task_condition.csv"
    pair_path = output_dir / "pairs.csv"
    markdown_path = output_dir / "summary.md"
    _write_csv(long_path, records, fieldnames=list(LANGUAGE_CONTROL_EXPORT_COLUMNS))
    _write_csv(
        cell_path,
        cells,
        fieldnames=[
            "task_slug",
            "instruction_condition",
            "successes",
            "n",
            "success_rate",
            "wilson_low",
            "wilson_high",
            "checkpoint_sha256",
        ],
    )
    _write_csv(pair_path, pairs, fieldnames=list(PAIR_COLUMNS))
    markdown_path.write_text(
        render_language_control_markdown(
            eval_root=eval_root, records=records, cells=cells, pairs=pairs
        ),
        encoding="utf-8",
    )
    return {
        "results_long": long_path,
        "summary_by_task_condition": cell_path,
        "pairs": pair_path,
        "summary_markdown": markdown_path,
    }


def load_language_control_records(eval_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return records
    for jsonl in sorted(eval_root.rglob("rollouts.jsonl")):
        if "report" in jsonl.parts:
            continue
        for record in RolloutStore(jsonl).records():
            if str(record.get("stage")) != "language_control":
                continue
            row = {column: record.get(column) for column in LANGUAGE_CONTROL_EXPORT_COLUMNS}
            records.append(row)
    records.sort(
        key=lambda item: (
            str(item.get("task_slug") or ""),
            int(item.get("eval_seed") or 0),
            str(item.get("instruction_condition") or ""),
        )
    )
    return records


def load_language_pairs(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return rows
    for path in sorted(eval_root.rglob("language_pairs.json")):
        if "report" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for pair in payload.get("pairs") or []:
            rows.append({column: pair.get(column) for column in PAIR_COLUMNS})
    rows.sort(
        key=lambda item: (
            str(item.get("task_slug") or ""),
            int(item.get("eval_seed") or 0),
        )
    )
    return rows


def summarize_language_control_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("task_slug") or ""),
            str(record.get("instruction_condition") or ""),
        )
        grouped.setdefault(key, []).append(record)
    rows: list[dict[str, Any]] = []
    slugs = [slug for slug in TARGET_TASK_SLUGS if any(slug == key[0] for key in grouped)]
    slugs.extend(slug for slug in sorted({key[0] for key in grouped}) if slug not in slugs)
    for slug in slugs:
        for condition in ("correct", "wrong"):
            items = grouped.get((slug, condition)) or []
            if not items:
                continue
            n = len(items)
            successes = sum(int(item.get("success") or 0) for item in items)
            rate, low, high = wilson_interval(successes, n)
            hashes = {str(item.get("checkpoint_sha256") or "") for item in items}
            rows.append(
                {
                    "task_slug": slug,
                    "instruction_condition": condition,
                    "successes": successes,
                    "n": n,
                    "success_rate": round(rate, 6),
                    "wilson_low": round(low, 6),
                    "wilson_high": round(high, 6),
                    "checkpoint_sha256": hashes.pop()
                    if len(hashes) == 1
                    else ",".join(sorted(hashes)),
                }
            )
    return rows


def render_language_control_markdown(
    *,
    eval_root: Path,
    records: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> str:
    generated = datetime.now(UTC).isoformat()
    protocols = sorted({str(row.get("protocol_id") or "") for row in records})
    normalization_suites = sorted(
        {str(row.get("normalization_suite") or "") for row in records}
    )
    lines = [
        "# Language-control export",
        "",
        f"- generated_at_utc: `{generated}`",
        f"- eval_root: `{eval_root}`",
        f"- rollouts: **{len(records)}**",
        f"- pairs: **{len(pairs)}**",
        f"- protocol: `{','.join(protocols)}`, n_demos=0, frozen seen checkpoint",
        f"- normalization_suite: `{','.join(normalization_suites)}`",
        "",
        "## Per-task condition success",
        "",
        "| task_slug | condition | successes | n | success_rate | wilson_95 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in cells:
        lines.append(
            f"| {row['task_slug']} | {row['instruction_condition']} | "
            f"{row['successes']}/{row['n']} | {row['n']} | "
            f"{float(row['success_rate']):.3f} | "
            f"[{float(row['wilson_low']):.3f}, {float(row['wilson_high']):.3f}] |"
        )
    if not cells:
        lines.append("|  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Pair divergence",
            "",
            "| task_slug | n_pairs | mean L2 | mean cosine gap | matching fingerprints |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    by_task: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_task.setdefault(str(pair.get("task_slug") or ""), []).append(pair)
    for slug in [slug for slug in TARGET_TASK_SLUGS if slug in by_task]:
        items = by_task[slug]
        l2 = sum(float(item["action_l2_divergence"]) for item in items) / len(items)
        cosine = sum(float(item["action_cosine_divergence"]) for item in items) / len(items)
        lines.append(
            f"| {slug} | {len(items)} | {l2:.4f} | {cosine:.4f} | {len(items)}/{len(items)} |"
        )
    if not pairs:
        lines.append("|  |  |  |  |  |")
    checkpoint = cells[0]["checkpoint_sha256"] if cells else ""
    lines.extend(["", f"- checkpoint_sha256: `{checkpoint}`", ""])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
