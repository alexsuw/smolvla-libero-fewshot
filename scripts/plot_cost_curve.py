"""Build observed cost curves with uncertainty from results_long.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_fewshot.reporting.plots import write_cost_curve_svg, write_language_control_svg
from vla_fewshot.reporting.constants import is_language_control_protocol
from vla_fewshot.reporting.tables import records_from_long


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        type=Path,
        default=Path("report/tables/results_long.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/figures"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.long.is_file():
        print(f"missing results_long.csv: {args.long}")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    macro = write_cost_curve_svg(args.long, args.output_dir / "cost_curve_macro.svg")
    write_cost_curve_svg(
        args.long,
        args.output_dir / "cost_curve_by_task.svg",
        title="Cost curve (task-level rates, same x ticks)",
    )
    records = records_from_long(args.long)
    language = [
        item
        for item in records
        if is_language_control_protocol(str(item.get("protocol_id") or ""))
    ]
    pairs: dict[str, dict[str, list[int]]] = {}
    for record in language:
        task = str(record["task_slug"])
        cond = str(record.get("instruction_condition") or "correct")
        pairs.setdefault(task, {"correct": [], "wrong": []})
        pairs[task].setdefault(cond, [])
        pairs[task][cond].append(int(record["success"]))
    bars = []
    for task, values in sorted(pairs.items()):
        correct = values.get("correct") or [0]
        wrong = values.get("wrong") or [0]
        bars.append(
            (
                task,
                sum(correct) / len(correct),
                sum(wrong) / len(wrong),
            )
        )
    if bars:
        write_language_control_svg(bars, args.output_dir / "language_control.svg")
    print(f"wrote {macro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
