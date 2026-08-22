"""Print seen-probe success from JSONL/summary.json. Does not start GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.calibration import load_calibration
from vla_fewshot.evaluation.select import (
    STAGE1_PROBE_ROLLOUTS,
    STAGE2_PROBE_ROLLOUTS,
    STAGED_PROBE_STEPS,
    parse_step_directory_name,
)
from vla_fewshot.storage.layout import step_directory_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument(
        "--pool-steps",
        default=",".join(str(step) for step in STAGED_PROBE_STEPS),
        help="Comma-separated steps in the selection pool. Others are leftover.",
    )
    return parser


def _cell_counts(jsonl: Path) -> tuple[int, int]:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows), sum(int(row.get("success") or 0) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    slugs = list(load_calibration().seen_probe_slugs)
    pool = {int(item.strip()) for item in args.pool_steps.split(",") if item.strip()}
    root: Path = args.probe_root
    print(f"probe_root={root}")
    print(
        f"selection pool: {sorted(pool)} | stage1={STAGE1_PROBE_ROLLOUTS} "
        f"seeds, stage2={STAGE2_PROBE_ROLLOUTS} seeds | tasks={slugs}"
    )
    print(
        f"planned new rollouts: stage1 {len(pool)*len(slugs)*STAGE1_PROBE_ROLLOUTS}, "
        f"stage2 {2*len(slugs)*(STAGE2_PROBE_ROLLOUTS-STAGE1_PROBE_ROLLOUTS)} extra"
    )
    if not root.is_dir():
        print("no probe root yet")
        return 0
    leftover = False
    for step_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        step = parse_step_directory_name(step_dir.name)
        if step is None:
            continue
        in_pool = step in pool
        tag = "POOL" if in_pool else "LEFTOVER"
        if not in_pool:
            leftover = True
        for slug in slugs:
            cell = step_dir / slug
            jsonl = cell / "rollouts.jsonl"
            summary = cell / "summary.json"
            if jsonl.is_file():
                n, s = _cell_counts(jsonl)
                extra = ""
                if summary.is_file():
                    payload = json.loads(summary.read_text(encoding="utf-8"))
                    extra = f" summary_rate={payload.get('success_rate')}"
                print(f"  [{tag}] {step_directory_name(step)}/{slug}: {s}/{n}{extra}")
            else:
                print(f"  [{tag}] {step_directory_name(step)}/{slug}: pending")
    ranking = root / "staged_selection.json"
    if ranking.is_file():
        print(f"staged_selection.json: {ranking}")
        print(ranking.read_text(encoding="utf-8"))
    elif leftover:
        print("leftover interrupt/5k cells are kept on disk and ignored for selection")
    print(
        "export every JSONL row (pool + leftover) for the report:\n"
        f"  uv run python scripts/export_seen_probe_report.py --probe-root {root}"
    )
    report_md = root / "report" / "summary.md"
    if report_md.is_file():
        print(f"existing report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
