"""Validate and collect completed rollout records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.data.gates import maybe_assert_no_leakage
from vla_fewshot.data.leakage import LeakageError
from vla_fewshot.reporting.collect import IncompleteGridError, collect_rollouts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/eval"),
        help="Directory tree containing rollouts.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/tables"),
        help="Where to write results_long.csv.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if the expected final grid is missing cells.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        maybe_assert_no_leakage(
            config_path=args.config,
            splits_path=args.split,
            output_root=args.output_root,
            stage="report",
        )
    except LeakageError as error:
        print(str(error))
        return 1
    try:
        report = collect_rollouts(
            args.runs_root,
            output_dir=args.output_dir,
            allow_incomplete=not args.require_complete,
        )
    except IncompleteGridError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
