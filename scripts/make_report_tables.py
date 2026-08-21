"""Build deterministic final report tables from results_long.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.reporting.bundle import write_report_bundle
from vla_fewshot.reporting.tables import write_report_tables


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
        default=Path("report/tables"),
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Also checksum a tar.gz of report markdown, tables, and figures.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.long.is_file():
        print(f"missing results_long.csv: {args.long}")
        return 1
    paths = write_report_tables(args.long, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    if args.bundle:
        bundle = write_report_bundle(args.report_dir)
        print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
