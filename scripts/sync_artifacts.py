"""Checksummed, dry-run-first artifact synchronization. Never deletes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.storage.sync import execute_local_mirror


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly opt in; default remains dry-run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = execute_local_mirror(
            args.source,
            args.destination,
            execute=args.execute,
        )
    except FileExistsError as error:
        print(str(error))
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2, sort_keys=True))
    print(f"dry_run={report['dry_run']} copied={report['copied']} deleted={report['deleted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
