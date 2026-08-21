"""Checksummed, dry-run-first artifact synchronization. Never deletes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.storage.object_sync import execute_object_sync
from vla_fewshot.storage.sync import execute_local_mirror
from vla_fewshot.storage.uri import ObjectUriError


def _looks_like_uri(value: str) -> bool:
    return value.startswith(("s3://", "file://"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        required=True,
        help="Local directory, file:///absolute/prefix, or s3://bucket/prefix.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly opt in; default remains dry-run.",
    )
    parser.add_argument(
        "--backup-status",
        type=Path,
        help="Local backup_status.json path after a verified object upload.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = str(args.destination)
    try:
        if _looks_like_uri(destination):
            report = execute_object_sync(
                args.source,
                destination,
                execute=args.execute,
                backup_status_path=args.backup_status,
            )
        else:
            report = execute_local_mirror(
                args.source,
                Path(destination),
                execute=args.execute,
            )
    except (FileExistsError, ObjectUriError, FileNotFoundError) as error:
        print(str(error))
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2, sort_keys=True))
    print(
        f"dry_run={report['dry_run']} copied={report['copied']} "
        f"deleted={report['deleted']} conflicts={report.get('conflicts', 0)}"
    )
    if report.get("conflicts"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
