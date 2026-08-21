"""Verify a remote COMPLETED.json backup against checksums. Never deletes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.storage.object_store import ObjectStoreError, open_object_store
from vla_fewshot.storage.object_sync import verify_completed_backup
from vla_fewshot.storage.uri import ObjectUriError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-uri",
        required=True,
        help="file:///prefix or s3://bucket/prefix that already has COMPLETED.json.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional local directory to compare against the remote checksums.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = open_object_store(args.object_uri)
        report = verify_completed_backup(store, source=args.source)
    except (ObjectStoreError, ObjectUriError, FileNotFoundError) as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
