"""Inventory safe retention candidates; never delete by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.storage.retention import inventory_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly opt in; deletion is still refused.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inventory_checkpoints(args.root)
    if args.execute:
        report["execute_requested"] = True
        report["deleted"] = 0
        report["note"] = (
            "prune --execute still refuses deletion; inventory only until a "
            "separate retention workflow is reviewed"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
