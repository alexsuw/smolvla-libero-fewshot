"""Verify checkpoint completeness, checksums, and fresh-instance load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.training.checkpoint import CheckpointError, verify_checkpoint_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = Path(args.checkpoint)
    try:
        report = verify_checkpoint_dir(directory)
    except (CheckpointError, FileNotFoundError, ValueError) as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
